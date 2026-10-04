"""Multi-stream milestone 3 on the GPU: full GLM-5.3's segmented kernels (several streams' rows in one launch) at the
real model's shapes (22 heads a rank, 512-wide latents, 64 rotary dims, 32 x 128 indexer, top-2048) against the
single-stream kernels per segment on the stream's extent views, bit for bit; a CUDA graph of the selection and the
attention captured once and replayed on other tables (positions, bases, extents) equal to eager; and the segmented
layer (``full_seg.FullSegVerify``) against ``verify.FullBatchedVerify``'s per-segment loop on the tiny model with
every real kernel."""

import os

import pytest
import torch

from tensorfold.families.glm5_next.cuda import dsa_full as F
from tensorfold.families.glm5_next.cuda import kv8, latent
from tensorfold.families.glm5_next.cuda.full_seg import FullRows, RowsScratch

DEV = "cuda"
THETA = 8_000_000.0
K = 2048
H, LW, ROPE, IH, ID = 22, 512, 64, 32, 128

# (base, pos, rows, capacity): dense; crossing the dense limit; sparse in a 16k extent; sparse deep in a 64k one
SEGS = [(0, 700, 3, 8192), (8192, 2044, 6, 4096), (12288, 9000, 4, 16384), (28672, 40000, 3, 65536)]
SEGS_B = [(28672, 61000, 5, 65536), (0, 2047, 2, 8192), (12288, 300, 4, 16384), (8192, 3000, 5, 4096)]
P = 28672 + 65536


def spans(segs):
    at = 0
    for base, pos, n, cap in segs:
        yield at, base, pos, n, cap
        at += n


def tables(segs, indexed=True):
    m = FullRows(16, DEV)
    R = m.set(segs, dense_limit=K, indexed=indexed)
    torch.cuda.synchronize()
    return m, R


def data(seed, R):
    g = torch.Generator().manual_seed(seed)
    d = dict(
        q=torch.randn(R, H, 256, generator=g).to(torch.bfloat16),
        qi=torch.randn(R, IH * ID, generator=g).to(torch.bfloat16),
        wts=torch.randn(R, ID + IH, generator=g).to(torch.bfloat16)[:, ID:],
        keys=torch.randn(P, ID, generator=g).to(torch.bfloat16),
        qa=(0.3 * torch.randn(R, H, LW, generator=g)).to(torch.bfloat16),
        qp=(0.3 * torch.randn(R, H, ROPE, generator=g)).to(torch.bfloat16),
        kr=torch.randn(P, ROPE, generator=g).to(torch.bfloat16),
        lat=torch.randn(P, LW, generator=g).to(torch.bfloat16),
        dp=torch.randn(R, 300, generator=g).to(torch.bfloat16),
        kraw=torch.randn(R, ID + IH, generator=g).to(torch.bfloat16),
        lnw=(1 + 0.1 * torch.randn(ID, generator=g)).to(torch.bfloat16),
        lnb=(0.1 * torch.randn(ID, generator=g)).to(torch.bfloat16),
    )
    return {k: v.to(DEV) for k, v in d.items()}


def test_rope_and_writes_equal_each_segment():
    m, R = tables(SEGS)
    d = data(1, R)
    freq = F.inv_freq(ROPE, THETA, DEV)
    qp = torch.empty(R, H, ROPE, dtype=torch.bfloat16, device=DEV)
    F.q_rope_rows(d["q"], qp, m.pos, freq, 192)
    qi = d["qi"].clone()
    F.index_q_rope_rows(qi, IH, m.pos, freq)
    kr, ik = d["kr"].clone(), d["keys"].clone()
    kr1, ik1 = d["kr"].clone(), d["keys"].clone()
    F.k_rope_write_rows(d["dp"][:, 300 - ROPE:], kr, m.pos, m.base, freq)
    F.index_write_rows(d["kraw"][:, :ID], d["lnw"], d["lnb"], ik, m.pos, m.base, freq)
    lcs = {kind: (kv8.zeros(P, LW, kind, DEV), kv8.zeros(P, LW, kind, DEV)) for kind in kv8.KINDS}
    for kind, (a, _) in lcs.items():
        latent.latent_write_rows(d["lat"][:R], a, m.pos, m.base)
    for at, base, pos, n, cap in spans(SEGS):
        rs = slice(at, at + n)
        pd = torch.tensor([pos], dtype=torch.int32, device=DEV)
        want = torch.empty(n, H, ROPE, dtype=torch.bfloat16, device=DEV)
        F.q_rope(d["q"][rs], want, pd, freq, 192)
        assert torch.equal(qp[rs], want)
        wi = d["qi"][rs].clone()
        F.index_q_rope(wi, IH, pd, freq)
        assert torch.equal(qi[rs], wi)
        F.k_rope_write(d["dp"][rs, 300 - ROPE:], kr1[base:base + cap], pd, freq)
        F.index_write(d["kraw"][rs, :ID], d["lnw"], d["lnb"], ik1[base:base + cap], pd, freq)
        for kind, (_, b) in lcs.items():
            latent.latent_write(d["lat"][rs], b[base:base + cap], pd)
    assert torch.equal(kr, kr1) and torch.equal(ik, ik1)
    for kind, (a, b) in lcs.items():
        assert torch.equal(a, b), kind


def select_and_attend(m, R, d, lc, NT, sel, out, s, nch):
    F.select_tokens_rows(d["qi"], d["wts"], d["keys"], m.pos, m.base, m.cap, R, NT, sel.scores, sel.tokens,
                         sel.counts, k=K)
    latent.attention_rows(d["qa"], lc, d["kr"], d["qp"], m.pos, m.base, m.sparse, sel.tokens, sel.counts, s,
                          scale=192 ** -0.5, nch=nch, out=out)


def reference(segs, d, lc):
    """Per segment on its extent views: forward.dsa_full_rows' selection and attention calls."""
    outs, sels = [], []
    for at, base, pos, n, cap in spans(segs):
        rs = slice(at, at + n)
        pd = torch.tensor([pos], dtype=torch.int32, device=DEV)
        t1, c1 = F.select_tokens(d["qi"][rs].contiguous(), d["wts"][rs], d["keys"][base:base + cap], pos, n, pd, k=K)
        o = torch.zeros(n, H, LW, dtype=torch.bfloat16, device=DEV)
        lv, kv = lc[base:base + cap], d["kr"][base:base + cap]
        if pos < K:
            dense = min(n, K - pos)
            nch = latent.chunks_for(pos + n)
            s1 = latent.LatentScratch(n, H, nch, DEV)
            latent.attention(d["qa"][rs][:dense], lv, pd, s1, scale=192 ** -0.5, nch=nch, out=o[:dense],
                             hb=latent.head_block(n), qp=d["qp"][rs][:dense], rope=kv)
        if pos + n - 1 >= K:
            latent.sparse_attention(d["qa"][rs], lv, t1, c1, o, 192 ** -0.5, qp=d["qp"][rs], rope=kv)
        outs.append(o)
        sels.append((t1, c1))
    return outs, sels


@pytest.mark.parametrize("kind", ["bf16", "fp8", "fp4"])
def test_selection_and_attention_equal_each_segment(kind):
    m, R = tables(SEGS)
    d = data(2, R)
    lc = kv8.zeros(P, LW, kind, DEV)
    latent.latent_write(d["lat"], lc, torch.tensor([0], dtype=torch.int32, device=DEV))
    NT = m.bucket()
    sel = RowsScratch(16, 65536, K, DEV)
    nch = latent.rows_chunks(K, K)
    s = latent.LatentScratch(16, H, nch, DEV)
    out = torch.empty(R, H, LW, dtype=torch.bfloat16, device=DEV)
    select_and_attend(m, R, d, lc, NT, sel, out, s, nch)
    outs, sels = reference(SEGS, d, lc)
    for (at, base, pos, n, cap), o, (t1, c1) in zip(spans(SEGS), outs, sels):
        rs = slice(at, at + n)
        assert torch.equal(sel.counts[rs], c1) and torch.equal(sel.tokens[rs], t1), (kind, pos)
        assert torch.equal(out[rs], o), (kind, pos)


def test_graph_replays_other_tables():
    """One graph of the selection and the attention (R rows, bucket NT), captured on one window's tables and
    replayed on another's (other streams' positions, bases and extents, the same R and NT): the eager bits."""
    m, R = tables(SEGS)
    m2 = FullRows(16, DEV)
    assert m2.set(SEGS_B, dense_limit=K, indexed=True) == R
    NT = 65536
    d = data(3, R)
    lc = kv8.zeros(P, LW, "bf16", DEV)
    latent.latent_write(d["lat"], lc, torch.tensor([0], dtype=torch.int32, device=DEV))
    sel = RowsScratch(16, 65536, K, DEV)
    nch = latent.rows_chunks(K, K)
    s = latent.LatentScratch(16, H, nch, DEV)
    out = torch.empty(R, H, LW, dtype=torch.bfloat16, device=DEV)
    select_and_attend(m, R, d, lc, NT, sel, out, s, nch)                  # compile / warm up
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select_and_attend(m, R, d, lc, NT, sel, out, s, nch)
    for segs in (SEGS_B, SEGS):
        m.set(segs, dense_limit=K, indexed=True)
        graph.replay()
        torch.cuda.synchronize()
        got_out, got_t, got_c = out.clone(), sel.tokens[:R].clone(), sel.counts[:R].clone()
        outs, sels = reference(segs, d, lc)
        for (at, base, pos, n, cap), o, (t1, c1) in zip(spans(segs), outs, sels):
            rs = slice(at, at + n)
            assert torch.equal(got_c[rs], c1) and torch.equal(got_t[rs], t1), pos
            assert torch.equal(got_out[rs], o), pos


# -- the layer on the tiny model with every real kernel -------------------------------------------------------------

@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    from full_fakes import write_checkpoint

    os.environ.setdefault("TF_GLM_DENSE", "bf16")
    return write_checkpoint(tmp_path_factory.mktemp("glm53seg"))


def test_layer_equals_the_per_segment_loop(folder):
    from test_multi_kernels import FOUR, THREE, TWO, assert_all, run

    from tensorfold.families.glm5_next.cuda import forward as F_
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=0, world=1, device=DEV, mtp=True)
    w.comm = None
    w.meta["long_context"] = True
    assert_all(run(F_, w, TWO, [(0, 1), (1, 0)], seed=50) + run(F_, w, THREE, [(2, 0, 1)], seed=70)
               + run(F_, w, FOUR, [(3, 1, 0, 2)], seed=90))
