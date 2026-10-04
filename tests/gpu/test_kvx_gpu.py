"""TF_GLM_KV=fp4x on the GPU (scripts/test-gpu.sh -k kvx): the write kernels store kv8's definition byte for byte;
Mia's sparse attention (msa.cu) with e4m3 rotary keys gives exactly the bits it gives on a bf16 rotary plane holding
the dequantized rows (the dequantize step writes the same bf16 key tile), close to the Triton kernels, the same bits
run to run and a row alone; the Triton readers keep each row's bits and stay close to the bf16 planes; and the tiny
checkpoint through the engine with fp4x: close to the reference, drafted == serial, CUDA graphs == eager."""

import os

import pytest
import torch

from tensorfold.families.glm5_next.cuda import dcp, kv8, latent
from tensorfold.families.glm5_next.cuda import dsa_full as F

DEV = "cuda"
THETA = 8_000_000.0
POS0 = torch.tensor([0], dtype=torch.int32)


def fp4x_planes(T, seed, dev=DEV):
    """An FP4 latent cache and its e4m3 rotary plane over T tokens written by the kernels, the rotary plane
    dequantized to bf16 (exact), and the bf16 plane the bf16 writer would hold."""
    g = torch.Generator().manual_seed(seed)
    lat = torch.randn(T, 512, generator=g).to(torch.bfloat16).to(dev)
    k = torch.randn(T, 64, generator=g).to(torch.bfloat16).to(dev)
    freq = F.inv_freq(64, THETA, dev)
    pos = POS0.to(dev)
    lc = kv8.zeros(T, 512, "fp4x", dev)
    latent.latent_write(lat, lc, pos)
    kr = kv8.rope_zeros(T, 64, "fp4x", dev)
    F.k_rope_write(k, kr, pos, freq, latent=lc)
    krb = torch.zeros(T, 64, dtype=torch.bfloat16, device=dev)
    F.k_rope_write(k, krb, pos, freq)
    return lc, kr, kv8.dequantize_rope8(kr, lc).to(torch.bfloat16), krb


def test_writers_store_the_definition():
    lc, kr, deq, krb = fp4x_planes(3000, 1)
    codes, s = kv8.quantize_codes8(krb.cpu())
    assert torch.equal(kr.cpu(), codes) and torch.equal(kv8.rope_scales(lc.cpu()), s)
    g = torch.Generator().manual_seed(2)
    raw = torch.randn(500, 160, generator=g).to(torch.bfloat16).to(DEV)
    ln_w = (1 + 0.1 * torch.randn(128, generator=g)).to(torch.bfloat16).to(DEV)
    ln_b = (0.1 * torch.randn(128, generator=g)).to(torch.bfloat16).to(DEV)
    freq = F.inv_freq(64, THETA, DEV)
    pos = torch.tensor([4097], dtype=torch.int32, device=DEV)
    ref = torch.zeros(5000, 128, dtype=torch.bfloat16, device=DEV)
    F.index_write(raw, ln_w, ln_b, ref, pos, freq)
    keys = kv8.index_zeros(5000, 128, "fp4x", DEV)
    F.index_write(raw, ln_w, ln_b, keys, pos, freq)
    assert torch.equal(keys.cpu(), kv8.quantize_index8(ref.cpu()))


def test_msa_fp8_rope_equals_dequantized_bf16_rope():
    """msa.prompt / prompt_lse with fp4x's e4m3 rotary plane: bit for bit what they give on the bf16 plane of its
    dequantized rows (the same key tile), close to latent.sparse_attention, rows alone keep their bits."""
    from tensorfold.families.glm5_next.cuda import msa

    torch.manual_seed(3)
    H, R, T, K = 22, 96, 6000, 2048
    qa = (0.3 * torch.randn(R, H, 512, device=DEV)).bfloat16()
    qp = (0.3 * torch.randn(R, H, 64, device=DEV)).bfloat16()
    lc, kr, deq, _ = fp4x_planes(T, 4)
    assert msa.supported(qa, lc, kr) and msa.supported(qa, lc, deq)
    assert not msa.supported(qa, kv8.zeros(T, 512, "fp8", DEV), kr)        # e4m3 rotary codes need FP4 rows
    tok = torch.full((R, K), -1, dtype=torch.int32, device=DEV)
    cnt = torch.zeros(R, dtype=torch.int32, device=DEV)
    for r in range(R):
        n = [K, 7, 1, 40, 2000][r % 5]
        tok[r, :n] = torch.randperm(T, device=DEV)[:n].sort().values.int()
        cnt[r] = n
    cnt[5] = 0
    got = torch.zeros(R, H, 512, dtype=torch.bfloat16, device=DEV)
    msa.prompt(qa, lc, tok, cnt, got, 0.0625, qp, kr)
    want = torch.zeros_like(got)
    msa.prompt(qa, lc, tok, cnt, want, 0.0625, qp, deq)
    assert torch.equal(got, want)
    ref = torch.zeros_like(got)
    latent.sparse_attention(qa, lc, tok, cnt, ref, 0.0625, qp=qp, rope=kr)
    live = cnt > 0
    err = (got[live].float() - ref[live].float()).abs().max().item()
    assert err <= 0.02 * ref[live].float().abs().max().item(), err
    again = torch.zeros_like(got)
    msa.prompt(qa, lc, tok, cnt, again, 0.0625, qp, kr)
    assert torch.equal(again, got)
    for r in (0, 3, 4):
        one = torch.zeros_like(got[:1])
        msa.prompt(qa[r:r + 1].contiguous(), lc, tok[r:r + 1].contiguous(), cnt[r:r + 1].contiguous(), one, 0.0625,
                   qp[r:r + 1].contiguous(), kr)
        assert torch.equal(one[0], got[r])
    # context parallelism's prompt partials
    out = torch.empty_like(got)
    lse = torch.empty(R, H, dtype=torch.float32, device=DEV)
    msa.prompt_lse(qa, lc, tok, cnt, out, lse, 0.0625, qp, kr)
    out_b = torch.empty_like(got)
    lse_b = torch.empty_like(lse)
    msa.prompt_lse(qa, lc, tok, cnt, out_b, lse_b, 0.0625, qp, deq)
    assert torch.equal(out, out_b) and torch.equal(lse, lse_b)
    o_ref, l_ref = dcp.attention_partial(qa, qp, lc, kr, tok, cnt, 0.0625)
    assert (out[live].float() - o_ref[live]).abs().max().item() <= 0.02 * o_ref[live].abs().max().item()
    assert (lse[live] - l_ref[live]).abs().max().item() < 2e-3


@pytest.mark.parametrize("pos0", [40, 3000])
def test_triton_readers_keep_row_bits(pos0):
    """Dense and sparse latent attention, and the index scores, on fp4x's planes: a window's rows equal each row
    alone, and stay within bf16 summation of the dequantized bf16 planes."""
    g = torch.Generator().manual_seed(5)
    H, R, T = 22, 8, pos0 + 8
    qa = (0.3 * torch.randn(R, H, 512, generator=g)).to(torch.bfloat16).to(DEV)
    qp = (0.3 * torch.randn(R, H, 64, generator=g)).to(torch.bfloat16).to(DEV)
    lc, kr, deq, _ = fp4x_planes(T, 6)
    nch = latent.chunks_for(T)
    s = latent.LatentScratch(R, H, nch, DEV)
    pos = torch.tensor([pos0], dtype=torch.int32, device=DEV)
    full = torch.empty(R, H, 512, dtype=torch.bfloat16, device=DEV)
    latent.attention(qa, lc, pos, s, scale=256 ** -0.5, nch=nch, out=full, qp=qp, rope=kr, hb=latent.HB)
    alt = torch.empty_like(full)
    latent.attention(qa, lc, pos, s, scale=256 ** -0.5, nch=nch, out=alt, qp=qp, rope=deq, hb=latent.HB)
    assert (full.float() - alt.float()).abs().max().item() <= 0.02 * alt.float().abs().max().item()
    for r in range(R):
        one = torch.empty(1, H, 512, dtype=torch.bfloat16, device=DEV)
        latent.attention(qa[r:r + 1].contiguous(), lc, pos + r, s, scale=256 ** -0.5, nch=nch, out=one,
                         qp=qp[r:r + 1].contiguous(), rope=kr, hb=latent.HB)
        assert torch.equal(one[0], full[r])
    tokens = torch.stack([torch.randperm(pos0 + 1, generator=g)[:min(2048, pos0 + 1)].sort().values
                          for _ in range(R)]).to(torch.int32).to(DEV)
    counts = torch.full((R,), tokens.shape[1], dtype=torch.int32, device=DEV)
    sp = torch.empty_like(full)
    latent.sparse_attention(qa, lc, tokens, counts, sp, 256 ** -0.5, qp=qp, rope=kr)
    for r in range(R):
        one = torch.empty(1, H, 512, dtype=torch.bfloat16, device=DEV)
        latent.sparse_attention(qa[r:r + 1].contiguous(), lc, tokens[r:r + 1].contiguous(), counts[r:r + 1], one,
                                256 ** -0.5, qp=qp[r:r + 1].contiguous(), rope=kr)
        assert torch.equal(one[0], sp[r])
    # index scores over e4m3 keys: rows alone, close to the dequantized bf16 keys
    qi = torch.randn(R, 32 * 128, generator=g).to(torch.bfloat16).to(DEV)
    wts = torch.randn(R, 32, generator=g).to(torch.bfloat16).to(DEV)
    NT = max(4096, T)
    keys = torch.zeros(NT, 132, dtype=torch.uint8, device=DEV)
    keys[:T] = kv8.quantize_index8(torch.randn(T, 128, generator=g).to(torch.bfloat16)).to(DEV)
    kb = kv8.dequantize_index8(keys).to(torch.bfloat16)
    a = torch.empty(R, NT, device=DEV)
    F.score(qi, wts, keys, pos, R, NT, a)
    b = torch.empty_like(a)
    F.score(qi, wts, kb, pos, R, NT, b)
    for r in range(R):                                        # rows alone
        one = torch.empty(1, NT, device=DEV)
        F.score(qi[r:r + 1].contiguous(), wts[r:r + 1], keys, pos + r, 1, NT, one)
        assert torch.equal(one[0], a[r])
    fin = torch.isfinite(b)
    assert torch.equal(torch.isfinite(a), fin)
    assert (a[fin] - b[fin]).abs().max().item() <= 1e-3 * b[fin].abs().max().item()


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    from full_fakes import write_checkpoint

    os.environ.setdefault("TF_GLM_DENSE", "bf16")
    return write_checkpoint(tmp_path_factory.mktemp("glm53kvx"))


def engine(folder, graphs=False):
    from tensorfold.families.glm5_next.cuda.decode import Engine
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=0, world=1, device="cuda", mtp=True)
    return Engine(w, capacity=4096 + 512, max_rows=8, prefill_rows=64, graphs=graphs, graph_rows=(1, 2, 3, 4),
                  long_context=True, mtp_rows=4, kv="fp4x")


def run(e, prompt, steps):
    from tensorfold.families.glm5_next.cuda.decode import prefill
    from tensorfold.families.glm5_next.cuda.forward import commit

    prefill(e, prompt, None, mtp=False, sample=False)
    rows = []
    for t in steps:
        rows.append(e.forward([t]).float().cpu().clone())
        commit(e.w, e.st, e.buf, 1, 1)
    return torch.cat(rows)


def test_engine_fp4x_matches_the_reference(folder):
    from test_full_forward import agree_kv, tokens
    from test_kvx import ReferenceX

    e = engine(folder)
    assert e.st.kr[0].dtype == torch.uint8 and e.st.index[0][2].dtype == torch.uint8
    ref = ReferenceX(folder)
    cache = ref.new_cache()
    prompt, steps = tokens(30, 1), tokens(8, 2)
    got = run(e, prompt, steps)
    ref.forward(prompt, cache, 0)
    want = torch.cat([ref.forward([t], cache, 30 + i)[0] for i, t in enumerate(steps)])
    ok, why = agree_kv(got, want, "fp4")
    assert ok, why


@pytest.mark.parametrize("start", [9, 60])
def test_engine_fp4x_drafted_window_equals_serial_steps(folder, start):
    from test_full_forward import tokens

    from tensorfold.families.glm5_next.cuda.decode import prefill
    from tensorfold.families.glm5_next.cuda.forward import commit

    e = engine(folder)
    prompt, draft = tokens(start, 3), tokens(6, 4)
    prefill(e, prompt, None, mtp=False, sample=False)
    window = e.forward(draft).float().cpu().clone()
    e.reset()
    prefill(e, prompt, None, mtp=False, sample=False)
    serial = []
    for t in draft:
        serial.append(e.forward([t]).float().cpu().clone())
        commit(e.w, e.st, e.buf, 1, 1)
    assert torch.equal(window, torch.cat(serial))


def test_engine_fp4x_graphs_replay_the_eager_bits(folder):
    from test_full_forward import tokens

    eager = engine(folder, graphs=False)
    graphed = engine(folder, graphs=True)
    a = run(eager, tokens(6, 7), tokens(4, 8))
    b = run(graphed, tokens(6, 7), tokens(4, 8))
    assert torch.equal(a, b) and graphed.replays["main"] >= 4
    eager.reset()
    graphed.reset()
    a = run(eager, tokens(4100, 5), tokens(6, 6))
    b = run(graphed, tokens(4100, 5), tokens(6, 6))
    assert torch.equal(a, b) and graphed.replays["sparse"] >= 6
