"""Multi-stream milestone 3 on CPU: full GLM-5.3's segmented kernels (several streams' rows in one launch, per-row
position / extent base / capacity / dense-or-sparse tables) against the single-stream kernels run per segment on that
stream's extent views, bit for bit (torch.equal) on random data: 2-4 segments at different positions and extents,
dense rows, sparse rows and windows crossing the dense limit, every cache format (bf16, fp8, fp4 latents; fp4x's
e4m3 rotary keys with their scales in the FP4 latent rows' pad and e4m3 index keys with an fp32 scale); then the
one-launch-per-piece layer (``full_seg.FullSegVerify``) against ``verify.FullBatchedVerify``'s per-segment loop on
the tiny model, 1 and 3 ranks, bf16 and fp4x caches."""

import multiprocessing as mp
import os
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.glm5_next.cuda import dsa_full as F
from tensorfold.families.glm5_next.cuda import kv8, latent
from tensorfold.families.glm5_next.cuda.full_seg import FullRows

THETA = 8_000_000.0
K = 520              # top-k (and dense limit): sparse lists span two 512-token chunks, dense rows up to 519 too

# (base, pos, rows, capacity): dense; crossing the dense limit (517..519 dense over two chunks, 520.. sparse);
# sparse; dense near the limit. Extents disjoint, unequal, not in base order.
SEGS4 = [(0, 3, 4, 1024), (1792, 517, 5, 768), (2560, 900, 3, 1024), (1024, 505, 2, 640)]
SEGS2 = [(1024, 600, 3, 1024), (0, 40, 2, 1024)]
SEGS3 = [(2048, 518, 4, 1024), (0, 1000, 1, 1024), (1024, 0, 3, 1024)]
P = 3584             # plane rows (every extent inside)


def rows_of(segs):
    out = []
    for k, (base, pos, n, cap) in enumerate(segs):
        out += [(k, pos + i, base, cap) for i in range(n)]
    return out


def meta(segs, indexed=True):
    m = FullRows(16, "cpu")
    R = m.set(segs, dense_limit=K, indexed=indexed)
    return m, R


def spans(segs):
    at = 0
    for base, pos, n, cap in segs:
        yield at, base, pos, n, cap
        at += n


def gen(seed):
    return torch.Generator().manual_seed(seed)


SCENARIOS = [SEGS2, SEGS3, SEGS4]


@pytest.mark.parametrize("segs", SCENARIOS)
def test_rope_and_cache_writes(segs):
    """Query / indexer-query RoPE and the latent, rotary-key and index-key writes: every row at its own position,
    into its stream's extent of the whole planes, equal to the single-stream kernels per segment on extent views."""
    g = gen(1)
    m, R = meta(segs)
    H, ROPE, LW, ID, IH = 6, 64, 128, 64, 4
    freq = F.inv_freq(ROPE, THETA, "cpu")
    q = torch.randn(R, H, 192, generator=g).to(torch.bfloat16)
    qi = torch.randn(R, IH * ID, generator=g).to(torch.bfloat16)
    dp = torch.randn(R, 300, generator=g).to(torch.bfloat16)               # [.. | rotary key] rows, strided
    kraw = torch.randn(R, ID + IH, generator=g).to(torch.bfloat16)
    lnw = (1 + 0.1 * torch.randn(ID, generator=g)).to(torch.bfloat16)
    lnb = (0.1 * torch.randn(ID, generator=g)).to(torch.bfloat16)
    lat = torch.randn(R, LW, generator=g).to(torch.bfloat16)

    qp = torch.empty(R, H, ROPE, dtype=torch.bfloat16)
    F.q_rope_rows(q, qp, m.pos, freq, 128)
    qi_rows = qi.clone()
    F.index_q_rope_rows(qi_rows, IH, m.pos, freq)
    kr = torch.randn(P, ROPE, generator=g).to(torch.bfloat16)                # stale rows everywhere
    ik = torch.randn(P, ID, generator=g).to(torch.bfloat16)
    lcs = {kind: kv8.zeros(P, LW, kind, "cpu") for kind in kv8.KINDS}
    kr_rows, ik_rows = kr.clone(), ik.clone()
    F.k_rope_write_rows(dp[:, 300 - ROPE:], kr_rows, m.pos, m.base, freq)
    F.index_write_rows(kraw[:, :ID], lnw, lnb, ik_rows, m.pos, m.base, freq)
    lc_rows = {kind: c.clone() for kind, c in lcs.items()}
    for kind in kv8.KINDS:
        latent.latent_write_rows(lat, lc_rows[kind], m.pos, m.base)
    # fp4x: e4m3 rotary codes (scales into the FP4 latent rows' pad, after the latents as forward writes them) and
    # e4m3 index rows with their fp32 scale; stale bytes everywhere
    kr8 = torch.randint(0, 120, (P, ROPE), generator=g, dtype=torch.uint8)
    ik8 = torch.randint(0, 120, (P, ID + 4), generator=g, dtype=torch.uint8)
    kr8_rows, ik8_rows = kr8.clone(), ik8.clone()
    F.k_rope_write_rows(dp[:, 300 - ROPE:], kr8_rows, m.pos, m.base, freq, latent=lc_rows["fp4x"])
    F.index_write_rows(kraw[:, :ID], lnw, lnb, ik8_rows, m.pos, m.base, freq)

    for at, base, pos, n, cap in spans(segs):
        rs = slice(at, at + n)
        pd = torch.tensor([pos], dtype=torch.int32)
        want = torch.empty(n, H, ROPE, dtype=torch.bfloat16)
        F.q_rope(q[rs], want, pd, freq, 128)
        assert torch.equal(qp[rs], want)
        want_i = qi[rs].clone()
        F.index_q_rope(want_i, IH, pd, freq)
        assert torch.equal(qi_rows[rs], want_i)
        F.k_rope_write(dp[rs, 300 - ROPE:], kr[base:base + cap], pd, freq)
        F.index_write(kraw[rs, :ID], lnw, lnb, ik[base:base + cap], pd, freq)
        for kind in kv8.KINDS:
            latent.latent_write(lat[rs], lcs[kind][base:base + cap], pd)
        F.k_rope_write(dp[rs, 300 - ROPE:], kr8[base:base + cap], pd, freq, latent=lcs["fp4x"][base:base + cap])
        F.index_write(kraw[rs, :ID], lnw, lnb, ik8[base:base + cap], pd, freq)
    assert torch.equal(kr_rows, kr) and torch.equal(ik_rows, ik)
    assert torch.equal(kr8_rows, kr8) and torch.equal(ik8_rows, ik8)
    for kind in kv8.KINDS:
        assert torch.equal(lc_rows[kind], lcs[kind]), kind
    assert not torch.equal(kr_rows, torch.zeros_like(kr_rows))
    # the fp4x rows are the bf16 rows' e4m3 codes and scales (the rotary scale in the latent row beside it)
    for base, pos, n, cap in segs:
        rows = slice(base + pos, base + pos + n)
        codes, sc = kv8.quantize_codes8(kr_rows[rows])
        assert torch.equal(kr8_rows[rows], codes) and torch.equal(kv8.rope_scales(lc_rows["fp4x"][rows]), sc)
        assert torch.equal(ik8_rows[rows], kv8.quantize_index8(ik_rows[rows]))


def select_case(segs, seed, fmt="bf16"):
    g = gen(seed)
    m, R = meta(segs)
    IH, ID = 4, 64
    qi = torch.randn(R, IH * ID, generator=g).to(torch.bfloat16)
    wts = torch.randn(R, ID + IH, generator=g).to(torch.bfloat16)[:, ID:]
    keys = torch.randn(P, ID, generator=g).to(torch.bfloat16)
    # ties: a few duplicated keys in every extent, so equal scores pick the lower stream-local token
    for base, pos, n, cap in segs:
        keys[base + 7] = keys[base + 2]
        keys[base + 300] = keys[base + 2]
    if fmt == "fp4x":                     # e4m3 rows and an fp32 scale (equal bf16 rows: equal codes, still ties)
        keys = kv8.quantize_index8(keys)
    return m, R, qi, wts, keys


@pytest.mark.parametrize("fmt", ["bf16", "fp4x"])
@pytest.mark.parametrize("segs", SCENARIOS)
def test_selection_equals_each_segment_alone(segs, fmt):
    """Each row's scores (the columns its selection reads), token list and count equal its segment's decode
    selection on the stream's extent alone (``select_tokens``: its own bucket), dense rows (count 0) included; bf16
    index keys and fp4x's e4m3 rows."""
    if fmt == "fp4x" and segs is SEGS2:
        pytest.skip("two scenarios on fp4x keys (the interpreter is slow)")
    m, R, qi, wts, keys = select_case(segs, 2, fmt)
    NT = m.bucket()
    scores = torch.full((16, P), 7.0)
    tokens = torch.full((16, K), -5, dtype=torch.int32)
    counts = torch.full((16,), -5, dtype=torch.int32)
    tr, cr = F.select_tokens_rows(qi, wts, keys, m.pos, m.base, m.cap, R, NT, scores, tokens, counts, k=K)
    sparse_rows = 0
    for at, base, pos, n, cap in spans(segs):
        rs = slice(at, at + n)
        pd = torch.tensor([pos], dtype=torch.int32)
        view = keys[base:base + cap]
        t1, c1 = F.select_tokens(qi[rs].contiguous(), wts[rs], view, pos, n, pd, k=K)
        assert torch.equal(cr[rs], c1) and torch.equal(tr[rs], t1), (base, pos)
        NT1 = F.bucket(pos, n, cap)
        s1 = torch.empty(n, NT1)
        F.score(qi[rs].contiguous(), wts[rs], view, pd, n, NT1, s1)
        for i in range(n):
            npr = min(NT, cap, max(pos + i + 1, K))
            assert torch.equal(scores[at + i, :npr], s1[i, :npr])
            assert (c1[i].item() == K) == (pos + i >= K)
            sparse_rows += int(c1[i].item() == K)
    if any(pos + n > K for _, pos, n, _ in segs):
        assert sparse_rows


def attn_case(segs, kind, seed, indexed):
    g = gen(seed)
    m, R = meta(segs, indexed)
    H, LW, ROPE = 20, 128, 64                                             # two 16-head blocks, the second padded
    qa = (0.3 * torch.randn(R, H, LW, generator=g)).to(torch.bfloat16)
    qp = (0.3 * torch.randn(R, H, ROPE, generator=g)).to(torch.bfloat16)
    lc = kv8.zeros(P, LW, kind, "cpu")
    latent.latent_write(torch.randn(P, LW, generator=g).to(torch.bfloat16), lc, torch.tensor([0], dtype=torch.int32))
    kr = torch.randn(P, ROPE, generator=g).to(torch.bfloat16)
    if kv8.x8(kind):                      # fp4x: the rotary plane as e4m3 codes, their scales in the latent rows' pad
        codes, s = kv8.quantize_codes8(kr)
        word = kv8.rope_scale_word(LW)
        lc.view(torch.uint8)[:, 4 * word:4 * word + 4] = s[:, None].contiguous().view(torch.uint8)
        kr = codes
        assert torch.equal(kv8.dequantize_rope8(kr, lc), codes.view(torch.float8_e4m3fn).float() * s[:, None])
    return m, R, qa, qp, lc, kr


def single_segment_attention(qa, qp, lc, kr, tokens, counts, pos, n, indexed, scale):
    """forward.dsa_full_rows' attention of one segment on its extent views: dense rows (``attention``, the window's
    chunks and head block), then the sparse rows (``sparse_attention``, rows with count k)."""
    H, LW = qa.shape[1], qa.shape[2]
    out = torch.zeros(n, H, LW, dtype=torch.bfloat16)
    pd = torch.tensor([pos], dtype=torch.int32)
    all_sparse = indexed and pos >= K
    sparse_rows = indexed and pos + n - 1 >= K
    if not all_sparse:
        dense = min(n, max(0, K - pos)) if sparse_rows else n
        nch = latent.chunks_for(pos + n)
        s = latent.LatentScratch(n, H, nch, "cpu", lw=LW)
        latent.attention(qa[:dense], lc, pd, s, scale=scale, nch=nch, out=out[:dense], hb=latent.head_block(n),
                         qp=qp[:dense], rope=kr)
    if sparse_rows:
        latent.sparse_attention(qa, lc, tokens, counts, out, scale, qp=qp, rope=kr)
    return out


@pytest.mark.parametrize("kind", ["bf16", "fp8", "fp4", "fp4x"])
@pytest.mark.parametrize("segs", SCENARIOS)
def test_attention_equals_each_segment_alone(segs, kind):
    """Dense and sparse rows with the rotary plane in one chunk pass and merge: every row equals its segment's
    single-stream dense / sparse attention on the stream's extent views (its own selection); fp4x: the e4m3 rotary
    plane with its scales in the FP4 latent rows."""
    if kind in ("fp4", "fp4x") and segs is not SEGS4:
        pytest.skip("one scenario on FP4 (the interpreter is slow on its tiles)")
    m, R, qa, qp, lc, kr = attn_case(segs, kind, 3, True)
    _, _, qi, wts, keys = select_case(segs, 4)
    scale = 192 ** -0.5
    tokens = torch.zeros(16, K, dtype=torch.int32)
    counts = torch.zeros(16, dtype=torch.int32)
    scores = torch.empty(16, P)
    NT = m.bucket()
    F.select_tokens_rows(qi, wts, keys, m.pos, m.base, m.cap, R, NT, scores, tokens, counts, k=K)
    nch = latent.rows_chunks(K, K)
    s = latent.LatentScratch(16, qa.shape[1], nch, "cpu", lw=qa.shape[2])
    out = torch.full((R, qa.shape[1], qa.shape[2]), 3.0, dtype=torch.bfloat16)
    latent.attention_rows(qa, lc, kr, qp, m.pos, m.base, m.sparse, tokens, counts, s, scale=scale, nch=nch, out=out)
    for at, base, pos, n, cap in spans(segs):
        rs = slice(at, at + n)
        t1, c1 = F.select_tokens(qi[rs].contiguous(), wts[rs], keys[base:base + cap], pos, n,
                                 torch.tensor([pos], dtype=torch.int32), k=K)
        want = single_segment_attention(qa[rs], qp[rs], lc[base:base + cap], kr[base:base + cap], t1, c1, pos, n,
                                        True, scale)
        assert torch.equal(out[rs], want), (kind, base, pos)
        assert torch.isfinite(out[rs].float()).all()


@pytest.mark.parametrize("segs", [SEGS2, SEGS4])
def test_dense_only_attention_equals_each_segment_alone(segs):
    """An engine without index caches: every row dense at any position (here up to 902: two chunks, a segment's own fewer), no lists."""
    m, R, qa, qp, lc, kr = attn_case(segs, "bf16", 5, False)
    assert not m.any_sparse and m.bucket() == 0
    scale = 192 ** -0.5
    nch = latent.chunks_for(max(pos + n for _, pos, n, _ in segs))
    s = latent.LatentScratch(16, qa.shape[1], nch, "cpu", lw=qa.shape[2])
    out = torch.empty(R, qa.shape[1], qa.shape[2], dtype=torch.bfloat16)
    latent.attention_rows(qa, lc, kr, qp, m.pos, m.base, m.sparse, None, None, s, scale=scale, nch=nch, out=out)
    for at, base, pos, n, cap in spans(segs):
        rs = slice(at, at + n)
        want = single_segment_attention(qa[rs], qp[rs], lc[base:base + cap], kr[base:base + cap], None, None, pos,
                                        n, False, scale)
        assert torch.equal(out[rs], want), (base, pos)


def test_rows_tables():
    """FullRows: per-row position, base, capacity, sparse flag; the bucket; refusals."""
    m, R = meta(SEGS4)
    assert R == 14
    assert m.pos.tolist()[:R] == [3, 4, 5, 6, 517, 518, 519, 520, 521, 900, 901, 902, 505, 506]
    assert m.base.tolist()[:R] == [0] * 4 + [1792] * 5 + [2560] * 3 + [1024] * 2
    assert m.sparse.tolist()[:R] == [0] * 7 + [1] * 5 + [0] * 2
    assert m.bucket() == 1024 and m.bucket(4096) == 4096 and m.bucket(1 << 20) == 4096
    with pytest.raises(ValueError):
        m.set([(0, 1020, 5, 1024)], dense_limit=K, indexed=True)          # past the extent
    with pytest.raises(ValueError):
        m.set([(0, 0, 9, 1024), (1024, 0, 8, 1024)], dense_limit=K, indexed=True)    # 17 rows


# -- the layer on the tiny model: FullSegVerify against FullBatchedVerify's per-segment loop -----------------------

from full_fakes import TEXT, install_fake_experts, write_checkpoint  # noqa: E402

CAP = 128
TAPS = (1, 3)
# per stream a prompt length, then rounds of (window width, rows kept); index_topk 16: dense below 16
TWO = ([9, 30], [[(3, 2), (5, 5)], [(5, 3), (1, 1)], [(4, 4), (2, 1)], [(1, 1), (1, 1)]])
THREE = ([13, 6, 40], [[(4, 2), (1, 1), (5, 3)], [(5, 5), (3, 1), (2, 2)], [(2, 1), (5, 4), (4, 4)],
                       [(1, 1), (1, 1), (1, 1)]])
FOUR = ([14, 3, 50, 15], [[(3, 3), (4, 2), (2, 1), (4, 4)], [(2, 1), (5, 5), (3, 3), (1, 1)],
                          [(4, 2), (1, 1), (4, 4), (3, 2)]])


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_checkpoint(tmp_path_factory.mktemp("glm53seg"))


@pytest.fixture
def fakes(monkeypatch):
    install_fake_experts(monkeypatch)
    monkeypatch.setenv("TF_GLM_DENSE", "bf16")


def tokens_of(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, TEXT["vocab_size"], (n,), generator=g).tolist()


class Pool:
    """``n`` streams on one shared pool (stream k's extent at a permuted k x CAP, slot k) and a verifier."""

    def __init__(self, F_, w, n, seg: bool, kv: str = "bf16"):
        from tensorfold.families.glm5_next.cuda.full_seg import FullSegVerify
        from tensorfold.families.glm5_next.cuda.verify import FullBatchedVerify

        self.F, self.w = F_, w
        self.caches = F_.Caches(w, n * CAP, streams=n, kv=kv)
        slots = F_.Slots(w, n, 8)
        place = [(k + 1) % n for k in range(n)]                               # extents not in stream order
        self.st = [F_.State(w, CAP, 8, caches=self.caches, base=place[k] * CAP, slots=slots, slot=k)
                   for k in range(n)]
        cls = FullSegVerify if seg else FullBatchedVerify
        self.verify = cls(SimpleNamespace(w=w, caches=self.caches), taps=TAPS)

    def copy_from(self, other):
        for a, b in zip(self.caches.arena.planes, other.caches.arena.planes):
            a.tensor.copy_(b.tensor)
        for a, b in zip(self.st, other.st):
            a.set_pos(b.pos)

    def prefill(self, k, toks):
        F_, st = self.F, self.st[k]
        pbuf = F_.Buffers(self.w, 64, CAP, prefill=True)
        pbuf.ids[:len(toks)] = torch.tensor(toks, dtype=torch.int32)
        F_.compute(self.w, st, pbuf, len(toks), nch=F_.chunks_for(st, len(toks)), host_pos=st.pos)
        F_.commit(self.w, st, pbuf, len(toks), len(toks))

    def batched(self, order, windows, keeps):
        from tensorfold.families.glm5_next.cuda.verify import Segment

        segs = [Segment(self.st[k], windows[k]) for k in order]
        out = self.verify.forward(segs)
        got = {k: (out.logits[i].clone(), out.taps[i].clone()) for i, k in enumerate(order)}
        self.verify.commit(segs, [keeps[k] for k in order])
        return got


def run(F_, w, scenario, orders, seed, kv="bf16"):
    """The per-segment loop (FullBatchedVerify) and the segmented layer (FullSegVerify), each stream order on pools
    of their own from the same prompts, through the rounds: (what, equal) checks of logits, taps, positions and every
    cache plane."""
    prompts, rounds = scenario
    n = len(prompts)
    base = Pool(F_, w, n, seg=False, kv=kv)
    for k, length in enumerate(prompts):
        base.prefill(k, tokens_of(length, seed + k))
    pairs = []
    for _ in orders:
        a, b = Pool(F_, w, n, seg=False, kv=kv), Pool(F_, w, n, seg=True, kv=kv)
        a.copy_from(base)
        b.copy_from(base)
        pairs.append((a, b))
    checks = []
    for r, rnd in enumerate(rounds):
        windows = [tokens_of(width, seed + 100 * (r + 1) + k) for k, (width, _) in enumerate(rnd)]
        keeps = [keep for _, keep in rnd]
        for order, (a, b) in zip(orders, pairs):
            pos = [st.pos for st in a.st]
            want = a.batched(order, windows, keeps)
            got = b.batched(order, windows, keeps)
            for k in range(n):
                what = f"round {r} order {order} stream {k} (pos {pos[k]}, {len(windows[k])} rows)"
                checks.append((what + " logits", torch.equal(got[k][0], want[k][0])))
                checks.append((what + " taps", torch.equal(got[k][1], want[k][1])))
    for order, (a, b) in zip(orders, pairs):
        for k in range(n):
            checks.append((f"order {order} stream {k} position", a.st[k].pos == b.st[k].pos))
        checks.append((f"order {order} cache planes", all(torch.equal(x.tensor, y.tensor) for x, y in
                                                          zip(a.caches.arena.planes, b.caches.arena.planes))))
    return checks


def _engine(folder, rank=0, world=1, comm=None):
    from tensorfold.families.glm5_next.cuda import forward as F_
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=rank, world=world, device="cpu", mtp=True)
    w.comm = comm
    w.meta["long_context"] = True
    return F_, w


def assert_all(checks):
    bad = [what for what, ok in checks if not ok]
    assert not bad, f"{len(bad)} of {len(checks)} differ: {bad[:8]}"


def test_layer_two_streams(folder, fakes):
    """Two streams (dense, sparse), both orders: the segmented layer's logits / taps / caches equal the loop's."""
    F_, w = _engine(folder)
    assert_all(run(F_, w, TWO, [(0, 1), (1, 0)], seed=50))


def test_layer_three_and_four_streams(folder, fakes):
    """Three and four streams (crossing the dense limit inside windows, dense, sparse), permuted extents."""
    F_, w = _engine(folder)
    assert_all(run(F_, w, THREE, [(2, 0, 1)], seed=70) + run(F_, w, FOUR, [(3, 1, 0, 2)], seed=90))


def test_layer_fp4x(folder, fakes):
    """TF_GLM_KV=fp4x caches (FP4 latents, e4m3 rotary keys with their scales in the latent rows, e4m3 index keys):
    the segmented layer's logits / taps / caches equal the per-segment loop's, two and three streams."""
    F_, w = _engine(folder)
    assert_all(run(F_, w, TWO, [(1, 0)], seed=150, kv="fp4x") + run(F_, w, THREE, [(2, 0, 1)], seed=170, kv="fp4x"))


def test_layer_without_index_caches(folder, fakes):
    """An engine without index caches (short contexts): every row dense, no selection."""
    F_, w = _engine(folder)
    w.meta["long_context"] = False
    short = ([9, 4], [[(3, 2), (5, 5)], [(2, 1), (1, 1)]])
    assert_all(run(F_, w, short, [(1, 0)], seed=30))


def _rank_seg(folder, rank, world, port, out_q):
    import torch.distributed as dist

    os.environ.setdefault("TRITON_INTERPRET", "1")
    torch.set_num_threads(1)
    import conftest  # noqa: F401  (the interpreter patches)
    import pytest as _pt

    mp_ = _pt.MonkeyPatch()
    install_fake_experts(mp_)
    os.environ["TF_GLM_DENSE"] = "bf16"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)

    class Comm:
        world_size = world

        def all_gather(self, send, recv):
            dist.all_gather(list(recv.view(world, -1).unbind(0)), send.contiguous().view(-1))

    F_, w = _engine(folder, rank, world, Comm())
    checks = run(F_, w, THREE, [(2, 0, 1)], seed=70) + run(F_, w, TWO, [(1, 0)], seed=50)
    out_q.put((rank, checks))
    dist.destroy_process_group()


def test_layer_three_ranks(folder, fakes):
    """Three ranks (heads 2/2/1): on every rank the segmented layer equals the per-segment loop."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 37000 + os.getpid() % 2000
    procs = [ctx.Process(target=_rank_seg, args=(folder, r, 3, port, q)) for r in range(3)]
    for p in procs:
        p.start()
    got = sorted((q.get(timeout=2400) for _ in procs), key=lambda x: x[0])
    for p in procs:
        p.join(timeout=60)
    for rank, checks in got:
        bad = [what for what, ok in checks if not ok]
        assert not bad, f"rank {rank}: {len(bad)} of {len(checks)} differ: {bad[:8]}"
