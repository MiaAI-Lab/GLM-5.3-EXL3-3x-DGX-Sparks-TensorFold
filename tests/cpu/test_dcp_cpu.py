"""The context-parallel (CP3) cache's kernels (dcp) against the replicated path, on CPU (Triton's interpreter): the
merged selection equals dsa_full.select_tokens split by owner bit for bit (ties included), the ranks' merged partial
attention equals latent.sparse_attention / attention within bf16 rounding, and every kernel is deterministic and
row-independent."""

import pytest
import torch

from tensorfold.families.glm5_next.cuda import dcp, kv8, latent
from tensorfold.families.glm5_next.cuda import dsa_full as F

G = 3
LW, ROPE = 512, 64
SCALE = 256 ** -0.5
MASK64 = (1 << 64) - 1


def index_case(seed, R, cap, ties=None, H=32, D=128):
    g = torch.Generator().manual_seed(seed)
    qi = torch.randn(R, H * D, generator=g).to(torch.bfloat16)
    wts = torch.randn(R, H + 128, generator=g).to(torch.bfloat16)[:, 128:]      # strided rows, unit-stride columns
    keys = torch.randn(cap, D, generator=g).to(torch.bfloat16)
    if ties == "zeros":
        # most keys zero: exact-zero scores (relu of 0), +0 and -0 (a row of negative weights), straddling the top-k
        keep = torch.rand(cap, generator=g) < 0.2
        keys = torch.where(keep[:, None], keys, torch.zeros_like(keys))
        wts[0] = -wts[0].abs()
    elif ties == "dups":
        # equal keys at neighbouring positions (other ranks): equal scores across ranks, ties to the lower position
        keys[1::2] = keys[0::2][:keys[1::2].shape[0]]
        keys[::9] = 0
    return qi, wts, keys


def order_key(s: torch.Tensor) -> torch.Tensor:
    """dsa_full._order_key in torch: int64 holding the uint32."""
    bits = (s + 0.0).view(torch.int32).to(torch.int64)
    return (bits ^ ((bits >> 31) | -2147483648)) & 0xFFFFFFFF


def run_select(qi, wts, keys, pos0, R, k, cols=None):
    """Every rank's candidates (stacked in rank order) and every rank's (slots, counts)."""
    pos = torch.tensor([pos0], dtype=torch.int32)
    cands = torch.stack([dcp.select_local(qi, wts, keys[r::G].contiguous(), r, G, pos, R, k=k, cols=cols)
                         for r in range(G)]).contiguous()
    return cands, [dcp.select_merge(cands, r, G, k) for r in range(G)]


def reference_own(tok, cnt, pos0, R, k, rank):
    """The replicated selection's tokens of ``rank`` as local slots a row (all visible ones for dense rows)."""
    out = []
    for r in range(R):
        ref = tok[r].long() if cnt[r] == k else torch.arange(pos0 + r + 1)
        out.append((ref[ref % G == rank] // G).to(torch.int32))
    return out


CASES = [  # seed, pos0, R, cap, k, ties
    (1, 300, 4, 384, 64, None),
    (2, 300, 4, 384, 64, "zeros"),
    (3, 250, 4, 384, 64, "dups"),
    (4, 60, 8, 192, 64, None),          # the window crosses the dense limit (rows see 61 .. 68 tokens)
    (5, 0, 3, 64, 64, "zeros"),         # first positions: ranks 1, 2 see nothing in row 0
    (6, 2300, 2, 2432, 2048, "zeros"),  # the real top-k
]


@pytest.mark.parametrize("seed,pos0,R,cap,k,ties", CASES)
def test_local_scores_have_replicated_bits(seed, pos0, R, cap, k, ties):
    qi, wts, keys = index_case(seed, R, cap, ties)
    pos = torch.tensor([pos0], dtype=torch.int32)
    full = torch.empty(R, cap)
    F.score(qi, wts, keys, pos, R, cap, full)
    for rank in range(G):
        loc_keys = keys[rank::G].contiguous()
        loc = torch.empty(R, loc_keys.shape[0])
        dcp.score_local(qi, wts, loc_keys, rank, G, pos, R, loc_keys.shape[0], loc)
        assert torch.equal(loc.view(torch.int32), full[:, rank::G].contiguous().view(torch.int32))


@pytest.mark.parametrize("seed,pos0,R,cap,k,ties", CASES)
def test_candidates_are_local_topk_keys(seed, pos0, R, cap, k, ties):
    qi, wts, keys = index_case(seed, R, cap, ties)
    pos = torch.tensor([pos0], dtype=torch.int32)
    full = torch.empty(R, cap)
    F.score(qi, wts, keys, pos, R, cap, full)
    cands, _ = run_select(qi, wts, keys, pos0, R, k)
    for rank in range(G):
        for r in range(R):
            g = torch.arange(pos0 + r + 1)[rank::G]
            packed = [(int(u) << 32) | (0xFFFFFFFF - int(t)) for u, t in zip(order_key(full[r, g]), g)]
            best = sorted(packed, reverse=True)[:k]
            want = sorted(best, key=lambda x: 0xFFFFFFFF - (x & 0xFFFFFFFF)) + [0] * (k - len(best))
            assert [int(x) & MASK64 for x in cands[rank, r]] == want


@pytest.mark.parametrize("prompt", [False, True])
@pytest.mark.parametrize("seed,pos0,R,cap,k,ties", CASES)
def test_merge_equals_replicated_selection(seed, pos0, R, cap, k, ties, prompt):
    qi, wts, keys = index_case(seed, R, cap, ties)
    pos = torch.tensor([pos0], dtype=torch.int32)
    if prompt:
        tok, cnt = F.select_tokens(qi, wts, keys, pos0, R, pos, prompt=True, k=k)
    else:
        tok, cnt = F.select_tokens(qi, wts, keys, pos0, R, pos, bucket_cols=cap, k=k)
    _, parts = run_select(qi, wts, keys, pos0, R, k)
    total = torch.zeros(R, dtype=torch.int64)
    for rank, (slots, counts) in enumerate(parts):
        own = reference_own(tok, cnt, pos0, R, k, rank)
        for r in range(R):
            n = int(counts[r])
            assert n == own[r].numel()
            assert torch.equal(slots[r, :n], own[r])
            assert not slots[r, n:].any()
        total += counts.long()
    vis = pos0 + torch.arange(R) + 1
    assert torch.equal(total, torch.minimum(vis, torch.tensor(k)))


def test_merge_decode_bucket_columns():
    """A decode window scoring only the window's visible local columns selects as scoring all local slots does."""
    seed, pos0, R, cap, k = 7, 200, 3, 384, 64
    qi, wts, keys = index_case(seed, R, cap, "dups")
    a, pa = run_select(qi, wts, keys, pos0, R, k)
    b, pb = run_select(qi, wts, keys, pos0, R, k, cols=dcp.visible_local(pos0 + R - 1, 0, G))
    assert torch.equal(a, b)
    for (sa, ca), (sb, cb) in zip(pa, pb):
        assert torch.equal(sa, sb) and torch.equal(ca, cb)


def attn_case(seed, R, H, cap, fp8):
    g = torch.Generator().manual_seed(seed)
    qa = (0.3 * torch.randn(R, H, LW, generator=g)).to(torch.bfloat16)
    qp = (0.3 * torch.randn(R, H, ROPE, generator=g)).to(torch.bfloat16)
    lat = torch.randn(cap, LW, generator=g).to(torch.bfloat16)
    kr = torch.randn(cap, ROPE, generator=g).to(torch.bfloat16)
    cache = kv8.quantize_rows(lat) if fp8 else lat
    return qa, qp, cache, kr


def run_attention(qa, qp, cache, kr, parts):
    o, lse = [], []
    for rank, (slots, counts) in enumerate(parts):
        a, b = dcp.attention_partial(qa, qp, cache[rank::G].contiguous(), kr[rank::G].contiguous(), slots, counts,
                                     SCALE)
        o.append(a)
        lse.append(b)
    o, lse = torch.stack(o).contiguous(), torch.stack(lse).contiguous()
    return o, lse, dcp.merge_ranks(o, lse)


def close(got, ref):
    d = (got.float() - ref.float()).abs()
    scale = max(1.0, ref.float().abs().max().item())
    return d.max().item() <= 2 ** -6 * scale and d.mean().item() <= 2 ** -10 * scale


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("seed,pos0,R,cap,k,ties", [(11, 300, 3, 384, 64, None), (12, 250, 2, 384, 64, "dups"),
                                                    (13, 2300, 1, 2432, 2048, None)])
def test_attention_equals_replicated_sparse(seed, pos0, R, cap, k, ties, fp8):
    H = 16
    qi, wts, keys = index_case(seed, R, cap, ties)
    pos = torch.tensor([pos0], dtype=torch.int32)
    tok, cnt = F.select_tokens(qi, wts, keys, pos0, R, pos, bucket_cols=cap, k=k)
    assert cnt.tolist() == [k] * R
    _, parts = run_select(qi, wts, keys, pos0, R, k)
    qa, qp, cache, kr = attn_case(seed, R, H, cap, fp8)
    ref = torch.zeros(R, H, LW, dtype=torch.bfloat16)
    latent.sparse_attention(qa, cache, tok, cnt, ref, SCALE, qp=qp, rope=kr)
    o, lse, out = run_attention(qa, qp, cache, kr, parts)
    assert torch.isfinite(lse).all()
    assert close(out, ref)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("pos0,R", [(0, 3), (30, 4)])
def test_attention_dense_rows_equal_replicated_dense(pos0, R, fp8):
    """Rows under the dense limit: every own visible slot; ranks 1 and 2 have none at position 0 (lse -inf, o 0)."""
    H, cap, k = 16, 96, 64
    qi, wts, keys = index_case(21, R, cap)
    _, parts = run_select(qi, wts, keys, pos0, R, k)
    qa, qp, cache, kr = attn_case(22, R, H, cap, fp8)
    s = latent.LatentScratch(R, H, 1, "cpu")
    ref = torch.zeros(R, H, LW, dtype=torch.bfloat16)
    latent.attention(qa, cache, torch.tensor([pos0], dtype=torch.int32), s, scale=SCALE, nch=1, out=ref, qp=qp,
                     rope=kr)
    o, lse, out = run_attention(qa, qp, cache, kr, parts)
    if pos0 == 0:
        assert torch.equal(lse[1:, 0], torch.full((G - 1, H), float("-inf"))) and not o[1:, 0].any()
        assert torch.equal(out[0], kv8.dequantize(cache)[0].to(torch.bfloat16).expand(H, LW))
    assert close(out, ref)


def full_pipeline(qi, wts, keys, pos0, R, k, qa, qp, cache, kr):
    cands, parts = run_select(qi, wts, keys, pos0, R, k)
    o, lse, out = run_attention(qa, qp, cache, kr, parts)
    return cands, parts, o, lse, out


def same(a, b):
    return torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("fp8", [False, True])
def test_deterministic(fp8):
    seed, pos0, R, cap, k = 31, 280, 3, 384, 64
    qi, wts, keys = index_case(seed, R, cap, "dups")
    qa, qp, cache, kr = attn_case(seed, R, 16, cap, fp8)
    a = full_pipeline(qi, wts, keys, pos0, R, k, qa, qp, cache, kr)
    b = full_pipeline(qi, wts, keys, pos0, R, k, qa, qp, cache, kr)
    assert same(a[0], b[0])
    for (sa, ca), (sb, cb) in zip(a[1], b[1]):
        assert same(sa, sb) and same(ca, cb)
    for x, y in zip(a[2:], b[2:]):
        assert same(x, y)


@pytest.mark.parametrize("fp8", [False, True])
def test_rows_independent(fp8):
    """A row alone (its position, R = 1) gets the window's bits at every stage."""
    seed, pos0, R, cap, k = 41, 270, 4, 384, 64
    qi, wts, keys = index_case(seed, R, cap, "zeros")
    qa, qp, cache, kr = attn_case(seed, R, 16, cap, fp8)
    cands, parts, o, lse, out = full_pipeline(qi, wts, keys, pos0, R, k, qa, qp, cache, kr)
    for r in range(R):
        c1, p1, o1, l1, out1 = full_pipeline(qi[r:r + 1].contiguous(), wts[r:r + 1], keys, pos0 + r, 1, k,
                                             qa[r:r + 1].contiguous(), qp[r:r + 1].contiguous(), cache, kr)
        assert same(c1[:, 0], cands[:, r])
        for (s1, n1), (s, n) in zip(p1, parts):
            assert same(s1[0], s[r]) and same(n1[0], n[r])
        assert same(o1[:, 0], o[:, r]) and same(l1[:, 0], lse[:, r]) and same(out1[0], out[r])
