"""Full GLM-5.3's RoPE and DSA indexer kernels (dsa_full) against transformers' formulas, on CPU."""

import math

import pytest
import torch

from tensorfold.families.glm5_next.cuda import dsa_full as F

THETA = 8_000_000.0


def close(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Within a bf16 rounding (the interpreter's libm cos / sin are not torch's)."""
    return torch.allclose(a.float(), b.float(), rtol=2 ** -7, atol=2 ** -7)


def rope_ref(x: torch.Tensor, pos: torch.Tensor, dim: int) -> torch.Tensor:
    """transformers' apply_rotary_pos_emb_interleave on x [..., dim] (fp32 math, bf16 cos / sin)."""
    inv = F.inv_freq(dim, THETA, "cpu")
    ang = pos.to(torch.float32)[:, None] * inv[None, :]                       # [n, dim / 2]
    c, s = ang.cos().to(torch.bfloat16).float(), ang.sin().to(torch.bfloat16).float()
    while c.dim() < x.dim():
        c, s = c.unsqueeze(1), s.unsqueeze(1)
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def test_inv_freq_is_transformers():
    inv = F.inv_freq(64, THETA, "cpu")
    ref = 1.0 / (THETA ** (torch.arange(0, 64, 2, dtype=torch.int64).to(dtype=torch.float) / 64))
    assert torch.equal(inv, ref)


@pytest.mark.parametrize("pos0", [0, 5, 131_000, 1_000_000])
def test_q_rope(pos0):
    g = torch.Generator().manual_seed(1)
    R, H = 3, 5
    q = torch.randn(R, H, 256, generator=g).to(torch.bfloat16)
    out = torch.empty(R, H, 64, dtype=torch.bfloat16)
    F.q_rope(q, out, torch.tensor([pos0], dtype=torch.int32), F.inv_freq(64, THETA, "cpu"), 192)
    ref = rope_ref(q[..., 192:], torch.arange(pos0, pos0 + R), 64).to(torch.bfloat16)
    assert close(out, ref)


def test_rope_keeps_interleaved_dot_products():
    """Half-split outputs of q and k dot like interleaved rotations do: the relative-position property."""
    g = torch.Generator().manual_seed(2)
    q = torch.randn(1, 1, 64, generator=g).to(torch.bfloat16)
    k = torch.randn(1, 64, generator=g).to(torch.bfloat16)
    freq = F.inv_freq(64, THETA, "cpu")
    dots = []
    for shift in (0, 7):
        qo = torch.empty(1, 1, 64, dtype=torch.bfloat16)
        F.q_rope(q, qo, torch.tensor([10 + shift], dtype=torch.int32), freq, 0)
        kc = torch.zeros(20, 64, dtype=torch.bfloat16)
        F.k_rope_write(k, kc, torch.tensor([3 + shift], dtype=torch.int32), freq)
        dots.append(float((qo[0, 0].float() * kc[3 + shift].float()).sum()))
    assert dots[0] == pytest.approx(dots[1], rel=0.05, abs=0.05)


@pytest.mark.parametrize("pos0", [0, 4000])
def test_k_rope_write(pos0):
    g = torch.Generator().manual_seed(3)
    R = 4
    raw = torch.randn(R, 2048 + 576, generator=g).to(torch.bfloat16)
    k = raw[:, 2048 + 512:]                                                    # the rotary key, strided rows
    cache = torch.zeros(pos0 + 8, 64, dtype=torch.bfloat16)
    F.k_rope_write(k, cache, torch.tensor([pos0], dtype=torch.int32), F.inv_freq(64, THETA, "cpu"))
    ref = rope_ref(k, torch.arange(pos0, pos0 + R), 64).to(torch.bfloat16)
    assert close(cache[pos0:pos0 + R], ref)
    assert not cache[:pos0].any() and not cache[pos0 + R:].any()


def test_index_write():
    g = torch.Generator().manual_seed(4)
    R, pos0 = 5, 9
    raw = torch.randn(R, 160, generator=g).to(torch.bfloat16)                 # [wk | weights_proj] rows
    w = (1 + 0.1 * torch.randn(128, generator=g)).to(torch.bfloat16)
    b = (0.1 * torch.randn(128, generator=g)).to(torch.bfloat16)
    keys = torch.zeros(32, 128, dtype=torch.bfloat16)
    F.index_write(raw, w, b, keys, torch.tensor([pos0], dtype=torch.int32), F.inv_freq(64, THETA, "cpu"))
    x = raw[:, :128].float()
    y = torch.nn.functional.layer_norm(x, (128,), w.float(), b.float(), eps=1e-6).to(torch.bfloat16)
    ref = torch.cat([rope_ref(y[:, :64], torch.arange(pos0, pos0 + R), 64).to(torch.bfloat16), y[:, 64:]], dim=1)
    got = keys[pos0:pos0 + R]
    assert (got.float() - ref.float()).abs().max() <= 0.02                    # fp32 statistics' summation order
    assert (got[:, 64:] == ref[:, 64:]).float().mean() > 0.95


def scores_ref(qi, wts, keys, pos0, R, NT):
    H, D = wts.shape[1], keys.shape[1]
    q = qi.view(R, H, D).float()
    dots = torch.einsum("rhd,td->rht", q, keys[:NT].float()) * D ** -0.5
    sc = (torch.relu(dots) * (wts.float() * H ** -0.5)[:, :, None]).sum(1)
    t = torch.arange(NT)
    vis = t[None, :] <= (pos0 + torch.arange(R))[:, None]
    return torch.where(vis, sc, torch.full_like(sc, float("-inf")))


def _index_case(seed, R, pos0, cap, H=32, D=128):
    g = torch.Generator().manual_seed(seed)
    qi = torch.randn(R, H * D, generator=g).to(torch.bfloat16)
    wts = torch.randn(R, H + 128, generator=g).to(torch.bfloat16)[:, 128:]  # strided rows, unit-stride columns
    keys = torch.randn(cap, D, generator=g).to(torch.bfloat16)
    return qi, wts, keys


@pytest.mark.parametrize("rows_per_program", [1, 4])
def test_scores_match_reference(rows_per_program):
    R, pos0, cap, NT = 5, 100, 256, 192
    qi, wts, keys = _index_case(5, R, pos0, cap)
    out = torch.full((R, NT), 7.0)
    F.score(qi, wts, keys, torch.tensor([pos0], dtype=torch.int32), R, NT, out, rows_per_program)
    ref = scores_ref(qi, wts, keys, pos0, R, NT)
    fin = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(out), fin)
    assert torch.allclose(out[fin], ref[fin], rtol=1e-4, atol=1e-4)


def test_scores_rows_are_independent():
    """A row's scores have the same bits alone, in a decode window and in prompt row blocks."""
    R, pos0, cap, NT = 6, 70, 192, 128
    qi, wts, keys = _index_case(6, R, pos0, cap)
    pos = torch.tensor([pos0], dtype=torch.int32)
    full = torch.empty(R, NT)
    F.score(qi, wts, keys, pos, R, NT, full)
    blocks = torch.empty(R, NT)
    F.score(qi, wts, keys, pos, R, NT, blocks, rows_per_program=4)
    assert torch.equal(full, blocks)
    for r in range(R):
        one = torch.empty(1, NT)
        F.score(qi[r:r + 1].contiguous(), wts[r:r + 1], keys, pos + r, 1, NT, one)
        assert torch.equal(one[0], full[r])


def select_ref(scores, pos0, k):
    """k best per row, ties to the lower token, ascending; count k when the row sees more than k tokens."""
    R, NT = scores.shape
    toks, cnts = [], []
    for r in range(R):
        vis = pos0 + r + 1
        n = min(NT, max(vis, k))
        s = scores[r, :n]
        order = sorted(range(n), key=lambda t: (-float(s[t]) if math.isfinite(float(s[t])) else math.inf, t))
        toks.append(sorted(order[:k]))
        cnts.append(k if vis > k else 0)
    return toks, cnts


@pytest.mark.parametrize("pos0,NT", [(40, 64), (200, 256), (63, 128)])
def test_select_matches_reference_with_ties(pos0, NT):
    k = 64
    g = torch.Generator().manual_seed(7)
    R = 4
    scores = torch.randint(0, 6, (R, NT), generator=g).float()                # many ties
    scores[:, ::7] = 0.0
    scores[0, 3] = -0.0
    vis = torch.arange(NT)[None, :] <= (pos0 + torch.arange(R))[:, None]
    scores = torch.where(vis, scores, torch.full_like(scores, float("-inf")))
    tokens = torch.full((R, k), -5, dtype=torch.int32)
    counts = torch.full((R,), -5, dtype=torch.int32)
    F.select(scores, torch.tensor([pos0], dtype=torch.int32), NT, tokens, counts, k)
    toks, cnts = select_ref(scores, pos0, k)
    for r in range(R):
        assert counts[r].item() == cnts[r]
        if cnts[r]:
            assert tokens[r].tolist() == toks[r]


def test_select_tokens_decode_and_prompt_agree():
    """The same rows selected as a decode window (bucketed) and as a prompt chunk (visible columns) agree."""
    k, R, pos0, cap = 64, 5, 90, 256
    qi, wts, keys = _index_case(8, R, pos0, cap)
    pos = torch.tensor([pos0], dtype=torch.int32)
    a_tok, a_cnt = F.select_tokens(qi, wts, keys, pos0, R, pos, bucket_cols=256, k=k)
    b_tok, b_cnt = F.select_tokens(qi, wts, keys, pos0, R, pos, prompt=True, k=k)
    assert torch.equal(a_cnt, b_cnt) and a_cnt.tolist() == [k] * R
    assert torch.equal(a_tok, b_tok)
    ref = scores_ref(qi, wts, keys, pos0, R, 256)
    toks, _ = select_ref(ref, pos0, k)
    agree = sum(len(set(a_tok[r].tolist()) & set(toks[r])) for r in range(R)) / (R * k)
    assert agree > 0.97                                     # fp32 summation order may swap near-ties


def test_buckets():
    assert F.bucket(2048, 1, 1 << 20) == 4096 and F.bucket(4095, 1, 1 << 20) == 4096
    assert F.bucket(4096, 1, 1 << 20) == 8192 and F.bucket(100_000, 16, 131_072) == 131_072
    assert F.buckets(131_072)[-1] == 131_072 and F.buckets(131_072)[0] == 4096
    assert all(F.bucket(p, 16, 131_072) in F.buckets(131_072) for p in range(2048, 131_000, 977))
