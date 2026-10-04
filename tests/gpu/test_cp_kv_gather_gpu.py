"""0141 on one GPU: Mia's sparse attention's FP4 tiles dequantized by the codes' bits give the constant table's bits
(FP4 and fp4x caches); prompt rows over the gathered planes of a context-parallel layer (forward.cp_prompt_kvg's
rank-major layout) give the bits of the same rows over the position-ordered cache; and timings (printed with -s) of a
3,072-row chunk's attention: FP8 against FP4 (table and bits), and context parallelism's three head groups over a
third of the tokens against one call over the whole top-k."""

import time

import pytest
import torch

from tensorfold.families.glm5_next.cuda import kv8, latent, msa

DEV = "cuda"
G = 3


def _case(R, T, K, kind, seed=5, H=22):
    torch.manual_seed(seed)
    lat = (2.0 * torch.randn(T, 512, device=DEV)).bfloat16()
    lat[::17] = 0                                  # zero rows / blocks: zero scales, signed zeros
    lat[1::13, :64] = -lat[1::13, :64].abs() * 1e-3
    kr = torch.randn(T, 64, device=DEV).bfloat16()
    cache = kv8.zeros(T, 512, kind, DEV)
    latent.latent_write(lat, cache, torch.tensor([0], dtype=torch.int32, device=DEV))
    rope = kr
    if kind == "fp4x":
        rope = kv8.rope_zeros(T, 64, kind, DEV)
        from tensorfold.families.glm5_next.cuda import dsa_full

        freq = torch.ones(32, device=DEV)
        dp = torch.zeros(T, 64, device=DEV).bfloat16()
        dp.copy_(kr)
        dsa_full.k_rope_write(dp, rope, torch.tensor([0], dtype=torch.int32, device=DEV), freq, latent=cache)
    qa = (0.3 * torch.randn(R, H, 512, device=DEV)).bfloat16()
    qp = (0.3 * torch.randn(R, H, 64, device=DEV)).bfloat16()
    tok = torch.full((R, K), -1, dtype=torch.int32, device=DEV)
    cnt = torch.zeros(R, dtype=torch.int32, device=DEV)
    for r in range(R):
        m = [K, K // 3, 9, 1][r % 4]
        tok[r, :m] = torch.randperm(T, device=DEV)[:m].sort().values.int()
        cnt[r] = m
    return qa, qp, cache, rope, tok, cnt


@pytest.mark.parametrize("kind", ["fp4", "fp4x"])
def test_fp4_bits_equal_table(kind):
    qa, qp, cache, rope, tok, cnt = _case(64, 4096, 2048, kind)
    outs = []
    for lut in (True, False):
        msa.set_fp4_lut(lut)
        o = torch.empty_like(qa)
        lse = torch.empty(qa.shape[:2], dtype=torch.float32, device=DEV)
        msa.prompt_lse(qa, cache, tok, cnt, o, lse, 0.07, qp, rope)
        o2 = torch.empty_like(qa)
        msa.prompt(qa, cache, tok, cnt, o2, 0.07, qp, rope)
        outs.append((o, lse, o2))
    msa.set_fp4_lut(False)
    for a, b in zip(*outs):
        assert torch.equal(a.view(torch.int16) if a.dtype == torch.bfloat16 else a.view(torch.int32),
                           b.view(torch.int16) if b.dtype == torch.bfloat16 else b.view(torch.int32))


@pytest.mark.parametrize("kind", ["fp8", "fp4"])
def test_gathered_planes_have_position_bits(kind):
    """Rows over the gathered planes (row (p % G) S + p // G) equal the rows over the position-ordered cache."""
    T = 3000
    qa, qp, cache, rope, tok, cnt = _case(48, T, 2048, kind)
    S = -(-T // G)
    glat = torch.zeros((G * S, cache.shape[1]), dtype=cache.dtype, device=DEV)
    grope = torch.zeros((G * S, rope.shape[1]), dtype=rope.dtype, device=DEV)
    p = torch.arange(T, device=DEV)
    rows = (p % G) * S + p // G
    glat[rows] = cache
    grope[rows] = rope
    gtok = torch.where(tok >= 0, rows[tok.clamp(min=0).long()].int(), -1).contiguous()
    a = torch.empty_like(qa)
    b = torch.empty_like(qa)
    msa.prompt(qa, cache, tok, cnt, a, 0.07, qp, rope)
    msa.prompt(qa, glat, gtok, cnt, b, 0.07, qp, grope)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def _time(fn, n=5):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def test_chunk_attention_timing():
    """One layer of a 3,072-row chunk at 22 heads and 2,048 selected tokens a row (32k context)."""
    R, T, K, H = 3072, 32768, 2048, 22
    torch.manual_seed(1)
    tok = torch.stack([torch.randperm(T, device=DEV)[:K].sort().values for _ in range(R)]).int().contiguous()
    cnt = torch.full((R,), K, dtype=torch.int32, device=DEV)
    qa = (0.3 * torch.randn(R, H, 512, device=DEV)).bfloat16()
    qp = (0.3 * torch.randn(R, H, 64, device=DEV)).bfloat16()
    lat = torch.randn(T, 512, device=DEV).bfloat16()
    kr = torch.randn(T, 64, device=DEV).bfloat16()
    out = torch.empty_like(qa)
    for kind in ("fp8", "fp4"):
        cache = kv8.zeros(T, 512, kind, DEV)
        latent.latent_write(lat, cache, torch.tensor([0], dtype=torch.int32, device=DEV))
        luts = (False,) if kind == "fp8" else (True, False)
        for lut in luts:
            msa.set_fp4_lut(lut)
            whole = _time(lambda: msa.prompt(qa, cache, tok, cnt, out, 0.07, qp, kr))
            # context parallelism: three head groups (22 heads) over a third of the tokens each (683)
            loc = (tok[:, ::3] // 3).contiguous()
            lc = torch.full((R,), loc.shape[1], dtype=torch.int32, device=DEV)
            lse = torch.empty((R, H), dtype=torch.float32, device=DEV)
            third = cache[:T // 3 + 1].contiguous()
            groups = _time(lambda: [msa.prompt_lse(qa, third, loc, lc, out, lse, 0.07, qp, kr[:T // 3 + 1].contiguous())
                                    for _ in range(G)])
            print(f"\n[msa] {kind}{' table' if lut else ''}: one call 22 heads x 2048 tokens {whole:.1f} ms; "
                  f"CP 3 x (22 heads x 683 tokens) {groups:.1f} ms (3,072 rows)", flush=True)
    msa.set_fp4_lut(False)
