"""The latent (MLA) attention kernels with full GLM-5.3's rotary term, against a torch reference, on CPU."""

import pytest
import torch

from tensorfold.families.glm5_next.cuda import kv8, latent

LW, ROPE = 512, 64
SCALE = 256 ** -0.5


def case(seed, R, H, T):
    g = torch.Generator().manual_seed(seed)
    qa = (0.3 * torch.randn(R, H, LW, generator=g)).to(torch.bfloat16)
    qp = (0.3 * torch.randn(R, H, ROPE, generator=g)).to(torch.bfloat16)
    lat = torch.randn(T, LW, generator=g).to(torch.bfloat16)
    kr = torch.randn(T, ROPE, generator=g).to(torch.bfloat16)
    return qa, qp, lat, kr


def reference(qa, qp, lat, kr, keys_of_row):
    """out[r, h] = sum_t softmax_t(scale (qa . lat_t + qp . kr_t)) lat_t over the row's keys (fp32)."""
    R, H, _ = qa.shape
    out = torch.zeros(R, H, LW)
    for r in range(R):
        idx = torch.tensor(keys_of_row(r), dtype=torch.long)
        s = (qa[r].float() @ lat[idx].float().T + (qp[r].float() @ kr[idx].float().T if qp is not None else 0)) * SCALE
        out[r] = torch.softmax(s, dim=-1) @ lat[idx].float()
    return out


def close(got, ref, tol=2e-2):
    return (got.float() - ref).abs().max().item() <= tol * max(1.0, ref.abs().max().item())


@pytest.mark.parametrize("pos0,R", [(0, 1), (37, 3), (600, 2)])
def test_dense_attention_with_rope(pos0, R):
    H = 22                                                        # a rank's heads at three ranks (rank 0)
    T = pos0 + R + 5
    qa, qp, lat, kr = case(1, R, H, T)
    s = latent.LatentScratch(R, H, latent.chunks_for(T), "cpu")
    out = torch.empty(R, H, LW, dtype=torch.bfloat16)
    pos = torch.tensor([pos0], dtype=torch.int32)
    latent.attention(qa, lat, pos, s, scale=SCALE, nch=latent.chunks_for(pos0 + R), out=out, qp=qp, rope=kr)
    ref = reference(qa, qp, lat, kr, lambda r: range(pos0 + r + 1))
    assert close(out, ref)
    # without the rotary term the kernels are the Flash engine's: the reference without it
    plain = torch.empty_like(out)
    latent.attention(qa, lat, pos, s, scale=SCALE, nch=latent.chunks_for(pos0 + R), out=plain)
    assert close(plain, reference(qa, None, lat, kr, lambda r: range(pos0 + r + 1)))
    if pos0 + R > 1:                                              # one key: softmax 1 either way
        assert not close(plain, ref, tol=1e-3)


def test_dense_rows_keep_their_bits():
    """A window's rows equal each row run alone (drafted == serial at the kernel level)."""
    H, pos0, R = 22, 40, 4
    qa, qp, lat, kr = case(2, R, H, pos0 + R)
    nch = latent.chunks_for(pos0 + R)
    s = latent.LatentScratch(R, H, nch, "cpu")
    full = torch.empty(R, H, LW, dtype=torch.bfloat16)
    latent.attention(qa, lat, torch.tensor([pos0], dtype=torch.int32), s, scale=SCALE, nch=nch, out=full, qp=qp,
                     rope=kr, hb=latent.HB)
    for r in range(R):
        one = torch.empty(1, H, LW, dtype=torch.bfloat16)
        latent.attention(qa[r:r + 1].contiguous(), lat, torch.tensor([pos0 + r], dtype=torch.int32), s, scale=SCALE,
                         nch=nch, out=one, qp=qp[r:r + 1].contiguous(), rope=kr, hb=latent.HB)
        assert torch.equal(one[0], full[r])


@pytest.mark.parametrize("onepass", [False, True])
def test_sparse_attention_with_rope(onepass):
    H, R, T, K = 21, 3, 300, 64
    qa, qp, lat, kr = case(3, R, H, T)
    g = torch.Generator().manual_seed(9)
    lists = [sorted(torch.randperm(T, generator=g)[:K].tolist()) for _ in range(R)]
    tokens = torch.tensor(lists, dtype=torch.int32)
    counts = torch.tensor([K, 0, K], dtype=torch.int32)              # row 1 dense: left alone
    out = torch.full((R, H, LW), 3.0, dtype=torch.bfloat16)
    if onepass:
        latent.sparse_onepass(qa, lat, tokens, counts, out, SCALE, qp=qp, rope=kr)
    else:
        latent.sparse_attention(qa, lat, tokens, counts, out, SCALE, qp=qp, rope=kr)
    ref = reference(qa, qp, lat, kr, lambda r: lists[r])
    assert close(out[0], ref[0]) and close(out[2], ref[2])
    assert torch.all(out[1] == 3.0)


def test_fp8_latent_cache_keeps_rope_exact():
    """TF_GLM_KV=fp8: the latent rows quantized, the rotary keys still bf16 in their own plane."""
    H, pos0, R = 16, 20, 2
    T = pos0 + R
    qa, qp, lat, kr = case(4, R, H, T)
    cache = kv8.zeros(T, LW, "fp8", "cpu")
    pos = torch.tensor([0], dtype=torch.int32)
    latent.latent_write(lat, cache, pos)
    deq = kv8.dequantize(cache) if hasattr(kv8, "dequantize") else None
    s = latent.LatentScratch(R, H, latent.chunks_for(T), "cpu")
    out = torch.empty(R, H, LW, dtype=torch.bfloat16)
    latent.attention(qa, cache, torch.tensor([pos0], dtype=torch.int32), s, scale=SCALE, nch=latent.chunks_for(T),
                     out=out, qp=qp, rope=kr)
    base = deq if deq is not None else lat
    ref = reference(qa, qp, base.to(torch.bfloat16), kr, lambda r: range(pos0 + r + 1))
    assert close(out, ref)
