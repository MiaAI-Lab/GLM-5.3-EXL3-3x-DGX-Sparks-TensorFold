"""TF_GLM_KV=fp4: the latent rows as e2m1 codes with an e4m3 scale a block of 16 and a power-of-two row scale. The
write kernel stores what the torch definition (kv8.quantize_rows4) does, byte for byte; the readers' tiles are its
values exactly; the round trip's error is what 4 bits allow."""

import pytest
import torch

from tensorfold.families.glm5_next.cuda import kv8, latent


@pytest.mark.parametrize("seed", [0, 1])
def test_write_kernel_is_the_definition(seed):
    g = torch.Generator().manual_seed(seed)
    R, W, T = 9, 512, 24
    lat = (torch.randn(R, W, generator=g) * torch.linspace(0.01, 3.0, W)).to(torch.bfloat16)
    lat[3] = 0                                                         # a zero row
    lat[4, :16] = 0                                                    # a zero block
    cache = kv8.zeros(T, W, "fp4", "cpu")
    latent.latent_write(lat, cache, torch.tensor([5], dtype=torch.int32))
    want = kv8.quantize_rows4(lat)
    assert torch.equal(cache[5:5 + R], want)
    assert (cache[:5] == 0).all() and (cache[5 + R:] == 0).all()
    assert kv8.width(cache) == W and kv8.kind_of(cache) == "fp4"


def test_round_trip_error():
    """Relative error of a row of normal values: within what e2m1 with an e4m3 block scale gives (~8-10%)."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(256, 512, generator=g).to(torch.bfloat16)
    y = kv8.dequantize4(kv8.quantize_rows4(x))
    rel = ((y - x.float()).norm(dim=1) / x.float().norm(dim=1))
    y8 = kv8.dequantize(kv8.quantize_rows(x))
    rel8 = ((y8 - x.float()).norm(dim=1) / x.float().norm(dim=1))
    print(f"\n[kv4] relative row error fp4 {rel.mean():.4f} (max {rel.max():.4f}), fp8 {rel8.mean():.4f}")
    assert rel.mean() < 0.12 and rel8.mean() < rel.mean()


def test_readers_see_the_exact_values():
    """latent.sparse_attention on an FP4 cache equals it on a bf16 cache holding the dequantized rows (bf16-exact),
    within fp32 summation order."""
    g = torch.Generator().manual_seed(4)
    T, H, W, R = 96, 4, 128, 5
    lat = torch.randn(T, W, generator=g).to(torch.bfloat16)
    c4 = kv8.zeros(T, W, "fp4", "cpu")
    latent.latent_write(lat, c4, torch.tensor([0], dtype=torch.int32))
    deq = kv8.dequantize4(c4).to(torch.bfloat16)
    assert torch.equal(deq.float(), kv8.dequantize4(c4))                 # bf16-exact values
    cb = deq.clone()
    qa = (0.3 * torch.randn(R, H, W, generator=g)).to(torch.bfloat16)
    tok = torch.stack([torch.randperm(T, generator=g)[:32].sort().values for _ in range(R)]).int()
    cnt = torch.full((R,), 32, dtype=torch.int32)
    o4 = torch.zeros(R, H, W, dtype=torch.bfloat16)
    ob = torch.zeros_like(o4)
    latent.sparse_attention(qa, c4, tok, cnt, o4, 0.1)
    latent.sparse_attention(qa, cb, tok, cnt, ob, 0.1)
    assert torch.allclose(o4.float(), ob.float(), atol=2e-2, rtol=2e-2), (o4.float() - ob.float()).abs().max()
