"""Context parallelism's cache writes (TF_GLM_CP): every rank keeps the positions p with p % cp == rank at slot
p // cp; the ranks' rows put together are the replicated cache's, byte for byte (latent bf16 and FP8, rotary keys,
index keys)."""

import pytest
import torch

from tensorfold.families.glm5_next.cuda import dsa_full as F
from tensorfold.families.glm5_next.cuda import kv8, latent

THETA = 8_000_000.0
CP = 3


@pytest.mark.parametrize("pos0", [0, 7])
@pytest.mark.parametrize("kind", ["bf16", "fp8"])
def test_owner_writes_put_together_are_the_replicated_cache(pos0, kind):
    g = torch.Generator().manual_seed(pos0 + (kind == "fp8"))
    R, T = 11, 32
    pos = torch.tensor([pos0], dtype=torch.int32)
    freq = F.inv_freq(64, THETA, "cpu")
    lat = torch.randn(R, 512, generator=g).bfloat16()
    kr = torch.randn(R, 64, generator=g).bfloat16()
    raw = torch.randn(R, 128, generator=g).bfloat16()
    lw, lb = torch.randn(128, generator=g).bfloat16(), torch.randn(128, generator=g).bfloat16()
    full = kv8.zeros(T, 512, kind, "cpu")
    full_kr = torch.zeros(T, 64, dtype=torch.bfloat16)
    full_ik = torch.zeros(T, 128, dtype=torch.bfloat16)
    latent.latent_write(lat, full, pos)
    F.k_rope_write(kr, full_kr, pos, freq)
    F.index_write(raw, lw, lb, full_ik, pos, freq)
    local = -(-T // CP) + 1
    for rank in range(CP):
        c = kv8.zeros(local, 512, kind, "cpu")
        ckr = torch.zeros(local, 64, dtype=torch.bfloat16)
        cik = torch.zeros(local, 128, dtype=torch.bfloat16)
        latent.latent_write(lat, c, pos, cp=CP, rank=rank)
        F.k_rope_write(kr, ckr, pos, freq, cp=CP, rank=rank)
        F.index_write(raw, lw, lb, cik, pos, freq, cp=CP, rank=rank)
        mine = torch.arange(rank, T, CP)
        assert torch.equal(c[:len(mine)], full[mine])
        assert torch.equal(ckr[:len(mine)], full_kr[mine])
        assert torch.equal(cik[:len(mine)], full_ik[mine])
        assert (c[len(mine):] == 0).all()
