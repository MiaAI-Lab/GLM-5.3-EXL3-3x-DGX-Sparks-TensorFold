"""0132 (TF_GLM_CP_RAW_Q) on one GPU: a peer's absorption of raw queries (the key dims only, DQ = 192, its heads padded to
HP with zero rows, rows in parts) gives the owner's absorb_q bits (the whole 256-dim query head over kv_b's key rows
padded with zero rotary rows, all of a chunk's rows at once), bit for bit, on the tensor cores; and a timing of the
extra absorption (two peer groups of a 2,048-row chunk) printed with -s."""

import time

import pytest
import torch

from tensorfold.families.glm5_next.cuda import latent

DEV = "cuda"


@pytest.mark.parametrize("HL,HP", [(22, 22), (21, 22)])
def test_peer_absorb_equals_owner_absorb(HL, HP):
    torch.manual_seed(HL)
    R, nope, rope, LW = 2048, 192, 64, 512
    q = torch.randn(R, HL, nope + rope, device=DEV).bfloat16()
    q[:, :, nope:] *= 1000                          # rotary dims: their products with the zero rows must not matter
    k_rows = (torch.randn(HL, nope, LW, device=DEV) * 0.05).bfloat16()
    wk = torch.zeros(HL, nope + rope, LW, dtype=torch.bfloat16, device=DEV)
    wk[:, :nope] = k_rows
    owner = latent.AbsorbW(wk, torch.zeros(HL, 256, LW, dtype=torch.bfloat16, device=DEV))
    want = torch.empty(R, HL, LW, dtype=torch.bfloat16, device=DEV)
    latent.absorb_q(q, owner, want, prompt=True)
    peer_w = torch.zeros(HP, nope, LW, dtype=torch.bfloat16, device=DEV)
    peer_w[:HL] = k_rows
    peer = latent.AbsorbW(peer_w, torch.empty(HP, 0, LW, dtype=torch.bfloat16, device=DEV))
    for r0, r1 in [(0, 256), (256, 768), (768, 1280), (1280, 1792), (1792, 2048), (100, 133)]:
        n = r1 - r0
        raw = torch.zeros(n, HP, nope, dtype=torch.bfloat16, device=DEV)
        raw[:, :HL] = q[r0:r1, :, :nope]
        got = torch.empty(n, HP, LW, dtype=torch.bfloat16, device=DEV)
        latent.absorb_q(raw, peer, got, prompt=True)
        assert torch.equal(got[:, :HL], want[r0:r1]), (r0, r1)
        assert (got[:, HL:] == 0).all()


def test_peer_absorb_timing():
    R, HP, nope, LW = 2048, 22, 192, 512
    peer = latent.AbsorbW((torch.randn(HP, nope, LW, device=DEV) * 0.05).bfloat16(),
                          torch.empty(HP, 0, LW, dtype=torch.bfloat16, device=DEV))
    raw = torch.randn(R, HP, nope, device=DEV).bfloat16()
    out = torch.empty(R, HP, LW, dtype=torch.bfloat16, device=DEV)

    def two():
        for _ in range(2):
            latent.absorb_q(raw, peer, out, prompt=True)

    two()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        two()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / 10 * 1e3
    print(f"\n[cp raw q] two peer groups' absorption of a 2,048-row chunk: {ms:.2f} ms a layer")
    assert ms < 5
