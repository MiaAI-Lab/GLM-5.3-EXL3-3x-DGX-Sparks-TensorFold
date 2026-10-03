"""The real checkpoint's mixed-width routed experts (CHECKPOINT: the snapshot folder; one layer, one rank's share,
as load_full slices and packs it) through the universal EXL3 kernels: each slot's rows against a float64 reference
of the stored trellis, and each row's bits alone and in windows."""

import os
from pathlib import Path

import pytest
import torch

CHECKPOINT = os.environ.get("CHECKPOINT", "")
pytestmark = pytest.mark.skipif(not CHECKPOINT, reason="CHECKPOINT (the real snapshot folder) not given")


@pytest.mark.parametrize("rank", [0, 2])
def test_layer_3_experts_of_a_rank(rank):
    from tensorfold.cuda.exl3 import experts as generic
    from tensorfold.cuda.exl3 import format as fmt
    from tensorfold.families.glm5_next.cuda.split import RankReader
    from tensorfold.families.glm5_next.cuda.weights import Config, expert_tensor_names, pack_experts

    folder = Path(CHECKPOINT)
    cfg = Config.read(folder)
    rd = RankReader(folder, rank, 3)
    layer, E = 3, cfg.experts
    names = expert_tensor_names(cfg.prefix, layer, E)
    rd.prefetch(names, None)
    parts = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for e in range(E):
            n = f"{cfg.prefix}layers.{layer}.mlp.experts.{e}.{proj}."
            parts[(proj, e)] = (rd.get(n + "trellis"), rd.get(n + "suh"), rd.get(n + "svh"))
    rd.close()
    ex = pack_experts(parts, E, torch.device("cuda"))
    widths = sorted({parts[("down_proj", e)][0].shape[-1] // 16 for e in range(E)})
    assert len(widths) >= 2, widths                  # mixed widths in this layer
    R, slots = 16, cfg.top_k + 1
    g = torch.Generator().manual_seed(rank)
    x = (torch.randn(R, cfg.hidden, generator=g) * 0.5).to(torch.bfloat16).cuda()
    pick = torch.stack([torch.randperm(E, generator=g)[:slots] for _ in range(R)]).to(torch.int32)
    pick[:, -1] = E                                  # the shared expert's slot: skipped by the routed kernels
    pick = pick.cuda()
    s = generic.Scratch(ex, R, slots, device="cuda")
    y = generic.routed(x, pick, None, ex, s, None, R, act_mode=generic.ACT_BF16).view(R, slots, -1).clone()
    for r in range(R):                               # each row alone: the same bits
        one = generic.routed(x[r:r + 1], pick[r:r + 1].contiguous(), None, ex, s, None, 1,
                             act_mode=generic.ACT_BF16).view(1, slots, -1)
        assert torch.equal(one[0, :-1], y[r, :-1])
    # float64 reference of a few (row, slot) pairs
    for r, k in ((0, 0), (5, 3), (15, 7)):
        e = int(pick[r, k])
        xf = x[r].double().cpu().numpy()

        def mat(proj):
            t, su, sv = parts[(proj, e)]
            return fmt.dequantize(t.numpy(), su.numpy(), sv.numpy(), t.shape[-1] / 16, "mcg")
        gt, up = xf @ mat("gate_proj"), xf @ mat("up_proj")
        act = (gt / (1 + __import__("numpy").exp(-gt))) * up
        want = torch.from_numpy(act @ mat("down_proj")).float()
        got = y[r, k].float().cpu()
        assert (got - want).abs().max() <= 0.02 * want.abs().max() + 1e-3, (r, k, e)


def test_a_prompt_chunk_of_routed_experts_runs_in_blocks():
    """A 2048-row prompt chunk's routed experts through the 2-tile kernel in one block (the grouping kernel opts in
    past 48 KiB of shared memory): each row's bits equal the row alone through the decode kernel."""
    from tensorfold.cuda.exl3 import experts as generic
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda.split import RankReader
    from tensorfold.families.glm5_next.cuda.weights import Config, expert_tensor_names, pack_experts

    folder = Path(CHECKPOINT)
    cfg = Config.read(folder)
    rd = RankReader(folder, 1, 3)
    rd.prefetch(expert_tensor_names(cfg.prefix, 4, cfg.experts), None)
    parts = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for e in range(cfg.experts):
            n = f"{cfg.prefix}layers.4.mlp.experts.{e}.{proj}."
            parts[(proj, e)] = (rd.get(n + "trellis"), rd.get(n + "suh"), rd.get(n + "svh"))
    rd.close()
    ex = pack_experts(parts, cfg.experts, torch.device("cuda"))
    R, slots, B = 2048, cfg.top_k + 1, F.EXL3_BLOCK_ROWS
    g = torch.Generator().manual_seed(5)
    x = (torch.randn(R, cfg.hidden, generator=g) * 0.5).to(torch.bfloat16).cuda()
    pick = torch.stack([torch.randperm(cfg.experts, generator=g)[:slots] for _ in range(R)]).to(torch.int32)
    pick[:, -1] = cfg.experts
    pick = pick.cuda()
    s = generic.Scratch(ex, R, slots, device="cuda")
    out = torch.empty(R, slots, cfg.hidden, device="cuda")
    for a in range(0, R, B):
        z = min(R, a + B)
        y = generic.routed(x[a:z], pick[a:z], None, ex, s, None, z - a, act_mode=generic.ACT_BF16, mt=2)
        out[a:z] = y.view(z - a, slots, -1)
    for r in (0, 1500, 2047):                     # a row alone, through the one-tile decode kernel: the same bits
        one = generic.routed(x[r:r + 1], pick[r:r + 1].contiguous(), None, ex, s, None, 1,
                             act_mode=generic.ACT_BF16).view(1, slots, -1)
        assert torch.equal(one[0, :-1], out[r, :-1])


def test_prompt_kernels_on_a_real_layer():
    """The prompt kernels (pairs mode) on a 2048-row chunk of a real layer: the same bits run to run and a row alone,
    and a few (row, slot) pairs against the float64 reference of the stored trellis."""
    import numpy as np

    from tensorfold.cuda.exl3 import experts as generic
    from tensorfold.cuda.exl3 import format as fmt
    from tensorfold.cuda.exl3 import prompt_experts as pe
    from tensorfold.families.glm5_next.cuda.split import RankReader
    from tensorfold.families.glm5_next.cuda.weights import Config, expert_tensor_names, pack_experts

    folder = Path(CHECKPOINT)
    cfg = Config.read(folder)
    rd = RankReader(folder, 2, 3)
    layer, E = 5, cfg.experts
    rd.prefetch(expert_tensor_names(cfg.prefix, layer, E), None)
    parts = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for e in range(E):
            n = f"{cfg.prefix}layers.{layer}.mlp.experts.{e}.{proj}."
            parts[(proj, e)] = (rd.get(n + "trellis"), rd.get(n + "suh"), rd.get(n + "svh"))
    rd.close()
    ex = pack_experts(parts, E, torch.device("cuda"))
    assert pe.supported(ex)
    R, slots = 2048, cfg.top_k + 1
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(R, cfg.hidden, generator=g) * 0.5).to(torch.bfloat16).cuda()
    w = torch.rand(E, generator=g) ** 3 + 0.05            # skewed like a real prompt: some experts take many rows
    pick = torch.stack([torch.multinomial(w, slots - 1, generator=g) for _ in range(R)]).to(torch.int32)
    pick = torch.cat([pick, torch.full((R, 1), E, dtype=torch.int32)], 1).cuda()
    s = pe.PromptScratch(ex, R, slots, device="cuda")
    y = torch.zeros(R * slots, cfg.hidden, device="cuda")
    a = pe.prompt_pairs(x, pick, ex, s, y, act_mode=generic.ACT_BF16).view(R, slots, -1)[:, :-1].clone()
    y.fill_(float("nan"))
    b = pe.prompt_pairs(x, pick, ex, s, y, act_mode=generic.ACT_BF16).view(R, slots, -1)[:, :-1]
    assert torch.equal(a, b)
    one = pe.PromptScratch(ex, 1, slots, device="cuda")
    y1 = torch.zeros(slots, cfg.hidden, device="cuda")
    for r in (0, 777, 2047):
        got = pe.prompt_pairs(x[r:r + 1], pick[r:r + 1].contiguous(), ex, one, y1, act_mode=generic.ACT_BF16)
        assert torch.equal(got.view(slots, -1)[:-1], a[r])
    for r, k in ((0, 0), (900, 4), (2047, 7)):
        e = int(pick[r, k])
        xf = x[r].double().cpu().numpy()

        def mat(proj):
            t, su, sv = parts[(proj, e)]
            return fmt.dequantize(t.numpy(), su.numpy(), sv.numpy(), t.shape[-1] / 16, "mcg")
        gt, up = xf @ mat("gate_proj"), xf @ mat("up_proj")
        want = torch.from_numpy(((gt / (1 + np.exp(-gt))) * up) @ mat("down_proj")).float()
        got = a[r, k].float().cpu()
        assert (got - want).abs().max() <= 0.02 * want.abs().max() + 1e-3, (r, k, e)


def test_mia_prompt_experts_on_a_real_layer():
    """Mia's prompt experts (cuda/exl3/mpe) on a 2048-row chunk of a real layer: the layer's experts share their
    gate/up input scales, the same bits run to run and a row alone, a few (row, slot) pairs against the float64
    reference of the stored trellis, and within fp16 rounding of the universal kernels."""
    import numpy as np

    from tensorfold.cuda.exl3 import experts as generic
    from tensorfold.cuda.exl3 import format as fmt
    from tensorfold.cuda.exl3 import mpe as pe
    from tensorfold.families.glm5_next.cuda.split import RankReader
    from tensorfold.families.glm5_next.cuda.weights import Config, expert_tensor_names, pack_experts

    folder = Path(CHECKPOINT)
    cfg = Config.read(folder)
    rd = RankReader(folder, 2, 3)
    layer, E = 6, cfg.experts
    rd.prefetch(expert_tensor_names(cfg.prefix, layer, E), None)
    parts = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for e in range(E):
            n = f"{cfg.prefix}layers.{layer}.mlp.experts.{e}.{proj}."
            parts[(proj, e)] = (rd.get(n + "trellis"), rd.get(n + "suh"), rd.get(n + "svh"))
    rd.close()
    ex = pack_experts(parts, E, torch.device("cuda"))
    assert pe.supported(ex)
    R, slots = 2048, cfg.top_k + 1
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(R, cfg.hidden, generator=g) * 0.5).to(torch.bfloat16).cuda()
    w = torch.rand(E, generator=g) ** 3 + 0.05            # skewed like a real prompt: some experts take many rows
    pick = torch.stack([torch.multinomial(w, slots - 1, generator=g) for _ in range(R)]).to(torch.int32)
    pick = torch.cat([pick, torch.full((R, 1), E, dtype=torch.int32)], 1).cuda()
    s = pe.Scratch(ex, R, slots, device="cuda")
    y = torch.zeros(R * slots, cfg.hidden, device="cuda")
    a = pe.pairs(x, pick, ex, s, y, act_mode=generic.ACT_BF16).view(R, slots, -1)[:, :-1].clone()
    y.fill_(float("nan"))
    b = pe.pairs(x, pick, ex, s, y, act_mode=generic.ACT_BF16).view(R, slots, -1)[:, :-1]
    assert torch.equal(a, b)
    gs = generic.Scratch(ex, R, slots, device="cuda")
    u = generic.routed(x, pick, None, ex, gs, None, R, act_mode=generic.ACT_BF16, mt=2).view(R, slots, -1)[:, :-1]
    assert ((a - u).norm() / u.norm()).item() < 1e-3
    one = pe.Scratch(ex, 1, slots, device="cuda")
    y1 = torch.zeros(slots, cfg.hidden, device="cuda")
    for r in (0, 777, 2047):
        got = pe.pairs(x[r:r + 1], pick[r:r + 1].contiguous(), ex, one, y1, act_mode=generic.ACT_BF16)
        assert torch.equal(got.view(slots, -1)[:-1], a[r])
    for r, k in ((0, 0), (900, 4), (2047, 7)):
        e = int(pick[r, k])
        xf = x[r].double().cpu().numpy()

        def mat(proj):
            t, su, sv = parts[(proj, e)]
            return fmt.dequantize(t.numpy(), su.numpy(), sv.numpy(), t.shape[-1] / 16, "mcg")
        gt, up = xf @ mat("gate_proj"), xf @ mat("up_proj")
        want = torch.from_numpy(((gt / (1 + np.exp(-gt))) * up) @ mat("down_proj")).float()
        got = a[r, k].float().cpu()
        assert (got - want).abs().max() <= 0.02 * want.abs().max() + 1e-3, (r, k, e)
