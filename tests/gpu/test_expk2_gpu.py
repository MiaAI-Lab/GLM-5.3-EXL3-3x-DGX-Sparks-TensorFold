"""expk2: the routed-expert decode variants (tensorfold/cuda/exl3/experts.py KNOBS: fused gate/up epilogue, fused down
epilogue + combine, grouping + rot_in in one launch, the one-buffer reduction with bounded registers, 4 n tiles a
program) give the default path's bits: y (per-slot rows), out (the combine), xd and the whole partial buffer z, at
1..33 rows (member tiles 1 and 3), random and repeated picks, mixed widths in both instance ranges, codebooks mcg and
3inst, eager and replayed from a CUDA graph; every fused counter is back at 0 after the calls. CHECKPOINT given: also
one real layer (rank 1, layer 4)."""

import math
import os
from pathlib import Path

import pytest
import torch

VARIANTS = {
    "nt4x1": {"nt": "4x1"},
    "nt4x2": {"nt": "4x2"},
    "occ": {"occ": 1},
    "occ+nt4x1": {"occ": 1, "nt": "4x1"},
    "fgu": {"fuse_gu": 1},
    "fdn": {"fuse_dn": 1},
    "prep": {"prep": 1},
    "fused": {"fuse_gu": 1, "fuse_dn": 1, "prep": 1},
    "fused+occ": {"fuse_gu": 1, "fuse_dn": 1, "prep": 1, "occ": 1},
    "fused+nt4x2": {"fuse_gu": 1, "fuse_dn": 1, "prep": 1, "nt": "4x2"},
    "fused+occ+nt4x1": {"fuse_gu": 1, "fuse_dn": 1, "prep": 1, "occ": 1, "nt": "4x1"},
}
OFF = {"fuse_gu": 0, "fuse_dn": 0, "prep": 0, "occ": 0, "nt": ""}
ROWS = (1, 2, 3, 5, 9, 16, 17, 33)


def _trellis(k, n, k2, gen):
    v = torch.randint(-32768, 32768, (k // 16, n // 16, 8 * k2), dtype=torch.int32, generator=gen)
    return v.to(torch.int16).cuda().contiguous()


def _scale(n, mag, gen):
    sign = torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1
    return (sign * (torch.rand((n,), generator=gen) + 0.5) * mag).half().cuda()


def _layer(E, D, I, k2s, cb, seed):
    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(seed)
    gate, up, down = [], [], []
    for e in range(E):
        kg, ku, kd = k2s[e % len(k2s)]
        gate.append((_trellis(D, I, kg, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        up.append((_trellis(D, I, ku, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        down.append((_trellis(I, D, kd, g), _scale(I, 1 / math.sqrt(I), g), _scale(D, 0.25, g)))
    return experts.prepare(gate, up, down, cb)


def _bits(t):
    return t.contiguous().view(torch.int32) if t.dtype == torch.float32 else t.contiguous().view(torch.int16)


def _outputs(G, ex, x, pk, wts, y0, R, slots, limit, graph=False):
    s = G.Scratch(ex, 64, slots)
    s.y.copy_(y0)
    out = torch.empty((R, ex.dims), dtype=torch.float32, device="cuda")
    if graph:
        G.routed(x, pk, None, ex, s, None, R, limit, G.ACT_BF16)     # build + counters outside the capture
        torch.cuda.synchronize()
        s.y.copy_(y0)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            G.routed(x, pk, None, ex, s, None, R, limit, G.ACT_BF16)
        gr.replay()
        gr.replay()
    else:
        G.routed(x, pk, None, ex, s, None, R, limit, G.ACT_BF16)
    res = {"y": s.y.clone(), "xd": s.xd.clone(), "z": s.z.clone()}
    s.y.copy_(y0)
    G.routed(x, pk, wts, ex, s, out, R, limit, G.ACT_BF16)
    res.update({"out": out.clone(), "y(out)": s.y.clone(), "xd(out)": s.xd.clone(), "z(out)": s.z.clone()})
    torch.cuda.synchronize()
    cnt = s._cnt
    res["_cnt_zero"] = cnt is None or int(cnt.abs().sum()) == 0
    return res


def _check(ex, E, slots, limit, seed, rows=ROWS, graph=False):
    from tensorfold.cuda.exl3 import experts as G

    old = G.set_knobs(**OFF)
    try:
        for R in rows:
            for kind in ("rand", "same"):
                g = torch.Generator().manual_seed(seed + R)
                if kind == "rand":
                    pk = torch.stack([torch.randperm(E, generator=g)[:slots - 1] for _ in range(R)]).int()
                else:
                    pk = torch.randperm(E, generator=g)[:slots - 1].int().repeat(R, 1)
                pk = torch.cat([pk, torch.full((R, 1), E, dtype=torch.int32)], 1).cuda().contiguous()
                x = (torch.randn(R, ex.dims, generator=g) * 0.5).bfloat16().cuda()
                wts = torch.rand(R, slots, generator=g).cuda()
                y0 = torch.randn(64 * slots, ex.dims, generator=g).cuda()
                G.set_knobs(**OFF)
                ref = _outputs(G, ex, x, pk, wts, y0, R, slots, limit, graph)
                for name, kv in VARIANTS.items():
                    G.set_knobs(**{**OFF, **kv})
                    got = _outputs(G, ex, x, pk, wts, y0, R, slots, limit, graph)
                    assert got["_cnt_zero"], (name, kind, R, "counters not reset")
                    for k in ref:
                        if k.startswith("_"):
                            continue
                        assert torch.equal(_bits(ref[k]), _bits(got[k])), (name, kind, R, k)
    finally:
        G.set_knobs(**old)


@pytest.mark.parametrize("cb,I,k2s", [
    (1, 640, [(4, 5, 6), (6, 6, 4), (8, 8, 8), (5, 4, 6)]),          # widths 2..10 instances, 5 128-blocks
    (1, 768, [(4, 12, 6), (6, 6, 11), (8, 8, 8), (3, 4, 16)]),       # 2..16 instances, 6 128-blocks
    (0, 768, [(4, 5, 6), (6, 6, 4)]),
])
def test_variants_equal_default_bits(cb, I, k2s):
    E, D, slots = 24, 1024, 9
    ex = _layer(E, D, I, k2s, cb, seed=11 + I + cb)
    _check(ex, E, slots, 10.0, seed=100 * cb + I)


def test_variants_equal_default_bits_in_graphs():
    E, D, I, slots = 16, 1024, 640, 9
    ex = _layer(E, D, I, [(4, 5, 6), (6, 6, 4)], 1, seed=5)
    _check(ex, E, slots, math.inf, seed=3, rows=(1, 5, 16), graph=True)


@pytest.mark.skipif(not os.environ.get("CHECKPOINT"), reason="CHECKPOINT (the real snapshot folder) not given")
def test_variants_real_layer():
    from tensorfold.families.glm5_next.cuda.split import RankReader
    from tensorfold.families.glm5_next.cuda.weights import Config, expert_tensor_names, pack_experts

    folder = Path(os.environ["CHECKPOINT"])
    cfg = Config.read(folder)
    L = 4
    rd = RankReader(folder, 1, 3)
    rd.prefetch(expert_tensor_names(cfg.prefix, L, cfg.experts), None)
    parts = {(p, e): tuple(rd.get(f"{cfg.prefix}layers.{L}.mlp.experts.{e}.{p}." + k)
                           for k in ("trellis", "suh", "svh"))
             for p in ("gate_proj", "up_proj", "down_proj") for e in range(cfg.experts)}
    rd.close()
    ex = pack_experts(parts, cfg.experts, torch.device("cuda"))
    _check(ex, cfg.experts, cfg.top_k + 1, cfg.limit, seed=77, rows=(1, 3, 9, 16))
