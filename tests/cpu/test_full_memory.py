"""Full GLM-5.3's startup memory estimate: the buffer and cache formulas (geometry.full_*) against what the engine
allocates (forward.Buffers / Caches, built on the meta device at the real dimensions, every rank of three), and
the per-rank weight estimate against the real checkpoint's headers (CHECKPOINT: its snapshot folder; only the
safetensors headers are read)."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from full_fakes import TEXT, write_checkpoint

CHECKPOINT = os.environ.get("CHECKPOINT", "")


def tensor_bytes(obj, seen=None) -> int:
    """Bytes of every tensor reachable from obj's attributes (each tensor once)."""
    seen = set() if seen is None else seen
    if isinstance(obj, torch.Tensor):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(x, seen) for x in obj)
    if isinstance(obj, dict):
        return sum(tensor_bytes(x, seen) for x in obj.values())
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return sum(tensor_bytes(x, seen) for k, x in vars(obj).items() if k != "w")
    return 0


def stand_in(cfg, rank, world, device="meta", dense="q4"):
    """What Buffers and Caches read of a Weights: config, layout, device, a first projection of TF_GLM_DENSE's kind,
    the head's rows, the MTP head (present), the layers' kinds and indexers."""
    from tensorfold.families.glm5_next.cuda import qmm
    from tensorfold.families.glm5_next.cuda.tp import Layout

    lay = Layout.from_config(cfg, rank, world)
    proj = object.__new__(qmm.Q4 if dense == "q4" else qmm.B16)       # only its type is read (prompt_split_k)
    layers = [SimpleNamespace(index=i, kind="dsa", kda=None,
                              dsa=SimpleNamespace(proj=proj, index=object() if k == "full" else None))
              for i, k in enumerate(cfg.index_kinds)]
    return SimpleNamespace(cfg=cfg, tp=lay, rank=rank, world=world, device=torch.device(device), layers=layers,
                           head=SimpleNamespace(n=lay.vocab), mtp=object(), meta={"long_context": True})


def text_of(folder):
    from tensorfold.cuda.capacity import config

    return config(folder)


def check_buffers(folder, world, dense, rows_list, capacity):
    from tensorfold.cuda.geometry import full_buffer_bytes
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda.weights import Config

    cfg = Config.read(folder)
    t = text_of(folder)
    for rank in range(world):
        w = stand_in(cfg, rank, world, dense=dense)
        share = lambda fam, total: w.tp.size(fam) if fam in w.tp.totals else int(total) // world   # noqa: E731
        for rows, prefill in rows_list:
            got = tensor_bytes(F.Buffers(w, rows, capacity, prefill=prefill))
            est = full_buffer_bytes(t, rows, world=world, share=share, capacity=capacity, prefill=prefill,
                                    prompt_split_k=dense != "q4")
            assert est >= got, f"rank {rank}, {rows} rows, prefill {prefill}: estimate {est} < allocated {got}"
            assert est <= got * 1.02 + (1 << 20), f"rank {rank}, {rows} rows: estimate {est} >> allocated {got}"


def test_buffer_formula_tiny(tmp_path):
    folder = write_checkpoint(tmp_path / "ck")
    check_buffers(folder, 3, "bf16", [(8, False), (64, True)], 128)
    check_buffers(folder, 1, "q4", [(16, False), (64, True)], 4096)


def test_cache_formula_tiny(tmp_path):
    from tensorfold.cuda.geometry import full_token_bytes
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda.weights import Config

    folder = write_checkpoint(tmp_path / "ck")
    cfg = Config.read(folder)
    for kv in ("bf16", "fp8"):
        w = stand_in(cfg, 0, 3)
        c = F.Caches(w, 1000, streams=1, ring=64, kv=kv)
        got = tensor_bytes(c.arena) if hasattr(c, "arena") else tensor_bytes(c)
        assert got == 1000 * full_token_bytes(text_of(folder), kv, True), kv


@pytest.mark.skipif(not CHECKPOINT, reason="CHECKPOINT (the real snapshot folder) not given")
def test_buffer_formula_real_dimensions():
    """At the real dimensions, every rank of three, decode, MTP and prompt buffers (meta tensors: nothing is
    allocated)."""
    check_buffers(Path(CHECKPOINT), 3, "q4", [(16, False), (8, False), (2048, True)], 131072 + 2560)
    check_buffers(Path(CHECKPOINT), 3, "bf16", [(16, False), (2048, True)], 65536)


@pytest.mark.skipif(not CHECKPOINT, reason="CHECKPOINT (the real snapshot folder) not given")
def test_real_cache_bytes_a_token():
    from tensorfold.cuda.geometry import full_token_bytes

    t = text_of(Path(CHECKPOINT))
    # 78 + 1 layers x (512 latent + 64 rotary) and 21 + 1 indexers x 128 index dims
    assert full_token_bytes(t, "bf16", True) == 79 * (1024 + 128) + 22 * 256 == 96_640
    assert full_token_bytes(t, "fp8", True) == 79 * (528 + 128) + 22 * 256 == 57_456


@pytest.mark.skipif(not CHECKPOINT, reason="CHECKPOINT (the real snapshot folder) not given")
def test_real_weights_split_whole():
    """Every split tensor's three shares add up to the tensor, and the rotated expert blocks give each rank a third
    of the routed bytes; per-rank totals by TF_GLM_DENSE."""
    from tensorfold.cuda.capacity import estimate_weights, headers
    from tensorfold.cuda.geometry import split_weights
    from tensorfold.families.glm5_next.cuda import tp
    from tensorfold.families.glm5_next.cuda.engine import full_weights
    from tensorfold.families.glm5_next.cuda.qmm import q4_weights
    from tensorfold.families.glm5_next.cuda.split import rank_cut, rule
    from tensorfold.families.glm5_next.cuda.weights import Config

    folder = Path(CHECKPOINT)
    cfg = Config.read(folder)
    h = headers(folder)
    lays = [tp.Layout.from_config(cfg, r, 3) for r in range(3)]

    def cut_for(lay):
        def cut(name, shape, kind):
            if kind == "vocab":
                return lay.vocab_offset, lay.vocab_offset + lay.vocab
            return rank_cut(lay, name, shape, kind)
        return cut

    routed = [0, 0, 0]
    for name, info in h.items():
        kind = rule(name)
        if kind in ("rep", "drop"):
            continue
        axis = {"row": 0, "col": 1, "dim1": 1}[kind]
        spans = [rank_cut(lay, name, info["shape"], kind) for lay in lays]
        assert spans[0][0] == 0 and spans[-1][1] == info["shape"][axis], name
        assert all(spans[i][1] == spans[i + 1][0] for i in range(2)), name
        if ".experts." in name and name.endswith(".trellis"):
            other = info["shape"][0] if axis == 1 else info["shape"][1]       # the axis not split
            for r, (a, b) in enumerate(spans):
                routed[r] += (b - a) * other * info["shape"][2] * 2
    total = sum(routed)
    assert max(routed) - min(routed) < 0.01 * total / 3, routed       # the rotation balances the routed bytes
    totals = []
    for r, lay in enumerate(lays):
        tr = full_weights(q4_weights(split_weights(rule, 3, cut_for(lay))), cfg, lay)
        totals.append(estimate_weights(folder, tr).resident)
    print("per-rank weights (q4 dense), GiB:", [round(x / 2 ** 30, 2) for x in totals])
    assert max(totals) - min(totals) < 2 * 2 ** 30
