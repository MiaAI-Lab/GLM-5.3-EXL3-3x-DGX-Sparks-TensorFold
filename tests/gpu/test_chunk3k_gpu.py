"""3,072-row prompt chunks on the GPU: the universal grouping kernel against its host reference at every size (16-bit
picks staged in shared memory up to ~5,600 rows of 9 slots, read from global memory past that), MPE's route against
its reference at 3,072 rows, one MPE call (and one universal 2-tile call) of a real layer's 3,072 rows against the
former two blocks (2,176 + 896) bit for bit, and the tiny checkpoint through the engine with every real kernel:
3,072-row chunks leave a 3,100-token prompt's head row, next-step logits and caches as 2,048-row chunks, bit for bit.
CHECKPOINT (the real snapshot folder) for the real-layer tests."""

import os
from pathlib import Path

import pytest
import torch

CHECKPOINT = os.environ.get("CHECKPOINT", "")
real = pytest.mark.skipif(not CHECKPOINT, reason="CHECKPOINT (the real snapshot folder) not given")
E, SLOTS = 256, 9


def _picks(R, seed, E=E, slots=SLOTS):
    g = torch.Generator().manual_seed(seed)
    pick = torch.stack([torch.randperm(E, generator=g)[:slots] for _ in range(R)]).to(torch.int32)
    pick[:, -1] = E                                   # the shared expert's slot: no routed expert
    return pick


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    a, b = a.contiguous(), b.contiguous()
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.reshape(-1).view(torch.uint8),
                                                                     b.reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("R", [1, 16, 2048, 2176, 3072, 5000, 5800, 8192, 16386])
def test_grouping_kernel_matches_the_reference(R):
    """ext.group at R rows of 9 slots (staged up to the opt-in limit, global past it) == experts.group_reference:
    the same experts, the same members in the same order; a few picks out of range belong to no expert."""
    from tensorfold.cuda.exl3 import experts as generic

    pick = _picks(R, R)
    pick[0, 0] = -1
    if R > 3:
        pick[3, 1] = 70000
    maxu = min(R * SLOTS, E)
    ids = torch.zeros((maxu,), dtype=torch.int32, device="cuda")
    count = torch.zeros((1,), dtype=torch.int32, device="cuda")
    members = torch.full((maxu, R), -7, dtype=torch.int32, device="cuda")
    pc = pick.cuda().contiguous()
    generic._ext().group(pc, ids, count, members, R, SLOTS, E)
    torch.cuda.synchronize()
    n = int(count.item())
    uids, want = generic.group_reference(pick, E, R)
    assert n == len(uids)
    assert torch.equal(ids[:n].cpu(), uids)
    assert torch.equal(members[:n].cpu(), want)


def test_mpe_route_matches_the_reference():
    """mpe's route of a 3,072-row chunk == mpe.route_reference: the same items; each expert's pairs the same set."""
    from tensorfold.cuda.exl3 import mpe

    R = 3072
    pick = _picks(R, 3)
    ext = mpe._ext()
    P = R * SLOTS
    max_items = -(-P // ext.item_rows()) + min(P, E)
    sorted_ = torch.zeros((P,), dtype=torch.int32, device="cuda")
    items = torch.zeros((3 * max_items,), dtype=torch.int32, device="cuda")
    count = torch.zeros((1,), dtype=torch.int32, device="cuda")
    ext.route(pick.cuda().contiguous(), E, sorted_, items, count, max_items)
    torch.cuda.synchronize()
    want_items, spans = mpe.route_reference(pick, E, ext.item_rows())
    n = int(count.item())
    got = [tuple(items[3 * i:3 * i + 3].tolist()) for i in range(n)]
    assert got == want_items
    s = sorted_.cpu().tolist()
    pos = 0
    for e in range(E):
        k = len(spans[e])
        assert set(s[pos:pos + k]) == spans[e], e
        pos += k


def _real_layer(layer=4, rank=1):
    from tensorfold.families.glm5_next.cuda.split import RankReader
    from tensorfold.families.glm5_next.cuda.weights import Config, expert_tensor_names, pack_experts

    folder = Path(CHECKPOINT)
    cfg = Config.read(folder)
    rd = RankReader(folder, rank, 3)
    rd.prefetch(expert_tensor_names(cfg.prefix, layer, cfg.experts), None)
    parts = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for e in range(cfg.experts):
            n = f"{cfg.prefix}layers.{layer}.mlp.experts.{e}.{proj}."
            parts[(proj, e)] = (rd.get(n + "trellis"), rd.get(n + "suh"), rd.get(n + "svh"))
    rd.close()
    return cfg, pack_experts(parts, cfg.experts, torch.device("cuda"))


@real
def test_mpe_one_call_of_3072_rows_equals_two_blocks():
    """A real layer's 3,072 rows through MPE in one call (the new EXL3_BLOCK_ROWS) and in the former two blocks
    (2,176 + 896 rows): every pair's fp32 row the same bits; and through the universal 2-tile kernels likewise."""
    from tensorfold.cuda.exl3 import experts as generic
    from tensorfold.cuda.exl3 import mpe

    cfg, ex = _real_layer()
    R, B, S = 3072, 2176, cfg.top_k + 1
    g = torch.Generator().manual_seed(11)
    x = (torch.randn(R, cfg.hidden, generator=g) * 0.5).to(torch.bfloat16).cuda()
    pick = _picks(R, 12, cfg.experts, S).cuda()
    if mpe.supported(ex):
        sc = mpe.Scratch(ex, R, S)
        y = torch.empty((R * S, cfg.hidden), dtype=torch.float32, device="cuda")
        whole = mpe.pairs(x, pick, ex, sc, y, act_mode=generic.ACT_BF16).clone()
        parts = []
        for a, z in ((0, B), (B, R)):
            parts.append(mpe.pairs(x[a:z], pick[a:z].contiguous(), ex, sc, y, act_mode=generic.ACT_BF16).clone())
        blocks = torch.cat(parts)
        routed = (pick < cfg.experts).reshape(-1)          # the shared slot's rows are not written
        assert torch.equal(whole[routed], blocks[routed])
    s = generic.Scratch(ex, R, S, device="cuda")
    whole = generic.routed(x, pick, None, ex, s, None, R, act_mode=generic.ACT_BF16, mt=2).clone()
    parts = [generic.routed(x[a:z], pick[a:z].contiguous(), None, ex, s, None, z - a, act_mode=generic.ACT_BF16,
                            mt=2).clone() for a, z in ((0, B), (B, R))]
    view = whole.view(R, S, -1)[:, :-1]
    assert torch.equal(view, torch.cat(parts).view(R, S, -1)[:, :-1])


# -- the tiny checkpoint through the engine ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    os.environ.setdefault("TF_GLM_DENSE", "bf16")
    from full_fakes import write_checkpoint

    return write_checkpoint(tmp_path_factory.mktemp("glm53chunk3k"))


def _run(folder, rows, prompt, steps):
    from tensorfold.families.glm5_next.cuda import decode as D
    from tensorfold.families.glm5_next.cuda.decode import Engine
    from tensorfold.families.glm5_next.cuda.forward import commit
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=0, world=1, device="cuda", mtp=True)
    e = Engine(w, capacity=4096 + 512, max_rows=8, prefill_rows=rows, long_context=True, mtp_rows=4)
    D.prefill(e, prompt, None, mtp=True, keep_head=True)
    out = [e.head.float().cpu().clone()]
    for t in steps:
        out.append(e.forward([t])[:1].float().cpu().clone())
        commit(e.w, e.st, e.buf, 1, 1)
    views = [v.cpu().clone() for v in D._row_views(e.st, e.st.pos, e.st.mtp_len)]
    return out, views


def test_engine_3072_row_chunks_equal_2048(folder):
    """A 3,100-token prompt in 3,072-row chunks (3,072 + 28) and in 2,048-row chunks (2,048 + 1,052), every real
    kernel: the same head row, next-step logits and cache rows, bit for bit."""
    g = torch.Generator().manual_seed(5)
    prompt = torch.randint(2, 256, (3100,), generator=g).tolist()
    steps = torch.randint(2, 256, (3,), generator=g).tolist()
    a = _run(folder, 3072, prompt, steps)
    b = _run(folder, 2048, prompt, steps)
    assert all(same(x, y) for x, y in zip(a[0], b[0]))
    assert len(a[1]) == len(b[1]) and all(same(x, y) for x, y in zip(a[1], b[1]))
