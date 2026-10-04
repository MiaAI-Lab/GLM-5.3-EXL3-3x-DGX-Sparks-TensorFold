"""0132: context-parallel prompt chunks gathering raw queries (TF_GLM_CP_RAW_Q: the key dims and rotated rotary dims
of every head, absorbed by the receiving rank with the other ranks' kv_b key rows replicated) and half-length first
and last parts (TF_GLM_CP_PARTS_EDGE). At three ranks the prompt's logits are the same bits with raw queries or
absorbed ones (0131) or the 0129 path, for any part size and chunking; drafted windows still equal serial steps; the
own-partial merge with real heads only and the memory estimate's replicated rows."""

import multiprocessing as mp
import os

import pytest
import torch

from full_fakes import TEXT, install_fake_experts
from test_full_forward import Rig, folder, tokens  # noqa: F401  (the checkpoint fixture)

from tensorfold.families.glm5_next.cuda import dcp

G = 3


def test_parts_halve_the_ends():
    from tensorfold.families.glm5_next.cuda.forward import cp_parts

    assert cp_parts(2048, 512) == [(0, 256), (256, 768), (768, 1280), (1280, 1792), (1792, 2048)]
    assert cp_parts(512, 512) == [(0, 512)]
    assert cp_parts(600, 512) == [(0, 256), (256, 600)]
    assert cp_parts(40, 8) == [(0, 4), (4, 12), (12, 20), (20, 28), (28, 36), (36, 40)]
    for R, step in [(2048, 512), (1000, 512), (13, 4), (64, 64), (65, 64)]:
        parts = cp_parts(R, step)
        assert parts[0][0] == 0 and parts[-1][1] == R and all(a[1] == b[0] for a, b in zip(parts, parts[1:]))
        assert all(0 < z - a <= step for a, z in parts)


@pytest.mark.parametrize("HO,H,HP", [(21, 21, 22), (2, 1, 2), (22, 22, 22)])
def test_merge_with_real_own_heads(HO, H, HP):
    """The own partial at its real heads (HO) gives merge_ranks' bits on the HP-padded partials."""
    g = torch.Generator().manual_seed(HO + 100 * HP)
    n, LW = 4, 512
    o = torch.randn(G, n, HP, LW, generator=g).to(torch.bfloat16)
    lse = torch.randn(G, n, HP, generator=g) * 3
    lse[2, 1] = float("-inf")
    o[2, 1] = 0
    want = dcp.merge_ranks(o.float().contiguous(), lse.contiguous())
    for rank in range(G):
        got_o = torch.stack([o[p] for p in range(G) if p != rank]).contiguous()
        got_l = torch.stack([lse[p] for p in range(G) if p != rank]).contiguous()
        out = torch.empty((n, H, LW), dtype=torch.bfloat16)
        dcp.merge_ranks_bf16(o[rank, :, :HO].contiguous(), lse[rank, :, :HO].contiguous(), got_o, got_l, rank, out)
        assert torch.equal(out, want[:, :H]), rank


def test_estimate_counts_replicated_rows():
    from tensorfold.cuda.geometry import full_cp_absorb_bytes

    full = {"num_attention_heads": 64, "qk_nope_head_dim": 192, "kv_lora_rank": 512, "num_hidden_layers": 78,
            "num_nextn_predict_layers": 1}
    assert full_cp_absorb_bytes(full, 3, 3) == 79 * 2 * 22 * 192 * 512 * 2
    assert full_cp_absorb_bytes(full, 3, 3, mtp=False) == 78 * 2 * 22 * 192 * 512 * 2
    assert full_cp_absorb_bytes(TEXT, 3, 3) == (TEXT["num_hidden_layers"] + TEXT.get("num_nextn_predict_layers", 0)) \
        * 2 * 2 * 64 * TEXT["kv_lora_rank"] * 2


def _rank(folder, rank, world, port, prompt, steps, out_q, kv):
    import torch.distributed as dist

    os.environ.setdefault("TRITON_INTERPRET", "1")
    torch.set_num_threads(1)
    import conftest  # noqa: F401  (the interpreter patches)
    import pytest as _pt

    mp_ = _pt.MonkeyPatch()
    install_fake_experts(mp_)
    os.environ["TF_GLM_DENSE"] = "bf16"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)

    class Comm:
        world_size = world

        def all_gather(self, send, recv):
            dist.all_gather(list(recv.view(world, -1).unbind(0)), send.contiguous().view(-1))

        def exchange_all(self, sends, recvs):
            work = []
            for p in sorted(sends):
                for t in sends[p]:
                    work.append(dist.isend(t.contiguous(), p))
                for t in recvs[p]:
                    tmp = torch.empty_like(t)
                    work.append((dist.irecv(tmp, p), t, tmp))
            for item in work:
                if isinstance(item, tuple):
                    item[0].wait()
                    item[1].copy_(item[2])
                else:
                    item.wait()

    from tensorfold.families.glm5_next.cuda import forward as F

    def rig(raw):
        r = Rig(folder, rank, world, Comm(), kv=kv, cp=world)
        if raw:
            F.cp_replicate_absorb(r.w)
        return r

    def run(chunks, rows, *, raw=True, edge=True, pipe=True, decode=False):
        F.CP_PROMPT_ROWS, F.CP_PIPE, F.CP_RAW_Q, F.CP_PARTS_EDGE = rows, pipe, raw, edge
        r = rig(raw)
        a = 0
        for n in chunks:
            last = r.prefill(prompt[a:a + n])
            a += n
        if not decode:
            return last, None, None
        rows_ = [last] + [r.window([t])[0] for t in steps]
        d = rig(raw)
        a = 0
        for n in chunks:
            d.prefill(prompt[a:a + n])
            a += n
        return last, torch.stack(rows_), d.window(steps)

    out = {}
    out["raw"], out["serial"], out["drafted"] = run([len(prompt)], 512, decode=True)
    out["raw8"] = run([len(prompt)], 8)[0]
    out["raw5_even"] = run([len(prompt)], 5, edge=False)[0]
    out["raw_chunks"] = run([13, 27, len(prompt) - 40], 8)[0]
    out["absorbed8"] = run([len(prompt)], 8, raw=False)[0]
    out["p0129"] = run([len(prompt)], 512, raw=False, pipe=False)[0]
    out_q.put((rank, out))
    dist.destroy_process_group()


@pytest.mark.parametrize("kv", ["bf16", "fp8"])
def test_raw_queries_keep_every_bit(folder, kv):  # noqa: F811
    prompt, steps = tokens(52, 41), tokens(5, 42)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 37000 + os.getpid() % 2000 + (kv == "fp8") * 7
    procs = [ctx.Process(target=_rank, args=(folder, r, 3, port, prompt, steps, q, kv)) for r in range(3)]
    for p in procs:
        p.start()
    got = sorted((q.get(timeout=1200) for _ in procs), key=lambda x: x[0])
    for p in procs:
        p.join(timeout=60)
    for rank, out in got:
        for key in ("raw8", "raw5_even", "raw_chunks", "absorbed8", "p0129"):
            assert torch.equal(out[key], out["raw"]), (rank, key)
        assert torch.equal(out["drafted"][:len(steps) - 1], out["serial"][1:len(steps)]), rank
