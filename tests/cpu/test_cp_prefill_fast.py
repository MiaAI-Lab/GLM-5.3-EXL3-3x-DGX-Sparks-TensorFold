"""Context parallelism's faster prompt path (forward.cp_prompt, TF_GLM_CP_PIPE=1): the selection merged by row
owners (dcp.select_threshold / select_keep) equals dcp.select_merge bit for bit (ties, short rows, any row count),
the bf16 partial merge (dcp.merge_ranks_bf16) equals merge_ranks on the widened partials bit for bit, and at three
ranks the prompt's logits are the same bits for any part size (CP_PROMPT_ROWS) and any chunking of the prompt and
the 0129 path's (TF_GLM_CP_PIPE=0), and a drafted window after it equals serial steps."""

import multiprocessing as mp
import os

import pytest
import torch

from full_fakes import TEXT, install_fake_experts
from test_dcp_cpu import G, index_case
from test_full_forward import Rig, folder, tokens  # noqa: F401  (the checkpoint fixture)

from tensorfold.families.glm5_next.cuda import dcp


def owner_merge(cands, R, k):
    """Every rank's (slots, counts) through the owner merge: each rank's candidates padded to G M rows, rank p
    receiving every rank's rows p M .. (p + 1) M, its thresholds gathered back in rank order."""
    M = dcp.owner_rows(R, G)
    padded = torch.zeros((G, G * M, k), dtype=torch.int64)
    padded[:, :R] = cands
    th_all = torch.empty((G * M,), dtype=torch.int64)
    for p in range(G):
        every = padded[:, p * M:(p + 1) * M].contiguous()
        th = torch.full((M,), 123, dtype=torch.int64)
        dcp.select_threshold(every, M, th)
        th_all[p * M:(p + 1) * M] = th
    return [dcp.select_keep(padded[r, :R].contiguous(), th_all, G) for r in range(G)]


SELECT_CASES = [  # seed, pos0, R, cap, k, ties
    (1, 300, 4, 384, 64, None),
    (2, 300, 5, 384, 64, "zeros"),
    (3, 280, 7, 384, 64, "dups"),
    (4, 30, 6, 384, 64, None),          # rows that see fewer than k tokens (every visible token kept)
    (5, 60, 1, 384, 64, "dups"),
    (6, 150, 8, 384, 128, "zeros"),
]


@pytest.mark.parametrize("seed,pos0,R,cap,k,ties", SELECT_CASES)
def test_owner_merge_equals_select_merge(seed, pos0, R, cap, k, ties):
    qi, wts, keys = index_case(seed, R, cap, ties)
    pos = torch.tensor([pos0], dtype=torch.int32)
    cands = torch.stack([dcp.select_local(qi, wts, keys[r::G].contiguous(), r, G, pos, R, k=k)
                         for r in range(G)]).contiguous()
    new = owner_merge(cands, R, k)
    for r in range(G):
        slots, counts = dcp.select_merge(cands, r, G, k)
        assert torch.equal(new[r][1], counts), r
        assert torch.equal(new[r][0], slots), r


def test_owner_merge_random_keys_with_ties():
    """Synthetic candidate lists (unique keys a rank, equal scores across ranks, padding) at k = 32."""
    g = torch.Generator().manual_seed(7)
    R, k = 11, 32
    cands = torch.zeros((G, R, k), dtype=torch.int64)
    for r in range(R):
        for rank in range(G):
            real = k if r % 4 else int(torch.randint(0, k, (1,), generator=g))
            score = torch.randint(0, 6, (real,), generator=g) + (1 << 31)      # few distinct scores: ties
            pos = torch.randperm(200, generator=g)[:real] * G + rank
            key = (score << 32) | (0xFFFFFFFF - pos)
            key = key[torch.argsort(pos)]
            cands[rank, r, :real] = key
    new = owner_merge(cands, R, k)
    for r in range(G):
        slots, counts = dcp.select_merge(cands.contiguous(), r, G, k)
        assert torch.equal(new[r][1], counts) and torch.equal(new[r][0], slots), r


@pytest.mark.parametrize("H,HP", [(2, 2), (1, 2), (3, 4)])
def test_merge_ranks_bf16_bits(H, HP):
    g = torch.Generator().manual_seed(H * 10 + HP)
    n, LW = 5, 512
    o = torch.randn(G, n, HP, LW, generator=g).to(torch.bfloat16)
    lse = torch.randn(G, n, HP, generator=g) * 4
    lse[1, 0] = float("-inf")                     # a rank without tokens for a row
    lse[:, 2, 0] = float("-inf")                  # no rank: zeros
    o[1, 0] = 0
    o[:, 2, 0] = 0
    want = dcp.merge_ranks(o.float().contiguous(), lse.contiguous())
    for rank in range(G):
        got_o = torch.stack([o[p] for p in range(G) if p != rank]).contiguous()
        got_l = torch.stack([lse[p] for p in range(G) if p != rank]).contiguous()
        out = torch.empty((n, H, LW), dtype=torch.bfloat16)
        dcp.merge_ranks_bf16(o[rank].contiguous(), lse[rank].contiguous(), got_o, got_l, rank, out)
        assert torch.equal(out, want[:, :H]), rank


def _rank(folder, rank, world, port, prompt, steps, out_q, kv):
    """One rank: the prompt at several part sizes and chunkings (new path), the 0129 path, then serial steps and a
    drafted window from the prompt's state."""
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

    def run(chunks, rows, pipe, decode=False):
        F.CP_PROMPT_ROWS, F.CP_PIPE = rows, pipe
        rig = Rig(folder, rank, world, Comm(), kv=kv, cp=world)
        a = 0
        for n in chunks:
            last = rig.prefill(prompt[a:a + n])
            a += n
        if not decode:
            return last, None, None
        rows_ = [last] + [rig.window([t])[0] for t in steps]
        drafted = Rig(folder, rank, world, Comm(), kv=kv, cp=world)
        b = 0
        for n in chunks:
            drafted.prefill(prompt[b:b + n])
            b += n
        return last, torch.stack(rows_), drafted.window(steps)

    out = {}
    out["one"], out["serial"], out["drafted"] = run([len(prompt)], 512, True, decode=True)
    out["parts8"] = run([len(prompt)], 8, True)[0]
    out["parts5"] = run([len(prompt)], 5, True)[0]
    out["chunks"] = run([13, 27, len(prompt) - 40], 8, True)[0]
    out["old"] = run([len(prompt)], 512, False)[0]
    out_q.put((rank, out))
    dist.destroy_process_group()


@pytest.mark.parametrize("kv", ["bf16", "fp8"])
def test_cp_prompt_parts_chunks_and_old_path(folder, kv):  # noqa: F811
    prompt, steps = tokens(52, 31), tokens(5, 32)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 35000 + os.getpid() % 2000 + (kv == "fp8") * 7
    procs = [ctx.Process(target=_rank, args=(folder, r, 3, port, prompt, steps, q, kv)) for r in range(3)]
    for p in procs:
        p.start()
    got = sorted((q.get(timeout=2400) for _ in procs), key=lambda x: x[0])
    for p in procs:
        p.join(timeout=60)
    for rank, out in got:
        # the same bits for any part size and any chunking of the prompt
        assert torch.equal(out["parts8"], out["one"]), rank
        assert torch.equal(out["parts5"], out["one"]), rank
        assert torch.equal(out["chunks"], out["one"]), rank
        # the 0129 path's bits (the same kernels on the same values; on CPU the fallback chunk programs too)
        assert torch.equal(out["one"], out["old"]), rank
        # drafted window == serial steps (the pending token's row and the steps)
        assert torch.equal(out["drafted"][:len(steps) - 1], out["serial"][1:len(steps)]), rank
