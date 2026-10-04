"""0141: context-parallel prompt chunks over the gathered context (forward.cp_prompt_kvg, TF_GLM_CP_KV_GATHER=1):
the row owners' whole lists (dcp.owner_lists) are the replicated selection's (dsa_full.select_tokens) bit for bit,
ties and short rows included; at three ranks the prompt's logits are the same bits for any chunking of the prompt,
agree with context parallelism's partial-merge path (TF_GLM_CP_KV_GATHER=0) and with one rank's, and a drafted window
after the prompt equals serial steps."""

import multiprocessing as mp
import os

import pytest
import torch

from full_fakes import install_fake_experts
from test_dcp_cpu import CASES, G, index_case
from test_full_forward import Rig, agree_kv, fakes, folder, tokens  # noqa: F401  (fixtures)

from tensorfold.families.glm5_next.cuda import dcp
from tensorfold.families.glm5_next.cuda import dsa_full as F


def owner_lists_all(cands, R, k, S):
    """Every row's (rows, count) through the owners: candidates padded to G M rows, rank p merging rows p M ..
    (p + 1) M - 1 (its threshold, then owner_lists), the lists put together in rank order."""
    M = dcp.owner_rows(R, G)
    padded = torch.zeros((G, G * M, k), dtype=torch.int64)
    padded[:, :R] = cands
    rows, counts = [], []
    for p in range(G):
        every = padded[:, p * M:(p + 1) * M].contiguous()
        th = torch.empty((M,), dtype=torch.int64)
        dcp.select_threshold(every, M, th)
        t, c = dcp.owner_lists(every, th, G, S)
        rows.append(t)
        counts.append(c)
    return torch.cat(rows)[:R], torch.cat(counts)[:R]


def positions(rows, S):
    """Rows of the gathered planes back to global positions: row (p % G) S + p // G."""
    rows = rows.long()
    return (rows % S) * G + rows // S


@pytest.mark.parametrize("seed,pos0,R,cap,k,ties", CASES + [(8, 100, 9, 384, 64, "dups"), (9, 10, 7, 64, 16, None)])
def test_owner_lists_equal_replicated_selection(seed, pos0, R, cap, k, ties):
    qi, wts, keys = index_case(seed, R, cap, ties)
    pos = torch.tensor([pos0], dtype=torch.int32)
    tok, cnt = F.select_tokens(qi, wts, keys, pos0, R, pos, prompt=True, k=k)
    cands = torch.stack([dcp.select_local(qi, wts, keys[r::G].contiguous(), r, G, pos, R, k=k)
                         for r in range(G)]).contiguous()
    S = -(-(pos0 + R) // G)
    rows, counts = owner_lists_all(cands, R, k, S)
    assert rows.dtype == torch.int32 and counts.dtype == torch.int32 and rows.is_contiguous()
    for r in range(R):
        vis = pos0 + r + 1
        want = tok[r].long().sort().values if vis > k else torch.arange(vis)
        n = int(counts[r])
        assert n == min(vis, k), (r, n)
        got = positions(rows[r, :n], S)
        assert torch.equal(got, want), r                     # ascending global positions, the replicated set
        assert (rows[r, n:] == -1).all()
        assert int(rows[r, :n].max()) < G * S


def test_owner_lists_short_rows_match_all_visible():
    """A chunk within the top-k (cp_kvg_all_visible) gives a row the list a longer chunk's selection gives it."""
    from tensorfold.families.glm5_next.cuda import forward as FW

    k, pos0, R = 64, 20, 30
    qi, wts, keys = index_case(11, R, 192, "zeros")
    pos = torch.tensor([pos0], dtype=torch.int32)
    cands = torch.stack([dcp.select_local(qi, wts, keys[r::G].contiguous(), r, G, pos, R, k=k)
                         for r in range(G)]).contiguous()
    S = -(-(pos0 + R) // G)
    rows, counts = owner_lists_all(cands, R, k, S)
    short = min(R, k - pos0)                              # rows that see at most k tokens
    t2, c2 = FW.cp_kvg_all_visible(None, short, pos0, G, S, k, torch.device("cpu"))
    for r in range(short):
        n = int(counts[r])
        assert n == int(c2[r]) and torch.equal(rows[r, :n], t2[r, :n]), r


def _rank(folder, rank, world, port, prompt, steps, out_q, kv):
    """One rank: the prompt over the gathered context at several chunkings, the partial-merge path, then serial
    steps and a drafted window after the gathered prompt."""
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

    from tensorfold.families.glm5_next.cuda import forward as FW

    def run(chunks, gather, decode=False):
        FW.CP_KV_GATHER, FW.CP_PIPE, FW.CP_PROMPT_ROWS = gather, True, 8
        rig = Rig(folder, rank, world, Comm(), kv=kv, cp=world)
        a = 0
        rows_ = []
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
    out["one"], out["serial"], out["drafted"] = run([len(prompt)], True, decode=True)
    out["chunks"] = run([13, 27, len(prompt) - 40], True)[0]
    out["chunks2"] = run([5, 11, len(prompt) - 16], True)[0]
    out["old"] = run([len(prompt)], False)[0]
    out["vocab"] = Rig(folder, rank, world, Comm(), kv=kv, cp=world).w.vocab_offset
    out_q.put((rank, out))
    dist.destroy_process_group()


@pytest.mark.parametrize("kv", ["bf16", "fp8", "fp4"])
def test_cp_kv_gather_prompt(folder, fakes, kv):  # noqa: F811
    prompt, steps = tokens(52, 41), tokens(5, 42)
    one = Rig(folder, kv=kv)
    whole = one.prefill(prompt)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 37000 + os.getpid() % 2000 + ("bf16", "fp8", "fp4").index(kv) * 7
    procs = [ctx.Process(target=_rank, args=(folder, r, 3, port, prompt, steps, q, kv)) for r in range(3)]
    for p in procs:
        p.start()
    got = sorted((q.get(timeout=2400) for _ in procs), key=lambda x: x[0])
    for p in procs:
        p.join(timeout=60)
    for rank, out in got:
        # the same bits for any chunking of the prompt
        assert torch.equal(out["chunks"], out["one"]), rank
        assert torch.equal(out["chunks2"], out["one"]), rank
        # close to the partial-merge path (other summation order, the same caches and selection)
        ok, why = agree_kv(out["one"][None], out["old"][None], kv)
        assert ok, (rank, why)
        # drafted window == serial steps (the pending token's row and the steps)
        assert torch.equal(out["drafted"][:len(steps) - 1], out["serial"][1:len(steps)]), rank
    # the ranks' vocabulary slices together agree with one rank's prompt (no context parallelism)
    parts = torch.cat([out["one"] for _, out in got], dim=-1)
    ok, why = agree_kv(parts[None], whole[None], kv)
    assert ok, why
