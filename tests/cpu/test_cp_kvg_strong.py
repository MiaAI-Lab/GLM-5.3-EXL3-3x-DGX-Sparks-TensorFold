"""Stronger checks for 0141's context-parallel kv gather (TF_GLM_CP_KV_GATHER=1), test-only (2026-10-06).
Motivation: test_cp_kv_gather_prompt[fp8] compares ONE bf16 logits row per
rank, so a near tie at the top (two logits rounding to the same bf16 value) decides the test.

1. test_gathered_planes_are_the_owners_bytes[kv]: tie-free and exact. Every layer's gathered latent and rotary planes
   on every rank must equal, BYTE FOR BYTE, the concatenation in rank order of every rank's own first S local cache
   slots (what each owner wrote, fp8 codes and scales included). This is the gather's whole job; any transport,
   offset or dtype-view error fails it, and it does not depend on summation order or bf16 rounding.
2. test_gather_vs_merge_logits_six_rows[fp8]: compares six rows per rank (the prompt's last row and 5 serial decode
   steps) using the fixed eight-bf16-step tie allowance (agree_fp8_tie) and the 0.12 x reference-scale magnitude limit.
   At least six of the eighteen rank-rows must have a reference top-two margin larger than the fixed allowance.
"""

import multiprocessing as mp
import os

import pytest
import torch

from full_fakes import install_fake_experts
from test_full_forward import Rig, fakes, folder, tokens  # noqa: F401  (fixtures)
from test_cp_kv_gather import agree_fp8_tie, bf16_step


class _Comm:
    def __init__(self, world):
        self.world_size = world

    def all_gather(self, send, recv):
        import torch.distributed as dist
        dist.all_gather(list(recv.view(self.world_size, -1).unbind(0)), send.contiguous().view(-1))

    def exchange_all(self, sends, recvs):
        import torch.distributed as dist
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


def _setup(rank, world, port):
    import torch.distributed as dist
    os.environ.setdefault("TRITON_INTERPRET", "1")
    torch.set_num_threads(1)
    import conftest  # noqa: F401  (the interpreter patches)
    import pytest as _pt
    mp_ = _pt.MonkeyPatch()
    install_fake_experts(mp_)
    os.environ["TF_GLM_DENSE"] = "bf16"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)


def _rank_planes(folder, rank, world, port, prompt, chunks, out_q, kv):
    """Prefill over the gathered context, recording per call (one per layer and chunk) the gathered planes and this
    rank's own first S slots, as raw bytes."""
    import torch.distributed as dist
    _setup(rank, world, port)
    from tensorfold.families.glm5_next.cuda import forward as FW

    FW.CP_KV_GATHER, FW.CP_PIPE, FW.CP_PROMPT_ROWS = True, True, 8
    calls = []
    orig = FW.cp_prompt_kvg

    def spy(layer, w, lc, kr, pos_dev, b, R, keys, host_pos, qp, within):
        out = orig(layer, w, lc, kr, pos_dev, b, R, keys, host_pos, qp, within)
        G = int(w.meta["cp"])
        S = -(-(int(host_pos) + R) // G)
        gl, gr = FW.cp_kvg_buffers(b, lc, kr, G, S)      # the held planes this call gathered into (no new gather)
        u8 = lambda t: t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()  # noqa: E731  (plain bytes: queue-safe)
        calls.append({"host_pos": int(host_pos), "R": int(R), "S": S, "G": G, "lc_dtype": str(lc.dtype),
                      "kr_dtype": str(kr.dtype), "row_lc": lc[0].numel() * lc.element_size(),
                      "row_kr": kr[0].numel() * kr.element_size(),
                      "own_lc": u8(lc[:S]), "own_kr": u8(kr[:S]), "gl": u8(gl), "gr": u8(gr)})
        return out

    FW.cp_prompt_kvg = spy
    rig = Rig(folder, rank, world, _Comm(world), kv=kv, cp=world)
    a = 0
    for n in chunks:
        rig.prefill(prompt[a:a + n])
        a += n
    out_q.put((rank, calls))
    dist.destroy_process_group()


def _run(target, args_of, world=3, timeout=2400):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 39000 + os.getpid() % 2000
    procs = [ctx.Process(target=target, args=args_of(r, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    try:
        got = sorted((q.get(timeout=timeout) for _ in procs), key=lambda x: x[0])
        for p in procs:
            p.join(timeout=120)
        assert [r for r, _ in got] == list(range(world)), "exactly one result a rank"
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
        return got
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=30)


@pytest.mark.parametrize("kv", ["fp8", "bf16", "fp4"])
def test_gathered_planes_are_the_owners_bytes(folder, fakes, kv):  # noqa: F811
    prompt = tokens(52, 41)
    chunks = [13, 27, len(prompt) - 40]                  # several chunks: S grows, earlier slots must stay put
    got = _run(_rank_planes, lambda r, port, q: (folder, r, 3, port, prompt, chunks, q, kv))
    n_calls = {len(calls) for _, calls in got}
    assert len(n_calls) == 1 and n_calls.pop() > 0, "every rank must gather the same number of times (layers x chunks)"
    checked = 0
    for i in range(len(got[0][1])):
        per = [calls[i] for _, calls in got]
        S, G = per[0]["S"], per[0]["G"]
        assert all(c["S"] == S and c["host_pos"] == per[0]["host_pos"] for c in per), i
        want_lc = b"".join(c["own_lc"] for c in per)            # rank-major: rank p's first S slots at rows p S ..
        want_kr = b"".join(c["own_kr"] for c in per)
        for rank, c in enumerate(per):
            assert len(c["gl"]) == len(want_lc), (i, rank, "latent plane size", len(c["gl"]), len(want_lc))
            assert c["gl"] == want_lc, (i, rank, "latent bytes differ from the owners'", c["lc_dtype"])
            assert c["gr"] == want_kr, (i, rank, "rotary bytes differ from the owners'", c["kr_dtype"])
            checked += 1
    print(f"\n[{kv}] gathered planes equal the owners' bytes: {checked} rank-calls, {len(got[0][1])} calls a rank "
          f"(latent {got[0][1][0]['lc_dtype']}, {got[0][1][0]['row_lc']} B a row; rotary {got[0][1][0]['kr_dtype']})")


def _rank_rows(folder, rank, world, port, prompt, steps, out_q, kv):
    """The prompt's last row and serial decode steps, over the gathered context and over the partial-merge path."""
    import torch.distributed as dist
    _setup(rank, world, port)
    from tensorfold.families.glm5_next.cuda import forward as FW

    def run(gather):
        FW.CP_KV_GATHER, FW.CP_PIPE, FW.CP_PROMPT_ROWS = gather, True, 8
        rig = Rig(folder, rank, world, _Comm(world), kv=kv, cp=world)
        rows = [rig.prefill(prompt)]
        rows += [rig.window([t])[0] for t in steps]
        return torch.stack(rows)

    out = {"gather": run(True).tolist(), "merge": run(False).tolist()}     # plain lists: queue-safe
    out_q.put((rank, out))
    dist.destroy_process_group()


BF16_ULP_2_4 = 2.0 ** -6                                 # bf16 spacing of logits in [2, 4)


def _bf16_ulp(x: float) -> float:
    import math
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else 2.0 ** -133


@pytest.mark.parametrize("kv", ["fp8"])
def test_gather_vs_merge_logits_six_rows(folder, fakes, kv):  # noqa: F811
    """6 rows a rank (the prompt's last row + 5 predetermined serial steps), gather vs partial-merge path: the fixed
    8-bf16-step tie rule (agree_fp8_tie) on every row, and at least 6 of the 18 rank-rows decided by a reference margin
    beyond the budget (so the tie allowance cannot make the check vacuous)."""
    prompt, steps = tokens(52, 41), tokens(5, 42)
    got = _run(_rank_rows, lambda r, port, q: (folder, r, 3, port, prompt, steps, q, kv))
    decided, report = 0, []
    for rank, out in got:
        g, m = torch.tensor(out["gather"]), torch.tensor(out["merge"])
        assert g.shape == m.shape and g.shape[0] == 1 + len(steps)
        ok, why = agree_fp8_tie(g[None], m[None])
        assert ok, (rank, why)
        for r in range(g.shape[0]):
            top = m[r].topk(2).values
            margin = (top[0] - top[1]).item()
            budget = 8 * bf16_step(top[0].item())
            decided += margin > budget
            report.append(f"rank {rank} row {r}: ref margin {margin:.4f} ({margin / bf16_step(top[0].item()):.0f} steps), "
                          f"best same {int(g[r].argmax()) == int(m[r].argmax())}")
    print("\n" + "\n".join(report) + f"\nrows decided beyond the 8-step budget: {decided} of 18")
    assert decided >= 6, f"only {decided} rows beyond the tie budget: vacuous"
