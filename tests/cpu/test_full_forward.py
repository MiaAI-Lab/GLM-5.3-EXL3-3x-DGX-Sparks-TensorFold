"""Full GLM-5.3 through the patched CUDA engine's forward on CPU (Triton's interpreter; the EXL3 expert kernels'
torch stand-in), one rank and three: loading and splitting the checkpoint, MLA with RoPE on the latent cache, the
DSA indexer on full layers and its top-k reused on shared ones, dense and sparse rows, plain residuals, the MTP
head; against an independent reference forward, and drafted windows against serial steps bit for bit."""

import multiprocessing as mp
import os

import pytest
import torch

from full_fakes import LAYERS, TEXT, Reference, install_fake_experts, write_checkpoint

CAP = 128            # cache slots of the tests' engine (the reference's sparse rows start past index_topk = 16)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_checkpoint(tmp_path_factory.mktemp("glm53full"))


@pytest.fixture(scope="module")
def ref(folder):
    return Reference(folder)


class Rig:
    """The engine's forward pieces on one rank: weights, decode and prompt buffers, one sequence's caches."""

    def __init__(self, folder, rank=0, world=1, comm=None, kv="bf16", cp=1):
        from tensorfold.families.glm5_next.cuda import forward as F
        from tensorfold.families.glm5_next.cuda.weights import load

        self.F = F
        self.w = load(folder, rank=rank, world=world, device="cpu", mtp=True)
        self.w.comm = comm
        self.w.meta["long_context"] = True
        if cp > 1:                                    # context parallelism (TF_GLM_CP): the engine's settings
            from tensorfold.families.glm5_next.cuda.tp import split_sizes

            self.w.meta["cp"] = cp
            self.w.meta["cp_heads"] = max(split_sizes(TEXT["num_attention_heads"], world, 1))
        self.buf = F.Buffers(self.w, 8, CAP)
        self.pbuf = F.Buffers(self.w, 64, CAP, prefill=True)
        self.mbuf = F.Buffers(self.w, 8, CAP)
        self.caches = F.Caches(self.w, CAP, streams=1, ring=F.index_ring(CAP, 64), kv=kv)
        self.st = F.State(self.w, CAP, 8, caches=self.caches, slots=F.Slots(self.w, 1, 8))

    def _ids(self, b, tokens):
        b.ids[:len(tokens)] = torch.tensor(tokens, dtype=torch.int32)

    def prefill(self, tokens):
        """A prompt chunk: committed; the last row's logits (this rank's vocabulary)."""
        b, st, R = self.pbuf, self.st, len(tokens)
        self._ids(b, tokens)
        out = self.F.compute(self.w, st, b, R, nch=self.F.chunks_for(st, R), host_pos=st.pos)
        self.F.commit(self.w, st, b, R, R)
        return out[0].float().clone()

    def window(self, tokens, keep=None):
        """A decode window: logits of every row; commits ``keep`` rows (default all)."""
        b, st, R = self.buf, self.st, len(tokens)
        self._ids(b, tokens)
        out = self.F.compute(self.w, st, b, R, nch=self.F.chunks_for(st, R), host_pos=st.pos).float().clone()
        self.F.commit(self.w, st, b, R, keep or R)
        return out

    def hidden(self, b, R):
        return b.fnormed[:R].clone()

    def mtp(self, next_tokens, hidden):
        from tensorfold.families.glm5_next.cuda import mtp as M

        b, st, n = self.mbuf, self.st, len(next_tokens)
        b.zero_first = st.mtp_len == 0
        self._ids(b, next_tokens)
        b.hin[:n].copy_(hidden)
        out = M.mtp_compute(self.w, st, b, n, last_only=False, nch=-(-(st.mtp_len + n) // 512),
                            host_pos=st.mtp_len).float().clone()
        st.set_mtp_len(st.mtp_len + n)
        return out


def tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, TEXT["vocab_size"], (n,), generator=g).tolist()


def agree(got, want):
    """Logits within bf16 noise of the reference: close values, the same best token on nearly every row. (The bf16
    paths of the two drift apart along a sequence; in this random model a near tie in an indexer's top-k, one token
    of 16 swapped, moves later logits by up to ~10%.)"""
    got, want = got.float(), want.float()
    scale = want.abs().max().item()
    err = (got - want).abs().max().item()
    same = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    return err <= 0.12 * scale and same >= 0.8, f"max err {err:.4f} of {scale:.3f}, argmax agreement {same:.2f}"


def agree_kv(got, want, kv):
    """``agree``, and for TF_GLM_KV=fp4 a wider band: a 4-bit code flips on the tiny bf16 differences between the
    engine's and the reference's latent rows before quantization, moving logits ~2-3x fp8's (measured per step);
    the best tokens must still agree wherever the reference's margin exceeds the error."""
    if kv != "fp4":
        return agree(got, want)
    got, want = got.float(), want.float()
    scale = want.abs().max().item()
    err = (got - want).abs().max().item()
    top = want.topk(2, dim=-1).values
    clear = (top[..., 0] - top[..., 1]) > 2 * err
    same = (got.argmax(-1) == want.argmax(-1))[clear].all().item() if clear.any() else True
    return err <= 0.15 * scale and same, f"max err {err:.4f} of {scale:.3f}, clear-margin argmax agree {same}"


@pytest.fixture
def fakes(monkeypatch):
    install_fake_experts(monkeypatch)
    monkeypatch.setenv("TF_GLM_DENSE", "bf16")


def test_config_reads_full_glm(folder):
    from tensorfold.families.glm5_next.cuda.weights import Config

    c = Config.read(folder)
    assert c.full and c.kinds == ["dsa"] * LAYERS and c.streams == 1 and c.kpool == 1
    assert c.rope == 16 and c.nope == 48 and c.qk_dim == 64 and c.dense_limit == 16 and c.limit == float("inf")
    assert c.index_kinds == ("full", "shared", "shared", "full", "shared") and c.prefix == "model."


@pytest.mark.parametrize("kv", ["bf16", "fp8", "fp4"])
def test_one_rank_matches_the_reference(folder, ref, fakes, kv):
    """A prompt chunk within the top-k, decode steps past it (sparse), a chunk crossing it, against the reference
    (TF_GLM_KV=fp8: the latent rows quantized, within the same tolerance)."""
    rig = Rig(folder, kv=kv)
    ref = Reference(folder, kv=kv) if kv != "bf16" else ref
    cache = ref.new_cache()
    prompt = tokens(12, 1)
    got = rig.prefill(prompt)
    want, _ = ref.forward(prompt, cache, 0)
    ok, why = agree_kv(got[None], want[-1:], kv)
    assert ok, f"prompt: {why}"
    pos = len(prompt)
    for step, t in enumerate(tokens(10, 2)):                    # positions 12 .. 21: crosses index_topk (16)
        got = rig.window([t])
        want, _ = ref.forward([t], cache, pos)
        ok, why = agree_kv(got, want, kv)
        assert ok, f"decode step {step} at {pos}: {why}"
        pos += 1
    more = tokens(30, 3)                                         # a prompt chunk of sparse rows
    got = rig.prefill(more)
    want, _ = ref.forward(more, cache, pos)
    ok, why = agree_kv(got[None], want[-1:], kv)
    assert ok, f"sparse chunk: {why}"
    pos += len(more)
    got = rig.window(tokens(4, 4))
    want, _ = ref.forward(tokens(4, 4), cache, pos)
    ok, why = agree_kv(got, want, kv)
    assert ok, f"sparse window: {why}"


@pytest.mark.parametrize("kv", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("start", [9, 40])
def test_drafted_window_equals_serial_steps(folder, fakes, start, kv):
    """A verify window's rows have the bits of the serial steps (dense and sparse positions), and a window that keeps
    fewer rows leaves the state of those rows' steps."""
    prompt = tokens(start, 5)
    draft = tokens(6, 6)
    a, b = Rig(folder, kv=kv), Rig(folder, kv=kv)
    a.prefill(prompt)
    b.prefill(prompt)
    serial = torch.cat([a.window([t]) for t in draft])
    window = b.window(draft, keep=3)
    assert torch.equal(serial, window)
    b2 = b.window(draft[3:])                                     # after keeping 3: the rest, as serial steps did
    assert torch.equal(b2, serial[3:])


def test_mtp_head_matches_the_reference(folder, ref, fakes):
    rig = Rig(folder)
    rig.w.draft_head = None              # the 4-bit draft head's matmul is a CUDA kernel: the exact head here
    cache = ref.new_cache()
    prompt = tokens(20, 7)
    rig.prefill(prompt)
    _, hidden = ref.forward(prompt, cache, 0)
    nxt = prompt[1:] + [5]
    got = rig.mtp(nxt[:8], rig.pbuf.fnormed[:8])
    want = ref.mtp(nxt[:8], rig.pbuf.fnormed[:8].float(), cache, 0)
    ok, why = agree(got, want)
    assert ok, f"MTP: {why}"
    assert torch.allclose(rig.pbuf.fnormed[:20].float(), hidden, atol=0.08 * hidden.abs().max().item())


def _rank(folder, rank, world, port, prompt, steps, out_q):
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

    rig = Rig(folder, rank, world, Comm())
    rows = [rig.prefill(prompt)]
    for t in steps:
        rows.append(rig.window([t])[0])
    out_q.put((rank, torch.stack(rows), rig.w.vocab_offset))
    dist.destroy_process_group()


def test_three_ranks_match_one(folder, fakes):
    """Three ranks (heads 2/2/1, expert blocks 2/1/1 rotated by layer, vocabulary 128/64/64): their logits slices,
    put together, are one rank's within fp32 summation order."""
    prompt, steps = tokens(14, 8), tokens(6, 9)
    one = Rig(folder)
    rows = [one.prefill(prompt)] + [one.window([t])[0] for t in steps]
    whole = torch.stack(rows)
    ctx = mp.get_context("spawn")             # forked children of a threaded parent can hang
    q = ctx.Queue()
    port = 29000 + os.getpid() % 2000
    procs = [ctx.Process(target=_rank, args=(folder, r, 3, port, prompt, steps, q)) for r in range(3)]
    for p in procs:
        p.start()
    got = sorted((q.get(timeout=900) for _ in procs), key=lambda x: x[0])
    for p in procs:
        p.join(timeout=60)
    parts = torch.cat([g[1] for g in got], dim=1)
    assert [g[2] for g in got] == [0, 128, 192]
    ok, why = agree(parts, whole)
    assert ok, why
    assert (parts.argmax(-1) == whole.argmax(-1)).all()


def _rank_split(folder, rank, world, port, prompt, steps, exchange, out_q):
    """One rank: a prompt chunk unsplit and row-split (``hcsplit``, plain-residual glue), then decode steps on each."""
    import torch.distributed as dist

    os.environ.setdefault("TRITON_INTERPRET", "1")
    torch.set_num_threads(1)
    import conftest  # noqa: F401  (the interpreter patches)
    import pytest as _pt

    from tensorfold.families.glm5_next.cuda.hcsplit import HcSplit, SplitSettings

    torch.cuda.current_stream = lambda: None    # no overlap: the split never uses the main stream object on CPU
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

    out = []
    for split in (False, True):
        rig = Rig(folder, rank, world, Comm())
        if split:
            from tensorfold.families.glm5_next.cuda import forward as F

            rig.pbuf = F.Buffers(rig.w, 64, CAP, prefill=True, pad=1)
            rig.pbuf.split = HcSplit(rig.w, rig.pbuf, SplitSettings(split=True, min_rows=2, exchange=exchange))
        rows = [rig.prefill(prompt)]
        hidden = rig.pbuf.fnormed[:len(prompt)].clone()
        for t in steps:
            rows.append(rig.window([t])[0])
        out.append((torch.stack(rows), hidden))
    out_q.put((rank, out))
    dist.destroy_process_group()


@pytest.mark.parametrize("exchange", ["p2p", "gather"])
def test_three_ranks_row_split_prompt_is_bit_identical(folder, fakes, exchange):
    """A prompt chunk with its rows split between three ranks (each rank's residual add and RMSNorm on its own rows,
    the partials summed rank 0 first, the normed rows swapped; a 14-row prompt: one pad row) gives the unsplit
    chunk's logits, final normed rows and later decode steps bit for bit, on every rank."""
    prompt, steps = tokens(14, 18), tokens(4, 19)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 31000 + os.getpid() % 2000 + (7 if exchange == "gather" else 0)
    procs = [ctx.Process(target=_rank_split, args=(folder, r, 3, port, prompt, steps, exchange, q)) for r in range(3)]
    for p in procs:
        p.start()
    got = sorted((q.get(timeout=900) for _ in procs), key=lambda x: x[0])
    for p in procs:
        p.join(timeout=60)
    for rank, ((plain, h0), (split, h1)) in got:
        assert torch.equal(h0, h1), rank
        assert torch.equal(plain, split), rank


def _rank_cp(folder, rank, world, port, prompt, steps, window, out_q, kv="bf16"):
    """One rank of a context-parallel engine: a prompt chunk, serial steps, then the same steps as one drafted
    window from the prompt's state."""
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

    serial = Rig(folder, rank, world, Comm(), kv=kv, cp=world)
    rows = [serial.prefill(prompt)] + [serial.window([t])[0] for t in steps]
    drafted = Rig(folder, rank, world, Comm(), kv=kv, cp=world)
    drafted.prefill(prompt)
    win = drafted.window(window)
    out_q.put((rank, torch.stack(rows), win, serial.w.vocab_offset))
    dist.destroy_process_group()


@pytest.mark.parametrize("kv", ["bf16", "fp8", "fp4"])
def test_context_parallel_three_ranks(folder, fakes, kv):
    """Context parallelism at three ranks (every rank keeps every third token; exact selection, attention partials
    merged by log-sum-exp in rank order): the logits slices put together agree with one rank's (within fp32 /
    bf16 summation order), and a drafted window equals the serial steps bit for bit on every rank."""
    prompt, steps = tokens(40, 28), tokens(6, 29)
    one = Rig(folder, kv=kv)
    whole = torch.stack([one.prefill(prompt)] + [one.window([t])[0] for t in steps])
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 33000 + os.getpid() % 2000
    procs = [ctx.Process(target=_rank_cp, args=(folder, r, 3, port, prompt, steps, steps, q, kv)) for r in range(3)]
    for p in procs:
        p.start()
    got = sorted((q.get(timeout=1500) for _ in procs), key=lambda x: x[0])
    for p in procs:
        p.join(timeout=60)
    parts = torch.cat([g[1] for g in got], dim=1)
    ok, why = agree_kv(parts, whole, kv)
    assert ok, why
    for rank, rows, win, _ in got:
        # the drafted window's rows (pending token and steps) against the serial steps' logits
        assert torch.equal(win[:len(steps) - 1], rows[1:len(steps)]), rank
