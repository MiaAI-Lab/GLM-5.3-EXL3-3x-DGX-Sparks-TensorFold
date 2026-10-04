"""Full GLM-5.3's batched verify window (``verify.FullBatchedVerify``) on CPU: several streams' verify windows, each a
stream's own tokens at its own position on its own extent of one shared cache pool, in ONE forward; every stream's
logits (this rank's vocabulary slice) and taps must be bit for bit those of its window run alone through the
single-stream forward at the same state, in any stream order, and after partial commits the next rounds too. One rank
and three (the fake three-rank harness of test_full_forward); streams in the dense region (before index_topk = 16),
past it (sparse, the selection carried through the shared indexer layers) and crossing it within one window."""

import multiprocessing as mp
import os
from types import SimpleNamespace

import pytest
import torch

from full_fakes import TEXT, install_fake_experts, write_checkpoint

CAP = 128            # cache tokens of each stream's extent
TAPS = (1, 3)        # tap layers both paths return (DFlash2-style inputs)

# scenarios: per stream a prompt length, then rounds of (window width, rows kept) per stream; widths 1 .. 5, a round's
# rows at most 16. Positions: 9 / 13 / 6 dense, 30 / 40 sparse, 13 + 4 and 14 + 4 windows crossing the dense limit.
TWO = ([9, 30], [[(3, 2), (5, 5)], [(5, 3), (1, 1)], [(4, 4), (2, 1)], [(1, 1), (1, 1)]])
THREE = ([13, 6, 40], [[(4, 2), (1, 1), (5, 3)], [(5, 5), (3, 1), (2, 2)], [(2, 1), (5, 4), (4, 4)],
                       [(1, 1), (1, 1), (1, 1)]])


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_checkpoint(tmp_path_factory.mktemp("glm53multi"))


@pytest.fixture
def fakes(monkeypatch):
    install_fake_experts(monkeypatch)
    monkeypatch.setenv("TF_GLM_DENSE", "bf16")


def tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, TEXT["vocab_size"], (n,), generator=g).tolist()


class Pool:
    """``n`` streams on one shared pool (stream k's extent at k x CAP, slot k), with the single-stream decode
    buffers (``serial``) and a batched verifier (``batched``) over the same states."""

    def __init__(self, F, w, n):
        from tensorfold.families.glm5_next.cuda.verify import FullBatchedVerify

        self.F, self.w = F, w
        self.caches = F.Caches(w, n * CAP, streams=n, ring=F.index_ring(CAP, 64))
        slots = F.Slots(w, n, 8)
        self.st = [F.State(w, CAP, 8, caches=self.caches, base=k * CAP, slots=slots, slot=k) for k in range(n)]
        self.buf = F.Buffers(w, 8, CAP)
        self.buf.set_taps(TAPS, w.cfg.hidden)
        self.verify = FullBatchedVerify(SimpleNamespace(w=w, caches=self.caches), taps=TAPS)

    def copy_from(self, other):
        """Another pool's caches and positions (prompts prefilled once)."""
        for a, b in zip(self.caches.arena.planes, other.caches.arena.planes):
            a.tensor.copy_(b.tensor)
        for a, b in zip(self.st, other.st):
            a.set_pos(b.pos)

    def prefill(self, k, toks):
        F, st = self.F, self.st[k]
        pbuf = F.Buffers(self.w, 64, CAP, prefill=True)
        pbuf.ids[:len(toks)] = torch.tensor(toks, dtype=torch.int32)
        F.compute(self.w, st, pbuf, len(toks), nch=F.chunks_for(st, len(toks)), host_pos=st.pos)
        F.commit(self.w, st, pbuf, len(toks), len(toks))

    def serial(self, k, toks, keep):
        """Stream k's window alone through the single-stream forward; its logits and taps, then its commit."""
        F, st, b, R = self.F, self.st[k], self.buf, len(toks)
        b.ids[:R] = torch.tensor(toks, dtype=torch.int32)
        logits = F.compute(self.w, st, b, R, nch=F.chunks_for(st, R), host_pos=st.pos).clone()
        taps = torch.cat([t[:R] for t in b.taps], dim=1).clone()
        F.commit(self.w, st, b, R, keep)
        return logits, taps

    def batched(self, order, windows, keeps):
        """The streams' windows (in ``order``) as one batched forward; per stream (by index) its logits and taps."""
        from tensorfold.families.glm5_next.cuda.verify import Segment

        segs = [Segment(self.st[k], windows[k]) for k in order]
        out = self.verify.forward(segs)
        got = {k: (out.logits[i].clone(), out.taps[i].clone()) for i, k in enumerate(order)}
        self.verify.commit(segs, [keeps[k] for k in order])
        return got

    def committed(self, k):
        """Every cache plane's rows of stream k's committed tokens (latents, rotary keys, index keys)."""
        st = self.st[k]
        return [p.tensor[st.base:st.base + st.pos].clone() for p in self.caches.arena.planes
                if p.tensor.shape[0] == self.caches.rows]


def run(F, w, scenario, orders, seed):
    """Serial and batched (each stream order on a pool of its own) through the scenario's rounds; returns a list of
    (what, equal) checks."""
    prompts, rounds = scenario
    n = len(prompts)
    serial = Pool(F, w, n)
    for k, length in enumerate(prompts):
        serial.prefill(k, tokens(length, seed + k))
    pools = []
    for _ in orders:
        p = Pool(F, w, n)
        p.copy_from(serial)
        pools.append(p)
    checks = []
    for r, rnd in enumerate(rounds):
        windows = [tokens(width, seed + 100 * (r + 1) + k) for k, (width, _) in enumerate(rnd)]
        keeps = [keep for _, keep in rnd]
        want = {k: serial.serial(k, windows[k], keeps[k]) for k in range(n)}
        for order, pool in zip(orders, pools):
            got = pool.batched(order, windows, keeps)
            for k in range(n):
                checks.append((f"round {r} order {order} stream {k} (pos {serial.st[k].pos - keeps[k]}, "
                               f"{len(windows[k])} rows) logits", torch.equal(got[k][0], want[k][0])))
                checks.append((f"round {r} order {order} stream {k} taps", torch.equal(got[k][1], want[k][1])))
    for order, pool in zip(orders, pools):
        for k in range(n):
            checks.append((f"order {order} stream {k} position", pool.st[k].pos == serial.st[k].pos))
            checks.append((f"order {order} stream {k} committed cache rows",
                           all(torch.equal(a, b) for a, b in zip(pool.committed(k), serial.committed(k)))))
    return checks


def _engine(folder, rank=0, world=1, comm=None):
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=rank, world=world, device="cpu", mtp=True)
    w.comm = comm
    w.meta["long_context"] = True
    return F, w


def assert_all(checks):
    bad = [what for what, ok in checks if not ok]
    assert not bad, f"{len(bad)} of {len(checks)} differ: {bad[:8]}"


def test_two_streams_equal_serial(folder, fakes):
    """Two streams (dense and sparse), unequal windows, partial commits, both stream orders: bit for bit serial."""
    F, w = _engine(folder)
    assert_all(run(F, w, TWO, [(0, 1), (1, 0)], seed=50))


def test_three_streams_equal_serial(folder, fakes):
    """Three streams (one crossing the dense limit inside a window, one dense, one sparse), two stream orders."""
    F, w = _engine(folder)
    assert_all(run(F, w, THREE, [(0, 1, 2), (2, 0, 1)], seed=70))


def test_batched_refusals(folder, fakes):
    """Windows the batched forward cannot run exactly are refused: past 16 rows, one stream twice, CP engines."""
    from tensorfold.families.glm5_next.cuda.verify import FullBatchedVerify, Segment

    F, w = _engine(folder)
    pool = Pool(F, w, 2)
    with pytest.raises(ValueError):
        pool.verify.forward([Segment(pool.st[0], [5] * 9), Segment(pool.st[1], [6] * 8)])
    with pytest.raises(ValueError):
        pool.verify.forward([Segment(pool.st[0], [5]), Segment(pool.st[0], [6])])
    with pytest.raises(ValueError):
        FullBatchedVerify(SimpleNamespace(w=w, caches=pool.caches), rows=17)
    w.meta["cp"] = 3
    try:
        with pytest.raises(ValueError):
            FullBatchedVerify(SimpleNamespace(w=w, caches=pool.caches))
    finally:
        w.meta.pop("cp")


def _rank_multi(folder, rank, world, port, out_q):
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

    F, w = _engine(folder, rank, world, Comm())
    checks = run(F, w, THREE, [(2, 0, 1)], seed=70) + run(F, w, TWO, [(1, 0)], seed=50)
    out_q.put((rank, checks))
    dist.destroy_process_group()


def test_three_ranks_equal_serial(folder, fakes):
    """Three ranks (heads 2/2/1, experts rotated by layer, vocabulary 128/64/64): on every rank each stream's
    logits slice and taps from the batched window equal its serial window's, through partial commits."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 35000 + os.getpid() % 2000
    procs = [ctx.Process(target=_rank_multi, args=(folder, r, 3, port, q)) for r in range(3)]
    for p in procs:
        p.start()
    got = sorted((q.get(timeout=2400) for _ in procs), key=lambda x: x[0])
    for p in procs:
        p.join(timeout=60)
    for rank, checks in got:
        bad = [what for what, ok in checks if not ok]
        assert not bad, f"rank {rank}: {len(bad)} of {len(checks)} differ: {bad[:8]}"
