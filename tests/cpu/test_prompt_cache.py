"""Kept prompt states on disk (``pcache``, TF_GLM_DISK_CACHE) for full GLM-5.3, on CPU: a state file holds a rank's
slice of a ``decode.Snapshot`` bit for bit (bf16, FP8 and FP4 latent caches); a prompt resumed from a state read back
from disk and continued (more prompt, decode steps) has the logits and caches of the same prompt prefilled fresh;
the longest saved prefix is found; a foreign, flipped or truncated file falls back to a plain prefill; the budget and
the free-space floor evict least recently used states; context parallelism over three ranks (each rank its own
positions' rows) keeps prompt states in memory and on disk, and a resumed request equals a fresh one."""

import multiprocessing as mp
import os
import threading

import pytest
import torch

from full_fakes import TEXT, install_fake_experts, write_checkpoint

CAP = 128
ROWS = 64            # prompt chunk rows (a multiple of every grid)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_checkpoint(tmp_path_factory.mktemp("glm53pcache"))


@pytest.fixture
def fakes(monkeypatch):
    install_fake_experts(monkeypatch)
    monkeypatch.setenv("TF_GLM_DENSE", "bf16")
    from tensorfold.families.glm5_next.cuda import decode as D

    monkeypatch.setattr(D, "_sync", lambda w: None)      # torch.cuda.synchronize: no GPU here


def tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, TEXT["vocab_size"], (n,), generator=g).tolist()


class _NoEvent:
    def synchronize(self):
        pass

    def record(self):
        pass


def engine(folder, kv="bf16", rank=0, world=1, comm=None, cp=1, grid=0):
    """The single-stream engine (``decode.Engine``) on CPU: weights, buffers, caches of CAP tokens."""
    from tensorfold.families.glm5_next.cuda import decode as D
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=rank, world=world, device="cpu", mtp=True)
    w.draft_head = None                  # the 4-bit draft head's matmul is a CUDA kernel: the exact head here
    w.comm = comm
    if cp > 1:
        from tensorfold.families.glm5_next.cuda.tp import split_sizes

        w.meta["cp"] = cp
        w.meta["cp_heads"] = max(split_sizes(TEXT["num_attention_heads"], world, 1))
    e = D.Engine(w, capacity=CAP, max_rows=8, prefill_rows=ROWS, long_context=True, kv=kv, grid=grid)
    for b in (e.buf, e.pbuf, e.mbuf):
        b.staged = _NoEvent()            # the pinned ids copy's CUDA event
    return e


class OneComm:
    world = 1
    world_size = 1

    def all_gather(self, send, recv):
        recv.copy_(send.reshape(-1))


def glm(folder, e, *, disk=None, rank=0, world=1, comm=None, grid=0, cp=1, policy="0"):
    """A ``GlmEngine`` around ``e`` without its startup (NCCL, memory plan, calibration): the request paths only."""
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    g = GlmEngine.__new__(GlmEngine)
    g.torch = torch
    g.rank, g.world, g.cp = rank, world, cp
    g.comm = comm if comm is not None else OneComm()
    g._dev = "cpu"
    g.e, g.w = e, e.w
    g.limit = CAP - 16
    g.serial_only, g.policy, g.request = False, policy, threading.local()
    g.drafter = g.vision = g.copy = g.dump = g.costs = g.opener = None
    g.cache, g.live, g.cache_entries, g.cache_bytes = [], [], 8, 1 << 30
    g.grid, g.shared, g.eos, g.disk, g.model_dir, g.mtp_on = grid, 0, tuple(e.w.cfg.eos), disk, folder, True
    return g


def disk_cache(root, identity="test", rank=0, world=1, cp=1, kv="bf16", **kw):
    from tensorfold.families.glm5_next.cuda.pcache import DiskCache

    kw.setdefault("keep_free", 0)
    return DiskCache(root, identity, rank=rank, world=world, cp=cp, kv=kv, threads=4, **kw)


def same(a, b):
    """Bit for bit (any dtype, NaNs included)."""
    if a is None or b is None:
        return a is None and b is None
    a, b = a.contiguous(), b.contiguous()
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.reshape(-1).view(torch.uint8),
                                                                     b.reshape(-1).view(torch.uint8))


def views(e):
    """The state's rows of every cache plane (this rank's own rows under context parallelism)."""
    from tensorfold.families.glm5_next.cuda.decode import _row_views

    return [v.clone() for v in _row_views(e.st, e.st.pos, e.st.mtp_len)]


# -- the file format ------------------------------------------------------------------------------------------------
def test_local_rows():
    from tensorfold.families.glm5_next.cuda.decode import local_rows

    for n in range(0, 20):
        for r in range(3):
            assert local_rows(n, r, 3) == sum(1 for p in range(n) if p % 3 == r)
        assert local_rows(n, 0, 1) == n


@pytest.mark.parametrize("kv", ["bf16", "fp8", "fp4"])
def test_round_trip_is_bit_identical(folder, fakes, tmp_path, kv):
    """Every section (token ids, MTP rows, the head's row, every cache plane's rows) comes back with its bits."""
    from tensorfold.families.glm5_next.cuda import decode as D
    from tensorfold.families.glm5_next.cuda import pcache as P

    e = engine(folder, kv)
    prompt = tokens(40, 1)
    D.prefill(e, prompt, None, mtp=True, keep_head=True)
    snap = D.take_snapshot(e, prompt, e.last_hidden, mtp=True)
    snap.head = e.head
    path = tmp_path / "state.tfpc"
    size = P.save_snapshot(path, e, snap, identity="id", kv=kv)
    assert size == path.stat().st_size and size % P.PAGE == 0
    meta = P.read_header(path)
    assert meta["n"] == 40 and meta["kv"] == kv and meta["mtp_len"] == snap.mtp_len == 39
    assert all(s["offset"] % P.PAGE == 0 for s in meta["sections"])
    back = P.open_snapshot(path, e, prompt, identity="id", kv=kv)
    want = D._saved_views(e, snap)
    assert len(back.rows) == len(want) and len(want) == 15      # 5 latents, 5 rotary, 2 index keys, MTP's 3
    for a, b in zip(back.rows, want):
        assert same(a, b)
    if kv != "bf16":
        assert back.rows[0].dtype == (torch.uint8 if kv == "fp8" else torch.int8)
    for name in ("rec", "conv", "pending", "head"):
        assert same(getattr(back, name), getattr(snap, name)), name
    assert back.ids == prompt and back.mtp_len == snap.mtp_len and back.drafter_end == -1
    # loaded into the live caches of another engine: the same rows there
    other = engine(folder, kv)
    live = P.open_snapshot(path, other, prompt, identity="id", kv=kv, live=True)
    assert live.rows is None
    for a, b in zip(D._saved_views(other, live), want):
        assert same(a, b)


@pytest.mark.parametrize("kv", ["bf16", "fp8", "fp4"])
def test_resumed_from_disk_equals_fresh(folder, fakes, tmp_path, kv):
    """A prompt resumed from a state read back from disk, then more prompt and decode steps: the logits (the
    prompt end's head row, every step's) and the caches of the fresh prefill of the same prompt, bit for bit; the
    file is written by the background writer from the live caches."""
    from tensorfold.families.glm5_next.cuda import decode as D
    from tensorfold.families.glm5_next.cuda.forward import commit

    head_ids, prompt, steps = None, tokens(56, 2), tokens(4, 3)
    head_ids = prompt[:32]

    def finish(e, resume):
        D.prefill(e, prompt, None, mtp=True, resume=resume, keep_head=True)
        rows = [e.head.clone()]
        for t in steps:
            logits = e.forward([t])
            rows.append(logits[:1].float().clone())
            commit(e.w, e.st, e.buf, 1, 1)
        return rows, views(e)

    # fresh: through the state at 32 in memory (as the engine keeps it), and straight from 0
    a = engine(folder, kv)
    D.prefill(a, head_ids, None, mtp=True, sample=False)
    snap = D.take_snapshot(a, head_ids, a.last_hidden, mtp=True)
    cache = disk_cache(tmp_path, kv=kv)
    assert cache.submit(a, snap)
    cache.writer.drain()
    assert cache.stats["written"] == 1
    fresh_rows, fresh_views = finish(a, snap)
    z = engine(folder, kv)
    zero_rows, zero_views = finish(z, None)
    # resumed from disk in a new engine: rows read straight into its caches
    from tensorfold.families.glm5_next.cuda.pcache import load_snapshot

    b = engine(folder, kv)
    meta = cache.check(head_ids)
    back = load_snapshot(cache, b, head_ids, meta, live=True)
    disk_rows, disk_views = finish(b, back)
    for x, y in zip(disk_rows, fresh_rows):
        assert same(x, y)
    for x, y in zip(disk_views, fresh_views):
        assert same(x, y)
    for x, y in zip(zero_rows, fresh_rows):
        assert same(x, y)
    for x, y in zip(zero_views, fresh_views):
        assert same(x, y)


def fake_snapshot(e, ids, mtp=True):
    """A kept state of ``ids`` without running them (the caches' rows as they are): the index and eviction tests."""
    from tensorfold.families.glm5_next.cuda import decode as D

    e.st.set_pos(len(ids))
    e.st.set_mtp_len(len(ids) - 1 if mtp else 0)
    pending = torch.zeros((1, TEXT["hidden_size"]), dtype=torch.bfloat16) if mtp else None
    return D.take_snapshot(e, ids, pending, mtp=mtp)


def test_longest_saved_prefix(folder, fakes, tmp_path):
    e = engine(folder)
    cache = disk_cache(tmp_path)
    base = tokens(100, 4)
    for n in (16, 32, 48):
        assert cache.submit(e, fake_snapshot(e, base[:n]))
        cache.writer.drain()
    other = tokens(64, 5)
    assert cache.submit(e, fake_snapshot(e, other[:64], mtp=False))
    cache.writer.drain()
    yes = lambda entry: True            # noqa: E731
    assert cache.lookup(base[:60], yes)["n"] == 48
    assert cache.lookup(base[:48], yes)["n"] == 48
    assert cache.lookup(base[:47], yes)["n"] == 32
    assert cache.lookup(base[:20] + [1] * 40, yes)["n"] == 16
    assert cache.lookup(base[:15], yes) is None
    assert cache.lookup(other + [3, 4], yes)["n"] == 64
    assert cache.lookup(other + [3, 4], lambda entry: entry["mtp"]) is None
    assert cache.lookup(base[:60], lambda entry: entry["n"] < 48)["n"] == 32
    # another engine identity never sees them
    stranger = disk_cache(tmp_path, identity="other model")
    assert stranger.lookup(base[:60], yes) is None
    # the engine's candidate rule: longer than the memory hit, on the grid, the replay needs the head row
    g = glm(folder, e, disk=cache, grid=16)
    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    code = encode_policy("0")
    assert g._disk_candidate(base[:60], code, None) == 48
    assert g._disk_candidate(base[:48], code, None) == 32           # 48 is the whole prompt: no head row kept
    g.grid = 32
    assert g._disk_candidate(base[:60], code, None) == 32


def test_corrupt_or_foreign_files_fall_back(folder, fakes, tmp_path):
    from tensorfold.families.glm5_next.cuda import pcache as P

    e = engine(folder)
    ids = tokens(40, 6)
    cache = disk_cache(tmp_path)
    cache.submit(e, fake_snapshot(e, ids))
    cache.writer.drain()
    path = cache.path(cache.key(ids))
    good = path.read_bytes()
    meta = cache.check(ids)

    other = engine(folder)

    def load():
        m = cache.check(ids)
        return P.load_snapshot(cache, other, ids, m, live=False)

    assert load().rows is not None
    # another identity, rank, world, cp, kv kind, or token ids: the header says no
    for k, v in (("identity", "x"), ("rank", 1), ("world", 3), ("cp", 3), ("kv", "fp8")):
        with pytest.raises(P.PCacheError):
            P.check_header(meta, dict(cache.expect(ids), **{k: v}))
    with pytest.raises(P.PCacheError):          # a cache of another identity looks for another file
        disk_cache(tmp_path, identity="another model").check(ids)
    with pytest.raises(P.PCacheError):
        P.check_header(meta, cache.expect(ids[:-1] + [ids[-1] ^ 1]))
    # a flipped byte in a row, in the token ids, in the header; a truncated file; a missing one
    start = meta["data_start"]
    rows0 = next(s for s in meta["sections"] if s["name"] == "rows.3")
    ids_s = next(s for s in meta["sections"] if s["name"] == "ids")
    for at in (start + rows0["offset"] + 17, start + ids_s["offset"] + 2, 40):
        bad = bytearray(good)
        bad[at] ^= 0x10
        path.write_bytes(bytes(bad))
        with pytest.raises(P.PCacheError):
            load()
    path.write_bytes(good[:len(good) - P.PAGE])
    with pytest.raises(P.PCacheError):
        load()
    path.write_bytes(good[:len(good) - 100])
    with pytest.raises(P.PCacheError):
        load()
    path.unlink()
    with pytest.raises(P.PCacheError, match="missing"):
        load()


def test_budget_and_free_space_evict_least_recently_used(folder, fakes, tmp_path):
    e = engine(folder)
    probe = disk_cache(tmp_path / "probe")
    probe.submit(e, fake_snapshot(e, tokens(32, 7)))
    probe.writer.drain()
    one = probe.held()                                   # one 32-token state's file
    cache = disk_cache(tmp_path / "c", budget=2 * one + one // 2)
    a, b, c, d = (tokens(32, s) for s in (8, 9, 10, 11))
    for ids in (a, b):
        assert cache.submit(e, fake_snapshot(e, ids))
        cache.writer.drain()
    cache.touch(cache.key(a))                            # a used last: b goes first
    cache.submit(e, fake_snapshot(e, c))
    cache.writer.drain()
    assert cache.contains(cache.key(a)) and cache.contains(cache.key(c)) and not cache.contains(cache.key(b))
    assert not cache.path(cache.key(b)).exists() and cache.stats["evicted"] == 1
    assert cache.held() <= cache.budget
    # the free-space floor: the disk "has" 2.5 states' room above the floor, so one more evicts the oldest
    cache.keep_free = 10 ** 12
    cache.free_bytes = lambda: cache.keep_free + one + one // 2
    cache.submit(e, fake_snapshot(e, d))
    cache.writer.drain()
    assert cache.contains(cache.key(d)) and cache.stats["written"] == 4
    # no room at all: nothing is written, everything else evicted trying
    cache.free_bytes = lambda: cache.keep_free + one // 2
    cache.submit(e, fake_snapshot(e, tokens(32, 12)))
    cache.writer.drain()
    assert cache.stats["written"] == 4 and not cache.entries and cache.stats["skipped"] == 1
    # a state larger than the budget is skipped
    small = disk_cache(tmp_path / "s", budget=one // 2)
    small.submit(e, fake_snapshot(e, a))
    small.writer.drain()
    assert not small.entries and not list((tmp_path / "s").glob("*.tfpc"))


def test_partial_and_orphan_files_never_count(folder, fakes, tmp_path):
    e = engine(folder)
    cache = disk_cache(tmp_path)
    keep, lost = tokens(32, 13), tokens(32, 14)
    for ids in (keep, lost):
        cache.submit(e, fake_snapshot(e, ids))
        cache.writer.drain()
    part = cache.path(cache.key(tokens(32, 15))).with_name(cache.path("x").name + ".77.1.part")
    part.write_bytes(b"\0" * 8192)                       # a write cut short by a crash
    orphan = cache.path("f" * 40)
    orphan.write_bytes(b"\0" * 8192)                     # renamed, the index not yet written
    cache.path(cache.key(lost)).unlink()                 # an entry whose file is gone
    again = disk_cache(tmp_path)
    assert not part.exists() and not orphan.exists()
    assert again.contains(again.key(keep)) and not again.contains(again.key(lost))
    # a cancelled write (the next request changes its rows) leaves nothing either
    cache2 = disk_cache(tmp_path / "x")
    job_ids = tokens(40, 16)
    with cache2.writer.cv:                               # the writer cannot take it before the fence runs
        assert cache2.submit(e, fake_snapshot(e, job_ids))
        assert cache2.fence(0) == 1
    cache2.writer.drain()
    assert not cache2.entries and not list((tmp_path / "x").glob("*.tfpc*"))
    # a running write cancelled between pieces: no file, partial or whole
    from tensorfold.families.glm5_next.cuda import pcache as P

    fields, tensors = P.snapshot_sections(e, fake_snapshot(e, job_ids))
    job = P.Job("k", tmp_path / "x" / "k.w1r0.tfpc", dict(fields, n=40), tensors, {})
    job.cancel = True
    with pytest.raises(P.Cancelled):
        P.write_file(job.path, job.meta, tensors, staging=P.Staging(2, False), pool=None, job=job)
    assert not list((tmp_path / "x").glob("*.tfpc*"))


# -- through the engine's request path ------------------------------------------------------------------------------
def run(g, prompt, n=4):
    got = []
    g.e.sampled = []
    stats = g.generate(list(prompt), n, None, lambda t: got.extend(t))
    return got, stats


def record_logits(e):
    """Every sampled logits row (this rank's slice), in order."""
    orig = e.sample

    def sample(logits, positions, sampling, **kw):
        e.sampled.append(logits.float().clone())
        return orig(logits, positions, sampling, **kw)

    e.sample = sample
    e.sampled = []


@pytest.mark.parametrize("grid", [0, 16])
def test_engine_resumes_from_disk_as_fresh(folder, fakes, tmp_path, grid):
    """One rank: a request keeps its prompt state, the writer puts it on disk; a new engine (nothing in memory)
    resumes a longer prompt from that file: the replies, every sampled logits row and the caches equal a fresh
    engine's; a corrupt file falls back to a plain prefill with the same result."""
    p1 = tokens(40, 17)
    p2 = p1 + tokens(16, 18)
    x = glm(folder, engine(folder, grid=grid), disk=disk_cache(tmp_path), grid=grid)
    x.generate(p1, 4, None, lambda t: None)
    x.disk.writer.drain()
    kept = 32 if grid else 40
    assert x.disk.stats["written"] == 1 and x.disk.lookup(p2, lambda en: True)["n"] == kept

    def fresh_engine(disk):
        e = engine(folder, grid=grid)
        record_logits(e)
        return glm(folder, e, disk=disk, grid=grid)

    z = fresh_engine(None)
    want, zstats = run(z, p2)
    y = fresh_engine(disk_cache(tmp_path))
    got, ystats = run(y, p2)
    assert ystats.get("disk_cached") == kept and ystats["cached"] == kept and zstats["cached"] == 0
    assert got == want
    assert all(same(a, b) for a, b in zip(y.e.sampled, z.e.sampled)) and len(y.e.sampled) == len(z.e.sampled)
    assert all(same(a, b) for a, b in zip(views(y.e), views(z.e)))
    # the state read back is kept in memory too: the same request again resumes there without the disk
    again, s3 = run(y, p2)
    assert again == want and "disk_cached" not in s3 and s3["cached"] >= kept
    # a flipped byte in the rows: the load fails its checksum, the request prefills from scratch, same reply
    y.disk.writer.drain()
    d = disk_cache(tmp_path)
    for k in [k for k in d.entries if k != d.key(p2[:kept])]:     # y kept p2's own state too: only the first
        d.remove(k)
    path = d.path(d.key(p2[:kept]))
    raw = bytearray(path.read_bytes())
    meta = d.check(p2[:kept])
    last = max(meta["sections"], key=lambda s: s["offset"])
    raw[meta["data_start"] + last["offset"] + 1] ^= 0x40
    path.write_bytes(bytes(raw))
    w = fresh_engine(disk_cache(tmp_path))
    got2, wstats = run(w, p2)
    assert wstats.get("disk_miss") == kept and wstats["cached"] == 0 and got2 == want
    assert all(same(a, b) for a, b in zip(w.e.sampled, z.e.sampled))
    assert not path.exists()                         # forgotten


# -- context parallelism, three ranks ----------------------------------------------------------------------------------
def _rank_cp(folder, rank, world, port, root, p1, p2, out_q):
    try:
        _rank_cp_body(folder, rank, world, port, root, p1, p2, out_q)
    except BaseException:                   # a rank that fails says so now (the others then hang: the test ends)
        import traceback

        out_q.put((rank, {"error": traceback.format_exc()}))
        raise


def _rank_cp_body(folder, rank, world, port, root, p1, p2, out_q):
    import torch.distributed as dist

    os.environ.setdefault("TRITON_INTERPRET", "1")
    torch.set_num_threads(1)
    import conftest  # noqa: F401  (the interpreter patches)
    import pytest as _pt

    from tensorfold.families.glm5_next.cuda import decode as D

    mp_ = _pt.MonkeyPatch()
    install_fake_experts(mp_)
    mp_.setattr(D, "_sync", lambda w: None)
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

    Comm.world = world                       # the sampler's gather sizes its buffer by it
    comm = Comm()
    grid = 16

    def make(disk):
        e = engine(folder, rank=rank, world=world, comm=comm, cp=world, grid=grid)
        record_logits(e)
        return glm(folder, e, disk=disk, rank=rank, world=world, comm=comm, grid=grid, cp=world)

    def request(g, prompt):
        if rank == 0:
            return run(g, prompt)
        g.e.sampled = []
        g.follow_one()
        return None, None

    out = {}
    # x: the first prompt, then the longer one resumed from the state kept in memory (CP: the rank's own rows)
    x = make(disk_cache(root, rank=rank, world=world, cp=world))
    request(x, p1)
    x.disk.writer.drain()
    out["files"] = len(x.disk.entries)
    out["mem"], out["mem_stats"] = request(x, p2)
    x.disk.writer.drain()                                # p2's state at 48 goes to disk too
    out["mem_views"], out["mem_logits"] = views(x.e), list(x.e.sampled)
    dist.barrier()
    # y: a new engine resumes the longer prompt from this rank's file; z: fresh
    y = make(disk_cache(root, rank=rank, world=world, cp=world))
    out["disk"], out["disk_stats"] = request(y, p2)
    out["disk_views"], out["disk_logits"] = views(y.e), list(y.e.sampled)
    out["y_hits"] = y.disk.stats["hits"]
    z = make(None)
    out["fresh"], out["fresh_stats"] = request(z, p2)
    out["fresh_views"], out["fresh_logits"] = views(z.e), list(z.e.sampled)
    out["local"] = [v.shape[0] for v in out["fresh_views"]]
    import io

    buf = io.BytesIO()                       # bytes, not fd-shared tensors: this process may be gone when read
    torch.save(out, buf)
    out_q.put((rank, buf.getvalue()))
    dist.destroy_process_group()


def test_context_parallel_three_ranks_keep_and_resume(folder, fakes, tmp_path):
    p1, p2 = tokens(40, 21), None
    p2 = p1 + tokens(16, 22)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 35000 + os.getpid() % 2000
    procs = [ctx.Process(target=_rank_cp, args=(folder, r, 3, port, str(tmp_path), p1, p2, q)) for r in range(3)]
    for p in procs:
        p.start()
    got = {}
    for _ in procs:
        rank, o = q.get(timeout=2400)
        if isinstance(o, bytes):
            import io

            o = torch.load(io.BytesIO(o), weights_only=False)
        if "error" in o:
            for p in procs:
                p.kill()
            pytest.fail(f"rank {rank}:\n{o['error']}")
        got[rank] = o
    for p in procs:
        p.join(timeout=60)
    zero = got[0]
    assert zero["mem_stats"]["cached"] == 32 and zero["disk_stats"].get("disk_cached") == 48
    assert zero["fresh_stats"]["cached"] == 0
    assert zero["mem"] == zero["fresh"] and zero["disk"] == zero["fresh"]
    for rank, o in sorted(got.items()):
        assert o["files"] == 1 and o["y_hits"] == 1, rank
        # each rank's own positions: 59 (56 and 3 decode steps); rows p < 59 with p % 3 == rank
        assert o["local"][0] == sum(1 for p in range(59) if p % 3 == rank), (rank, o["local"])
        for key in ("mem", "disk"):
            assert len(o[f"{key}_logits"]) == len(o["fresh_logits"]), (rank, key)
            assert all(same(a, b) for a, b in zip(o[f"{key}_logits"], o["fresh_logits"])), (rank, key)
            assert all(same(a, b) for a, b in zip(o[f"{key}_views"], o["fresh_views"])), (rank, key)
