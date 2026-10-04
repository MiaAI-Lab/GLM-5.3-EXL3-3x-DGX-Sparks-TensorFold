"""Multi-stream milestone 4 on CPU: the segmented verify (``full_seg.FullSegVerify``) wired into the scheduler
(``multi.MultiDecoder``, TF_GLM_MULTI_VERIFY=seg by default, segments / serial selectable), its CUDA graphs and the
DSpark contexts' graphs, and their memory accounting.

CPU runs no CUDA graphs, so the graphs are checked through their structure: a capture and a forward call the same
``FullSegVerify._run(R, NT)``, and a fake capturer stands in for ``full_seg.capture_graph`` whose "graph" records
every launch of the captured run (Triton kernels with their grids, the torch ops; tensors as the persistent storage
and offset they point into, or as a temporary by shape) and, on each replay, re-runs the run with the capture's own
R and NT and asserts its launches are exactly the captured ones: so every per-window value reaches the kernels through
the staged device tables and ids (a graph's replay on another window is the eager run on it), and nothing syncs with
the host (no ``_local_scalar_dense``). The replies of streams served with these "graphs" must equal each request
served alone, and the logits / taps / caches equal ``verify.FullBatchedVerify``'s per-segment loop; the real
replay-vs-eager bit checks are tests/gpu/test_multi_graphs_gpu.py."""

import gc
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import test_full_multi_engine as fm
import test_multi_kernels as mk
from full_fakes import install_fake_experts, write_checkpoint


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_checkpoint(tmp_path_factory.mktemp("glm53graphs"))


@pytest.fixture(scope="module")
def dspark(tmp_path_factory):
    return fm.write_dspark(tmp_path_factory.mktemp("dspark4") / "d")


@pytest.fixture
def fakes(monkeypatch):
    install_fake_experts(monkeypatch)
    fm._env(monkeypatch)
    for k in ("TF_GLM_MULTI_PREFILL", "TF_GLM_MULTI_VERIFY", "TF_GLM_MULTI_DEPTH", "TF_GLM_FILL_BUDGET_MS",
              "TF_GLM_DSPARK_MULTI_POLICY", "TF_GLM_FILL_ROWS", "TF_GLM_MULTI_GRAPHS", "TF_GLM_MULTI_DRAFT_GRAPHS"):
        monkeypatch.delenv(k, raising=False)
    install_hooks(monkeypatch)


# -- the launch recorder and the fake graphs ----------------------------------------------------------------------------
REC = None


def live_storages() -> set:
    """Storage addresses of every tensor alive now (the persistent ones: weights, caches, buffers, tables)."""
    out = set()
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor):
                out.add(o.untyped_storage().data_ptr())
        except Exception:                                  # noqa: BLE001  (objects that refuse isinstance / storage)
            pass
    return out


class Recorder(TorchDispatchMode):
    """Every Triton launch and torch op of a run, its tensors canonical: ("P", storage, offset, shape, stride,
    dtype) for storage alive before the run, ("T", offset, shape, stride, dtype) for temporaries (a graph's pool
    memory). Inside a launch or the fake experts nothing is recorded (their own internals: the interpreter's host
    copies; the EXL3 stand-in's per-row loop, which the real kernels replace)."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.depth = 0
        self.persistent = live_storages()

    def canon(self, x):
        if isinstance(x, torch.Tensor):
            try:
                ptr = x.untyped_storage().data_ptr()
            except Exception:                              # noqa: BLE001
                ptr = None
            where = ("P", ptr) if ptr in self.persistent else ("T",)
            return where + (x.storage_offset(), tuple(x.shape), tuple(x.stride()), str(x.dtype))
        if isinstance(x, (list, tuple)):
            return tuple(self.canon(v) for v in x)
        if isinstance(x, dict):
            return tuple((str(k), self.canon(v)) for k, v in sorted(x.items(), key=lambda kv: str(kv[0])))
        if isinstance(x, float):
            return ("f", x if x == x else "nan")
        if x is None or isinstance(x, (bool, int, str)):
            return x
        if isinstance(x, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
            return str(x)
        return type(x).__name__

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.depth == 0:
            self.events.append(("op", str(func), self.canon(args), self.canon(kwargs)))
        return func(*args, **kwargs)


def install_hooks(monkeypatch):
    """Record Triton interpreter launches (``GridExecutor.__call__``) and the routed experts as single events."""
    from triton.runtime import interpreter as itp

    from tensorfold.cuda.exl3 import experts as generic

    ge = getattr(itp, "GridExecutor", None)
    assert ge is not None, "cannot hook Triton's interpreter launches (no GridExecutor)"
    call = ge.__call__

    def launch(self, *args, **kwargs):
        rec = REC
        if rec is None or rec.depth:
            return call(self, *args, **kwargs)
        grid = getattr(self, "grid", None)
        rec.events.append(("launch", getattr(getattr(self, "fn", None), "__name__", "?"),
                           "callable" if callable(grid) else rec.canon(grid), rec.canon(args), rec.canon(kwargs)))
        rec.depth += 1
        try:
            return call(self, *args, **kwargs)
        finally:
            rec.depth -= 1

    monkeypatch.setattr(ge, "__call__", launch)
    routed = generic.routed

    def routed_once(*args, **kwargs):
        rec = REC
        if rec is None or rec.depth:
            return routed(*args, **kwargs)
        rec.events.append(("routed", rec.canon(args), rec.canon(kwargs)))
        rec.depth += 1
        try:
            return routed(*args, **kwargs)
        finally:
            rec.depth -= 1

    monkeypatch.setattr(generic, "routed", routed_once)


def record(fn):
    global REC
    rec = Recorder()
    REC = rec
    try:
        with rec:
            fn()
    finally:
        REC = None
    return rec.events


def same_launches(want, got, what):
    if want == got:
        return
    i = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
    show = lambda ev: (str(ev[i]) if i < len(ev) else "(none)")[:600]       # noqa: E731
    raise AssertionError(f"{what}: launch {i} of {len(want)} captured / {len(got)} replayed differs:\n"
                         f"  captured {show(want)}\n  replayed {show(got)}")


class FakeGraph:
    """A CUDA graph's stand-in: ``replay`` re-runs the captured callable (its R and NT are the capture's) and asserts
    the same launches as the capture."""

    count = 0                      # replays, every graph
    captured = 0

    def __init__(self, fn, trace, key):
        self.fn, self.trace, self.key = fn, trace, key
        self.replays = 0
        FakeGraph.captured += 1

    def replay(self):
        same_launches(self.trace, record(self.fn), f"graph {self.key}")
        self.replays += 1
        FakeGraph.count += 1


def fake_capture(fn, pool, warm: int = 1):
    """``full_seg.capture_graph``'s stand-in (warm-up runs, then the recorded run)."""
    assert pool is not None
    for _ in range(warm):
        fn()
    trace = record(fn)
    assert any(ev[0] == "launch" for ev in trace), "no Triton launch recorded"
    syncs = [ev[1] for ev in trace if ev[0] == "op" and "_local_scalar_dense" in ev[1]]
    assert not syncs, f"a host sync inside a captured run: {syncs[:3]}"
    return FakeGraph(fn, trace, f"#{FakeGraph.captured} ({len(trace)} launches)")


# -- pools of streams with either verifier ------------------------------------------------------------------------------
class Pool:
    """``mk.Pool`` with an extent size ``cap`` and any verifier class (FullSegVerify's ``kw``)."""

    def __init__(self, F_, w, n, cls, cap=mk.CAP, kv="bf16", **kw):
        self.F, self.w, self.cap = F_, w, cap
        self.caches = F_.Caches(w, n * cap, streams=n, kv=kv)
        slots = F_.Slots(w, n, 8)
        place = [(k + 1) % n for k in range(n)]
        self.st = [F_.State(w, cap, 8, caches=self.caches, base=place[k] * cap, slots=slots, slot=k)
                   for k in range(n)]
        self.verify = cls(SimpleNamespace(w=w, caches=self.caches), taps=mk.TAPS, **kw)

    copy_from = mk.Pool.copy_from
    batched = mk.Pool.batched

    def prefill(self, k, toks):
        F_, st = self.F, self.st[k]
        pbuf = F_.Buffers(self.w, 64, self.cap, prefill=True)
        pbuf.ids[:len(toks)] = torch.tensor(toks, dtype=torch.int32)
        F_.compute(self.w, st, pbuf, len(toks), nch=F_.chunks_for(st, len(toks)), host_pos=st.pos)
        F_.commit(self.w, st, pbuf, len(toks), len(toks))


def run_graphs(F_, w, scenario, order, seed, *, rows, nts, mode="all", cap=mk.CAP, kv="bf16"):
    """The per-segment loop (FullBatchedVerify) and FullSegVerify with fake graphs of (rows x nts) captured before
    the streams arrive, from the same prompts, through the rounds: (what, equal) checks and the replay counts."""
    from tensorfold.families.glm5_next.cuda.full_seg import FullSegVerify
    from tensorfold.families.glm5_next.cuda.verify import FullBatchedVerify

    prompts, rounds = scenario
    n = len(prompts)
    base = Pool(F_, w, n, FullBatchedVerify, cap, kv)
    for k, length in enumerate(prompts):
        base.prefill(k, mk.tokens_of(length, seed + k))
    a, b = Pool(F_, w, n, FullBatchedVerify, cap, kv), Pool(F_, w, n, FullSegVerify, cap, kv)
    got_n = b.verify.capture(mode=mode, pool="cpu", rows=rows, nts=nts, capturer=fake_capture)
    assert got_n == len(rows) * len(nts)
    a.copy_from(base)
    b.copy_from(base)
    checks = []
    for r, rnd in enumerate(rounds):
        windows = [mk.tokens_of(width, seed + 100 * (r + 1) + k) for k, (width, _) in enumerate(rnd)]
        keeps = [keep for _, keep in rnd]
        pos = [st.pos for st in a.st]
        want = a.batched(order, windows, keeps)
        got = b.batched(order, windows, keeps)
        for k in range(n):
            what = f"round {r} stream {k} (pos {pos[k]}, {len(windows[k])} rows)"
            checks.append((what + " logits", torch.equal(got[k][0], want[k][0])))
            checks.append((what + " taps", torch.equal(got[k][1], want[k][1])))
    for k in range(n):
        checks.append((f"stream {k} position", a.st[k].pos == b.st[k].pos))
    checks.append(("cache planes", all(torch.equal(x.tensor, y.tensor) for x, y in
                                       zip(a.caches.arena.planes, b.caches.arena.planes))))
    return checks, dict(b.verify.replays), b.verify


def test_graph_replays_equal_the_loop(folder, fakes):
    """Three streams (sparse rows), fake graphs of 3 and 10 rows at the bucket: rounds of 10, 10 and 3 rows replay
    (their launches the captured ones on other streams' positions and extents), the 11-row round runs eagerly; logits,
    taps and caches equal the per-segment loop's."""
    F_, w = mk._engine(folder)
    checks, replays, v = run_graphs(F_, w, mk.THREE, (2, 0, 1), 70, rows=[3, 10], nts=[mk.CAP * 3])
    mk.assert_all(checks)
    assert replays == {"graph": 3, "eager": 1}, replays
    assert sum(g.replays for g in v.graphs.values()) == 3


def test_graph_replays_without_index_caches(folder, fakes):
    """No index caches: graphs of NT 0 only; the 8-row round replays, the 3-row one runs eagerly."""
    F_, w = mk._engine(folder)
    w.meta["long_context"] = False
    short = ([9, 4], [[(3, 2), (5, 5)], [(2, 1), (1, 1)]])
    checks, replays, v = run_graphs(F_, w, short, (1, 0), 30, rows=[8], nts=[0])
    mk.assert_all(checks)
    assert replays == {"graph": 1, "eager": 1}, replays
    assert v.graph_buckets() == [0]


def test_graph_replays_fp4x(folder, fakes):
    """TF_GLM_KV=fp4x caches under fake graphs (the e4m3 rotary and index planes, scales in the latent rows)."""
    F_, w = mk._engine(folder)
    checks, replays, _ = run_graphs(F_, w, mk.TWO, (1, 0), 150, rows=[8, 6], nts=[mk.CAP * 2], kv="fp4x")
    mk.assert_all(checks)
    assert replays == {"graph": 3, "eager": 1}, replays


def test_top_bucket_equals_the_loop(folder, fakes):
    """TF_GLM_MULTI_GRAPHS=top: windows whose bucket is 4,096 run (replayed) at the largest one (8,192, the pool):
    the same bits as the loop at each window's own bucket (a row's selection reads only its own columns)."""
    F_, w = mk._engine(folder)
    two = ([9, 30], [[(3, 2), (5, 5)], [(5, 3), (1, 1)]])
    checks, replays, v = run_graphs(F_, w, two, (0, 1), 50, rows=[8, 6], nts=[2 * 4096], mode="top", cap=4096)
    mk.assert_all(checks)
    assert v.top == 2 * 4096 and v.graph_buckets(mode="top") == [0, 2 * 4096]
    v.meta.set([(0, 40, 2, 4096)], dense_limit=w.cfg.dense_limit, indexed=True)
    assert v.meta.bucket(v.capacity) == 4096 and v.bucket() == 2 * 4096
    assert replays == {"graph": 2, "eager": 0}, replays


def test_bucket_lists():
    """The buckets a window reaches, the graphs' keys and counts."""
    from tensorfold.families.glm5_next.cuda import full_seg as S

    assert S.reach_buckets(384) == [384]
    assert S.reach_buckets(10240, 2048) == [4096]
    assert S.reach_buckets(10240) == [4096, 8192, 10240]
    assert S.reach_buckets(4 * 200_000, 200_016) == [4096 << i for i in range(7)]
    assert S.reach_buckets(300_000, 300_000) == [4096 << i for i in range(7)] + [300_000]
    assert S.graph_keys(2, [0, 4096]) == [(1, 0), (2, 0), (1, 4096), (2, 4096)]
    assert S.graph_count(16, 800_000, 200_016, indexed=True) == 16 * 8
    assert S.graph_count(16, 800_000, 200_016, indexed=True, mode="top") == 32
    assert S.graph_count(16, 800_000, 200_016, indexed=False) == 16
    assert S.graph_count(16, 800_000, 200_016, indexed=True, mode="0") == 0
    assert [S.graph_mode(v) for v in ("", "1", "all", "top", "0")] == ["all", "all", "all", "top", "0"]
    with pytest.raises(ValueError):
        S.graph_mode("yes")


# -- the scheduler's verifier -------------------------------------------------------------------------------------------
def multi_with(folder, streams, *, w=None, dspark=None, drafts=None, make_verify=None, taps=()):
    """``fm.multi`` with the verifier made on the engine (``make_verify(e)``), or the decoder's own choice."""
    from tensorfold.families.glm5_next.cuda import decode
    from tensorfold.families.glm5_next.cuda.multi import MultiDecoder

    w = w if w is not None else fm.load_w(folder)
    pool = (streams + 2) * fm.EXT
    drafter = None
    if dspark is not None:
        from tensorfold.families.glm5_next.cuda.dspark import Drafter

        drafter = Drafter(dspark, w, capacity=pool, tap_rows=16, ring=True)
    e = decode.Engine(w, capacity=pool, max_rows=16, prefill_rows=64, graphs=False, long_context=True,
                      taps=drafter.tap_layers if drafter is not None else tuple(taps), streams=streams,
                      pool_rows=pool, kv="bf16")
    host = fm.Host(w, e, drafter, 0, 1, None, limit=fm.EXT - 16)
    verify = make_verify(e) if make_verify is not None else None
    return MultiDecoder(host, streams, drafts=drafts, verify=verify), host


def test_verify_kinds(folder, fakes, monkeypatch):
    """TF_GLM_MULTI_VERIFY on full GLM-5.3: seg (default, also "batched") picks FullSegVerify, segments the
    per-segment FullBatchedVerify, serial SerialVerify; GLM-5.3-Flash keeps batched / serial."""
    from tensorfold.families.glm5_next.cuda.multi import verify_kind

    assert [verify_kind(v, full=True) for v in ("", "batched", "seg", "SEG ", "segments", "serial")] == \
        ["seg", "seg", "seg", "seg", "segments", "serial"]
    assert [verify_kind(v) for v in ("", "batched", "serial")] == ["batched", "batched", "serial"]
    with pytest.raises(ValueError, match="full GLM"):
        verify_kind("seg")
    with pytest.raises(ValueError):
        verify_kind("both", full=True)
    w = fm.load_w(folder)
    for value, name in (("", "FullSegVerify"), ("segments", "FullBatchedVerify"), ("serial", "SerialVerify"),
                        ("seg", "FullSegVerify")):
        monkeypatch.setenv("TF_GLM_MULTI_VERIFY", value)
        dec, _ = multi_with(folder, 2, w=w)
        assert type(dec.verify).__name__ == name, (value, type(dec.verify).__name__)
        assert dec.graph_counts == {"verify": 0, "drafts": 0}           # CPU: no graphs captured


def test_seg_dspark_streams_equal_serial(folder, dspark, fakes):
    """Two DSpark streams through the default (segmented) verifier: every reply equals each request alone."""
    from tensorfold.families.glm5_next.cuda.full_seg import FullSegVerify

    reqs = fm.two_requests()
    dec, host = multi_with(folder, 2, dspark=dspark)
    assert isinstance(dec.verify, FullSegVerify)
    seen = fm.watch(dec)
    streams = fm.drive(dec, host, reqs)
    assert max(n for n, _ in seen) == 2
    assert dec.verify.replays["eager"] == len(seen) and dec.verify.replays["graph"] == 0
    ref = fm.Serial(dec.w)
    fm.check(fm.replies(streams), [ref.reply(r) for r in reqs], reqs)


def test_oracle_rounds_on_fake_graphs_equal_serial(folder, fakes):
    """Three oracle-drafting streams (rounds of 12 to 16 rows, partial commits) through FullSegVerify with fake
    graphs of 8 .. 16 rows at the bucket the streams reach: the full rounds replay; every reply equals serial."""
    from tensorfold.families.glm5_next.cuda.full_seg import FullSegVerify

    reqs = [fm.request(fm.tokens(18, 21), 16, policy="fc7:0"),
            fm.request(fm.tokens(11, 22), 14, fm.sampled(4242), policy="fc7:0", stop_eos=False),
            fm.request(fm.tokens(40, 23), 12, fm.sampled(31, top_k=0, top_p=0.8), policy="fc7:0", at=1)]
    w = fm.load_w(folder)
    ref = fm.Serial(w)
    want = [ref.reply(r) for r in reqs]
    oracle = fm.Oracle(3, {tuple(r["prompt"]): t for r, t in zip(reqs, want)})

    def make(e):
        v = FullSegVerify(e, taps=(1, 3))
        assert v.graph_buckets(most=fm.EXT) == [0, 4096]
        v.capture(most=fm.EXT, pool="cpu", rows=range(8, 17), nts=[4096], capturer=fake_capture)
        return v

    dec, host = multi_with(folder, 3, w=w, drafts=oracle, make_verify=make, taps=(1, 3))
    oracle.dec = dec
    streams = fm.drive(dec, host, reqs)
    fm.check(fm.replies(streams), want, reqs)
    assert dec.verify.replays["graph"] >= 2, dec.verify.replays
    assert max(oracle.rows) <= 16


def test_dspark_contexts_on_fake_graphs(folder, dspark, fakes):
    """Each DSpark stream context's block pass and context updates captured in one shared pool (fake graphs: the
    replays' launches are the captures'): the contexts draft what the solo drafter drafts eagerly, independently."""
    from tensorfold.families.glm5_next.cuda.dflash2_multi import DraftRequest
    from tensorfold.families.glm5_next.cuda.dspark import Drafter
    from tensorfold.families.glm5_next.cuda.dspark_multi import DSparkMulti

    w = fm.load_w(folder)
    d = Drafter(dspark, w, capacity=4 * fm.EXT, tap_rows=16, ring=True)
    md = DSparkMulti(d, streams=2)
    n = md.capture(pool="cpu", capturer=fake_capture)
    a, b = md.contexts
    assert n == 2 * 17 and a.pool == "cpu" and b.block_graph is not None and len(a.tap_graphs) == 16
    assert a.context_end == 0 and int(a.pos_dev[0]) == 0 and d.block_graph is None
    g = torch.Generator().manual_seed(3)
    taps = [(torch.randn(n_, 2 * fm.D, generator=g) * 2).to(torch.bfloat16) for n_ in (30, 7)]
    before = FakeGraph.count
    md.commit([(a, taps[0]), (b, taps[1])])
    got = md.propose([DraftRequest(a, 17, 8, None), DraftRequest(b, 17, 8, None)])
    assert FakeGraph.count - before >= 3                       # b's 7-row update, both block passes
    d.add_taps(taps[0])
    assert got[0] == d.propose(17, 8, None) and len(got[0]) == 8
    other = Drafter(dspark, w, capacity=4 * fm.EXT, tap_rows=16, ring=True)
    other.add_taps(taps[1])
    assert got[1] == other.propose(17, 8, None)
    md.drop_graphs()
    assert a.block_graph is None and not b.tap_graphs


# -- memory -------------------------------------------------------------------------------------------------------------
def test_memory_terms(folder, fakes):
    """FullSegVerify's own buffers are ``seg_bytes``; the engine's --parallel term (``full_seg_multi_bytes``):
    those over the pool, plus (graphs) the verify graphs (count x per-graph executable), the DSpark contexts'
    graphs and the pool slack; printed for the real model's 78 layers."""
    from tensorfold.cuda.capacity import config
    from tensorfold.families.glm5_next.cuda import forward as F_
    from tensorfold.families.glm5_next.cuda import full_seg as S
    from tensorfold.families.glm5_next.cuda.engine import GRAPH_POOL_SLACK, full_seg_multi_bytes

    w = fm.load_w(folder)
    t = config(folder)
    n, slots = 3, 2560
    caches = F_.Caches(w, n * slots, streams=n)
    v = S.FullSegVerify(SimpleNamespace(w=w, caches=caches), taps=(1, 3))
    k = int(t["index_topk"])
    assert v.nbytes() == S.seg_bytes(16, n * slots, k, indexed=True)
    assert full_seg_multi_bytes(t, 1, slots, graphs=True) == 0
    eager = full_seg_multi_bytes(t, n, slots, graphs=False)
    assert eager == v.nbytes()
    assert full_seg_multi_bytes(t, n, slots, graphs=False, verify="segments") == 0
    layers = int(t["num_hidden_layers"])
    graphs = S.graph_count(16, n * slots, slots + 16, indexed=True)
    assert graphs == 16 * (1 + len(S.reach_buckets(n * slots, slots + 16)))
    full = full_seg_multi_bytes(t, n, slots, graphs=True, draft_layers=2, tap_rows=16)
    assert full == eager + S.verify_graph_bytes(layers, graphs) + S.draft_graph_bytes(2, n, 16) + GRAPH_POOL_SLACK
    top = full_seg_multi_bytes(t, n, slots, graphs=True, mode="top")
    assert top == eager + S.verify_graph_bytes(layers, 32) + GRAPH_POOL_SLACK
    # the real model: 78 layers, a stream of 200k tokens at --parallel 4 (every bucket 4,096 .. 262,144)
    real = 16 * (1 + len(S.reach_buckets(4 * 200_000, 200_016)))
    per = S.verify_graph_bytes(78, 1)
    print(f"\n[multi graphs] full GLM-5.3, --parallel 4 x 200k: {real} verify graphs x {per / 2 ** 20:.1f} MiB = "
          f"{real * per / 2 ** 30:.2f} GiB (top: 32 x = {32 * per / 2 ** 30:.2f} GiB); DSpark contexts' graphs "
          f"{S.draft_graph_bytes(3, 4, 16) / 2 ** 20:.1f} MiB; selection scores {S.seg_bytes(16, 800_000, 2048, indexed=True) / 2 ** 20:.1f} MiB")
    assert real == 128 and per == (78 * S.LAYER_LAUNCHES + S.HEAD_LAUNCHES) * S.GRAPH_NODE_BYTES
