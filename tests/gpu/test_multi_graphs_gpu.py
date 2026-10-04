"""Multi-stream milestone 4 on the GPU: ``full_seg.FullSegVerify``'s CUDA graphs (a graph a window's rows R and token
bucket NT, in one shared pool) replayed against the same verifier's eager forward, bit for bit (logits, taps, every
cache plane), for R = 1 .. 16 and buckets 0 / 4,096 / 8,192 / 16,384 over windows of 1 to 4 streams at other
positions and extents than the capture's, bf16 and TF_GLM_KV=fp4x caches; the DSpark stream contexts' graphs against
the solo drafter's eager drafts; concurrent requests served by ``multi.MultiDecoder`` with every graph on (the
engine's, the contexts', the batched verify's) against each request served alone; and the graphs' memory and kernel
counts against ``full_seg``'s estimates (printed). The tiny checkpoint with every real kernel (EXL3 experts too)."""

import os
from types import SimpleNamespace

import pytest
import torch

DEV = "cuda"
EXTENT = 16384
STREAMS = 4
TAPS = (1, 3)
BUCKETS = (0, 4096, 8192, 16384)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    from full_fakes import write_checkpoint

    os.environ.setdefault("TF_GLM_DENSE", "bf16")
    return write_checkpoint(tmp_path_factory.mktemp("glm53graphs"))


def load_w(folder, mtp=False):
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=0, world=1, device=DEV, mtp=mtp)
    w.comm = None
    w.meta["long_context"] = True
    return w


def fill(w, caches, seed=0):
    """Every DSA layer's latent, rotary and index planes written through the writers (any format) from random rows."""
    from tensorfold.families.glm5_next.cuda import dsa_full, latent

    g = torch.Generator(device=DEV).manual_seed(seed)
    c = w.cfg
    P = caches.rows
    ps = caches.arena.planes
    pos0 = torch.zeros((1,), dtype=torch.int32, device=DEV)
    freq = w.meta["rope_freq"]
    for di in range(len(caches.kc)):
        lc = ps[caches.kc[di]].tensor
        latent.latent_write(torch.randn(P, c.kv_lora, generator=g, device=DEV).to(torch.bfloat16), lc, pos0)
        dsa_full.k_rope_write(torch.randn(P, c.rope, generator=g, device=DEV).to(torch.bfloat16),
                              ps[caches.kr[di]].tensor, pos0, freq, latent=lc)
    for _, _, i in caches.index or ():
        D = c.index_dim
        dsa_full.index_write(torch.randn(P, D, generator=g, device=DEV).to(torch.bfloat16),
                             torch.ones(D, dtype=torch.bfloat16, device=DEV),
                             torch.zeros(D, dtype=torch.bfloat16, device=DEV), ps[i].tensor, pos0, freq)
    torch.cuda.synchronize()


class Rig:
    """STREAMS streams of EXTENT tokens on one pool (extents permuted), random caches, two verifiers on it: one
    eager, one with graphs of every (R, NT) in BUCKETS."""

    def __init__(self, folder, kv="bf16", rows=range(1, 17)):
        from tensorfold.families.glm5_next.cuda import forward as F_
        from tensorfold.families.glm5_next.cuda.full_seg import FullSegVerify
        from tensorfold.families.glm5_next.cuda.verify import FullBatchedVerify

        self.w = w = load_w(folder)
        self.caches = F_.Caches(w, STREAMS * EXTENT, streams=STREAMS, kv=kv)
        slots = F_.Slots(w, STREAMS, 16)
        place = [(k + 2) % STREAMS for k in range(STREAMS)]
        self.st = [F_.State(w, EXTENT, 16, caches=self.caches, base=place[k] * EXTENT, slots=slots, slot=k)
                   for k in range(STREAMS)]
        e = SimpleNamespace(w=w, caches=self.caches)
        self.graph = FullSegVerify(e, taps=TAPS)
        free0 = torch.cuda.mem_get_info()[0]
        reserved0 = torch.cuda.memory_reserved()
        self.count = self.graph.capture(most=EXTENT, rows=rows)
        torch.cuda.synchronize()
        self.free_delta = free0 - torch.cuda.mem_get_info()[0]
        self.reserved_delta = torch.cuda.memory_reserved() - reserved0
        assert sorted({nt for _, nt in self.graph.graphs}) == list(BUCKETS)
        self.eager = FullSegVerify(e, taps=TAPS)
        self.loop = FullBatchedVerify(e, taps=TAPS)
        fill(w, self.caches)

    def window(self, spec, seed):
        """Segments [(stream, position, rows)] with random tokens."""
        from tensorfold.families.glm5_next.cuda.verify import Segment

        g = torch.Generator().manual_seed(seed)
        segs = []
        for k, pos, n in spec:
            self.st[k].set_pos(pos)
            segs.append(Segment(self.st[k], torch.randint(2, self.w.cfg.vocab, (n,), generator=g).tolist()))
        return segs

    def compare(self, spec, seed, NT):
        """The window eager (FullSegVerify, then the per-segment loop) and replayed: logits, taps, planes equal."""
        segs = self.window(spec, seed)
        out = []
        for v in (self.eager, self.loop, self.graph):
            before = dict(getattr(v, "replays", {}))
            got = v.forward(segs)
            torch.cuda.synchronize()
            out.append(([x.clone() for x in got.logits], [x.clone() for x in got.taps],
                        [p.tensor.clone() for p in self.caches.arena.planes]))
            if v is self.graph:
                assert v.replays["graph"] == before["graph"] + 1, (spec, "not replayed")
                assert v.NT == NT, (spec, v.NT, NT)
            v.segments = []
        (le, te, pe), (ll, tl, pl), (lg, tg, pg) = out
        for i in range(len(segs)):
            assert torch.equal(lg[i], le[i]) and torch.equal(tg[i], te[i]), (spec, i, "graph vs eager")
            assert torch.equal(ll[i], le[i]) and torch.equal(tl[i], te[i]), (spec, i, "seg vs loop")
        assert all(torch.equal(a, b) for a, b in zip(pg, pe)), (spec, "planes")


def specs(R, NT):
    """Windows of R rows over 1 .. 4 streams whose bucket is NT (0: every row dense, below the dense limit 16)."""
    parts = [R] if R < 3 else [R // 3, R // 3, R - 2 * (R // 3)] if R < 8 else [R // 4] * 3 + [R - 3 * (R // 4)]
    parts = [n for n in parts if n]
    lead = {0: 0, 4096: 3000, 8192: 6000, 16384: EXTENT - 16}[NT]
    out = []
    for j, n in enumerate(parts):
        if NT == 0:
            pos = max(0, 16 - n - j)
        else:
            pos = lead if j == 0 else (j * 977) % max(1, lead - n)        # others anywhere below the lead
        out.append((j, pos, n))
    return [out, [(k, p, n) for (k, p, n) in reversed(out)]]


@pytest.mark.parametrize("kv", ["bf16", "fp4x"])
def test_graph_replay_equals_eager(folder, kv):
    """Every R = 1 .. 16 at every bucket (bf16; fp4x: R 1, 5, 11, 16), two windows each (streams in either order):
    the replay's logits / taps / planes equal the eager forward's and the per-segment loop's."""
    rows = range(1, 17) if kv == "bf16" else (1, 5, 11, 16)
    rig = Rig(folder, kv, rows=rows)
    seed = 0
    for R in rows:
        for NT in BUCKETS:
            for spec in specs(R, NT):
                seed += 1
                rig.compare(spec, seed, NT)
    print(f"\n[multi graphs] {kv}: {rig.count} graphs, free memory -{rig.free_delta / 2 ** 20:.1f} MiB, "
          f"allocator reserved +{rig.reserved_delta / 2 ** 20:.1f} MiB")


def test_unmatched_shape_runs_eagerly(folder):
    """A window whose (R, NT) has no graph runs eagerly (the same bits as the eager verifier)."""
    rig = Rig(folder, rows=(4,))
    segs = rig.window([(0, 3000, 2), (1, 100, 3)], 7)
    before = dict(rig.graph.replays)
    got = rig.graph.forward(segs)
    rig.graph.segments = []
    want = rig.eager.forward(segs)
    rig.eager.segments = []
    assert rig.graph.replays["eager"] == before["eager"] + 1
    assert all(torch.equal(a, b) for a, b in zip(got.logits, want.logits))


def test_graph_memory_and_launches(folder):
    """The graphs' memory (free memory and the allocator's reserve around the captures) and an eager window's kernel
    count against full_seg's estimates; printed for calibration."""
    from tensorfold.families.glm5_next.cuda import full_seg as S
    from tensorfold.families.glm5_next.cuda.engine import GRAPH_POOL_SLACK

    rig = Rig(folder)
    layers = len(rig.w.layers)
    exec_bytes = max(0, rig.free_delta - rig.reserved_delta)
    per = exec_bytes / rig.count
    est = S.verify_graph_bytes(layers, 1)
    segs = rig.window([(0, 3000, 5), (1, 700, 6), (2, 20, 5)], 3)
    rig.eager.forward(segs)
    rig.eager.segments = []
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        rig.eager.forward(segs)
        torch.cuda.synchronize()
    rig.eager.segments = []
    kernels = sum(1 for ev in prof.events() if str(ev.device_type).endswith("CUDA"))
    print(f"\n[multi graphs] {rig.count} verify graphs ({layers} layers): executables ~{per / 1024:.0f} KiB a graph "
          f"(estimate {est / 1024:.0f} KiB), pool and allocator +{rig.reserved_delta / 2 ** 20:.1f} MiB (slack "
          f"{GRAPH_POOL_SLACK >> 20} MiB); a 16-row window launches {kernels} kernels, "
          f"{(kernels - S.HEAD_LAUNCHES) / layers:.1f} a layer (estimate {S.LAYER_LAUNCHES})")
    assert per <= est, (per, est)
    assert rig.reserved_delta <= GRAPH_POOL_SLACK
    assert kernels <= layers * S.LAYER_LAUNCHES + S.HEAD_LAUNCHES


def test_dspark_context_graphs(folder, tmp_path, monkeypatch):
    """The DSpark stream contexts' graphs (block pass, updates of 1 .. 16 rows; one pool) draft what the solo
    drafter drafts eagerly on the same taps, each context independently."""
    import test_full_multi_engine as fm

    from tensorfold.families.glm5_next.cuda.dflash2_multi import DraftRequest
    from tensorfold.families.glm5_next.cuda.dspark import Drafter
    from tensorfold.families.glm5_next.cuda.dspark_multi import DSparkMulti

    monkeypatch.setenv("TF_GLM_DSPARK_QUANT", "bf16")
    d_dir = fm.write_dspark(tmp_path / "d")
    w = load_w(folder)
    d = Drafter(d_dir, w, capacity=4 * fm.EXT, tap_rows=16, ring=True)
    md = DSparkMulti(d, streams=3)
    assert md.capture() == 3 * 17
    a, b, c = md.contexts
    g = torch.Generator().manual_seed(3)
    taps = [(torch.randn(n, 2 * fm.D, generator=g) * 2).to(torch.bfloat16).to(DEV) for n in (30, 7, 16)]
    md.commit([(a, taps[0]), (b, taps[1]), (c, taps[2])])
    got = md.propose([DraftRequest(x, 17, 8, None) for x in (a, b, c)])
    for t, drafts in zip(taps, got):
        solo = Drafter(d_dir, w, capacity=4 * fm.EXT, tap_rows=16, ring=True)
        solo.add_taps(t)
        assert drafts == solo.propose(17, 8, None) and len(drafts) == 8


def test_served_concurrent_replies_equal_serial(folder, tmp_path, monkeypatch):
    """Three DSpark streams and a fourth request served by MultiDecoder with every graph on (the engine's one-stream
    graphs, the contexts', the batched verify's in the engine's pool): every reply equals the request alone."""
    import test_full_multi_engine as fm

    from tensorfold.families.glm5_next.cuda import decode
    from tensorfold.families.glm5_next.cuda.dspark import Drafter
    from tensorfold.families.glm5_next.cuda.full_seg import FullSegVerify
    from tensorfold.families.glm5_next.cuda.multi import MultiDecoder

    fm._env(monkeypatch)
    for k in ("TF_GLM_MULTI_VERIFY", "TF_GLM_MULTI_GRAPHS", "TF_GLM_MULTI_DRAFT_GRAPHS", "TF_GLM_MULTI_PREFILL"):
        monkeypatch.delenv(k, raising=False)
    d_dir = fm.write_dspark(tmp_path / "d")
    w = load_w(folder)
    streams = 3
    pool = (streams + 2) * fm.EXT
    drafter = Drafter(d_dir, w, capacity=pool, tap_rows=16, ring=True)
    e = decode.Engine(w, capacity=pool, max_rows=16, prefill_rows=64, graphs=True, graph_rows=(1, 2, 3, 4),
                      long_context=True, taps=drafter.tap_layers, streams=streams, pool_rows=pool, kv="bf16")
    drafter.capture()
    host = fm.Host(w, e, drafter, 0, 1, None, limit=fm.EXT - 16)
    dec = MultiDecoder(host, streams)
    assert isinstance(dec.verify, FullSegVerify) and dec.graph_counts["verify"] == 16 * 2
    assert dec.graph_counts["drafts"] == streams * 17
    reqs = fm.three_plus_one()
    streams_ = fm.drive(dec, host, reqs)
    ref = fm.Serial(w)
    fm.check(fm.replies(streams_), [ref.reply(r) for r in reqs], reqs)
    assert dec.verify.replays["graph"] > 0, dec.verify.replays
    print(f"\n[multi graphs] served: verify windows {dec.verify.replays}")
