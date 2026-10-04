"""Full GLM-5.3 under --parallel on CPU (multi-stream milestone 2): the serving machinery (``multi.MultiDecoder``: its
admissions, prompt chunks, decode rounds through ``verify.FullBatchedVerify``, keyed sampling, commits, kept prompts;
``dspark_multi.DSparkMulti``: a DSpark context per stream; ``multi_prefill`` for full GLM) on the tiny checkpoint of
``full_fakes``, one rank and three (gloo, rank 0 deciding and the others applying its messages as rank 1 does).

Every reply (its token list) must equal the same request served alone: ``decode.prefill`` and ``decode.serial_decode``
on an engine of its own (--parallel 1, no drafts), whatever else was in the rounds: 2 and 3 concurrent requests with
different prompts (dense and sparse positions, one prompt over two chunks, one resuming from another's kept prompt),
lengths, max_tokens, seeds, greedy, top-k and nucleus sampling, stop at the end token or not, arriving at different
iterations, one finishing early; with no drafts, with a fake DSpark speculator (random weights: drafts rarely kept)
and with an oracle drafter (the reference reply, corrupted now and then: long kept windows, partial commits, every
round's rows at the 16-row cap). Plus the refusals, the memory accounting and DSpark's 8 drafts a stream."""

import json
import multiprocessing as mp
import os
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from full_fakes import TEXT, install_fake_experts, write_checkpoint

EXT = 2048                 # an extent (pool.ALIGN): the pool holds two more than its streams (admission's headroom)
STOP = -7                  # the tests' end of rank 0's messages (followers return)

# the fake DSpark speculator on the tiny model (D 256, vocabulary 256, 5 layers): taps after layers 1 and 3
D, VOCAB = TEXT["hidden_size"], TEXT["vocab_size"]
DS_HEADS, DS_HD, DS_INTER, DS_LAYERS, DS_BLOCK, DS_WINDOW, DS_RANK = 4, 64, 384, 2, 8, 16, 32
DS_AUX = [2, 4]
DS_MASK = 250


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_checkpoint(tmp_path_factory.mktemp("glm53multi2"))


def write_dspark(folder):
    from safetensors.torch import save_file

    cfg = {"architectures": ["DSparkDraftModel"], "aux_hidden_state_layer_ids": DS_AUX, "block_size": DS_BLOCK,
           "confidence_head_with_markov": True, "draft_vocab_size": VOCAB, "dtype": "bfloat16",
           "enable_confidence_head": True, "markov_head_type": "vanilla", "markov_rank": DS_RANK,
           "mask_token_id": DS_MASK, "sample_from_anchor": True, "sliding_window_non_causal": False,
           "speculators_model_type": "dspark", "tie_word_embeddings": False,
           "transformer_layer_config": {
               "attention_bias": False, "head_dim": DS_HD, "hidden_act": "silu", "hidden_size": D,
               "intermediate_size": DS_INTER, "layer_types": ["sliding_attention"] * DS_LAYERS, "model_type": "qwen3",
               "num_attention_heads": DS_HEADS, "num_hidden_layers": DS_LAYERS, "num_key_value_heads": DS_HEADS,
               "rms_norm_eps": 1e-5, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
               "sliding_window": DS_WINDOW, "use_sliding_window": True, "vocab_size": VOCAB}}
    g = torch.Generator().manual_seed(11)

    def r(*shape, std=1.0, mean=0.0):
        return (torch.randn(*shape, generator=g) * std + mean).to(torch.bfloat16)

    p = {"fc.weight": r(D, len(DS_AUX) * D, std=0.03), "hidden_norm.weight": r(D, std=0.1, mean=1.0),
         "norm.weight": r(D, std=0.1, mean=1.0),
         "markov_head.markov_w1.weight": r(VOCAB, DS_RANK, std=0.3),
         "markov_head.markov_w2.weight": r(VOCAB, DS_RANK, std=0.3),
         "confidence_head.proj.weight": r(1, D + DS_RANK, std=0.05), "confidence_head.proj.bias": r(1, std=0.5)}
    q = DS_HEADS * DS_HD
    for i in range(DS_LAYERS):
        a = f"layers.{i}."
        p.update({a + "input_layernorm.weight": r(D, std=0.1, mean=1.0),
                  a + "post_attention_layernorm.weight": r(D, std=0.1, mean=1.0),
                  a + "self_attn.q_proj.weight": r(q, D, std=0.08), a + "self_attn.k_proj.weight": r(q, D, std=0.08),
                  a + "self_attn.v_proj.weight": r(q, D, std=0.06), a + "self_attn.o_proj.weight": r(D, q, std=0.06),
                  a + "self_attn.q_norm.weight": r(DS_HD, std=0.1, mean=1.0),
                  a + "self_attn.k_norm.weight": r(DS_HD, std=0.1, mean=1.0),
                  a + "mlp.gate_proj.weight": r(DS_INTER, D, std=0.06),
                  a + "mlp.up_proj.weight": r(DS_INTER, D, std=0.06),
                  a + "mlp.down_proj.weight": r(D, DS_INTER, std=0.06)})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(cfg))
    save_file({k: v.contiguous() for k, v in p.items()}, str(folder / "model.safetensors"))
    return folder


@pytest.fixture(scope="module")
def dspark(tmp_path_factory):
    return write_dspark(tmp_path_factory.mktemp("dspark") / "d")


def _env(env) -> None:
    """The settings every rank runs with (``env``: ``os.environ`` or a MonkeyPatch)."""
    put = env.setenv if hasattr(env, "setenv") else env.__setitem__
    put("TF_GLM_DENSE", "bf16")
    put("TF_GLM_DSPARK_QUANT", "bf16")       # the 4-bit draft matmuls are CUDA kernels
    put("TF_GLM_MULTI_WATCHDOG_S", "0")


@pytest.fixture
def fakes(monkeypatch):
    install_fake_experts(monkeypatch)
    _env(monkeypatch)
    for k in ("TF_GLM_MULTI_PREFILL", "TF_GLM_MULTI_VERIFY", "TF_GLM_MULTI_DEPTH", "TF_GLM_FILL_BUDGET_MS",
              "TF_GLM_DSPARK_MULTI_POLICY", "TF_GLM_FILL_ROWS"):
        monkeypatch.delenv(k, raising=False)


def tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, VOCAB, (n,), generator=g).tolist()


# -- the CPU engine and what MultiDecoder reads of GlmEngine ----------------------------------------------------------
class Host:
    """``engine.GlmEngine``'s part ``multi.MultiDecoder`` uses, on CPU: the decode engine, the drafter, the settings
    and rank 0's message channel (``_share``: a length, then the values, through the communicator)."""

    def __init__(self, w, e, drafter, rank, world, comm, limit):
        self.w, self.e, self.drafter = w, e, drafter
        self.rank, self.world, self.comm = rank, world, comm
        self.grid, self.limit, self.serial_only, self.shared = 0, limit, False, 0
        self.copy = self.costs = self.opener = self.vision = self.model_dir = None
        self.cache_entries = 8

    def _ring(self):
        pass

    def _await_bell(self):
        pass

    def _share(self, values):
        if self.comm is None:
            return list(values)
        n = torch.tensor([len(values) if self.rank == 0 else 0], dtype=torch.int32)
        got = torch.zeros((self.world,), dtype=torch.int32)
        self.comm.all_gather(n, got)
        count = int(got[0])
        if count == 0:
            return []
        buf = (torch.tensor(values, dtype=torch.int32) if self.rank == 0 else torch.zeros((count,), dtype=torch.int32))
        allv = torch.zeros((self.world * count,), dtype=torch.int32)
        self.comm.all_gather(buf, allv)
        return [int(v) for v in allv[:count].tolist()]


def load_w(folder, rank=0, world=1, comm=None):
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=rank, world=world, device="cpu", mtp=False)      # --parallel: no MTP head
    w.comm = comm
    w.meta["long_context"] = True
    return w


def multi(folder, streams, *, w=None, rank=0, world=1, comm=None, dspark=None, drafts=None, prefill_rows=64,
          verify=None, taps=()):
    """A ``MultiDecoder`` over ``streams`` slots and a pool of streams + 2 extents, as the engine builds it."""
    from tensorfold.families.glm5_next.cuda import decode
    from tensorfold.families.glm5_next.cuda.multi import MultiDecoder

    w = w if w is not None else load_w(folder, rank, world, comm)
    pool = (streams + 2) * EXT
    drafter = None
    if dspark is not None:
        from tensorfold.families.glm5_next.cuda.dspark import Drafter

        drafter = Drafter(dspark, w, capacity=pool, tap_rows=16, ring=True)
    e = decode.Engine(w, capacity=pool, max_rows=16, prefill_rows=prefill_rows, graphs=False, long_context=True,
                      taps=drafter.tap_layers if drafter is not None else tuple(taps), streams=streams,
                      pool_rows=pool, kv="bf16")
    host = Host(w, e, drafter, rank, world, comm, limit=EXT - 16)
    if taps and verify is None:            # an injected drafter's taps (the engine's drafter gives its own)
        from tensorfold.families.glm5_next.cuda.verify import FullBatchedVerify

        verify = FullBatchedVerify(e, taps=tuple(taps))
    return MultiDecoder(host, streams, drafts=drafts, verify=verify), host


class Serial:
    """The reference: a request alone on an engine of its own (--parallel 1), no drafts."""

    def __init__(self, w, prefill_rows=64):
        from tensorfold.families.glm5_next.cuda import decode

        self.e = decode.Engine(w, capacity=EXT, max_rows=16, prefill_rows=prefill_rows, graphs=False,
                               long_context=True, kv="bf16")

    def reply(self, r):
        from tensorfold.families.glm5_next.cuda import decode

        from tensorfold.families.glm5_next.cuda.forward import commit

        e = self.e
        out = [decode.prefill(e, r["prompt"], r["sampling"], mtp=False)]
        # ``decode.serial_decode``'s loop (its CUDA clock aside): a token a step, sampled at its absolute position
        while len(out) < r["count"] and not (r["stop_eos"] and out[-1] in e.w.cfg.eos):
            logits = e.forward([out[-1]])
            out.append(e.sample(logits[:1], [e.st.pos + 1], r["sampling"])[0])
            commit(e.w, e.st, e.buf, 1, 1)
        return out


def request(prompt, count, sampling=None, *, at=0, stop_eos=True, policy="auto", draft=True):
    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    return dict(prompt=list(prompt), count=count, sampling=sampling, at=at, stop_eos=stop_eos, draft=draft,
                code=encode_policy(policy))


def drive(dec, host, reqs, *, most=2000):
    """Rank 0: requests admitted at their iterations (when a slot and room are free, else later, in order), one
    ``round`` an iteration until every reply is done; each request's stream. Followers are told to stop."""
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.glm5_next.cuda.multi import NoRoom

    streams = [None] * len(reqs)
    waiting = list(range(len(reqs)))
    for it in range(most):
        for i in list(waiting):
            r = reqs[i]
            if r["at"] > it or dec.live() >= dec.count:
                break
            s = Stream(list(r["prompt"]), r["count"], r["sampling"], draft=r["draft"], stop_eos=r["stop_eos"])
            s.glm = {"code": r["code"]}
            if not dec.fits(s):
                break
            try:
                dec.admit(s)
            except NoRoom:
                break
            streams[i] = s
            waiting.remove(i)
        done = dec.round()
        dec.finish(done)
        if not waiting and not dec.lanes:
            break
    else:
        raise AssertionError("the requests did not finish")
    if host.world > 1:
        host._share([STOP])
    return streams


def follow(dec, host):
    """A follower rank: rank 0's messages applied until its STOP; every stream it held, by sid."""
    from tensorfold.families.glm5_next.cuda.multi import unseal

    seen = {}
    while True:
        msg = host._share(None)
        if msg == [STOP]:
            return seen
        dec.apply(unseal(msg, dec.received, dec.rank))
        dec.received += 1
        for sid, lane in dec.lanes.items():
            seen[sid] = lane.s


def watch(dec):
    """Record every batched verify window's segments (streams) and rows."""
    seen = []
    forward = dec.verify.forward

    def wrapped(segments):
        seen.append((len(segments), sum(len(s.tokens) for s in segments)))
        return forward(segments)

    dec.verify.forward = wrapped
    return seen


def replies(streams):
    return [list(s.out[:s.count]) for s in streams]


def check(got, want, reqs):
    bad = [(i, g, w_) for i, (g, w_) in enumerate(zip(got, want)) if g != w_]
    assert not bad, "replies differ from serial: " + "; ".join(
        f"request {i} ({len(reqs[i]['prompt'])}-token prompt) got {g} want {w_}" for i, g, w_ in bad[:4])


def sampled(seed, top_k=20, top_p=0.95, temperature=0.8):
    from tensorfold.engine.exact_sampling import Sampling

    return Sampling(seed=seed, temperature=temperature, top_k=top_k, top_p=top_p)


# -- the scenarios ---------------------------------------------------------------------------------------------------
def two_requests():
    """A dense greedy request and a sparse sampled one arriving two iterations later."""
    return [request(tokens(9, 1), 10), request(tokens(30, 2), 8, sampled(1234), at=2, stop_eos=False)]


def three_plus_one():
    """Three slots, four requests: greedy sparse; a two-chunk nucleus-sampled prompt; a short top-k one finishing
    early; then (arriving later, waiting for a slot) the first prompt extended, which resumes from its kept prompt."""
    a = tokens(20, 3)
    return [request(a, 12),
            request(tokens(70, 4), 9, sampled(77, top_k=0, top_p=0.9), stop_eos=False),
            request(tokens(13, 5), 3, sampled(5, top_k=8), at=1),
            request(a + tokens(9, 6), 6, sampled(99), at=3, policy="fc3:0")]


def test_two_streams_no_drafts_equal_serial(folder, fakes):
    """Serial multi-stream decoding (no drafter: one row a stream a round) of two requests equals each alone."""
    reqs = two_requests()
    dec, host = multi(folder, 2)
    seen = watch(dec)
    got = replies(drive(dec, host, reqs))
    assert max(n for n, _ in seen) == 2, "the two streams shared rounds"
    ref = Serial(dec.w)
    check(got, [ref.reply(r) for r in reqs], reqs)


def test_three_streams_dspark_equal_serial(folder, dspark, fakes):
    """Three DSpark streams (a context each, one weight set) and a fourth request reusing a freed slot and resuming
    from a kept prompt: every reply equals serial; and equals the same requests without drafts."""
    from tensorfold.families.glm5_next.cuda.dspark_multi import DSparkMulti

    reqs = three_plus_one()
    dec, host = multi(folder, 3, dspark=dspark)
    assert isinstance(dec.drafts, DSparkMulti) and type(dec.verify).__name__ == "FullSegVerify"   # the default (milestone 4)
    seen = watch(dec)
    streams = drive(dec, host, reqs)
    assert max(n for n, _ in seen) == 3 and max(r for _, r in seen) <= 16, seen
    got = replies(streams)
    assert streams[3].cached == 20, "the extended prompt resumes from the first one's kept state"
    assert sum(s.drafted for s in streams) > 0, "the DSpark streams drafted"
    ref = Serial(dec.w)
    check(got, [ref.reply(r) for r in reqs], reqs)
    plain, host2 = multi(folder, 3, w=dec.w)
    assert replies(drive(plain, host2, reqs)) == got


class Oracle:
    """A drafter that proposes each stream's reference continuation (``refs`` by prompt), one draft in three
    corrupted: long kept windows, partial commits, the round's rows at the cap. Its contexts stand in for DSpark's."""

    def __init__(self, streams, refs):
        self.refs, self.dec, self.block = refs, None, 8
        self.contexts = [SimpleNamespace(slot=i, context_end=0, pos_dev=torch.zeros((1,), dtype=torch.int64),
                                         kc=[], vc=[], ring=0, window=0, capacity=1 << 30) for i in range(streams)]
        for c in self.contexts:
            c.reset = (lambda c=c: setattr(c, "context_end", 0))
            c.add_taps = (lambda taps, c=c: setattr(c, "context_end", c.context_end + taps.shape[0]))
        self.rows = []

    def propose(self, reqs):
        out = []
        for r in reqs:
            lane = next(l for l in self.dec.lanes.values() if l.slot == r.ctx.slot)
            ref = self.refs[tuple(lane.s.prompt)]
            j = len(lane.s.out) - 1
            assert ref[j] == r.pending
            d = list(ref[j + 1:j + 1 + r.depth])
            if d and (j + r.ctx.slot) % 3 == 0:
                k = len(d) // 2
                d[k] = (d[k] + 1) % VOCAB or 2
            out.append(d)
        self.rows.append(sum(len(d) + 1 for d in out))
        return out

    def commit(self, items):
        for c, taps in items:
            c.add_taps(taps)


def test_oracle_drafts_equal_serial(folder, fakes):
    """Three streams whose drafts are mostly right (an oracle): windows of up to 8 rows a stream, each stream capped
    to its share of the 16 rows, partial commits: every reply equals serial."""
    reqs = [request(tokens(18, 21), 16, policy="fc7:0"),
            request(tokens(11, 22), 14, sampled(4242), policy="fc7:0", stop_eos=False),
            request(tokens(40, 23), 12, sampled(31, top_k=0, top_p=0.8), policy="fc7:0", at=1)]
    w = load_w(folder)
    ref = Serial(w)
    want = [ref.reply(r) for r in reqs]
    oracle = Oracle(3, {tuple(r["prompt"]): t for r, t in zip(reqs, want)})
    dec, host = multi(folder, 3, w=w, drafts=oracle, taps=(1, 3))
    oracle.dec = dec
    streams = drive(dec, host, reqs)
    check(replies(streams), want, reqs)
    assert sum(s.accepted for s in streams) >= 10, [s.accepted for s in streams]
    assert max(oracle.rows) <= 16 and max(oracle.rows) >= 12, oracle.rows


def test_grouped_prompts_equal_serial(folder, dspark, fakes, monkeypatch):
    """TF_GLM_MULTI_PREFILL=1: three prompts arriving together share prompt chunks (each piece its own stream's
    caches, selection and DSpark context): every reply equals serial."""
    monkeypatch.setenv("TF_GLM_MULTI_PREFILL", "1")
    reqs = [request(tokens(25, 31), 6), request(tokens(50, 32), 5, sampled(8)), request(tokens(7, 33), 5)]
    dec, host = multi(folder, 3, dspark=dspark, prefill_rows=128)
    streams = drive(dec, host, reqs)
    assert dec.grouped["chunks"] >= 1 and dec.grouped["pieces"] >= 2, dec.grouped
    ref = Serial(dec.w)
    check(replies(streams), [ref.reply(r) for r in reqs], reqs)


def test_scheduler_threads_equal_serial(folder, fakes):
    """The serving path: ``GlmScheduler``'s worker thread, three clients submitting at different times into two
    slots (the third waits for one): every reply equals serial."""
    from tensorfold.families.glm5_next.cuda.engine import encode_policy
    from tensorfold.families.glm5_next.cuda.multi import GlmScheduler

    reqs = [request(tokens(12, 41), 7), request(tokens(26, 42), 6, sampled(5150)),
            request(tokens(5, 43), 5, sampled(6, top_k=0))]
    dec, host = multi(folder, 2)
    sched = GlmScheduler(dec, max_streams=2)
    got: list = [None] * len(reqs)

    def client(i):
        time.sleep(0.5 * i)
        out = []
        r = reqs[i]
        sched.submit(r["prompt"], r["count"], r["sampling"], True, lambda new: out.extend(new), stop_eos=r["stop_eos"],
                     glm={"code": encode_policy("auto")})
        got[i] = out

    threads = [threading.Thread(target=client, args=(i,)) for i in range(len(reqs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2400)
    assert all(g is not None for g in got)
    ref = Serial(dec.w)
    check(got, [ref.reply(r) for r in reqs], reqs)


# -- three ranks ---------------------------------------------------------------------------------------------------
def _rank(folder, dspark_dir, rank, world, port, out_q):
    import torch.distributed as dist

    os.environ.setdefault("TRITON_INTERPRET", "1")
    torch.set_num_threads(1)
    import conftest  # noqa: F401  (the interpreter patches)
    import pytest as _pt

    mp_ = _pt.MonkeyPatch()
    install_fake_experts(mp_)
    _env(os.environ)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)

    ranks = world

    class Comm:
        world_size = ranks
        world = ranks                        # (the nucleus rule's gathers read it)

        def all_gather(self, send, recv):
            dist.all_gather(list(recv.view(ranks, -1).unbind(0)), send.contiguous().view(-1))

    try:
        reqs = [request(tokens(20, 51), 7), request(tokens(9, 52), 5, sampled(321), at=1),
                request(tokens(66, 53), 4, sampled(9, top_k=0, top_p=0.9), at=2, stop_eos=False)]
        dec, host = multi(folder, 3, rank=rank, world=world, comm=Comm(), dspark=dspark_dir)
        if rank == 0:
            streams = drive(dec, host, reqs)
            got = {i: list(s.out[:s.count]) for i, s in enumerate(streams)}
            sids = {i: s.sid for i, s in enumerate(streams)}
        else:
            seen = follow(dec, host)
            got, sids = {sid: list(s.out[:s.count]) for sid, s in seen.items()}, None
        ref = Serial(dec.w)
        want = [ref.reply(r) for r in reqs]
        out_q.put((rank, got, sids, want, None))
    except Exception as exc:                     # noqa: BLE001
        import traceback

        out_q.put((rank, None, None, None, traceback.format_exc()))
        raise
    finally:
        dist.destroy_process_group()


def test_three_ranks_equal_serial(folder, dspark, fakes):
    """Three ranks (heads 2/2/1, vocabulary 128/64/64; DSpark heads 2/1/1): rank 0 decides, ranks 1 and 2 apply its
    messages; every rank's replies equal the requests served alone on the three ranks."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 37000 + os.getpid() % 2000
    procs = [ctx.Process(target=_rank, args=(folder, dspark, r, 3, port, q)) for r in range(3)]
    for p in procs:
        p.start()
    got = []
    try:
        while len(got) < len(procs):
            got.append(q.get(timeout=2900))
            if got[-1][4] is not None:                 # a failed rank: the others would wait in a collective
                break
    finally:
        for p in procs:
            p.join(timeout=60 if len(got) == len(procs) and got[-1][4] is None else 1)
            if p.is_alive():
                p.terminate()
    got.sort(key=lambda x: x[0])
    for rank, _, _, _, err in got:
        assert err is None, f"rank {rank}: {err}"
    assert len(got) == len(procs)
    _, zero, sids, want, _ = got[0]
    for i, tokens_ in zero.items():
        assert tokens_ == want[i], f"rank 0 request {i}: {tokens_} != serial {want[i]}"
    for rank, mine, _, w_, _ in got[1:]:
        assert w_ == want, f"rank {rank}'s serial replies differ from rank 0's"
        for i, sid in sids.items():
            assert mine[sid] == want[i], f"rank {rank} request {i}: {mine[sid]} != {want[i]}"


# -- pieces ----------------------------------------------------------------------------------------------------------
def test_dspark_contexts_draft_as_the_solo_drafter(folder, dspark, fakes):
    """A stream context shares the drafter's weights and drafts what the solo drafter drafts on the same taps, all
    8 of a pass (DFlash2's multi-stream drafter caps at block - 1; DSpark's anchor predicts the first draft); the
    contexts are independent; DFlash2's MultiDrafter refuses a DSpark drafter."""
    from tensorfold.families.glm5_next.cuda.dflash2_multi import DraftRequest, MultiDrafter
    from tensorfold.families.glm5_next.cuda.dspark import Drafter
    from tensorfold.families.glm5_next.cuda.dspark_multi import DSparkMulti

    w = load_w(folder)
    d = Drafter(dspark, w, capacity=4 * EXT, tap_rows=16, ring=True)
    md = DSparkMulti(d, streams=2)
    a, b = md.contexts
    assert a.layers is d.layers and a.kc[0].data_ptr() != d.kc[0].data_ptr() != b.kc[0].data_ptr()
    g = torch.Generator().manual_seed(3)
    taps = [(torch.randn(n, 2 * D, generator=g) * 2).to(torch.bfloat16) for n in (30, 7)]
    d.add_taps(taps[0])
    solo = d.propose(17, 8, None)
    md.commit([(a, taps[0]), (b, taps[1])])
    got = md.propose([DraftRequest(a, 17, 8, None), DraftRequest(b, 17, 8, None)])
    assert len(got[0]) == 8 and got[0] == solo
    other = Drafter(dspark, w, capacity=4 * EXT, tap_rows=16, ring=True)
    other.add_taps(taps[1])
    assert got[1] == other.propose(17, 8, None)
    with pytest.raises(ValueError):
        MultiDrafter(d, streams=2)


def test_refusals():
    """--parallel's refusals: context parallelism (not built yet), full GLM without a DSpark speculator (unless
    --no-drafts); full GLM with DSpark or no drafts is served."""
    from tensorfold.families.glm5_next.cuda.engine import parallel_refusal

    assert parallel_refusal(1, full=True, drafter=False, dspark=False, serial_only=False, cp=3) is None
    why = parallel_refusal(2, full=True, drafter=True, dspark=True, serial_only=False, cp=3)
    assert why and "TF_GLM_CP=1" in why and "--parallel 1" in why
    assert "DSpark" in parallel_refusal(3, full=True, drafter=False, dspark=False, serial_only=False)
    assert "DFlash2" in parallel_refusal(3, full=False, drafter=False, dspark=False, serial_only=False)
    assert parallel_refusal(4, full=True, drafter=True, dspark=True, serial_only=False) is None
    assert parallel_refusal(2, full=True, drafter=False, dspark=False, serial_only=True) is None


def test_multi_refusals(folder, fakes, monkeypatch):
    """The decoder refuses a context-parallel engine and TF_GLM_MULTI_DEPTH=joint on full GLM."""
    w = load_w(folder)
    monkeypatch.setenv("TF_GLM_MULTI_DEPTH", "joint")
    with pytest.raises(ValueError, match="joint"):
        multi(folder, 2, w=w)
    monkeypatch.delenv("TF_GLM_MULTI_DEPTH")
    w.meta["cp"] = 3
    try:
        with pytest.raises(ValueError):
            multi(folder, 2, w=w, verify=object())
    finally:
        w.meta.pop("cp")


def test_memory_accounting(folder, dspark, fakes):
    """The startup estimate's --parallel terms against what is allocated: the other streams' caches and the batched
    verify buffers (``engine.full_multi_bytes``), the DSpark contexts (``engine.dspark_multi_bytes``), the pool
    split (``engine.full_pool_rows``)."""
    from tensorfold.cuda.capacity import config
    from tensorfold.cuda.geometry import full_token_bytes
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda.dspark import Drafter
    from tensorfold.families.glm5_next.cuda.dspark_multi import DSparkMulti
    from tensorfold.families.glm5_next.cuda.engine import dspark_multi_bytes, full_multi_bytes, full_pool_rows
    from tensorfold.families.glm5_next.cuda.verify import FullBatchedVerify

    from test_full_memory import tensor_bytes

    t = config(folder)
    w = load_w(folder)
    share = lambda fam, total: w.tp.size(fam) if fam in w.tp.totals else int(total)   # noqa: E731
    n, slots = 3, 2560
    pool = F.Caches(w, n * slots)
    one = F.Caches(w, slots)
    caches = sum(p.tensor.numel() * p.tensor.element_size() for p in pool.arena.planes)
    alone = sum(p.tensor.numel() * p.tensor.element_size() for p in one.arena.planes)
    assert caches - alone == (n - 1) * slots * full_token_bytes(t, "bf16", mtp=False)
    v = FullBatchedVerify(SimpleNamespace(w=w, caches=pool), taps=(1, 3))
    est = full_multi_bytes(t, n, slots, world=1, share=share, kv="bf16", mtp=False, prompt_split_k=True)
    got = caches - alone + tensor_bytes(v.b)
    assert got <= est <= got * 1.05 + (1 << 20), (got, est)
    d = Drafter(dspark, w, capacity=n * slots, tap_rows=16, ring=True)
    held = DSparkMulti(d, streams=n).nbytes()
    est = dspark_multi_bytes(dspark, n, 0, 1, capacity=n * slots, ring=True)
    assert held <= est <= held * 2 + (1 << 20), (held, est)
    tb = full_token_bytes(t, "bf16", mtp=False)
    pool_rows, each = full_pool_rows(65536 + 16, 2, 10 * 2048 * tb, tb, 2048)
    assert pool_rows == 2 * 65536 + 10 * 2048 and each == 65536 + 5 * 2048
    pool_rows, each = full_pool_rows(2560, 4, 0, tb, 2048)
    assert (pool_rows, each) == (10240, 2048)
