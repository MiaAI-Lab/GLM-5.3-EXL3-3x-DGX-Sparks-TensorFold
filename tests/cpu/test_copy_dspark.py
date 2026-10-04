"""Copy drafts vetted by DSpark (TF_GLM_COPY_HYBRID: copy_drafts.hybrid / walk / agreement, CopyDrafts.plan /
continuations / match_length / trusted continuations, dspark.Drafter.slot_scorer / propose_hybrid,
decode.hybrid_round inside dflash_decode, copy_sim's record replay): the rule on synthetic slots; the drafter's slot
scores equal Drafter.chain's arithmetic and a dump record's; drafted decoding with hybrid copies equals serial
decoding token for token (greedy and keyed sampling, a real tiny DSpark and a fake target); three ranks propose the
same drafts."""

import multiprocessing as mp
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_dspark import BLOCK, _drafter, _taps, folder  # noqa: F401  (the tiny DSpark checkpoint fixture)
from test_dspark_sim import FakeDrafter, make_record

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import copy_drafts as cd


# -- the rule on synthetic slots ------------------------------------------------------------------------------------
def synthetic(D=8, K=6, seed=0):
    """Candidates [D, K] (distinct ids) and a slot whose scores depend on the previous candidate (a Markov-like
    term), fixed confidences; the slot counts its calls."""
    rng = np.random.default_rng(seed)
    tokens = np.stack([rng.choice(1000, K, replace=False) for _ in range(D)])
    base = rng.normal(size=(D, K))
    extra = rng.normal(size=(D, K + 1, K)) * 0.5
    confs = np.linspace(0.95, 0.6, D)
    calls = []

    def slot(d, prev):
        calls.append((d, prev))
        return base[d] + extra[d, prev], float(confs[d])

    slot.calls = calls
    return tokens, slot, confs


def greedy_chain(tokens, slot, n):
    out, idx = [], []
    for d in range(n):
        pick, _ = slot(d, idx[-1] if d else -1)
        j = int(np.argmax(pick))
        idx.append(j)
        out.append(int(tokens[d, j]))
    return out, idx


def test_walk_is_the_chain_rule():
    tokens, slot, confs = synthetic()
    full, idx = greedy_chain(tokens, slot, 8)
    assert cd.walk(tokens, slot, 8) == (full, idx)
    assert cd.walk(tokens, slot, 3)[0] == full[:3]
    for thr in (0.9, 0.5, 0.2, 0.01):
        alive, n = 1.0, 0
        for d, c in enumerate(confs):
            alive *= c
            if d > 0 and alive < thr:
                break
            n += 1
        assert cd.walk(tokens, slot, 8, thr)[0] == full[:n]
    ms = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6)
    assert cd.walk(tokens, slot, 8, round_ms=ms)[0] == full[:cd.best_depth(list(confs), ms)]


def test_best_depth_equals_dflash2():
    from tensorfold.families.glm5_next.cuda.dflash2 import best_depth

    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.uniform(0.1, 1.0, size=8)
        ms = np.cumsum(rng.uniform(5, 15, size=9))
        assert cd.best_depth(q, ms) == best_depth(q, ms)


def test_hybrid_no_agreement_is_dspark():
    tokens, slot, _ = synthetic()
    full, idx = greedy_chain(tokens, slot, 8)
    p0, _ = slot(0, -1)
    second = int(tokens[0, np.argsort(-p0)[1]])
    absent = 123456                                         # not a candidate at all
    for conts in ([[second] + full[1:]], [[absent] + full[1:]], [[second], [absent, 1, 2]]):
        drafts, kind, k, a = cd.hybrid(tokens, slot, conts, chain_most=5, confidence=0.3)
        assert (kind, k, a) == ("d", -1, 0)
        assert drafts == cd.walk(tokens, slot, 5, 0.3)[0]


def test_hybrid_full_agreement_extends_past_the_block():
    tokens, slot, _ = synthetic()
    full, _ = greedy_chain(tokens, slot, 8)
    cont = full + [7, 8, 9, 10, 11, 12, 13]
    drafts, kind, k, a = cd.hybrid(tokens, slot, [cont], chain_most=5, confidence=0.3, most=15)
    assert (kind, k, a) == ("x", 0, 8) and drafts == cont
    assert cd.hybrid(tokens, slot, [cont], chain_most=5, confidence=0.3, most=10)[0] == cont[:10]


def test_hybrid_partial_agreement_then_dspark():
    tokens, slot, confs = synthetic()
    full, idx = greedy_chain(tokens, slot, 8)
    p3, _ = slot(3, idx[2])
    other = int(tokens[3, np.argsort(-p3)[2]])            # third best at slot 3: rank 1 and 2 reject it
    cont = full[:3] + [other] + full[4:] + [5, 6]
    drafts, kind, k, a = cd.hybrid(tokens, slot, [cont], chain_most=5, confidence=0.0)
    assert (kind, a) == ("h", 3)
    assert drafts == full[:5]                             # the agreed prefix is DSpark's chain: then its own picks
    # an agreed prefix longer than DSpark's confidence cut is kept whole (it is the copy's evidence)
    drafts, kind, _, a = cd.hybrid(tokens, slot, [cont], chain_most=5, confidence=0.99)
    assert kind == "h" and a == 3 and drafts == full[:3]
    # rank 3 accepts the third best: the copy then agrees on
    drafts, kind, _, a = cd.hybrid(tokens, slot, [cont], chain_most=5, confidence=0.0, rank=3)
    assert a >= 4 and drafts[:4] == cont[:4]


def test_agreement_rank_ties_and_filter():
    tokens = np.array([[10, 11, 12, 13]])
    pick = np.array([1.0, 2.0, 2.0, -np.inf])

    def slot(d, prev):
        return pick, 0.9

    assert cd.agreement(tokens, slot, [11], 1) == [1]      # argmax: the first of the tied best
    assert cd.agreement(tokens, slot, [12], 1) == []       # tied, but after 11 in argmax's order
    assert cd.agreement(tokens, slot, [12], 2) == [2]
    assert cd.agreement(tokens, slot, [10], 2) == [] and cd.agreement(tokens, slot, [10], 3) == [0]
    assert cd.agreement(tokens, slot, [13], 4) == []       # excluded by the target's rule: never
    assert cd.agreement(tokens, slot, [99], 4) == []       # not a candidate


def test_hybrid_picks_the_best_agreeing_continuation():
    tokens, slot, _ = synthetic()
    full, idx = greedy_chain(tokens, slot, 8)

    def off(d):                                           # a candidate DSpark ranks 4th at slot d after the chain
        return int(tokens[d, np.argsort(-slot(d, idx[d - 1] if d else -1)[0])[3]])

    short = full[:1] + [off(1)] + full[2:]                # agrees at 1 slot
    longer = full[:8] + [1, 2]                            # agrees at all
    drafts, kind, k, _ = cd.hybrid(tokens, slot, [short, longer], chain_most=5, most=15)
    assert (kind, k) == ("x", 1) and drafts == longer
    # equal agreement: the longer continuation, then the earlier one in the list (the latest occurrence)
    a, b = full[:2] + [off(2)], full[:2] + [off(2), 5, 6]
    assert cd.hybrid(tokens, slot, [a, b], chain_most=5)[1:3] == ("h", 1)
    assert cd.hybrid(tokens, slot, [b, a], chain_most=5)[1:3] == ("h", 0)
    assert cd.hybrid(tokens, slot, [full[:8] + [1], full[:8] + [2]], chain_most=5)[1:3] == ("x", 0)


def test_walk_prefix_counts_its_confidences():
    tokens, slot, confs = synthetic()
    full, idx = greedy_chain(tokens, slot, 8)
    # forced prefix of 2: the product after it is confs[0] * confs[1]; the chain stops where it falls below
    thr = float(confs[0] * confs[1] * confs[2]) + 1e-9
    drafts, got = cd.walk(tokens, slot, 5, thr, prefix=idx[:2])
    assert drafts == full[:2] and got == idx[:2]
    drafts, _ = cd.walk(tokens, slot, 5, 0.0, prefix=idx[:2])
    assert drafts == full[:5]
    drafts, _ = cd.walk(tokens, slot, 2, 0.0, prefix=idx[:4])     # a prefix longer than most is kept whole
    assert drafts == full[:4]


# -- the index: continuations, match lengths, the plan ---------------------------------------------------------------
def test_continuations_latest_first_distinct():
    ctx = [1, 2, 3, 9, 1, 2, 3, 8, 1, 2, 3, 9, 5, 1, 2, 3]
    c = cd.CopyDrafts(ctx, match=3, most=4)
    assert c.continuations() == [(8, [9, 5, 1, 2]), (4, [8, 1, 2, 3]), (0, [9, 1, 2, 3])]
    assert c.continuations(sources=2) == [(8, [9, 5, 1, 2]), (4, [8, 1, 2, 3])]
    d = cd.CopyDrafts([1, 2, 3, 9, 1, 2, 3, 9, 1, 2, 3], match=3, most=1)
    assert d.continuations() == [(4, [9])]                # the same continuation twice: once, the latest
    assert c.continuations(room=2, sources=1) == [(8, [9, 5])]
    assert c.continuations(room=0) == []
    assert cd.CopyDrafts([4, 5, 6, 7], match=3).continuations() == []


def test_match_length():
    ctx = [7, 1, 2, 3, 4, 5, 9, 0, 2, 3, 4, 5]
    c = cd.CopyDrafts(ctx, match=3, most=4)
    (start, cont), = c.continuations()
    assert cont == [9, 0, 2, 3] and c.match_length(start) == 4          # 2 3 4 5 (1 vs 0 before it)
    assert c.match_length(start, most=3) == 3


def plan_settings(**kw):
    kw.setdefault("match", 3)
    return cd.HybridSettings(**kw)


def test_plan_trust_by_source():
    prompt = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    h = plan_settings(trust_prompt=5, trust_reply=8)
    c = cd.CopyDrafts(prompt + [14], match=3, most=6, prompt=len(prompt), hybrid=h)
    c.extend([15, 16])                                     # suffix 14 15 16: a 3-token match in the prompt
    p = c.plan(h, least=2)
    assert p.kind == "v" and p.conts == [(3, [17, 18, 19, 20, 14, 15])]
    c.extend([17, 18])                                     # 14..18: 5 tokens, the prompt's trust
    p = c.plan(h)
    assert p.kind == "c" and p.drafts == [19, 20, 14, 15, 16, 17] and p.copied == 6 and p.start == 5
    # a reply occurrence needs trust_reply
    r = cd.CopyDrafts([1] + [30, 31, 32, 33, 34, 35, 36, 40, 30, 31, 32], match=3, most=4, prompt=1, hybrid=h)
    assert r.plan(h).kind == "v"
    r2 = cd.CopyDrafts([1] + list(range(30, 40)) + [41] + list(range(30, 38)), match=3, most=4, prompt=1, hybrid=h)
    p = r2.plan(h)
    assert p.kind == "c" and p.drafts == [38, 39, 41, 30]
    never = plan_settings(trust_prompt=0, trust_reply=0)
    assert r2.plan(never).kind == "v"


def test_plan_skips_the_block_after_a_whole_kept_trusted_copy():
    h = plan_settings(trust_prompt=0, trust_reply=0)
    text = list(range(100, 130))
    c = cd.CopyDrafts(text + text[:3], match=3, most=5, prompt=len(text), hybrid=h)
    p = c.plan(h, least=2)
    assert p.kind == "v" and p.conts[0] == (0, text[3:8])
    c.follow(0, 5)                                         # DSpark agreed at every slot: "x"
    c.extend(text[3:9])                                    # all 5 kept and the bonus token
    p = c.plan(h, least=2)
    assert p.kind == "s" and p.start == 6 and p.drafts == text[9:14]
    c.follow(p.start, p.copied)
    c.extend(text[9:11] + [5])                             # 2 of 5 kept: no longer trusted
    assert c.plan(h, least=2).kind == ""                   # ... and 5 never occurred
    c2 = cd.CopyDrafts(text + text[:3], match=3, most=5, prompt=len(text), hybrid=h)
    c2.plan(h)
    c2.extend(text[3:9])                                   # kept, but nothing followed it (not a trusted copy)
    assert c2.plan(h, least=2).kind == "v"
    c3 = cd.CopyDrafts(text + text[:3], match=3, most=5, prompt=len(text), hybrid=plan_settings(skip=False))
    c3.follow(0, 5)
    c3.extend(text[3:9])
    assert c3.plan(plan_settings(skip=False, trust_prompt=0, trust_reply=0), least=2).kind == "v"


def test_a_trusted_copy_among_others_is_vetted():
    """Two prompt occurrences, both trusted, continuing differently: the block vets both ("v" with the trusted one);
    hybrid_round copies the trusted one unless DSpark agreed with one at every slot ("x")."""
    from tensorfold.families.glm5_next.cuda.decode import DepthPolicy, hybrid_round

    A = list(range(50, 58))
    prompt = A + [58, 59, 1] + A + [60, 61, 62, 63]
    h = plan_settings(trust_prompt=8, trust_reply=0)

    def fresh():
        c = cd.CopyDrafts(prompt + A[:1], match=3, most=4, prompt=len(prompt), hybrid=h)
        c.extend(A[1:])
        return c

    p = fresh().plan(h)
    assert p.kind == "v" and p.conts == [(16, [60, 61, 62, 63]), (5, [58, 59, 1, 50])] and p.trusted == 0

    class Stub:
        block = 8

        def __init__(self, answer):
            self.answer = answer

        def propose(self, *a, **k):
            return [7]

        def propose_hybrid(self, pending, conts, depth, sampling, confidence, **kw):
            return self.answer

    rule = DepthPolicy(5, fixed=True, confidence=0.3)
    for answer, want in ((([7, 8], "d", -1), ([60, 61, 62, 63], "t")), (([58, 9], "h", 1), ([60, 61, 62, 63], "t")),
                         (([58, 59, 1, 50], "x", 1), ([58, 59, 1, 50], "x"))):
        c = fresh()
        assert hybrid_round(Stub(answer), c, h, A[-1], 5, 10, None, rule) == want
        assert c.trail == ((16 if want[1] == "t" else 5), 4)
    alone = cd.CopyDrafts(A + [58, 59] + A[:1], match=3, most=4, prompt=10, hybrid=h)
    alone.extend(A[1:])
    p = alone.plan(h)
    assert p.kind == "c" and p.drafts == [58, 59, 50, 51]
    assert hybrid_round(Stub(None), alone, h, A[-1], 5, 10, None, rule) == ([58, 59, 50, 51], "c")


def test_settings_from_env():
    assert cd.HybridSettings.from_env({}) is None
    h = cd.HybridSettings.from_env({"TF_GLM_COPY_HYBRID": "1"})
    assert h == cd.HybridSettings() and h.match == cd.SHORT and h.trust_prompt == cd.MATCH
    h = cd.HybridSettings.from_env({"TF_GLM_COPY_HYBRID": "1", "TF_GLM_COPY_SHORT": "3", "TF_GLM_COPY_RANK": "2",
                                    "TF_GLM_COPY_SOURCES": "1", "TF_GLM_COPY_SKIP": "0",
                                    "TF_GLM_COPY_TRUST_PROMPT": "0", "TF_GLM_COPY_TRUST_REPLY": "32"})
    assert h == cd.HybridSettings(3, 2, 1, False, 0, 32)
    for bad in ({"TF_GLM_COPY_HYBRID": "2"}, {"TF_GLM_COPY_HYBRID": "1", "TF_GLM_COPY_RANK": "0"},
                {"TF_GLM_COPY_HYBRID": "1", "TF_GLM_COPY_SHORT": "6", "TF_GLM_COPY_TRUST_REPLY": "5"}):
        with pytest.raises(ValueError):
            cd.HybridSettings.from_env(bad)
    off = cd.CopySettings.from_env(15, {"TF_GLM_COPY_DRAFTS": "1"})
    on = cd.CopySettings.from_env(15, {"TF_GLM_COPY_DRAFTS": "1", "TF_GLM_COPY_HYBRID": "1"})
    assert off.hybrid is None and on.hybrid is not None
    assert len(off.code()) == len(on.code()) == len(cd.CopySettings().code()) and off.code() != on.code()
    ctx = list(range(20))
    assert on.drafts(ctx).match == cd.SHORT and on.drafts(ctx).hybrid is on.hybrid
    assert on.drafts(ctx, hybrid=False).match == cd.MATCH and on.drafts(ctx, hybrid=False).hybrid is None


# -- the drafter's slot scores ---------------------------------------------------------------------------------------
class SlotDrafter(FakeDrafter):
    from tensorfold.families.glm5_next.cuda.dspark import Drafter as _D

    slot_scorer = _D.slot_scorer


@pytest.fixture
def noise07(monkeypatch):
    import tensorfold.families.glm5_next.cuda.dspark as ds

    monkeypatch.setattr(ds, "DSPARK_NOISE", 0.7)
    monkeypatch.setattr(ds, "DSPARK_FILTER", True)


@pytest.mark.parametrize("sampling", [None, Sampling(seed=99, temperature=1.0, top_k=0, top_p=0.95),
                                      Sampling(seed=5, temperature=0.7, top_k=4)])
def test_slot_scores_equal_chain_and_record(noise07, sampling):
    """Drafter.slot_scorer's walk is Drafter.chain (confidence and cost cuts); copy_sim.record_slot (a dump record)
    gives the same scores and confidences for every (slot, previous candidate)."""
    from tensorfold.families.glm5_next.cuda.copy_sim import record_slot

    rng = np.random.default_rng(11)
    dr = SlotDrafter(3)
    r = make_record(dr, rng, sampling)
    D, K = r.cand.shape[1:]
    costs = (60.0, 70.0, 80.0, 91.0, 103.0, 116.0, 130.0)
    for i in range(len(r.cand)):
        anchor, first = int(r.tokens[i]), 100 + i + 1
        slot = dr.slot_scorer(r.cand[i], r.vals[i], r.hconf[i], anchor, first, sampling)
        rec = record_slot(r, i)
        for d in range(D):
            for prev in ([-1] if d == 0 else range(K)):
                a, b = slot(d, prev), rec(d, prev)
                assert np.array_equal(a[0], b[0]) and a[1] == b[1]
        for most, conf, ms in ((5, 0.3, None), (D, 0.0, None), (D, 0.0, costs)):
            want = dr.chain(r.cand[i, :most], r.vals[i, :most], r.hconf[i, :most], anchor, first, sampling, conf,
                            round_ms=ms)
            assert cd.walk(r.cand[i], slot, most, conf, ms)[0] == want


def test_propose_hybrid_on_the_tiny_dspark(folder):
    d = _drafter(folder)
    d.add_taps(_taps(30, seed=9))
    s = Sampling(seed=1234, temperature=0.8)
    for sampling in (None, s):
        own = d.propose(42, BLOCK, sampling)                          # the full chain (no cut)
        prod = d.propose(42, 5, sampling, 0.3)
        tail = [1, 2, 3, 4, 5]
        drafts, kind, k = d.propose_hybrid(42, [[(own[0] + 1) % 512], own + tail], 5, sampling, 0.3, most=12)
        assert (kind, k) == ("x", 1) and drafts == (own + tail)[:12]
        p0 = [t for t in range(512) if t != own[0]][:1]
        drafts, kind, _ = d.propose_hybrid(42, [p0 + own[1:]], 5, sampling, 0.3)
        assert kind in ("d", "h") and (kind == "h" or drafts == prod)
        drafts, kind, _ = d.propose_hybrid(42, [own[:3] + [(own[3] + 1) % 512]], 5, sampling, 0.3)
        assert kind == "h" and drafts[:3] == own[:3]


# -- drafted == serial -----------------------------------------------------------------------------------------------
VOC = 512


class Target:
    """A deterministic target over VOC tokens: a row's logits depend on the context's last two tokens (keyed random
    rows, a peak on a favourite drawn from a small alphabet so replies repeat themselves)."""

    def __init__(self, seed=3, peak=8.0, alphabet=14):
        self.seed, self.peak = seed, peak
        self.alpha = np.random.default_rng(seed).choice(VOC, alphabet, replace=False)
        self.memo = {}

    def row(self, ctx):
        key = (ctx[-2] if len(ctx) > 1 else -1, ctx[-1])
        if key not in self.memo:
            g = np.random.default_rng(self.seed * 1_000_003 + (key[0] + 1) * 1009 + key[1])
            x = g.normal(size=VOC).astype(np.float32)
            x[self.alpha[g.integers(len(self.alpha))]] += self.peak
            self.memo[key] = x
        return self.memo[key]


def taps_for(pos, tok, width):
    g = torch.Generator().manual_seed(pos * 7919 + tok)
    return (torch.randn(width, generator=g) * 2.0).to(torch.bfloat16)


class FakeEngine:
    """decode.dflash_decode's engine on a ``Target``: windows' rows are the target's logits of their contexts, the
    real sampler (``sample_rows``, one rank), commit keeps rows; taps a function of (position, token)."""

    def __init__(self, target, prompt, tap_width):
        self.target, self.ctx, self.width = target, list(prompt), tap_width
        self.w = SimpleNamespace(cfg=SimpleNamespace(eos=[]), comm=None, vocab_offset=0, world=1, rank=0)
        self.st = SimpleNamespace(pos=len(prompt), engine=self)
        self.buf = None
        self.vote = self.constraint = None
        self.window = []

    def verify_window(self, tokens):
        return tokens

    def forward(self, tokens):
        self.window = list(tokens)
        return torch.from_numpy(np.stack([self.target.row(self.ctx + self.window[:r + 1])
                                          for r in range(len(tokens))]))

    def sample(self, logits, positions, sampling):
        from tensorfold.families.glm5_next.cuda.decode import sample_rows

        return sample_rows(self.w, logits, positions, sampling)

    def follow(self, tokens):
        pass

    def tap_rows(self, n):
        return torch.stack([taps_for(p, self.ctx[p], self.width) for p in range(len(self.ctx) - n, len(self.ctx))])


def fake_commit(w, st, b, R, keep):
    e = st.engine
    e.ctx.extend(e.window[:keep])
    st.pos += keep


def serial(target, prompt, pending, count, sampling):
    from tensorfold.families.glm5_next.cuda.decode import sample_rows

    w = SimpleNamespace(comm=None, vocab_offset=0, world=1, rank=0)
    out = [pending]
    while len(out) < count:
        row = torch.from_numpy(target.row(list(prompt) + out)[None])
        out += sample_rows(w, row, [len(prompt) + len(out)], sampling)
    return out


class OracleDrafter:
    """A fast stand-in for DSpark (the hybrid's interface): each slot's candidates are the target's top ones along a
    noisy guess of the reply, scores the target's logits plus keyed noise; ``propose_hybrid`` is copy_drafts.hybrid
    over them, so the rounds take every kind ("d", "h", "x", "c", "s")."""

    block = 8

    def __init__(self, engine, K=6, seed=0):
        self.e, self.K, self.rng = engine, K, np.random.default_rng(seed)
        self.taps = 0

    def add_taps(self, taps):
        self.taps += taps.shape[0]

    def _slots(self, pending):
        ctx = self.e.ctx + [pending]
        guess, tokens, rows = [], [], []
        for d in range(self.block):
            row = self.e.target.row(ctx + guess) + self.rng.normal(size=VOC).astype(np.float32) * 1.5
            top = np.argsort(-row, kind="stable")[:self.K]
            tokens.append(top)
            rows.append(row)
            guess.append(int(top[0]))
        tokens = np.stack(tokens)

        def slot(d, prev):
            return np.asarray(rows[d][tokens[d]], dtype=np.float64) + 0.01 * (prev + 1), 0.8

        return tokens, slot

    def propose(self, pending, depth, sampling, confidence=0.0, **_):
        tokens, slot = self._slots(pending)
        return cd.walk(tokens, slot, min(depth, self.block), confidence)[0]

    def propose_hybrid(self, pending, conts, depth, sampling, confidence=0.0, *, rank=1, most=15, **_):
        tokens, slot = self._slots(pending)
        drafts, kind, k, _ = cd.hybrid(tokens, slot, conts, chain_most=min(depth, self.block), confidence=confidence,
                                       rank=rank, most=most)
        return drafts, kind, k


def run_drafted(monkeypatch, drafter_for, target, prompt, pending, count, sampling, settings, tap_width):
    from tensorfold.families.glm5_next.cuda import decode

    monkeypatch.setattr(decode, "commit", fake_commit)
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    e = FakeEngine(target, prompt, tap_width)
    drafter = drafter_for(e)
    drafter.add_taps(torch.stack([taps_for(p, t, tap_width) for p, t in enumerate(prompt)]))
    copies = settings.drafts(list(prompt) + [pending]) if settings is not None else None
    res = decode.dflash_decode(e, drafter, pending, count, sampling,
                               policy=decode.DepthPolicy(5, fixed=True, confidence=0.3), copies=copies)
    return res


SAMPLINGS = [None, Sampling(seed=17, temperature=1.0, top_k=20), Sampling(seed=4, temperature=0.8, top_k=0,
                                                                            top_p=0.9)]


def copy_settings(**env):
    base = {"TF_GLM_COPY_DRAFTS": "1", "TF_GLM_COPY_MAX": "15", "TF_GLM_COPY_HYBRID": "1"}
    base.update(env)
    return cd.CopySettings.from_env(15, base)


@pytest.mark.parametrize("sampling", SAMPLINGS)
@pytest.mark.parametrize("env", [{}, {"TF_GLM_COPY_SHORT": "3", "TF_GLM_COPY_RANK": "2", "TF_GLM_COPY_SKIP": "0"},
                                 {"TF_GLM_COPY_TRUST_PROMPT": "0", "TF_GLM_COPY_TRUST_REPLY": "0"}])
def test_drafted_equals_serial_oracle(monkeypatch, sampling, env):
    target = Target()
    prompt = serial(target, [3, 5], 9, 60, None)[:-1]           # the target's own greedy text: the reply quotes it
    pending = 9
    count = 160
    want = serial(target, prompt, pending, count, sampling)
    kinds = ""
    for settings in (copy_settings(**env), None, cd.CopySettings.from_env(15, {"TF_GLM_COPY_DRAFTS": "1",
                                                                               "TF_GLM_COPY_MAX": "15"})):
        res = run_drafted(monkeypatch, lambda e: OracleDrafter(e), target, prompt, pending, count, sampling,
                          settings, 16)
        assert res.tokens == want
        assert sum(res.keeps) >= count - 1 and res.rounds == len(res.keeps)
        if settings is not None and settings.hybrid is not None:
            kinds = res.arms
            assert len(kinds) == res.rounds
    assert "d" in kinds and set(kinds) - {"d"}, kinds          # copies took part


@pytest.mark.parametrize("sampling", SAMPLINGS[:2])
def test_drafted_equals_serial_tiny_dspark(monkeypatch, folder, sampling):
    """The real (tiny) DSpark drafter: hybrid rounds through its block, slot scores and taps; the reply is serial's."""
    target = Target(seed=5)
    prompt = serial(target, [3, 5], 9, 40, None)[:-1]
    pending, count = 9, 30
    want = serial(target, prompt, pending, count, sampling)
    width = 5 * 256
    for settings in (copy_settings(), copy_settings(TF_GLM_COPY_SHORT="2", TF_GLM_COPY_TRUST_PROMPT="0")):
        res = run_drafted(monkeypatch, lambda e: _drafter(folder), target, prompt, pending, count, sampling,
                          settings, width)
        assert res.tokens == want, res.arms
        assert len(res.arms) == res.rounds


# -- three ranks -----------------------------------------------------------------------------------------------------
CASES = [(20, 3), (60, 31)]


def _hybrid_cases(drafter):
    out = []
    for sampling in (None, Sampling(seed=77, temperature=1.0, top_k=0, top_p=0.95)):
        for p, anchor in CASES:
            drafter.reset()
            drafter.add_taps(_taps(p, seed=p))
            own = drafter.propose(anchor, BLOCK, sampling)
            conts = [own + [5, 6, 7], own[:2] + [(own[2] + 1) % 512, 4], [(own[0] + 3) % 512, 1]]
            for c in (conts, conts[1:], conts[2:]):
                out.append(drafter.propose_hybrid(anchor, c, 5, sampling, 0.3, most=10))
    return out


def _rank(folder, rank, world, port, out_q):
    import torch.distributed as dist

    os.environ.setdefault("TRITON_INTERPRET", "1")
    torch.set_num_threads(1)
    import conftest  # noqa: F401  (the interpreter patches)

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)

    class Comm:
        world_size = world

        def all_gather(self, send, recv):
            dist.all_gather(list(recv.view(world, -1).unbind(0)), send.contiguous().view(-1))

    d = _drafter(folder, rank, world, Comm(), ring=True)
    out_q.put((rank, _hybrid_cases(d)))
    dist.destroy_process_group()


def test_three_ranks_propose_the_same(folder):
    """Every rank proposes the same hybrid drafts (the merged candidates and the host rule are the same on all), and
    deterministically (a second one-rank run equals the first)."""
    one = _hybrid_cases(_drafter(folder, ring=True))
    assert one == _hybrid_cases(_drafter(folder, ring=True))
    assert {k for _, k, _ in one} >= {"x", "h"}
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 33000 + os.getpid() % 2000
    procs = [ctx.Process(target=_rank, args=(folder, r, 3, port, q)) for r in range(3)]
    for p in procs:
        p.start()
    got = sorted((q.get(timeout=900) for _ in procs), key=lambda x: x[0])
    for p in procs:
        p.join(timeout=60)
    assert got[0][1] == got[1][1] == got[2][1]
    assert [k for _, k, _ in got[0][1]] == [k for _, k, _ in one]


# -- the replay ------------------------------------------------------------------------------------------------------
def test_copy_sim_replays_the_engine_round(noise07):
    """copy_sim.simulate's hybrid rounds are decode.hybrid_round's (RecordDrafter); with no copies they are the
    recorded production chain; on a reply that repeats itself the hybrid keeps more tokens a round."""
    from tensorfold.families.glm5_next.cuda import copy_sim

    rng = np.random.default_rng(3)
    dr = SlotDrafter(2)
    r = make_record(dr, rng, None, n=40)
    r.tokens[:] = np.tile(r.tokens[:10], 5)[:41]               # a reply that repeats a 10-token phrase
    r.meta.update(count=41, seed=0)
    # the recorded production chain: what Drafter.chain (fc5:0.3) proposes at each record
    r.prod = np.full((40, r.cand.shape[1]), -1)
    for i in range(40):
        ch = copy_sim.RecordDrafter(r, i).propose(int(r.tokens[i]), 5, None, 0.3)
        r.prod[i, :len(ch)] = ch
    for p in copy_sim.POLICIES:
        out = copy_sim.simulate(r, p, check=True)
        assert out["tokens"] == 40 and sum(v[2] for v in out["kinds"].values()) + out["rounds"] == 40
    base = copy_sim.simulate(r, copy_sim.POLICIES[0])
    hyb = copy_sim.simulate(r, copy_sim.Policy("h", copies="hybrid", hybrid=cd.HybridSettings(3, trust_reply=6)))
    assert hyb["rounds"] < base["rounds"] and "c" in hyb["kinds"]
