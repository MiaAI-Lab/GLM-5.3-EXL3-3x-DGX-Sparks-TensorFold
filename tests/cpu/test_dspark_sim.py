"""The offline DSpark simulator (dspark_sim) rebuilds Drafter.chain's drafts bit for bit from draft_dump's recorded
terms (dspark_terms), greedy and sampled, for the production confidence cut and a cost-table cut; walk() follows a
reply's decode round by round."""
import numpy as np
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import dspark_sim as sim
from tensorfold.families.glm5_next.cuda.draft_dump import dspark_terms
from tensorfold.families.glm5_next.cuda.dspark import Drafter

V, R, D, K = 300, 16, 6, 8


class FakeDrafter:
    """Drafter's sequential stage (Markov head, confidence head) on random weights; chain is Drafter's own."""

    chain = Drafter.chain
    markov_embed = Drafter.markov_embed
    markov_bias = Drafter.markov_bias

    def __init__(self, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.w1 = (torch.randn((V, R), generator=g) * 0.3).to(torch.bfloat16)
        self.w2 = (torch.randn((V, R), generator=g) * 0.3).to(torch.bfloat16)
        self.conf_h = True
        self.conf_m = (torch.randn((R,), generator=g) * 0.2).double().numpy()
        self.conf_b = 0.4


def make_record(dr, rng, sampling, n=12):
    cand = np.stack([np.stack([rng.choice(V, K, replace=False) for _ in range(D)]) for _ in range(n)])
    vals = rng.normal(size=(n, D, K)) * 2.0
    hconf = rng.normal(size=(n, D))
    tokens = rng.choice(V, n + 1)
    terms = [dspark_terms(dr, cand[i], int(tokens[i])) for i in range(n)]
    noise = None
    if sampling is not None:
        from tensorfold.engine.exact_sampling import uniform_rows
        noise = np.stack([-np.log(-np.log(uniform_rows(sampling.seed, 100 + i + 1 + np.arange(D), cand[i])))
                          for i in range(n)])
    meta = dict(drafter="dspark", dspark_noise=0.7, dspark_filter=True, conf_b=dr.conf_b, learned_conf=True,
                prod_most=5, prod_confidence=0.3)
    r = sim.Record(meta, tokens, cand, vals, hconf, np.stack([t[0] for t in terms]), np.array([t[1] for t in terms]),
                   np.stack([t[2] for t in terms]), np.stack([t[3] for t in terms]), noise,
                   np.full((n, D), -1), sampling)
    return r


def drafter_chain(dr, r, i, most, confidence=0.0, round_ms=None):
    import tensorfold.families.glm5_next.cuda.dspark as ds
    old = ds.DSPARK_NOISE
    ds.DSPARK_NOISE = 0.7
    try:
        return dr.chain(r.cand[i, :most], r.vals[i, :most], r.hconf[i, :most], int(r.tokens[i]), 100 + i + 1,
                        r.sampling, confidence, round_ms=round_ms)
    finally:
        ds.DSPARK_NOISE = old


def test_replay_equals_drafter_chain_greedy_and_sampled():
    rng = np.random.default_rng(5)
    costs = (60.0, 70.0, 80.0, 91.0, 103.0, 116.0, 130.0, 145.0, 160.0)
    for sampling in (None, Sampling(seed=99, temperature=1.0, top_k=0, top_p=0.95)):
        dr = FakeDrafter(3)
        r = make_record(dr, rng, sampling)
        for i in range(len(r.cand)):
            assert sim.chain(r, i, sim.Policy("p", most=5, confidence=0.3)) == drafter_chain(dr, r, i, 5, 0.3)
            assert sim.chain(r, i, sim.Policy("c", most=D, costs=costs)) == drafter_chain(dr, r, i, D,
                                                                                        round_ms=costs)


def test_walk_counts_rounds():
    rng = np.random.default_rng(7)
    dr = FakeDrafter(4)
    r = make_record(dr, rng, None, n=10)
    # make the reply exactly the greedy chain from position 0 for 3 tokens: the first round keeps them
    first = sim.chain(r, 0, sim.Policy("p", most=3))
    r.tokens[1:1 + len(first)] = first
    out = sim.walk(r, sim.Policy("p", most=3))
    assert out["tokens"] == 10 and out["rounds"] <= 10 - len(first) + 1
