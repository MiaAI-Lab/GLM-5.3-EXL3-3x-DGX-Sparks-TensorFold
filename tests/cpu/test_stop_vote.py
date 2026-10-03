"""0122 (the GLM-5.3-Flash recipe's 0070, issue #38): a serial request rank 0 stops (client gone, stop string) ends on
every rank after the same round. Three ranks' StopVotes ride on one sample all-gather (threads and a barrier stand in
for the ranks); the gathered words and the sampled tokens are the same with or without the vote."""
import threading
from types import SimpleNamespace

import torch

from tensorfold.families.glm5_next.cuda.decode import StopVote, sample_rows, stopped

WORLD = 3


class Gather:
    """all_gather across WORLD threads: every rank's send, stacked in rank order, to every rank."""

    def __init__(self) -> None:
        self.slots = [None] * WORLD
        self.bar = threading.Barrier(WORLD)

    def comm(self, rank: int):
        outer = self

        class Comm:
            world_size = WORLD
            world = WORLD

            def all_gather(self, send, recv):
                outer.slots[rank] = send.clone()
                outer.bar.wait()
                recv.copy_(torch.cat([s.reshape(-1) for s in outer.slots]))
                outer.bar.wait()

        return Comm()


def run_ranks(fn):
    out, errs = [None] * WORLD, []

    def go(r):
        try:
            out[r] = fn(r)
        except Exception as e:      # noqa: BLE001  (reported below)
            errs.append(e)

    ts = [threading.Thread(target=go, args=(r,)) for r in range(WORLD)]
    [t.start() for t in ts]
    [t.join(60) for t in ts]
    assert not errs, errs
    return out


def logits_for(rank: int, step: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 * step + rank)
    return torch.randn((3, 16), generator=g)


def test_stop_reaches_every_rank_after_the_same_round():
    g = Gather()

    def rank_run(r):
        w = SimpleNamespace(comm=g.comm(r), world=WORLD, vocab_offset=16 * r)
        wish = {"at": 2}
        vote = StopVote((lambda toks: len(toks) and wish["at"] == 0) if r == 0 else None)
        e = SimpleNamespace(vote=vote)
        rounds, tokens = 0, []
        for step in range(6):
            if stopped(e):
                break
            tokens.append(sample_rows(w, logits_for(r, step), [10, 11, 12], None, vote=vote))
            rounds += 1
            wish["at"] -= 1
            vote(tokens[-1])                    # rank 0's on_tokens wishes once wish["at"] reaches 0
        return rounds, tokens

    got = run_ranks(rank_run)
    assert len({r for r, _ in got}) == 1, got           # every rank ended after the same round
    assert got[0][0] < 6                                # and early
    assert all(t == got[0][1] for _, t in got)          # with the same tokens

    def plain(r):                                       # no vote: the same tokens for those rounds
        w = SimpleNamespace(comm=Gather_plain.comm(r), world=WORLD, vocab_offset=16 * r)
        return [sample_rows(w, logits_for(r, s), [10, 11, 12], None) for s in range(got[0][0])]

    Gather_plain = Gather()
    ref = run_ranks(plain)
    assert ref[0] == got[0][1]


def test_no_wish_never_stops():
    g = Gather()

    def rank_run(r):
        w = SimpleNamespace(comm=g.comm(r), world=WORLD, vocab_offset=16 * r)
        vote = StopVote(lambda toks: False)
        e = SimpleNamespace(vote=vote)
        n = 0
        for step in range(4):
            assert not stopped(e)
            vote(sample_rows(w, logits_for(r, step), [0, 1, 2], None, vote=vote))
            n += 1
        return n

    assert run_ranks(rank_run) == [4] * WORLD
