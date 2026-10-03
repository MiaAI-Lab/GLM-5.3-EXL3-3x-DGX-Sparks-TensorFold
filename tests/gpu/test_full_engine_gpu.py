"""The tiny full GLM-5.3 checkpoint through the engine on one GPU with every real kernel (the universal EXL3 expert
kernels on its mixed-width random trellises): against the reference forward; drafted windows equal serial steps bit
for bit; CUDA graphs (dense and sparse buckets, MTP) replay the eager bits."""

import os

import pytest
import torch

from full_fakes import TEXT, Reference, write_checkpoint
from test_full_forward import agree, tokens

CAP = 4096 + 512       # past dsa_full.MIN_BUCKET: a sparse graph bucket of 4096 and the capacity's


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    os.environ.setdefault("TF_GLM_DENSE", "bf16")
    return write_checkpoint(tmp_path_factory.mktemp("glm53full"))


def engine(folder, graphs=False):
    from tensorfold.families.glm5_next.cuda.decode import Engine
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=0, world=1, device="cuda", mtp=True)
    return Engine(w, capacity=CAP, max_rows=8, prefill_rows=64, graphs=graphs, graph_rows=(1, 2, 3, 4),
                  long_context=True, mtp_rows=4)


def run(e, prompt, steps):
    """Prefill (no sample), then one forward a step: logits rows (this rank's vocabulary, fp32 on the host)."""
    from tensorfold.families.glm5_next.cuda.decode import prefill
    from tensorfold.families.glm5_next.cuda.forward import commit

    prefill(e, prompt, None, mtp=False, sample=False)
    rows = []
    for t in steps:
        rows.append(e.forward([t]).float().cpu().clone())
        commit(e.w, e.st, e.buf, 1, 1)
    return torch.cat(rows)


def test_matches_the_reference(folder):
    e = engine(folder)
    ref = Reference(folder)
    cache = ref.new_cache()
    prompt, steps = tokens(30, 1), tokens(8, 2)
    got = run(e, prompt, steps)
    ref.forward(prompt, cache, 0)
    want = torch.cat([ref.forward([t], cache, 30 + i)[0] for i, t in enumerate(steps)])
    ok, why = agree(got, want)
    assert ok, why


@pytest.mark.parametrize("start", [9, 60])
def test_drafted_window_equals_serial_steps(folder, start):
    from tensorfold.families.glm5_next.cuda.decode import prefill
    from tensorfold.families.glm5_next.cuda.forward import commit

    e = engine(folder)
    prompt, draft = tokens(start, 3), tokens(6, 4)
    prefill(e, prompt, None, mtp=False, sample=False)
    window = e.forward(draft).float().cpu().clone()
    e.reset()
    prefill(e, prompt, None, mtp=False, sample=False)
    serial = []
    for t in draft:
        serial.append(e.forward([t]).float().cpu().clone())
        commit(e.w, e.st, e.buf, 1, 1)
    assert torch.equal(window, torch.cat(serial))


def test_graphs_replay_the_eager_bits(folder):
    """Dense steps (main graphs), sparse steps (the 4096 bucket's graphs) and eager steps give the same rows."""
    eager = engine(folder, graphs=False)
    graphed = engine(folder, graphs=True)
    prompt = tokens(4100, 5)                          # past 4096: the sparse graphs' first bucket
    steps = tokens(6, 6)
    a = run(eager, tokens(6, 7), tokens(4, 8))          # positions 6 .. 9: within the tiny model's top-k (16)
    b = run(graphed, tokens(6, 7), tokens(4, 8))
    assert torch.equal(a, b) and graphed.replays["main"] >= 4
    eager.reset()
    graphed.reset()
    a = run(eager, prompt, steps)
    b = run(graphed, prompt, steps)
    assert torch.equal(a, b) and graphed.replays["sparse"] >= 6


def test_mtp_drafts_run(folder):
    """The MTP head drafts through its graphs and the 4-bit draft head: a chained draft of 3 tokens."""
    from tensorfold.families.glm5_next.cuda.decode import draft, prefill

    e = engine(folder, graphs=True)
    prompt = tokens(40, 9)
    first = prefill(e, prompt, None, mtp=True)
    d = draft(e, e.last_hidden, [first], len(prompt) + 1, 3, None)
    assert len(d) == 3 and all(0 <= t < TEXT["vocab_size"] for t in d)
