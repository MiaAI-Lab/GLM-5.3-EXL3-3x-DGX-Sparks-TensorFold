"""mp_wire: rank results that cross a multiprocessing queue survive the rank process exiting before the parent reads.

The multi-rank tests spawn one process a rank; each puts its result on a queue and exits. A tensor on the queue is
shared through a file descriptor the sending process serves, so a parent that reads after the child has gone fails
with EOFError or FileNotFoundError (multiprocessing.resource_sharer). mp_wire.pack / unpack send plain bytes."""
import multiprocessing as mp

import pytest
import torch

from mp_wire import pack, unpack


def _sample():
    g = torch.Generator().manual_seed(7)
    return {"f32": torch.randn(3, 5, generator=g), "bf16": torch.randn(4, 6, generator=g).to(torch.bfloat16),
            "i64": torch.arange(12).reshape(3, 4), "rows": [torch.randn(2, generator=g), 7], "scalar": 128}


def _same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def test_pack_unpack_round_trip_is_bit_exact():
    want = _sample()
    assert _same(unpack(pack(want)), want)


def _child(q, packed):
    out = (1, _sample())
    q.put(pack(out) if packed else out)


def _send_then_exit(packed):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, packed))
    p.start()
    p.join(timeout=120)                       # the child has exited before the parent reads
    assert p.exitcode == 0
    return q.get(timeout=60)


def test_packed_results_survive_the_rank_exiting_first():
    rank, got = unpack(_send_then_exit(packed=True))
    assert rank == 1 and _same(got, _sample())


def test_raw_tensors_fail_when_the_rank_exits_first():
    """The failure the packing avoids (documents why; skipped where torch shares tensors another way)."""
    try:
        _send_then_exit(packed=False)
    except (EOFError, FileNotFoundError, ConnectionRefusedError) as e:
        assert e is not None
        return
    pytest.skip("this platform's tensor sharing survived the sender's exit; packing is still harmless")
