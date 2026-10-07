"""mp_wire: rank results that cross a multiprocessing queue survive the rank process exiting before the parent reads.

The multi-rank tests spawn one process a rank; each puts its result on a queue and exits. A tensor on the queue is
shared through a file descriptor the sending process serves, so a parent that reads after the child has gone fails
with EOFError or FileNotFoundError (multiprocessing.resource_sharer). mp_wire.pack / unpack send plain bytes."""
import multiprocessing as mp

import pytest
import torch

from mp_wire import pack, unpack


BF16_SPECIAL = (0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0xFFC3)   # +0, -0, +inf, -inf, two NaN payloads


def _sample():
    g = torch.Generator().manual_seed(7)
    special = [0.0, -0.0, float("inf"), float("-inf"), float("nan")]
    nan_bits = torch.tensor([0x7FC00001, -0x00400001], dtype=torch.int32).view(torch.float32)  # two NaN payloads
    return {"f32": torch.randn(3, 5, generator=g), "bf16": torch.randn(4, 6, generator=g).to(torch.bfloat16),
            "f32_special": torch.cat([torch.tensor(special), nan_bits]),
            "bf16_special": torch.tensor([b - (b >> 15 << 16) for b in BF16_SPECIAL], dtype=torch.int16).view(torch.bfloat16),
            "i64": torch.arange(12).reshape(3, 4), "rows": [torch.randn(2, generator=g), 7], "scalar": 128}


def _bits(t):
    return t.contiguous().view(-1).view(torch.uint8)


def _same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def test_pack_unpack_round_trip_is_bit_exact():
    want = _sample()
    assert _same(unpack(pack(want)), want)


def test_bit_compare_sees_signed_zero_and_nan_payloads():
    """_same compares bytes: -0.0 differs from 0.0 and two NaN payloads differ, though torch.equal says otherwise."""
    z = torch.tensor([0.0])
    assert torch.equal(z, -z) and not _same(z, -z)
    a, b = torch.tensor([0x7FC00001, 0x7FC00002], dtype=torch.int32).view(torch.float32).split(1)
    assert _same(a, a.clone()) and not _same(a, b)


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
