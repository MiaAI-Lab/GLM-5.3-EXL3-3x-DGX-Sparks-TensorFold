"""Plain-bytes transport for results a spawned rank process sends to the test process.

A torch tensor put on a multiprocessing queue is shared through a file descriptor that the sending process serves;
when the rank process exits before the parent reads the queue, the parent fails with EOFError or FileNotFoundError
in multiprocessing.resource_sharer. ``pack`` turns tensors (inside tuples, lists and dicts) into (dtype, shape,
bytes) records the queue pickles by value; ``unpack`` rebuilds them bit for bit, any dtype (bf16 included)."""
import torch

_TAG = "__mp_wire_tensor__"


def pack(obj):
    if torch.is_tensor(obj):
        t = obj.detach().cpu().contiguous()
        return (_TAG, str(t.dtype).replace("torch.", ""), tuple(t.shape), t.view(-1).view(torch.uint8).numpy().tobytes())
    if isinstance(obj, tuple):
        return tuple(pack(o) for o in obj)
    if isinstance(obj, list):
        return [pack(o) for o in obj]
    if isinstance(obj, dict):
        return {k: pack(v) for k, v in obj.items()}
    return obj


def unpack(obj):
    if isinstance(obj, tuple) and len(obj) == 4 and obj[0] == _TAG:
        _, dtype, shape, data = obj
        flat = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        return flat.view(getattr(torch, dtype)).reshape(shape).clone()
    if isinstance(obj, tuple):
        return tuple(unpack(o) for o in obj)
    if isinstance(obj, list):
        return [unpack(o) for o in obj]
    if isinstance(obj, dict):
        return {k: unpack(v) for k, v in obj.items()}
    return obj
