"""3,072-row prompt chunks (TF_GLM_PREFILL_ROWS=3072) for full GLM-5.3, on CPU: the prompt buffers' rows (no split
pad at three ranks: 3,072 = 3 x 1,024; 2,048 keeps its one pad row), one routed-expert call a whole chunk
(forward.EXL3_BLOCK_ROWS), the grouping kernels' host references (a 3,072-row chunk grouped whole is its two former
blocks grouped apart, the same members in the same order; MPE's route keeps per-expert counts only), the startup
estimate against what the buffers allocate at 3,072 rows (the fake model and the real dimensions, meta tensors),
and a prompt's state not depending on the chunk size: chunks of 3 : 2 the rows (192 vs 128, the same pad pattern as
3,072 vs 2,048 at three ranks) leave the same caches, head row and next-step logits bit for bit, at one rank and at
three ranks with the row split on. The full-size run (3,072 vs 2,048 on a 3,100-token prompt, one rank) is opt-in:
it runs when ``-k full_size`` selects it (an hour of Triton interpreter)."""

import io
import json
import multiprocessing as mp
import os

import pytest
import torch

from full_fakes import TEXT, install_fake_experts, write_checkpoint

E_REAL, SLOTS_REAL = 256, 9


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_checkpoint(tmp_path_factory.mktemp("glm53chunk3k"))


@pytest.fixture
def fakes(monkeypatch):
    install_fake_experts(monkeypatch)
    monkeypatch.setenv("TF_GLM_DENSE", "bf16")
    from tensorfold.families.glm5_next.cuda import decode as D

    monkeypatch.setattr(D, "_sync", lambda w: None)


def tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, TEXT["vocab_size"], (n,), generator=g).tolist()


def same(a, b):
    """Bit for bit (any dtype, NaNs included)."""
    if a is None or b is None:
        return a is None and b is None
    a, b = a.contiguous(), b.contiguous()
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.reshape(-1).view(torch.uint8),
                                                                     b.reshape(-1).view(torch.uint8))


# -- rows -------------------------------------------------------------------------------------------------------------
def test_prompt_buffer_rows():
    """3,072-row chunks need no split pad at three ranks; 2,048 keeps its one pad row; one rank and two never pad."""
    from tensorfold.families.glm5_next.cuda.hcsplit import SplitSettings, buffer_rows, pad_rows

    on = SplitSettings(split=True)
    assert pad_rows(3072, 3) == 0 and buffer_rows(3072, 3, on) == 3072
    assert pad_rows(2048, 3) == 1 and buffer_rows(2048, 3, on) == 2049
    assert buffer_rows(3072, 1, on) == buffer_rows(3072, 3, None) == 3072
    assert buffer_rows(3072, 2, on) == 3072


def test_a_whole_chunk_is_one_routed_call():
    """forward.EXL3_BLOCK_ROWS holds every chunk the engine accepts (up to 16,384 rows) with its split pad: the
    routed experts of a 3,072-row chunk (2,049 at 2,048) run as one call, the MoE overlap path included."""
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda.hcsplit import pad_rows

    for rows in (2048, 3072, 4096, 16384):
        alloc = rows + pad_rows(rows, 3)
        assert alloc <= F.EXL3_BLOCK_ROWS, rows
        assert len(range(0, alloc, F.EXL3_BLOCK_ROWS)) == 1


def test_moe_block_calls_the_experts_once_for_3072_rows(folder, fakes, monkeypatch):
    """A 3,072-row prompt chunk through an MoE layer of the fake model: the routed experts take all 3,072 rows in
    one call (before: 2,176 + 896), and each row's output is the row's own (rows 0 and 3,071 alone: the same bits)."""
    from tensorfold.cuda.exl3 import experts as generic
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=0, world=1, device="cpu", mtp=False)
    w.meta["long_context"] = True
    b = F.Buffers(w, 3072, 128, prefill=True)
    assert b.alloc_rows == 3072 and b.gx.y.shape[0] == 3072 * b.pick.shape[1]
    layer = next(l for l in w.layers if l.moe is not None)
    g = torch.Generator().manual_seed(7)
    b.normed.copy_((torch.randn(b.normed.shape, generator=g) * 0.5).to(torch.bfloat16))
    calls = []
    real = generic.routed

    def counted(x, pick, wts, ex, s, out, R, *a, **k):
        calls.append(R)
        return real(x, pick, wts, ex, s, out, R, *a, **k)

    monkeypatch.setattr(generic, "routed", counted)
    F.moe_full_block(layer, w, b, 3072)                     # this rank's fp32 partial rows: b.part
    whole = b.part[:3072].clone()
    assert calls == [3072]
    for r in (0, 3071):
        one = F.Buffers(w, 8, 128, prefill=True)
        one.normed[:1].copy_(b.normed[r:r + 1])
        F.moe_full_block(layer, w, one, 1)
        assert same(one.part[0], whole[r]), r


# -- the grouping kernels' host references -----------------------------------------------------------------------------
def _picks(R, seed, E=E_REAL, slots=SLOTS_REAL):
    """Router picks as the engine's: top-k distinct experts a row, the last slot the shared expert's marker (E)."""
    g = torch.Generator().manual_seed(seed)
    pick = torch.stack([torch.randperm(E, generator=g)[:slots] for _ in range(R)]).to(torch.int32)
    pick[:, -1] = E
    return pick


def test_grouping_of_3072_rows_whole_equals_the_two_blocks():
    """The universal grouping kernel's members (``experts.group_reference``) of a 3,072-row chunk grouped whole are
    the members of its former blocks (2,176 + 896 rows) grouped apart, the second's offset by 2,176 rows: the same
    pairs a expert in the same order, so the per-row arithmetic and the members' order are those of the 2-call path.
    The kernel's 16-bit staging keeps every pick < E and maps the rest to -1: the same members."""
    from tensorfold.cuda.exl3.experts import group_reference

    R, B = 3072, 2176
    pick = _picks(R, 1)
    pick[5, 3] = -1                                        # picks outside [0, E) belong to no expert
    pick[9, 0] = 70000                                     # (16-bit staging maps them to -1, never to an expert)
    uids, members = group_reference(pick, E_REAL, R)
    u1, m1 = group_reference(pick[:B], E_REAL, B)
    u2, m2 = group_reference(pick[B:], E_REAL, R - B)
    assert uids.tolist() == sorted(set(u1.tolist()) | set(u2.tolist()))
    assert int((members >= 0).sum()) == R * (SLOTS_REAL - 1) - 2
    for k, e in enumerate(uids.tolist()):
        want = []
        if e in u1.tolist():
            row = m1[u1.tolist().index(e)]
            want += row[row >= 0].tolist()
        if e in u2.tolist():
            row = m2[u2.tolist().index(e)]
            want += (row[row >= 0] + B * 32).tolist()
        got = members[k]
        assert got[got >= 0].tolist() == want, e
        assert want == sorted(want)                         # pair order: row-major
    # the 16-bit staging's mapping changes no member
    staged = torch.where((pick >= 0) & (pick < E_REAL), pick, torch.full_like(pick, -1)).to(torch.int16).to(torch.int32)
    us, ms = group_reference(staged, E_REAL, R)
    assert torch.equal(us, uids) and torch.equal(ms, members)


def test_mpe_route_of_3072_rows():
    """MPE's route (``mpe.route_reference``): a 3,072-row chunk's pairs grouped by expert into items of <= 64 pairs,
    each expert's pairs those of its two former blocks together, the item count within the scratch's bound."""
    from tensorfold.cuda.exl3.mpe import route_reference

    R, B, S = 3072, 2176, SLOTS_REAL
    pick = _picks(R, 2)
    items, spans = route_reference(pick, E_REAL)
    _, s1 = route_reference(pick[:B], E_REAL)
    _, s2 = route_reference(pick[B:], E_REAL)
    for e in range(E_REAL):
        assert spans[e] == s1[e] | {p + B * S for p in s2[e]}, e
    P = R * S
    assert len(items) <= -(-P // 64) + min(P, E_REAL)          # mpe.Scratch.max_items at 3,072 rows
    assert sum(n for _, _, n in items) == R * (S - 1)
    assert all(0 < n <= 64 for _, _, n in items)
    firsts = [f for _, f, _ in items]
    assert firsts == sorted(firsts) and firsts[0] == 0


# -- the startup estimate ------------------------------------------------------------------------------------------------
def _real_dims_folder(tmp_path):
    """A config.json at the real checkpoint's dimensions (DESIGN.md: hidden 6144, 64 heads, 256 experts, ...),
    nothing else: the buffers are built on the meta device, no tensor is read."""
    t = dict(TEXT)
    t.update({"hidden_size": 6144, "num_hidden_layers": 78, "vocab_size": 154880, "intermediate_size": 12288,
              "moe_intermediate_size": 2048, "first_k_dense_replace": 3,
              "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
              "n_routed_experts": 256, "num_experts_per_tok": 8, "num_attention_heads": 64,
              "num_key_value_heads": 64, "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_head_dim": 192,
              "qk_rope_head_dim": 64, "qk_head_dim": 256, "v_head_dim": 256, "head_dim": 192,
              "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048, "max_position_embeddings": 202752,
              "indexer_types": ["full" if i < 3 or (i - 2) % 4 == 0 else "shared" for i in range(78)]})
    folder = tmp_path / "real"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(t))
    return folder


def test_estimate_covers_3072_row_buffers(tmp_path):
    """geometry.full_buffer_bytes against forward.Buffers at 3,072 and 2,049 rows (every rank of three, meta
    tensors): the estimate covers what is allocated, within 2 % + 1 MiB; on the fake model and at the real
    dimensions (q4 and bf16 dense)."""
    from test_full_memory import check_buffers

    small = write_checkpoint(tmp_path / "ck")
    check_buffers(small, 3, "bf16", [(3072, True), (2049, True)], 4096)
    try:
        real = _real_dims_folder(tmp_path)
        from tensorfold.families.glm5_next.cuda.weights import Config

        Config.read(real)
    except Exception as exc:  # noqa: BLE001 - a config field this tree reads that the stand-in lacks
        pytest.skip(f"real-dimension config stand-in not readable here: {exc}")
    check_buffers(real, 3, "q4", [(3072, True), (2049, True)], 163840 + 16)
    check_buffers(real, 3, "bf16", [(3072, True)], 65536)


def test_estimate_added_bytes_a_rank(tmp_path):
    """The prompt chunk's share of the startup estimate at the real dimensions, 3,072 vs 2,049 rows (rank 0 of three,
    the default q4 dense, a 163,840-token window): printed, and between 0.9 and 1.2 GiB more a rank."""
    from tensorfold.cuda.capacity import config
    from tensorfold.cuda.geometry import full_buffer_bytes, full_transient_bytes
    from tensorfold.families.glm5_next.cuda.tp import Layout
    from tensorfold.families.glm5_next.cuda.weights import Config

    real = _real_dims_folder(tmp_path)
    try:
        cfg = Config.read(real)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"real-dimension config stand-in not readable here: {exc}")
    t = config(real)
    added = []
    for rank in range(3):
        lay = Layout.from_config(cfg, rank, 3)
        share = lambda fam, total, lay=lay: lay.size(fam) if fam in lay.totals else int(total) // 3  # noqa: E731
        cap = 163840 + 16

        def at(rows):
            return (full_buffer_bytes(t, rows, world=3, share=share, capacity=cap, prefill=True, prompt_split_k=False)
                    + full_transient_bytes(t, rows, cap, prefill=True, share=share))
        added.append(at(3072) - at(2049))
    print(f"\n[chunk3k] prompt buffers + transient, 3,072 vs 2,049 rows, a rank of three: "
          f"{[round(a / 2**20, 1) for a in added]} MiB more")
    assert all(0.9 * 2**30 < a < 1.2 * 2**30 for a in added), added


# -- a prompt's state does not depend on the chunk size -------------------------------------------------------------
class _NoEvent:
    def synchronize(self):
        pass

    def record(self):
        pass


def _engine(folder, rows, cap, rank=0, world=1, comm=None, split=None):
    from tensorfold.families.glm5_next.cuda import decode as D
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=rank, world=world, device="cpu", mtp=True)
    w.draft_head = None
    w.comm = comm
    e = D.Engine(w, capacity=cap, max_rows=8, prefill_rows=rows, long_context=True, kv="bf16", split=split)
    for b in (e.buf, e.pbuf, e.mbuf):
        b.staged = _NoEvent()
    return e


def _run(e, prompt, steps):
    """The prompt prefilled in the engine's chunks, its head row, then decode steps: (rows of logits, cache views)."""
    from tensorfold.families.glm5_next.cuda import decode as D
    from tensorfold.families.glm5_next.cuda.forward import commit

    D.prefill(e, prompt, None, mtp=True, keep_head=True)
    rows = [e.head.clone()]
    hidden = e.last_hidden.clone() if e.last_hidden is not None else None
    for t in steps:
        logits = e.forward([t])
        rows.append(logits[:1].float().clone())
        commit(e.w, e.st, e.buf, 1, 1)
    return rows, [v.clone() for v in D._row_views(e.st, e.st.pos, e.st.mtp_len)], hidden


def test_one_rank_chunk_size_changes_no_bit(folder, fakes):
    """One rank: a 200-token prompt in 192-row chunks (192 + 8) and in 128-row chunks (128 + 72): the same head row,
    next-step logits, MTP pending row and every cache row (latents, rotary keys, index keys, the MTP layer's)."""
    prompt, steps = tokens(200, 41), tokens(3, 42)
    a = _run(_engine(folder, 192, 256), prompt, steps)
    b = _run(_engine(folder, 128, 256), prompt, steps)
    assert len(a[0]) == len(b[0]) and all(same(x, y) for x, y in zip(a[0], b[0]))
    assert len(a[1]) == len(b[1]) and all(same(x, y) for x, y in zip(a[1], b[1]))
    assert same(a[2], b[2])


def _rank_split(folder, rank, world, port, prompt, steps, out_q):
    try:
        _rank_split_body(folder, rank, world, port, prompt, steps, out_q)
    except BaseException:                   # a rank that fails says so now
        import traceback

        out_q.put((rank, {"error": traceback.format_exc()}))
        raise


def _rank_split_body(folder, rank, world, port, prompt, steps, out_q):
    import torch.distributed as dist

    os.environ.setdefault("TRITON_INTERPRET", "1")
    torch.set_num_threads(1)
    import conftest  # noqa: F401  (the interpreter patches)
    import pytest as _pt

    from tensorfold.families.glm5_next.cuda import decode as D
    from tensorfold.families.glm5_next.cuda.hcsplit import SplitSettings

    torch.cuda.current_stream = lambda: None    # no overlap: the split never uses the main stream object on CPU
    mp_ = _pt.MonkeyPatch()
    install_fake_experts(mp_)
    mp_.setattr(D, "_sync", lambda w: None)
    os.environ["TF_GLM_DENSE"] = "bf16"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)

    class Comm:
        world_size = world

        def all_gather(self, send, recv):
            dist.all_gather(list(recv.view(world, -1).unbind(0)), send.contiguous().view(-1))

        def exchange_all(self, sends, recvs):
            work = []
            for p in sorted(sends):
                for t in sends[p]:
                    work.append(dist.isend(t.contiguous(), p))
                for t in recvs[p]:
                    tmp = torch.empty_like(t)
                    work.append((dist.irecv(tmp, p), t, tmp))
            for item in work:
                if isinstance(item, tuple):
                    item[0].wait()
                    item[1].copy_(item[2])
                else:
                    item.wait()

    Comm.world = world
    split = SplitSettings(split=True, min_rows=2)
    out = {}
    for rows in (192, 128):
        e = _engine(folder, rows, 256, rank=rank, world=world, comm=Comm(), split=split)
        out[f"alloc{rows}"] = e.pbuf.alloc_rows
        out[f"split{rows}"] = e.pbuf.split is not None and e.pbuf.split.applies(rows)
        out[rows] = _run(e, prompt, steps)
        dist.barrier()
    buf = io.BytesIO()
    torch.save(out, buf)
    out_q.put((rank, buf.getvalue()))
    dist.destroy_process_group()


def test_three_ranks_row_split_chunk_size_changes_no_bit(folder, fakes):
    """Three ranks with the row split on: 192-row chunks (no pad row, as 3,072) and 128-row chunks (one pad row, as
    2,048) over a 200-token prompt leave the same head row, next-step logits and caches on every rank, bit for bit."""
    prompt, steps = tokens(200, 43), tokens(3, 44)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 37000 + os.getpid() % 2000
    procs = [ctx.Process(target=_rank_split, args=(folder, r, 3, port, prompt, steps, q)) for r in range(3)]
    for p in procs:
        p.start()
    got = []
    for _ in procs:
        rank, payload = q.get(timeout=2400)
        if isinstance(payload, dict) and "error" in payload:
            for p in procs:
                p.kill()
            pytest.fail(f"rank {rank}: {payload['error']}")
        got.append((rank, torch.load(io.BytesIO(payload), weights_only=False)))
    for p in procs:
        p.join(timeout=60)
    for rank, out in sorted(got, key=lambda x: x[0]):
        assert out["alloc192"] == 192 and out["alloc128"] == 129, rank
        assert out["split192"] and out["split128"], rank
        a, b = out[192], out[128]
        assert all(same(x, y) for x, y in zip(a[0], b[0])), rank
        assert len(a[1]) == len(b[1]) and all(same(x, y) for x, y in zip(a[1], b[1])), rank
        assert same(a[2], b[2]), rank


def test_full_size_3072_vs_2048_one_rank(folder, fakes, request):
    """The real chunk sizes on the fake model, one rank: a 3,100-token prompt in 3,072-row chunks (3,072 + 28) and in
    2,048-row chunks (2,048 + 1,052) leaves the same head row, next-step logits and caches, bit for bit. Opt-in
    (``-k full_size``): the Triton interpreter takes about an hour."""
    if "full_size" not in (request.config.option.keyword or ""):
        pytest.skip("opt-in: -k full_size")
    prompt, steps = tokens(3100, 45), tokens(2, 46)
    a = _run(_engine(folder, 3072, 3200), prompt, steps)
    b = _run(_engine(folder, 2048, 3200), prompt, steps)
    assert all(same(x, y) for x, y in zip(a[0], b[0]))
    assert len(a[1]) == len(b[1]) and all(same(x, y) for x, y in zip(a[1], b[1]))
    assert same(a[2], b[2])
