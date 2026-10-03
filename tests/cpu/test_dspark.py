"""DSpark drafter (families/glm5_next/cuda/dspark.py) on CPU (Triton's interpreter): a tiny random DSpark against
an independent plain-torch reference written from the spec (taps -> fc -> hidden_norm -> context K/V; the block
[anchor, mask x 7] through pre-norm Qwen3 layers with the anchor-block mask -> norm -> the verifier's lm_head ->
sequential Markov bias over the top-K -> confidence), one rank and three (gloo), ring == flat context,
determinism; and the real checkpoint's config and tensor names / shapes (CHECKPOINT or DSPARK_CKPT: the snapshot
folder, or the HF repo folder holding snapshots/*/ so its relative blob links resolve)."""

import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

D, HEADS, HD, INTER, LAYERS, VOCAB, BLOCK, WINDOW, RANK = 256, 4, 64, 384, 3, 512, 8, 16, 32
AUX = [2, 20, 39, 58, 75]
MASK = 500
TOPK = 64
EPS = 1e-5
THETA = 8e6


def _config() -> dict:
    return {
        "architectures": ["DSparkDraftModel"], "aux_hidden_state_layer_ids": AUX, "block_size": BLOCK,
        "confidence_head_with_markov": True, "draft_vocab_size": VOCAB, "dtype": "bfloat16",
        "enable_confidence_head": True, "markov_head_type": "vanilla", "markov_rank": RANK, "mask_token_id": MASK,
        "sample_from_anchor": True, "sliding_window_non_causal": False, "speculators_model_type": "dspark",
        "tie_word_embeddings": False,
        "transformer_layer_config": {
            "attention_bias": False, "head_dim": HD, "hidden_act": "silu", "hidden_size": D,
            "intermediate_size": INTER, "layer_types": ["sliding_attention"] * LAYERS, "model_type": "qwen3",
            "num_attention_heads": HEADS, "num_hidden_layers": LAYERS, "num_key_value_heads": HEADS,
            "rms_norm_eps": EPS, "rope_parameters": {"rope_theta": THETA, "rope_type": "default"},
            "sliding_window": WINDOW, "use_sliding_window": True, "vocab_size": VOCAB}}


def _params(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def r(*shape, std=1.0, mean=0.0):
        return (torch.randn(*shape, generator=g) * std + mean).to(torch.bfloat16)

    p = {"fc.weight": r(D, len(AUX) * D, std=0.03), "hidden_norm.weight": r(D, std=0.1, mean=1.0),
         "norm.weight": r(D, std=0.1, mean=1.0),
         "markov_head.markov_w1.weight": r(VOCAB, RANK, std=0.3), "markov_head.markov_w2.weight": r(VOCAB, RANK, std=0.3),
         "confidence_head.proj.weight": r(1, D + RANK, std=0.05), "confidence_head.proj.bias": r(1, std=0.5)}
    for i in range(LAYERS):
        q = f"layers.{i}."
        p.update({q + "input_layernorm.weight": r(D, std=0.1, mean=1.0),
                  q + "post_attention_layernorm.weight": r(D, std=0.1, mean=1.0),
                  q + "self_attn.q_proj.weight": r(HEADS * HD, D, std=0.08),
                  q + "self_attn.k_proj.weight": r(HEADS * HD, D, std=0.08),
                  q + "self_attn.v_proj.weight": r(HEADS * HD, D, std=0.06),
                  q + "self_attn.o_proj.weight": r(D, HEADS * HD, std=0.06),
                  q + "self_attn.q_norm.weight": r(HD, std=0.1, mean=1.0),
                  q + "self_attn.k_norm.weight": r(HD, std=0.1, mean=1.0),
                  q + "mlp.gate_proj.weight": r(INTER, D, std=0.06), q + "mlp.up_proj.weight": r(INTER, D, std=0.06),
                  q + "mlp.down_proj.weight": r(D, INTER, std=0.06)})
    return p


def _verifier(seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    embed = (torch.randn(VOCAB, D, generator=g) * 0.5).to(torch.bfloat16)
    head = (torch.randn(VOCAB, D, generator=g) * 0.06).to(torch.bfloat16)
    return embed, head


def _taps(n: int, seed: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, len(AUX) * D, generator=g) * 2.0).to(torch.bfloat16)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    from safetensors.torch import save_file

    d = tmp_path_factory.mktemp("dspark")
    (d / "config.json").write_text(json.dumps(_config()))
    save_file({k: v.contiguous() for k, v in _params().items()}, str(d / "model.safetensors"))
    return d


def _vocab_split(world: int) -> list[int]:
    from tensorfold.families.glm5_next.cuda.tp import split_sizes

    return split_sizes(VOCAB, world, 64)


def _fake_w(rank: int = 0, world: int = 1, comm=None):
    from tensorfold.families.glm5_next.cuda import qmm

    embed, head = _verifier()
    sizes = _vocab_split(world)
    off = sum(sizes[:rank])
    return SimpleNamespace(device=torch.device("cpu"), rank=rank, world=world, comm=comm, embed=embed,
                           head=qmm.make_b16(head[off:off + sizes[rank]]), draft_head=None, vocab_offset=off)


def _drafter(folder, rank=0, world=1, comm=None, **kw):
    from tensorfold.families.glm5_next.cuda.dspark import Drafter

    kw.setdefault("quant", "bf16")
    kw.setdefault("capacity", 256)
    return Drafter(folder, _fake_w(rank, world, comm), top_k=TOPK, **kw)


# -- the reference (plain torch, from the spec) -------------------------------------------------------------------
def _bf(x):
    return x.to(torch.bfloat16)


def _lin(x, w):
    return _bf(x.float() @ w.float().t())


def _rms(x, w, eps=EPS):
    x = x.float()
    y = _bf(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
    return _bf(w.float() * y.float())


def _rope(x, pos):
    """neox rotate_half on [heads, rows, hd] at positions ``pos``."""
    inv = 1.0 / THETA ** (torch.arange(HD // 2, dtype=torch.float32) * 2 / HD)
    ph = pos.float()[:, None] * inv[None, :]
    cos, sin = torch.cat([ph.cos()] * 2, -1), torch.cat([ph.sin()] * 2, -1)
    x = x.float()
    rot = torch.cat([-x[..., HD // 2:], x[..., :HD // 2]], -1)
    return _bf(x * cos + rot * sin)


def reference(taps: torch.Tensor, anchor: int):
    """(h [block, D] bf16, logits [block, V] fp32) for the block at p = len(taps)."""
    P = _params()
    embed, head = _verifier()
    p = taps.shape[0]
    ctx = _rms(_lin(taps, P["fc.weight"]), P["hidden_norm.weight"])
    ids = torch.tensor([anchor] + [MASK] * (BLOCK - 1))
    x = embed[ids]
    bpos = torch.arange(p, p + BLOCK)
    cpos = torch.arange(p)
    # the anchor-block mask: context [p - W, p), block causal
    allow = torch.zeros(BLOCK, p + BLOCK, dtype=torch.bool)
    allow[:, max(0, p - WINDOW):p] = True
    allow[:, p:] = torch.tril(torch.ones(BLOCK, BLOCK, dtype=torch.bool))
    for i in range(LAYERS):
        q_ = f"layers.{i}."

        def heads(t):
            return t.view(t.shape[0], HEADS, HD).transpose(0, 1)

        kc = _rope(_rms(heads(_lin(ctx, P[q_ + "self_attn.k_proj.weight"])), P[q_ + "self_attn.k_norm.weight"]), cpos)
        vc = heads(_lin(ctx, P[q_ + "self_attn.v_proj.weight"]))
        n = _rms(x, P[q_ + "input_layernorm.weight"])
        q = _rope(_rms(heads(_lin(n, P[q_ + "self_attn.q_proj.weight"])), P[q_ + "self_attn.q_norm.weight"]), bpos)
        kb = _rope(_rms(heads(_lin(n, P[q_ + "self_attn.k_proj.weight"])), P[q_ + "self_attn.k_norm.weight"]), bpos)
        vb = heads(_lin(n, P[q_ + "self_attn.v_proj.weight"]))
        k, v = torch.cat([kc, kb], 1).float(), torch.cat([vc, vb], 1).float()
        s = (q.float() @ k.transpose(1, 2)) / math.sqrt(HD)
        s = s.masked_fill(~allow[None], float("-inf"))
        a = _bf(torch.softmax(s, -1) @ v)                      # [heads, block, hd]
        a = a.transpose(0, 1).reshape(BLOCK, HEADS * HD)
        x = _bf(x.float() + _lin(a, P[q_ + "self_attn.o_proj.weight"]).float())
        n = _rms(x, P[q_ + "post_attention_layernorm.weight"])
        g, u = _lin(n, P[q_ + "mlp.gate_proj.weight"]), _lin(n, P[q_ + "mlp.up_proj.weight"])
        act = _bf(_bf(torch.nn.functional.silu(g.float())).float() * u.float())
        x = _bf(x.float() + _lin(act, P[q_ + "mlp.down_proj.weight"]).float())
    h = _rms(x, P["norm.weight"])
    return h, h.float() @ head.float().t()


def reference_chain(h, logits, anchor, k=TOPK, follow=None):
    """Greedy Markov chain over each slot's top-k base candidates: per step (scores dict id -> score, pick, conf).
    ``follow``: feed these picks as prev instead of the reference's own (teacher forcing)."""
    P = _params()
    w1 = P["markov_head.markov_w1.weight"].double()
    w2 = P["markov_head.markov_w2.weight"].double()
    cw = P["confidence_head.proj.weight"].float()[0]
    cb = P["confidence_head.proj.bias"].float()[0]
    prev, steps = anchor, []
    for d in range(logits.shape[0]):
        vals, ids = torch.topk(logits[d].double(), k)
        sc = vals + w2[ids] @ w1[prev]
        j = int(torch.argmax(sc))
        feat = torch.cat([h[d].float(), w1[prev].to(torch.bfloat16).float()])
        conf = float(torch.sigmoid(feat @ cw + cb))
        steps.append(({int(i): float(s) for i, s in zip(ids, sc)}, int(ids[j]), conf))
        prev = follow[d] if follow is not None else int(ids[j])
    return steps


def _check_against_reference(drafter, taps, anchor, chunks=(None,)):
    drafter.reset()
    drafter.debug = True
    start = 0
    for c in chunks:
        stop = taps.shape[0] if c is None else start + c
        drafter.add_taps(taps[start:stop])
        start = stop
    assert drafter.context_end == taps.shape[0]
    tokens, values, hconf = drafter.candidates(anchor, BLOCK)
    confs: list[float] = []
    drafts = drafter.chain(tokens, values, hconf, anchor, drafter.context_end + 1, None, confs=confs)
    h_ref, logits_ref = reference(taps, anchor)
    scale = logits_ref.abs().max().item()
    # hidden and base logits within bf16 tolerance
    assert torch.allclose(drafter.last_h.float(), h_ref.float(), atol=0.05, rtol=0.02), \
        (drafter.last_h.float() - h_ref.float()).abs().max()
    # bf16 rounding flips (one ulp of h) give ~0.04 max / ~0.009 mean here; a window off by one row gives
    # ~0.6 max / ~0.13 mean
    err = (drafter.last_logits - logits_ref).abs()
    assert err.max() < 0.03 * scale and err.mean() < 0.015, (err.max(), err.mean())
    # the drafts: equal to the reference's greedy chain wherever its margin exceeds the tolerance; where a near tie
    # remains the drafter's pick is one of the tied (and the reference then follows the drafter's prefix)
    tol = 0.02 * scale + 0.02
    steps = reference_chain(h_ref, logits_ref, anchor, follow=drafts)
    exact = 0
    for d, (scores, pick, conf) in enumerate(steps):
        best = max(scores.values())
        assert drafts[d] in scores and scores[drafts[d]] >= best - tol, (d, drafts[d], pick)
        second = sorted(scores.values())[-2]
        if best - second > tol:
            assert drafts[d] == pick, (d, drafts, pick)
            exact += 1
        assert abs(confs[d] - conf) < 2e-2, (d, confs[d], conf)       # through h's bf16 noise
    # the confidence head itself, exactly: on the drafter's own h, prev = anchor then the drafts
    own = reference_chain(drafter.last_h, logits_ref, anchor, follow=drafts)
    assert np.allclose(confs, [c for _, _, c in own], atol=1e-5), (confs, [c for _, _, c in own])
    assert exact >= BLOCK - 2
    # and the base candidate values match the reference's top-k (merged by value then id)
    ref_top = torch.topk(logits_ref, TOPK, dim=-1).values.numpy()
    assert np.allclose(values, ref_top, atol=tol)
    return drafts, values, confs


@pytest.mark.parametrize("p,chunks", [(5, (None,)), (40, (33, 1, 6)), (150, (64, 64, 22))])
def test_matches_reference(folder, p, chunks):
    """Short context (p < window), a window past the start (flat), and past the 64-row tap buffer."""
    taps = _taps(p)
    _check_against_reference(_drafter(folder), taps, anchor=17 + p, chunks=chunks)


def test_ring_equals_flat(folder):
    """The context in a ring of window + block + a tile rows: the same bits as the flat buffer."""
    taps = _taps(150, seed=5)
    flat, ring = _drafter(folder), _drafter(folder, ring=True)
    assert ring.ring and ring.ring < 150 + BLOCK and not flat.ring
    out = []
    for d in (flat, ring):
        d.add_taps(taps)
        t, v, hc = d.candidates(321, BLOCK)
        out.append((t, v, hc, d.cand_host.clone()))
    assert torch.equal(out[0][3], out[1][3])
    _check_against_reference(ring, taps, anchor=321)


def test_deterministic(folder):
    taps = _taps(40, seed=7)
    runs = []
    for _ in range(2):
        d = _drafter(folder)
        d.add_taps(taps)
        drafts = d.propose(99, BLOCK, None)
        runs.append((d.cand_host.clone(), drafts, list(d.last_confidence)))
        drafts2 = d.propose(99, BLOCK, None)             # again on the same drafter: the block rows are overwritten
        assert drafts2 == drafts and torch.equal(d.cand_host, runs[-1][0])
    assert torch.equal(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1] and runs[0][2] == runs[1][2]


def test_depth_rules(folder):
    """propose: depth caps the chain; the confidence product threshold keeps the first draft; round_ms cuts at
    best_depth of the learned confidences."""
    from tensorfold.families.glm5_next.cuda.dflash2 import best_depth

    d = _drafter(folder)
    d.add_taps(_taps(30, seed=9))
    full = d.propose(42, BLOCK, None)
    confs = list(d.last_confidence)
    assert len(full) == BLOCK and len(confs) == BLOCK
    assert d.propose(42, 3, None) == full[:3]
    assert d.propose(42, 20, None) == full                # at most block drafts
    for thr in (0.999, 0.5, 0.05):
        got = d.propose(42, BLOCK, None, thr)
        alive, n = 1.0, 0
        for i, c in enumerate(confs):
            alive *= c
            if i > 0 and alive < thr:
                break
            n += 1
        assert got == full[:n]
    for ms in ((1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8), (1.0, 3, 5, 7, 9, 11, 13, 15, 17)):
        assert d.propose(42, BLOCK, None, round_ms=ms) == full[:best_depth(confs, ms)]
    assert d.propose(42, BLOCK, None) == full


def test_sampled_drafts_keyed(folder):
    """temperature > 0: keyed noise (the target's uniform_rows) over the Markov-biased candidates; reproducible."""
    from tensorfold.engine.exact_sampling import Sampling

    d = _drafter(folder)
    d.add_taps(_taps(30, seed=11))
    s = Sampling(seed=1234, temperature=0.8)
    a, b = d.propose(7, BLOCK, s), d.propose(7, BLOCK, s)
    assert a == b and len(a) == BLOCK and all(0 <= t < VOCAB for t in a)
    other = d.propose(7, BLOCK, Sampling(seed=99, temperature=0.8))
    assert len(other) == BLOCK


def test_memory_estimate_matches_allocation(folder):
    """``dspark.memory`` (what an engine budget would reserve) against what the drafter allocates, q4 and bf16, one
    rank and rank 1 of 3. (The q4 matmul itself needs the CUDA extension: not run on CPU.)"""
    from tensorfold.families.glm5_next.cuda.dspark import DSparkConfig, memory

    c = DSparkConfig.read(folder)
    norms = (3 + 2 * LAYERS) * D * 2 + LAYERS * 4 * HD
    for rank, world in ((0, 1), (1, 3)):
        for quant in ("q4", "bf16"):
            for ring in (False, True):
                d = _drafter(folder, rank, world, quant=quant, ring=ring)
                m = memory(c, rank, world, quant=quant, capacity=256, ring=ring)
                assert m["weights"] - norms + m["kv"] == d.nbytes(), (rank, world, quant, ring)
                assert m["host"] == 2 * VOCAB * RANK * 2
    assert _drafter(folder, quant="q4").nbytes() < _drafter(folder).nbytes()


# -- three ranks ----------------------------------------------------------------------------------------------------
CASES = [(5, 3), (40, 31), (150, 400)]


def _run_cases(drafter):
    out = []
    for p, anchor in CASES:
        drafter.reset()
        taps = _taps(p, seed=p)
        drafter.add_taps(taps[:p - 3])
        drafter.add_taps(taps[p - 3:])
        tokens, values, hconf = drafter.candidates(anchor, BLOCK)
        confs: list[float] = []
        drafts = drafter.chain(tokens, values, hconf, anchor, drafter.context_end + 1, None, confs=confs)
        out.append((drafts, values, confs))
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
    out_q.put((rank, d.heads, d.inter, _run_cases(d)))
    dist.destroy_process_group()


def test_three_ranks_match_one(folder):
    """Heads 2/1/1, MLP 128 each, vocabulary 192/192/128: the merged candidates and drafts equal one rank's (values
    within fp32 summation order of the row-parallel partials)."""
    one = _run_cases(_drafter(folder, ring=True))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 31000 + os.getpid() % 2000
    procs = [ctx.Process(target=_rank, args=(folder, r, 3, port, q)) for r in range(3)]
    for p in procs:
        p.start()
    got = sorted((q.get(timeout=900) for _ in procs), key=lambda x: x[0])
    for p in procs:
        p.join(timeout=60)
    assert [g[1] for g in got] == [2, 1, 1] and [g[2] for g in got] == [128, 128, 128]
    for _, _, _, cases in got:                        # every rank holds the same merged view
        for (d0, v0, c0), (d1, v1, c1) in zip(cases, got[0][3]):
            assert d0 == d1 and np.array_equal(v0, v1) and c0 == c1
    for (d3, v3, c3), (d1, v1, c1) in zip(got[0][3], one):
        assert d3 == d1, (d3, d1)
        assert np.allclose(v3, v1, atol=0.02)
        assert np.allclose(c3, c1, atol=2e-3)


# -- the real checkpoint --------------------------------------------------------------------------------------------
def _real_snapshot() -> Path | None:
    for var in ("DSPARK_CKPT", "CHECKPOINT"):
        root = os.environ.get(var)
        if not root:
            continue
        root = Path(root)
        for cand in [root, *sorted(root.glob("snapshots/*"))]:
            cfg = cand / "config.json"
            try:
                if cfg.is_file() and json.loads(cfg.read_text()).get("speculators_model_type") == "dspark":
                    return cand
            except (OSError, ValueError):
                continue
    return None


@pytest.fixture(scope="module")
def real():
    snap = _real_snapshot()
    if snap is None:
        pytest.skip("CHECKPOINT / DSPARK_CKPT: no DSpark snapshot (RedHatAI/GLM-5.3-speculator.dspark)")
    return snap


def test_real_config_and_tensors(real):
    from safetensors import safe_open

    from tensorfold.families.glm5_next.cuda.dspark import DSparkConfig, expected_tensors, memory, ring_rows, shares

    c = DSparkConfig.read(real)
    assert (c.hidden, c.heads, c.kv_heads, c.head_dim, c.inter, c.layers) == (6144, 64, 64, 64, 12288, 3)
    assert (c.block, c.mask_id, c.window, c.vocab, c.markov_rank) == (8, 154856, 2048, 154880, 256)
    assert c.aux_ids == (2, 20, 39, 58, 75) and c.tap_layers == (1, 19, 38, 57, 74)
    assert c.theta == 8e6 and c.eps == 1e-5 and c.causal and c.confidence and c.confidence_markov
    want = expected_tensors(c)
    with safe_open(str(real / "model.safetensors"), framework="pt", device="cpu") as f:
        have = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
        assert have == want
        assert all(f.get_slice(k).get_dtype() in ("BF16", "bfloat16") for k in have)
        # this rank's shard of a layer reads only its rows / columns
        s = shares(c, 1, 3)
        assert s == {"q": (22, 21), "kv": (22, 21), "mlp": (4096, 4096)}
        q = f.get_slice("layers.0.self_attn.q_proj.weight")[22 * 64:43 * 64]
        o = f.get_slice("layers.0.self_attn.o_proj.weight")[:, 22 * 64:43 * 64]
        assert tuple(q.shape) == (1344, 6144) and tuple(o.shape) == (6144, 1344)
        cw = f.get_tensor("confidence_head.proj.weight")
        assert tuple(cw.shape) == (1, 6400) and torch.isfinite(cw.float()).all()
    assert [shares(c, r, 3)["q"][1] for r in range(3)] == [22, 21, 21]
    assert ring_rows(c) == 2176
    for quant in ("q4", "bf16"):
        m = memory(c, 0, 3, quant=quant)
        print(f"\n[dspark] rank 0 of 3, {quant}: weights {m['weights'] / 2**20:.0f} MiB, ring K/V "
              f"{m['kv'] / 2**20:.1f} MiB, host Markov {m['host'] / 2**20:.0f} MiB")
        assert m["weights"] < (0.5e9 if quant == "q4" else 1.2e9)
