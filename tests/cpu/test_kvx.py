"""TF_GLM_KV=fp4x: fp4's latent rows, plus the rotary key plane as e4m3 codes (the row's power-of-two scale in the FP4
latent row's pad) and the indexer's per-token keys as e4m3 codes with an fp32 power-of-two scale. The format's
properties (row independence, error bounds, zeros and subnormals), the write kernels against the torch definition
byte for byte, every reader's dequantization exact (the bf16 kernels on the dequantized planes, bit for bit), the
memory accounting, the full forward on one rank and three (tensor and context parallel), drafted == serial, and a
quality proxy against fp4 on the tiny model."""

import multiprocessing as mp
import os

import pytest
import torch
import triton
import triton.language as tl

from full_fakes import TEXT, Reference, _rb, install_fake_experts, write_checkpoint
from test_full_forward import Rig, _rank_cp, agree_kv, tokens
from mp_wire import unpack  # _rank_cp sends its results packed (mp_wire)

from tensorfold.families.glm5_next.cuda import dcp, kv8, latent
from tensorfold.families.glm5_next.cuda import dsa_full as F

LW, ROPE, D = 512, 64, 128
SCALE = 256 ** -0.5


# -- the format ---------------------------------------------------------------------------------------------------

def rows_case(seed, n=64, w=ROPE):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, w, generator=g) * torch.logspace(-3, 1, n)[:, None]            # rows of every magnitude
    return x.to(torch.bfloat16)


def test_codes_are_row_independent_and_pow2():
    x = rows_case(0)
    codes, s = kv8.quantize_codes8(x)
    for i in (0, 17, 63):
        c1, s1 = kv8.quantize_codes8(x[i:i + 1])
        assert torch.equal(c1[0], codes[i]) and torch.equal(s1, s[i:i + 1])
    m, e = torch.frexp(s)
    assert (m == 0.5).all()                                           # powers of two
    y = codes.view(torch.float8_e4m3fn).float()
    amax = y.abs().amax(dim=1)
    assert (amax >= 128).all() and (amax <= 256).all()                # the largest code in [128, 256]: never saturates


def test_round_trip_error_bounds():
    """e4m3 keeps 3 mantissa bits: |x - x~| <= 2^-4 |x| for values in the normal range (>= 2^-6 s), and at most half
    the subnormal step (2^-10 s) below it."""
    x = rows_case(1, n=256)
    codes, s = kv8.quantize_codes8(x)
    y = codes.view(torch.float8_e4m3fn).float() * s[:, None]
    xf = x.float()
    err = (y - xf).abs()
    normal = xf.abs() >= 2 ** -6 * s[:, None]
    assert (err[normal] <= 2 ** -4 * xf.abs()[normal]).all()
    assert (err[~normal] <= 2 ** -10 * s[:, None].expand_as(err)[~normal]).all()
    rel = (y - xf).norm(dim=1) / xf.norm(dim=1)
    print(f"\n[kvx] e4m3 relative row error mean {rel.mean():.4f} max {rel.max():.4f}")
    assert rel.mean() < 0.03


def test_zeros_and_subnormals():
    x = torch.zeros(4, ROPE, dtype=torch.bfloat16)
    x[1, 3] = 1e-30                                                   # a lone tiny value: its own scale
    x[2, 0], x[2, 1] = 100.0, 1e-5                                    # 2^-23 of the row's largest: flushes to 0
    x[3, :8] = torch.tensor([2.0 ** -k for k in range(8)])           # an exact geometric row
    codes, s = kv8.quantize_codes8(x)
    y = codes.view(torch.float8_e4m3fn).float() * s[:, None]
    assert (y[0] == 0).all() and s[0] == 2.0 ** -126
    assert y[1, 3] > 0 and abs(y[1, 3] - 1e-30) <= 2 ** -4 * 1e-30
    assert y[2, 0] == 100.0 or abs(y[2, 0] - 100) <= 100 * 2 ** -4
    assert y[2, 1] == 0
    assert torch.equal(y[3, :8], x[3, :8].float())                    # powers of two are exact
    idx = kv8.quantize_index8(x)
    assert idx.shape == (4, ROPE + 4) and torch.equal(kv8.dequantize_index8(idx), y)
    # the dequantized values are bf16 values (what the readers' tiles hold)
    assert torch.equal(y.to(torch.bfloat16).float(), y)


def edge_values():
    """fp32 values over e4m3's range and its edges: zeros of both signs, fp32 subnormals, e4m3 subnormals and their
    ties, every e4m3 value and the midpoints between neighbours (ties to even), powers of two, +-448 and past it."""
    g = torch.Generator().manual_seed(11)
    allc = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    vals = allc.view(torch.float8_e4m3fn).float()
    vals = vals[torch.isfinite(vals)]
    pos = vals[vals > 0].sort().values
    mids = (pos[1:] + pos[:-1]) / 2                                           # exact ties in fp32
    nudges = torch.cat([mids * (1 + 2.0 ** -20), mids * (1 - 2.0 ** -20)])
    pow2 = torch.tensor([2.0 ** k for k in range(-140, 10)])
    rnd = torch.randn(20000, generator=g) * torch.logspace(-12, 2.5, 20000)
    tiny = torch.tensor([0.0, 1e-45, 1e-40, 2.0 ** -10, 2.0 ** -10 * 3, 2.0 ** -11, 448.0, 464.0, 480.0, 500.0, 1e4])
    v = torch.cat([vals, mids, nudges, pow2, rnd, tiny])
    return torch.cat([v, -v])


def test_e4m3_encoder_is_round_to_nearest_even():
    """kv8.e4m3_bits (the definition, integer arithmetic) equals torch's float8_e4m3fn conversion wherever that one is
    defined the same way (|v| <= 448: round to nearest even), and saturates at +-448 past it."""
    v = edge_values()
    got = kv8.e4m3_bits(v)
    ok = v.abs() <= 448
    want = v[ok].to(torch.float8_e4m3fn).view(torch.uint8)
    bad = (got[ok] != want).nonzero()
    assert bad.numel() == 0, [(v[ok][i].item(), got[ok][i].item(), want[i].item()) for i in bad[:5, 0]]
    assert (got[~ok & (v > 0)] == 0x7E).all() and (got[~ok & (v < 0)] == 0xFE).all()
    assert got[(v == 0) & (torch.signbit(v))].eq(0x80).all() and got[(v == 0) & ~torch.signbit(v)].eq(0).all()


@triton.jit
def _encode(X, OUT, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(OUT + i, kv8.e4m3_code(tl.load(X + i, mask=i < N, other=0.0)), mask=i < N)


def test_triton_encoder_is_the_definition():
    """The write kernels' encoder (kv8.e4m3_code) against the definition on every edge and random value."""
    v = edge_values()
    out = torch.empty(v.numel(), dtype=torch.uint8)
    _encode[(triton.cdiv(v.numel(), 1024),)](v, out, v.numel(), B=1024)
    assert torch.equal(out, kv8.e4m3_bits(v))


# -- the write kernels against the definition ---------------------------------------------------------------------

def test_rope_write_is_the_definition():
    g = torch.Generator().manual_seed(2)
    T, R, p0 = 40, 7, 9
    freq = F.inv_freq(ROPE, 10000.0, "cpu")
    k = torch.randn(R, ROPE + 8, generator=g).to(torch.bfloat16)[:, 8:]        # strided rows
    lat = torch.randn(R, LW, generator=g).to(torch.bfloat16)
    pos = torch.tensor([p0], dtype=torch.int32)
    ref = torch.zeros(T, ROPE, dtype=torch.bfloat16)
    F.k_rope_write(k, ref, pos, freq)
    lc = kv8.zeros(T, LW, "fp4x", "cpu")
    kr = kv8.rope_zeros(T, ROPE, "fp4x", "cpu")
    latent.latent_write(lat, lc, pos)
    F.k_rope_write(k, kr, pos, freq, latent=lc)
    codes, s = kv8.quantize_codes8(ref[p0:p0 + R])
    assert torch.equal(kr[p0:p0 + R], codes) and (kr[:p0] == 0).all() and (kr[p0 + R:] == 0).all()
    assert torch.equal(kv8.rope_scales(lc)[p0:p0 + R], s)
    # the latent bytes are fp4's (the rotary scale only fills pad), whichever writer runs first
    assert torch.equal(kv8.dequantize4(lc), kv8.dequantize4(kv8.zeros(T, LW, "fp4", "cpu").index_copy_(
        0, torch.arange(p0, p0 + R), kv8.quantize_rows4(lat))))
    lc2 = kv8.zeros(T, LW, "fp4x", "cpu")
    kr2 = kv8.rope_zeros(T, ROPE, "fp4x", "cpu")
    F.k_rope_write(k, kr2, pos, freq, latent=lc2)
    latent.latent_write(lat, lc2, pos)
    assert torch.equal(lc, lc2) and torch.equal(kr, kr2)
    assert torch.equal(kv8.dequantize_rope8(kr, lc)[p0:p0 + R], codes.view(torch.float8_e4m3fn).float() * s[:, None])


def test_rope_write_context_parallel():
    g = torch.Generator().manual_seed(3)
    G, R, p0 = 3, 8, 4
    freq = F.inv_freq(ROPE, 10000.0, "cpu")
    k = torch.randn(R, ROPE, generator=g).to(torch.bfloat16)
    lat = torch.randn(R, LW, generator=g).to(torch.bfloat16)
    pos = torch.tensor([p0], dtype=torch.int32)
    whole_lc, whole_kr = kv8.zeros(16, LW, "fp4x", "cpu"), kv8.rope_zeros(16, ROPE, "fp4x", "cpu")
    latent.latent_write(lat, whole_lc, pos)
    F.k_rope_write(k, whole_kr, pos, freq, latent=whole_lc)
    for rank in range(G):
        lc, kr = kv8.zeros(7, LW, "fp4x", "cpu"), kv8.rope_zeros(7, ROPE, "fp4x", "cpu")
        latent.latent_write(lat, lc, pos, cp=G, rank=rank)
        F.k_rope_write(k, kr, pos, freq, cp=G, rank=rank, latent=lc)
        for p in range(p0, p0 + R):
            if p % G == rank:
                assert torch.equal(kr[p // G], whole_kr[p]) and torch.equal(lc[p // G], whole_lc[p])


@pytest.mark.parametrize("cp", [1, 3])
def test_index_write_is_the_definition(cp):
    g = torch.Generator().manual_seed(4)
    T, R, p0 = 32, 6, 5
    freq = F.inv_freq(ROPE, 10000.0, "cpu")
    raw = torch.randn(R, D + 16, generator=g).to(torch.bfloat16)
    ln_w = (1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
    ln_b = (0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
    pos = torch.tensor([p0], dtype=torch.int32)
    ref = torch.zeros(T, D, dtype=torch.bfloat16)
    F.index_write(raw, ln_w, ln_b, ref, pos, freq)
    keys = kv8.index_zeros(T, D, "fp4x", "cpu")
    assert keys.shape == (T, D + 4) and keys.dtype == torch.uint8
    if cp == 1:
        F.index_write(raw, ln_w, ln_b, keys, pos, freq)
        assert torch.equal(keys[p0:p0 + R], kv8.quantize_index8(ref[p0:p0 + R]))
        assert (keys[:p0] == 0).all() and (keys[p0 + R:] == 0).all()
    else:
        for rank in range(cp):
            loc = kv8.index_zeros(T, D, "fp4x", "cpu")
            F.index_write(raw, ln_w, ln_b, loc, pos, freq, cp=cp, rank=rank)
            for p in range(p0, p0 + R):
                if p % cp == rank:
                    assert torch.equal(loc[p // cp], kv8.quantize_index8(ref[p:p + 1])[0])


# -- the readers: the bf16 kernels on the dequantized planes, bit for bit --------------------------------------------

def index_case(seed, R, cap, H=32):
    g = torch.Generator().manual_seed(seed)
    qi = torch.randn(R, H * D, generator=g).to(torch.bfloat16)
    wts = torch.randn(R, H + 128, generator=g).to(torch.bfloat16)[:, 128:]
    keys = torch.randn(cap, D, generator=g).to(torch.bfloat16)
    keys[::7] = 0
    k8 = kv8.quantize_index8(keys)
    return qi, wts, k8, kv8.dequantize_index8(k8).to(torch.bfloat16)


@pytest.mark.parametrize("pos0,R,cap,k", [(300, 3, 384, 64), (50, 5, 128, 16)])
def test_scores_and_selection_dequantize_exactly(pos0, R, cap, k):
    qi, wts, k8, kb = index_case(5, R, cap)
    pos = torch.tensor([pos0], dtype=torch.int32)
    a, b = torch.empty(R, cap), torch.empty(R, cap)
    F.score(qi, wts, k8, pos, R, cap, a)
    F.score(qi, wts, kb, pos, R, cap, b)
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    ta, ca = F.select_tokens(qi, wts, k8, pos0, R, pos, bucket_cols=cap, k=k)
    tb, cb = F.select_tokens(qi, wts, kb, pos0, R, pos, bucket_cols=cap, k=k)
    assert torch.equal(ta, tb) and torch.equal(ca, cb)
    ta, ca = F.select_tokens(qi, wts, k8, pos0, R, pos, prompt=True, k=k)
    tb, cb = F.select_tokens(qi, wts, kb, pos0, R, pos, prompt=True, k=k)
    assert torch.equal(ta, tb) and torch.equal(ca, cb)
    for rank in range(3):
        la, lb = torch.empty(R, k8[rank::3].shape[0]), torch.empty(R, k8[rank::3].shape[0])
        dcp.score_local(qi, wts, k8[rank::3].contiguous(), rank, 3, pos, R, la.shape[1], la)
        dcp.score_local(qi, wts, kb[rank::3].contiguous(), rank, 3, pos, R, lb.shape[1], lb)
        assert torch.equal(la.view(torch.int32), lb.view(torch.int32))


def attn_case(seed, R, H, T):
    g = torch.Generator().manual_seed(seed)
    qa = (0.3 * torch.randn(R, H, LW, generator=g)).to(torch.bfloat16)
    qp = (0.3 * torch.randn(R, H, ROPE, generator=g)).to(torch.bfloat16)
    lc = kv8.zeros(T, LW, "fp4x", "cpu")
    latent.latent_write(torch.randn(T, LW, generator=g).to(torch.bfloat16), lc, torch.tensor([0], dtype=torch.int32))
    kr = kv8.rope_zeros(T, ROPE, "fp4x", "cpu")
    F.k_rope_write(torch.randn(T, ROPE, generator=g).to(torch.bfloat16), kr, torch.tensor([0], dtype=torch.int32),
                   F.inv_freq(ROPE, 10000.0, "cpu"), latent=lc)
    krb = kv8.dequantize_rope8(kr, lc).to(torch.bfloat16)
    return qa, qp, lc, kr, krb


def test_attention_dequantizes_rope_exactly():
    """latent.attention / sparse_attention / dcp.attention_partial with the e4m3 rotary plane equal them with a bf16
    rotary plane holding the dequantized rows, bit for bit (the same tiles)."""
    H, T, R, pos0 = 6, 80, 3, 60
    qa, qp, lc, kr, krb = attn_case(6, R, H, T)
    pos = torch.tensor([pos0], dtype=torch.int32)
    nch = latent.chunks_for(pos0 + R)
    s = latent.LatentScratch(R, H, nch, "cpu")
    a, b = torch.empty(R, H, LW, dtype=torch.bfloat16), torch.empty(R, H, LW, dtype=torch.bfloat16)
    latent.attention(qa, lc, pos, s, scale=SCALE, nch=nch, out=a, qp=qp, rope=kr)
    latent.attention(qa, lc, pos, s, scale=SCALE, nch=nch, out=b, qp=qp, rope=krb)
    assert torch.equal(a, b)
    g = torch.Generator().manual_seed(7)
    tok = torch.stack([torch.randperm(T, generator=g)[:24].sort().values for _ in range(R)]).int()
    cnt = torch.tensor([24, 10, 0], dtype=torch.int32)
    a.zero_(), b.zero_()
    latent.sparse_attention(qa, lc, tok, cnt, a, SCALE, qp=qp, rope=kr)
    latent.sparse_attention(qa, lc, tok, cnt, b, SCALE, qp=qp, rope=krb)
    assert torch.equal(a, b)
    oa, la = dcp.attention_partial(qa, qp, lc, kr, tok, cnt, SCALE)
    ob, lb = dcp.attention_partial(qa, qp, lc, krb, tok, cnt, SCALE)
    assert torch.equal(oa, ob) and torch.equal(la, lb)
    with pytest.raises(ValueError):                    # the e4m3 plane needs its FP4 latent rows
        latent.sparse_attention(qa, kv8.zeros(T, LW, "fp8", "cpu"), tok, cnt, a, SCALE, qp=qp, rope=kr)


def test_rotary_dot_error():
    """The rotary term's error at real dimensions: q_rope . k_rope with e4m3 keys against bf16 keys, over rotated keys
    of every position (RoPE spreads a key's energy over its 32 pairs: amax / rms ~ 2-3)."""
    g = torch.Generator().manual_seed(8)
    n = 4096
    freq = F.inv_freq(ROPE, 10000.0, "cpu")
    raw = torch.randn(n, ROPE, generator=g).to(torch.bfloat16)
    kb = torch.zeros(n, ROPE, dtype=torch.bfloat16)
    pos = torch.tensor([0], dtype=torch.int32)
    F.k_rope_write(raw, kb, pos, freq)                         # rows rotated at positions 0 .. n - 1
    codes, s = kv8.quantize_codes8(kb)
    k8 = codes.view(torch.float8_e4m3fn).float() * s[:, None]
    q = torch.randn(256, ROPE, generator=g).to(torch.bfloat16).float()
    db, d8 = q @ kb.float().T, q @ k8.T
    err = (d8 - db).abs()
    rms = db.pow(2).mean().sqrt()
    rel_rows = (k8 - kb.float()).norm(dim=1) / kb.float().norm(dim=1)
    print(f"\n[kvx] rotary dot: abs err mean {err.mean():.4f}, p99 {err.flatten().kthvalue(int(0.99 * err.numel())).values:.4f},"
          f" max {err.max():.4f} against dot rms {rms:.3f} (mean err / rms {err.mean() / rms:.4f}); key row rel err "
          f"mean {rel_rows.mean():.4f}")
    assert err.mean() / rms < 0.03


# -- memory -------------------------------------------------------------------------------------------------------

def test_token_bytes_real_dimensions():
    from tensorfold.cuda.geometry import full_token_bytes

    kinds = ["full"] * 21 + ["shared"] * 57
    t = {"num_hidden_layers": 78, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "index_head_dim": 128,
         "indexer_types": kinds}
    assert kinds.count("full") == 21
    assert full_token_bytes(t, "fp4", False) == 78 * (304 + 128) + 21 * 256 == 39_072
    assert full_token_bytes(t, "fp4x", False) == 78 * (304 + 64) + 21 * 132 == 31_476
    assert full_token_bytes(t, "fp4x", True) == 79 * 368 + 22 * 132 == 31_976
    print(f"\n[kvx] bytes a token a rank: fp4 {full_token_bytes(t, 'fp4', True)}, fp4x {full_token_bytes(t, 'fp4x', True)}"
          f" (with MTP): x{full_token_bytes(t, 'fp4', True) / full_token_bytes(t, 'fp4x', True):.4f} the window")


def test_caches_allocate_what_geometry_counts(tmp_path):
    from test_full_memory import stand_in, tensor_bytes, text_of
    from tensorfold.cuda.geometry import full_token_bytes
    from tensorfold.families.glm5_next.cuda import forward as FW
    from tensorfold.families.glm5_next.cuda.weights import Config

    folder = write_checkpoint(tmp_path / "ck")
    cfg = Config.read(folder)
    for kv in ("fp4", "fp4x"):
        w = stand_in(cfg, 0, 3)
        c = FW.Caches(w, 1000, streams=1, ring=64, kv=kv)
        assert tensor_bytes(c.arena) == 1000 * full_token_bytes(text_of(folder), kv, True), kv
    c = FW.Caches(stand_in(cfg, 0, 3), 1000, streams=1, ring=64, kv="fp4x")
    assert c.arena.planes[c.kr[0]].tensor.dtype == torch.uint8
    assert c.arena.planes[c.index[0][2]].tensor.shape[1] == TEXT["index_head_dim"] + 4


# -- the full forward ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_checkpoint(tmp_path_factory.mktemp("glm53kvx"))


@pytest.fixture
def fakes(monkeypatch):
    install_fake_experts(monkeypatch)
    monkeypatch.setenv("TF_GLM_DENSE", "bf16")


class ReferenceX(Reference):
    """The reference with fp4x's caches: fp4 latent rows, rotary keys and index keys through e4m3 (kv8's
    definitions on the bf16 rows the engine stores)."""

    def __init__(self, folder):
        super().__init__(folder, kv="fp4")

    @staticmethod
    def e4m3(x):
        codes, s = kv8.quantize_codes8(x.to(torch.bfloat16))
        return codes.view(torch.float8_e4m3fn).float() * s[:, None]

    def attention(self, i, x, cache, pos, topk_prev, indexed):
        c = self.c
        H, nope, rope, vd = c["num_attention_heads"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
        a = f"model.layers.{i}.self_attn."
        T = x.shape[0]
        qr = self.rms(_rb(x @ self.w(a + "q_a_proj.weight").T), self.w(a + "q_a_layernorm.weight"))
        q = _rb(qr @ self.w(a + "q_b_proj.weight").T).view(T, H, nope + rope)
        kv = _rb(x @ self.w(a + "kv_a_proj_with_mqa.weight").T)
        lat = self.rms(kv[:, :c["kv_lora_rank"]], self.w(a + "kv_a_layernorm.weight"))
        lat = kv8.dequantize4(kv8.quantize_rows4(lat.to(torch.bfloat16))).float()
        k_rot = self.e4m3(self.rope(kv[:, c["kv_lora_rank"]:], pos))
        q_rot = self.rope(q[..., nope:], pos)
        kvb = _rb(lat @ self.w(a + "kv_b_proj.weight").T).view(T, H, nope + vd)
        cache["k"].append(torch.cat([kvb[..., :nope], k_rot[:, None, :].expand(T, H, rope)], -1))
        cache["v"].append(kvb[..., nope:])
        K, V = torch.cat(cache["k"]), torch.cat(cache["v"])
        S = K.shape[0]
        topk = topk_prev
        if indexed:
            ih, idim = c["index_n_heads"], c["index_head_dim"]
            qi = _rb(qr @ self.w(a + "indexer.wq_b.weight").T).view(T, ih, idim)
            qi = torch.cat([self.rope(qi[..., :rope], pos), qi[..., rope:]], -1)
            k = _rb(torch.nn.functional.layer_norm(_rb(x @ self.w(a + "indexer.wk.weight").T), (idim,),
                                                   self.w(a + "indexer.k_norm.weight"),
                                                   self.w(a + "indexer.k_norm.bias"), eps=1e-6))
            k = self.e4m3(torch.cat([self.rope(k[:, :rope], pos), k[:, rope:]], -1))
            cache["ik"].append(k)
            IK = torch.cat(cache["ik"])
            sc = torch.relu(torch.einsum("thd,sd->ths", qi, IK) * idim ** -0.5)
            wts = _rb(x @ self.w(a + "indexer.weights_proj.weight").T) * ih ** -0.5
            score = (sc * wts[:, :, None]).sum(1)
            vis = torch.arange(S)[None, :] <= pos[:, None]
            score = score.masked_fill(~vis, float("-inf"))
            topk = score.topk(min(c["index_topk"], S), dim=-1).indices
        mask = torch.full((T, S), float("-inf"))
        mask.scatter_(1, topk[:, :min(c["index_topk"], S)], 0.0)
        mask = mask.masked_fill(torch.arange(S)[None, :] > pos[:, None], float("-inf"))
        qh = torch.cat([q[..., :nope], q_rot], -1)
        s = torch.einsum("thd,shd->hts", qh, K) * (nope + rope) ** -0.5 + mask[None]
        o = torch.einsum("hts,shd->thd", torch.softmax(s, -1), V)
        return _rb(_rb(o).reshape(T, H * vd) @ self.w(a + "o_proj.weight").T), topk


def test_one_rank_matches_the_reference(folder, fakes):
    """As test_full_forward's (prompt, decode steps across the top-k, a sparse chunk, a sparse window) with fp4x's
    caches against the reference holding the same quantized rows, within fp4's band."""
    rig = Rig(folder, kv="fp4x")
    ref = ReferenceX(folder)
    cache = ref.new_cache()
    prompt = tokens(12, 1)
    got = rig.prefill(prompt)
    want, _ = ref.forward(prompt, cache, 0)
    ok, why = agree_kv(got[None], want[-1:], "fp4")
    assert ok, f"prompt: {why}"
    pos = len(prompt)
    for step, t in enumerate(tokens(10, 2)):
        got = rig.window([t])
        want, _ = ref.forward([t], cache, pos)
        ok, why = agree_kv(got, want, "fp4")
        assert ok, f"decode step {step} at {pos}: {why}"
        pos += 1
    more = tokens(30, 3)
    got = rig.prefill(more)
    want, _ = ref.forward(more, cache, pos)
    ok, why = agree_kv(got[None], want[-1:], "fp4")
    assert ok, f"sparse chunk: {why}"
    pos += len(more)
    got = rig.window(tokens(4, 4))
    want, _ = ref.forward(tokens(4, 4), cache, pos)
    ok, why = agree_kv(got, want, "fp4")
    assert ok, f"sparse window: {why}"


def test_mtp_head_matches_the_reference(folder, fakes):
    """The MTP layer's caches (its FP4 latent, e4m3 rotary and index planes) through the MTP head."""
    rig = Rig(folder, kv="fp4x")
    rig.w.draft_head = None
    assert rig.st.mtp_kr.dtype == torch.uint8 and rig.st.index[-1][2].dtype == torch.uint8
    ref = ReferenceX(folder)
    cache = ref.new_cache()
    prompt = tokens(20, 7)
    rig.prefill(prompt)
    ref.forward(prompt, cache, 0)
    nxt = prompt[1:] + [5]
    got = rig.mtp(nxt[:8], rig.pbuf.fnormed[:8])
    want = ref.mtp(nxt[:8], rig.pbuf.fnormed[:8].float(), cache, 0)
    ok, why = agree_kv(got, want, "fp4")
    assert ok, f"MTP: {why}"


@pytest.mark.parametrize("start", [9, 40])
def test_drafted_window_equals_serial_steps(folder, fakes, start):
    prompt = tokens(start, 5)
    draft = tokens(6, 6)
    a, b = Rig(folder, kv="fp4x"), Rig(folder, kv="fp4x")
    a.prefill(prompt)
    b.prefill(prompt)
    serial = torch.cat([a.window([t]) for t in draft])
    window = b.window(draft, keep=3)
    assert torch.equal(serial, window)
    b2 = b.window(draft[3:])
    assert torch.equal(b2, serial[3:])
    st = b.st                                                    # the planes are fp4x's
    assert st.kr[0].dtype == torch.uint8 and st.index[0][2].dtype == torch.uint8 and kv8.fp4(st.kc[0])


def test_quality_proxy_against_fp4(folder, fakes):
    """fp4x against fp4 (and fp4 against bf16, for scale) on the tiny model: a prompt, decode steps across the
    top-k, a sparse chunk. The rotary and index planes' e4m3 costs less than the latent's e2m1 already does."""
    prompt, steps, more = tokens(14, 41), tokens(8, 42), tokens(24, 43)
    runs = {}
    for kv in ("bf16", "fp4", "fp4x"):
        rig = Rig(folder, kv=kv)
        rows = [rig.prefill(prompt)] + [rig.window([t])[0] for t in steps] + [rig.prefill(more)]
        runs[kv] = torch.stack(rows)
    scale = runs["bf16"].abs().max().item()

    def diff(a, b):
        d = (runs[a] - runs[b]).abs()
        same = (runs[a].argmax(-1) == runs[b].argmax(-1)).float().mean().item()
        return d.max().item(), d.mean().item(), same

    x4 = diff("fp4x", "fp4")
    f4 = diff("fp4", "bf16")
    xb = diff("fp4x", "bf16")
    print(f"\n[kvx] logits scale {scale:.3f}; fp4x vs fp4: max {x4[0]:.4f} mean {x4[1]:.5f} argmax agree {x4[2]:.2f}; "
          f"fp4 vs bf16: max {f4[0]:.4f} mean {f4[1]:.5f} agree {f4[2]:.2f}; fp4x vs bf16: max {xb[0]:.4f} "
          f"mean {xb[1]:.5f} agree {xb[2]:.2f}")
    assert x4[0] <= 0.15 * scale and x4[2] >= 0.8
    assert xb[1] <= 2.0 * f4[1] + 1e-3 * scale


def _rank_tp(folder, rank, world, port, prompt, steps, kv, out_q):
    import torch.distributed as dist

    os.environ.setdefault("TRITON_INTERPRET", "1")
    torch.set_num_threads(1)
    import conftest  # noqa: F401  (the interpreter patches)
    import pytest as _pt

    mp_ = _pt.MonkeyPatch()
    install_fake_experts(mp_)
    os.environ["TF_GLM_DENSE"] = "bf16"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)

    class Comm:
        world_size = world

        def all_gather(self, send, recv):
            dist.all_gather(list(recv.view(world, -1).unbind(0)), send.contiguous().view(-1))

    rig = Rig(folder, rank, world, Comm(), kv=kv)
    rows = [rig.prefill(prompt)]
    for t in steps:
        rows.append(rig.window([t])[0])
    out_q.put((rank, torch.stack(rows), rig.w.vocab_offset))
    dist.destroy_process_group()


def test_three_ranks_match_one(folder, fakes):
    """Tensor parallel at three ranks with fp4x's caches: the logits slices put together are one rank's within fp32
    summation order (each rank quantizes the same rows: the caches are replicated)."""
    prompt, steps = tokens(14, 8), tokens(6, 9)
    one = Rig(folder, kv="fp4x")
    whole = torch.stack([one.prefill(prompt)] + [one.window([t])[0] for t in steps])
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 35000 + os.getpid() % 2000
    procs = [ctx.Process(target=_rank_tp, args=(folder, r, 3, port, prompt, steps, "fp4x", q)) for r in range(3)]
    for p in procs:
        p.start()
    got = sorted((q.get(timeout=900) for _ in procs), key=lambda x: x[0])
    for p in procs:
        p.join(timeout=60)
    parts = torch.cat([g[1] for g in got], dim=1)
    assert [g[2] for g in got] == [0, 128, 192]
    ok, why = agree_kv(parts, whole, "fp4")
    assert ok, why


def test_context_parallel_three_ranks(folder, fakes):
    """Context parallelism at three ranks with fp4x: the slices put together agree with one rank's fp4x run within
    fp4's band (agree_kv), and a drafted window equals the serial steps bit for bit on every rank."""
    prompt, steps = tokens(40, 28), tokens(6, 29)
    one = Rig(folder, kv="fp4x")
    whole = torch.stack([one.prefill(prompt)] + [one.window([t])[0] for t in steps])
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 37000 + os.getpid() % 2000
    procs = [ctx.Process(target=_rank_cp, args=(folder, r, 3, port, prompt, steps, steps, q, "fp4x"))
             for r in range(3)]
    for p in procs:
        p.start()
    got = sorted((unpack(q.get(timeout=1500)) for _ in procs), key=lambda x: x[0])
    for p in procs:
        p.join(timeout=60)
    parts = torch.cat([g[1] for g in got], dim=1)
    ok, why = agree_kv(parts, whole, "fp4")
    assert ok, why
    for rank, rows, win, _ in got:
        assert torch.equal(win[:len(steps) - 1], rows[1:len(steps)]), rank
