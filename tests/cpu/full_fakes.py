"""A tiny random full GLM-5.3 (glm_moe_dsa) checkpoint in the real one's layout (EXL3 routed experts of mixed
widths, BF16 elsewhere, the same tensor names and dtypes), a torch stand-in for the universal EXL3 expert kernels
(CUDA only), and an independent reference forward written from transformers' GlmMoeDsa modelling code."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import torch

D, VOCAB, LAYERS = 256, 256, 5
TEXT = {
    "architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa",
    "hidden_size": D, "num_hidden_layers": LAYERS, "vocab_size": VOCAB, "rms_norm_eps": 1e-5,
    "first_k_dense_replace": 1, "intermediate_size": 384, "moe_intermediate_size": 512,
    "mlp_layer_types": ["dense"] + ["sparse"] * (LAYERS - 1),
    "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 1, "routed_scaling_factor": 2.5,
    "norm_topk_prob": True, "n_group": 1, "topk_group": 1, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
    "num_attention_heads": 5, "num_key_value_heads": 5, "q_lora_rank": 128, "kv_lora_rank": 128,
    "qk_nope_head_dim": 48, "qk_rope_head_dim": 16, "qk_head_dim": 64, "v_head_dim": 128, "head_dim": 48,
    "rope_interleave": True, "indexer_rope_interleave": True,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "index_n_heads": 4, "index_head_dim": 64, "index_topk": 16,
    "indexer_types": ["full", "shared", "shared", "full", "shared"],
    "num_nextn_predict_layers": 1, "max_position_embeddings": 4096, "eos_token_id": [1],
    "tie_word_embeddings": False, "dtype": "bfloat16",
    "quantization_config": {"quant_method": "exl3", "bits": 2.75, "codebook": "mcg", "head_bits": 16,
                            "scope": "glm53_routed_experts_only"},
}
WIDTHS = ((2, 2, 2), (2, 2, 3), (3, 3, 3), (3, 3, 4), (4, 4, 4))   # [gate, up, down] bits an expert, mixed in a layer


def _bf16(shape, g, scale):
    return (scale * torch.randn(*shape, generator=g)).to(torch.bfloat16)


def write_safetensors(path: Path, tensors: dict) -> None:
    header, blobs, at = {}, [], 0
    names = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int16: "I16",
             torch.int32: "I32"}
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype != torch.bfloat16 else \
            t.contiguous().view(torch.int16).numpy().tobytes()
        header[name] = {"dtype": names[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def write_checkpoint(folder: Path, seed: int = 0) -> Path:
    """The tiny checkpoint in two shards with an index (as the real one, 137 of them), config.json included."""

    g = torch.Generator().manual_seed(seed)
    c = TEXT
    H, nope, rope, vd = c["num_attention_heads"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    ql, kl, I, E = c["q_lora_rank"], c["kv_lora_rank"], c["moe_intermediate_size"], c["n_routed_experts"]
    t: dict = {}
    t["model.embed_tokens.weight"] = _bf16((VOCAB, D), g, 1.0)
    t["lm_head.weight"] = _bf16((VOCAB, D), g, 0.06)
    t["model.norm.weight"] = (1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
    rng = np.random.default_rng(seed)

    def layer(i: int, mtp: bool, indexed: bool) -> None:
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = (1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
        t[p + "post_attention_layernorm.weight"] = (1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
        a = p + "self_attn."
        t[a + "q_a_proj.weight"] = _bf16((ql, D), g, D ** -0.5)
        t[a + "q_a_layernorm.weight"] = (1 + 0.1 * torch.randn(ql, generator=g)).to(torch.bfloat16)
        t[a + "q_b_proj.weight"] = _bf16((H * (nope + rope), ql), g, ql ** -0.5)
        t[a + "kv_a_proj_with_mqa.weight"] = _bf16((kl + rope, D), g, D ** -0.5)
        t[a + "kv_a_layernorm.weight"] = (1 + 0.1 * torch.randn(kl, generator=g)).to(torch.bfloat16)
        t[a + "kv_b_proj.weight"] = _bf16((H * (nope + vd), kl), g, kl ** -0.5)
        t[a + "o_proj.weight"] = _bf16((D, H * vd), g, (H * vd) ** -0.5)
        if indexed:
            ih, idim = c["index_n_heads"], c["index_head_dim"]
            t[a + "indexer.wq_b.weight"] = _bf16((ih * idim, ql), g, ql ** -0.5)
            t[a + "indexer.wk.weight"] = _bf16((idim, D), g, D ** -0.5)
            t[a + "indexer.k_norm.weight"] = (1 + 0.1 * torch.randn(idim, generator=g)).to(torch.bfloat16)
            t[a + "indexer.k_norm.bias"] = (0.1 * torch.randn(idim, generator=g)).to(torch.bfloat16)
            t[a + "indexer.weights_proj.weight"] = _bf16((ih, D), g, D ** -0.5)
        m = p + "mlp."
        if not mtp and c["mlp_layer_types"][i] == "dense":
            w = c["intermediate_size"]
            t[m + "gate_proj.weight"] = _bf16((w, D), g, D ** -0.5)
            t[m + "up_proj.weight"] = _bf16((w, D), g, D ** -0.5)
            t[m + "down_proj.weight"] = _bf16((D, w), g, w ** -0.5)
            return
        t[m + "gate.weight"] = _bf16((E, D), g, D ** -0.5)
        t[m + "gate.e_score_correction_bias"] = (0.05 * torch.randn(E, generator=g)).float()
        for proj, k, n in (("gate_proj", D, I), ("up_proj", D, I), ("down_proj", I, D)):
            t[m + f"shared_experts.{proj}.weight"] = _bf16((n, k), g, k ** -0.5)
        for e in range(E):
            bits = WIDTHS[(e + i) % len(WIDTHS)] if not mtp else (4, 4, 4)
            for proj, k, n, b in (("gate_proj", D, I, bits[0]), ("up_proj", D, I, bits[1]),
                                  ("down_proj", I, D, bits[2])):
                q = f"{m}experts.{e}.{proj}."
                t[q + "trellis"] = torch.from_numpy(rng.integers(-32768, 32768, size=(k // 16, n // 16, 16 * b),
                                                                 dtype=np.int16))
                # suh / svh: the rotations' signs with a scale (fp16), as ExLlamaV3 stores them
                t[q + "suh"] = torch.from_numpy((rng.choice([-1.0, 1.0], size=k) * 0.02 * k ** -0.25)
                                                .astype(np.float16))
                t[q + "svh"] = torch.from_numpy((rng.choice([-1.0, 1.0], size=n) * 0.5 * n ** -0.25)
                                                .astype(np.float16))
                t[q + "mcg"] = torch.tensor([-878968851], dtype=torch.int32)        # 0xCBAC1FED as int32

    for i in range(LAYERS):
        layer(i, False, c["indexer_types"][i] == "full")
    i = LAYERS
    layer(i, True, True)
    p = f"model.layers.{i}."
    t[p + "enorm.weight"] = (1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
    t[p + "hnorm.weight"] = (1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
    t[p + "eh_proj.weight"] = _bf16((D, 2 * D), g, (2 * D) ** -0.5)
    t[p + "shared_head.norm.weight"] = (1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)

    folder.mkdir(parents=True, exist_ok=True)
    names = sorted(t)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    wmap = {}
    for fname, part in shards.items():
        write_safetensors(folder / fname, {n: t[n] for n in part})
        wmap.update({n: fname for n in part})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))
    (folder / "config.json").write_text(json.dumps(TEXT))
    return folder


# -- the universal EXL3 kernels' stand-in (they are CUDA only) --------------------------------------------------------

class FakeExperts:
    """What ``experts.prepare`` returns, holding each expert's dequantized fp32 weights (W [K, N]: y = x W)."""

    def __init__(self, gate, up, down, codebook):
        from tensorfold.cuda.exl3 import format as fmt

        def deq(m):
            tr, su, sv = m
            bits = tr.shape[-1] / 16
            return torch.from_numpy(fmt.dequantize(tr.cpu().numpy(), su.cpu().numpy(), sv.cpu().numpy(), bits,
                                                   codebook)).float()
        self.wg = [deq(m) for m in gate]
        self.wu = [deq(m) for m in up]
        self.wd = [deq(m) for m in down]
        self.suh_g = torch.stack([m[1].reshape(-1).to(torch.float16) for m in gate])   # as prepare stacks them
        self.suh_u = torch.stack([m[1].reshape(-1).to(torch.float16) for m in up])
        self.count = len(gate)
        self.dims = self.wg[0].shape[0]
        self.width = self.wg[0].shape[1]
        self.keep = []


class FakeScratch:
    def __init__(self, ex, rows, slots, cfg_gu=None, cfg_d=None, device="cpu"):
        self.rows, self.slots = rows, slots
        self.y = torch.zeros((rows * slots, ex.dims), dtype=torch.float32)


def fake_routed(x, pick, wts, ex, s, out, R, limit=math.inf, act_mode=0, group=True, mt=1):
    """Per (row, slot) fp32 rows of routed experts; picks >= E left as they are (the shared slot): the kernels' ACT_BF16
    SwiGLU roundings (bf16 gate and up, bf16 silu(gate), bf16 product), the down projection in fp32."""

    slots = pick.shape[1]
    y = s.y[:R * slots].view(R, slots, -1)
    xf = x[:R].float()
    for r in range(R):
        for k in range(slots):
            e = int(pick[r, k])
            if e >= ex.count:
                continue
            g = (xf[r] @ ex.wg[e]).to(torch.bfloat16).float()
            u = (xf[r] @ ex.wu[e]).to(torch.bfloat16).float()
            a = (torch.nn.functional.silu(g).to(torch.bfloat16).float() * u).to(torch.bfloat16).float()
            y[r, k] = a @ ex.wd[e]
    return s.y[:R * slots]


def install_fake_experts(monkeypatch) -> None:
    from tensorfold.cuda.exl3 import experts as generic

    monkeypatch.setattr(generic, "prepare", lambda g, u, d, cb, device="cuda": FakeExperts(g, u, d, cb))
    monkeypatch.setattr(generic, "Scratch", FakeScratch)
    monkeypatch.setattr(generic, "routed", fake_routed)


# -- the reference forward (transformers' GlmMoeDsa, in fp32 with bf16 activations between modules) ----------------

def _rb(x):
    return x.to(torch.bfloat16).float()


class Reference:
    """GlmMoeDsaForCausalLM's forward and the MTP head from the checkpoint's tensors, unsplit, one sequence."""

    def __init__(self, folder: Path, kv: str = "bf16"):
        from tensorfold.cuda.exl3 import format as fmt

        self.kv = kv                     # fp8 / fp4: the cached latent rows as TF_GLM_KV stores them (kv8)
        idx = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
        self.t = {}
        for fname in sorted(set(idx.values())):
            self.t.update(read_safetensors(folder / fname))
        self.c = TEXT
        self.fmt = fmt
        self._experts: dict = {}

    def w(self, name):
        return self.t[name].float()

    def expert(self, i, e, proj):
        key = (i, e, proj)
        if key not in self._experts:
            p = f"model.layers.{i}.mlp.experts.{e}.{proj}."
            tr = self.t[p + "trellis"]
            self._experts[key] = torch.from_numpy(self.fmt.dequantize(
                tr.numpy(), self.t[p + "suh"].numpy(), self.t[p + "svh"].numpy(), tr.shape[-1] / 16, "mcg")).float()
        return self._experts[key]

    @staticmethod
    def rms(x, w, eps=1e-5):
        x = x.float()
        return _rb(w * _rb(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)))

    def rope(self, x, pos):
        """transformers' interleaved rotation of x [..., n] at positions pos [T] (leading dim T)."""
        n = x.shape[-1]
        inv = 1.0 / (self.c["rope_parameters"]["rope_theta"] ** (torch.arange(0, n, 2).float() / n))
        ang = pos.float()[:, None] * inv[None, :]
        cos, sin = _rb(ang.cos()), _rb(ang.sin())
        while cos.dim() < x.dim():
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
        return _rb(torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1))

    def attention(self, i, x, cache, pos, topk_prev, indexed):
        """x [T, D] normed rows at positions pos; cache: this layer's growing lists (keys, values, index keys)."""
        c = self.c
        H, nope, rope, vd = c["num_attention_heads"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
        a = f"model.layers.{i}.self_attn."
        T = x.shape[0]
        qr = self.rms(_rb(x @ self.w(a + "q_a_proj.weight").T), self.w(a + "q_a_layernorm.weight"))
        q = _rb(qr @ self.w(a + "q_b_proj.weight").T).view(T, H, nope + rope)
        kv = _rb(x @ self.w(a + "kv_a_proj_with_mqa.weight").T)
        lat = self.rms(kv[:, :c["kv_lora_rank"]], self.w(a + "kv_a_layernorm.weight"))
        if self.kv == "fp8":
            from tensorfold.families.glm5_next.cuda import kv8

            lat = kv8.dequantize(kv8.quantize_rows(lat.to(torch.bfloat16))).float()
        elif self.kv == "fp4":
            from tensorfold.families.glm5_next.cuda import kv8

            lat = kv8.dequantize4(kv8.quantize_rows4(lat.to(torch.bfloat16))).float()
        k_rot = self.rope(kv[:, c["kv_lora_rank"]:], pos)
        q_rot = self.rope(q[..., nope:], pos)
        kvb = _rb(lat @ self.w(a + "kv_b_proj.weight").T).view(T, H, nope + vd)
        cache["k"].append(torch.cat([kvb[..., :nope], k_rot[:, None, :].expand(T, H, rope)], -1))
        cache["v"].append(kvb[..., nope:])
        K, V = torch.cat(cache["k"]), torch.cat(cache["v"])                    # [S, H, *]
        S = K.shape[0]
        topk = topk_prev
        if indexed:
            ih, idim = c["index_n_heads"], c["index_head_dim"]
            qi = _rb(qr @ self.w(a + "indexer.wq_b.weight").T).view(T, ih, idim)
            qi = torch.cat([self.rope(qi[..., :rope], pos), qi[..., rope:]], -1)
            k = _rb(torch.nn.functional.layer_norm(_rb(x @ self.w(a + "indexer.wk.weight").T), (idim,),
                                                   self.w(a + "indexer.k_norm.weight"),
                                                   self.w(a + "indexer.k_norm.bias"), eps=1e-6))
            k = torch.cat([self.rope(k[:, :rope], pos), k[:, rope:]], -1)
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
        out = _rb(_rb(o).reshape(T, H * vd) @ self.w(a + "o_proj.weight").T)
        return out, topk

    def mlp(self, i, x):
        c = self.c
        m = f"model.layers.{i}.mlp."
        if f"{m}gate_proj.weight" in self.t:
            g = _rb(x @ self.w(m + "gate_proj.weight").T)
            u = _rb(x @ self.w(m + "up_proj.weight").T)
            return _rb(_rb(_rb(torch.nn.functional.silu(g)) * u) @ self.w(m + "down_proj.weight").T)
        logits = x.float() @ self.w(m + "gate.weight").T
        sc = logits.sigmoid()
        choice = sc + self.t[m + "gate.e_score_correction_bias"]
        idx = choice.topk(c["num_experts_per_tok"], -1).indices
        wts = sc.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * c["routed_scaling_factor"]
        out = torch.zeros_like(x, dtype=torch.float32)
        for r in range(x.shape[0]):
            for j in range(idx.shape[1]):
                e = int(idx[r, j])
                g = _rb(x[r] @ self.expert(i, e, "gate_proj"))
                u = _rb(x[r] @ self.expert(i, e, "up_proj"))
                out[r] += wts[r, j] * (_rb(_rb(torch.nn.functional.silu(g)) * u) @ self.expert(i, e, "down_proj"))
        s = m + "shared_experts."
        g = _rb(x @ self.w(s + "gate_proj.weight").T)
        u = _rb(x @ self.w(s + "up_proj.weight").T)
        out += _rb(_rb(torch.nn.functional.silu(g)) * u) @ self.w(s + "down_proj.weight").T
        return _rb(out)

    def new_cache(self):
        return [{"k": [], "v": [], "ik": []} for _ in range(LAYERS + 1)]

    def forward(self, ids, cache, start):
        """Rows ids at positions start ..: (logits [T, V], final-normed hidden [T, D]); cache grows."""
        pos = torch.arange(start, start + len(ids))
        x = self.w("model.embed_tokens.weight")[torch.tensor(ids)]
        x = _rb(x)
        topk = None
        for i in range(LAYERS):
            p = f"model.layers.{i}."
            indexed = self.c["indexer_types"][i] == "full"
            h, topk = self.attention(i, self.rms(x, self.w(p + "input_layernorm.weight")), cache[i], pos, topk,
                                     indexed)
            x = _rb(x + h)
            x = _rb(x + self.mlp(i, self.rms(x, self.w(p + "post_attention_layernorm.weight"))))
        normed = self.rms(x, self.w("model.norm.weight"))
        return normed @ self.w("lm_head.weight").T, normed

    def mtp(self, next_ids, hidden, cache, start):
        """The MTP head: rows (next token, final-normed hidden) at MTP positions start .. -> logits."""
        i = LAYERS
        p = f"model.layers.{i}."
        pos = torch.arange(start, start + len(next_ids))
        e = _rb(self.w("model.embed_tokens.weight")[torch.tensor(next_ids)])
        if start == 0:
            e[0] = 0
        cat = torch.cat([self.rms(e, self.w(p + "enorm.weight")), self.rms(hidden, self.w(p + "hnorm.weight"))], -1)
        x = _rb(cat @ self.w(p + "eh_proj.weight").T)
        h, _ = self.attention(i, self.rms(x, self.w(p + "input_layernorm.weight")), cache[i], pos, None, True)
        x = _rb(x + h)
        x = _rb(x + self.mlp(i, self.rms(x, self.w(p + "post_attention_layernorm.weight"))))
        return self.rms(x, self.w(p + "shared_head.norm.weight")) @ self.w("lm_head.weight").T


def read_safetensors(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I16": torch.int16, "I32": torch.int32}
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        raw = torch.frombuffer(bytearray(data[a:b]), dtype=torch.uint8)
        d = dt[info["dtype"]]
        out[name] = (raw.view(torch.int16).view(torch.bfloat16) if d == torch.bfloat16 else raw.view(d)).reshape(
            info["shape"])
    return out
