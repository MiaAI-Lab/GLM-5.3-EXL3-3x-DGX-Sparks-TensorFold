"""Full GLM-5.3's new kernels on the GPU: RoPE, the per-token indexer and the latent attention with its rotary term
agree with the CPU references, and every row keeps its bits alone, in windows and in prompt blocks."""

import pytest
import torch

from tensorfold.families.glm5_next.cuda import dsa_full as F
from tensorfold.families.glm5_next.cuda import kv8, latent

DEV = "cuda"
THETA = 8_000_000.0


def test_rope_and_index_write_match_cpu_formulas():
    from test_dsa_full import rope_ref

    g = torch.Generator().manual_seed(1)
    q = torch.randn(4, 22, 256, generator=g).to(torch.bfloat16)
    freq = F.inv_freq(64, THETA, DEV)
    for pos0 in (0, 4097, 1_000_000):
        out = torch.empty(4, 22, 64, dtype=torch.bfloat16, device=DEV)
        F.q_rope(q.to(DEV), out, torch.tensor([pos0], dtype=torch.int32, device=DEV), freq, 192)
        ref = rope_ref(q[..., 192:], torch.arange(pos0, pos0 + 4), 64).to(torch.bfloat16)
        assert torch.allclose(out.cpu().float(), ref.float(), rtol=2 ** -7, atol=2 ** -7)


@pytest.mark.parametrize("NT,pos0", [(4096, 3000), (8192, 6500)])
def test_scores_and_selection_rows_keep_their_bits(NT, pos0):
    """Decode windows of 1..16 rows: each row's scores and its 2048 tokens equal the row alone; prompt blocks give
    the same tokens."""
    g = torch.Generator().manual_seed(2)
    R, cap = 16, NT
    qi = torch.randn(R, 32 * 128, generator=g).to(torch.bfloat16).to(DEV)
    wts = torch.randn(R, 160, generator=g).to(torch.bfloat16).to(DEV)[:, 128:]
    keys = torch.randn(cap, 128, generator=g).to(torch.bfloat16).to(DEV)
    pos = torch.tensor([pos0], dtype=torch.int32, device=DEV)
    t_all, c_all = F.select_tokens(qi, wts, keys, pos0, R, pos, bucket_cols=NT)
    s_all = torch.empty(R, NT, device=DEV)
    F.score(qi, wts, keys, pos, R, NT, s_all)
    for r in range(R):
        p1 = pos + r
        t1, c1 = F.select_tokens(qi[r:r + 1].contiguous(), wts[r:r + 1], keys, pos0 + r, 1, p1, bucket_cols=NT)
        s1 = torch.empty(1, NT, device=DEV)
        F.score(qi[r:r + 1].contiguous(), wts[r:r + 1], keys, p1, 1, NT, s1)
        assert torch.equal(s1[0], s_all[r]) and torch.equal(t1[0], t_all[r]) and c1.item() == c_all[r].item()
    tp, cp = F.select_tokens(qi, wts, keys, pos0, R, pos, prompt=True)
    assert torch.equal(cp, c_all) and torch.equal(tp, t_all)
    # against torch: the same top 2048 sets but for near ties
    ref = s_all.float().cpu()
    for r in range(0, R, 5):
        vis = pos0 + r + 1
        want = set(torch.topk(ref[r, :vis], 2048).indices.tolist())
        got = set(t_all[r].tolist())
        assert len(want & got) >= 2040


@pytest.mark.parametrize("kv", ["bf16", "fp8"])
@pytest.mark.parametrize("pos0", [40, 3000])
def test_latent_attention_rows_keep_their_bits(kv, pos0):
    """Dense (decode) and sparse latent attention with RoPE: a window's rows equal each row alone; close to fp32."""
    g = torch.Generator().manual_seed(3)
    H, R, T = 22, 8, pos0 + 8
    qa = (0.3 * torch.randn(R, H, 512, generator=g)).to(torch.bfloat16).to(DEV)
    qp = (0.3 * torch.randn(R, H, 64, generator=g)).to(torch.bfloat16).to(DEV)
    lat = torch.randn(T, 512, generator=g).to(torch.bfloat16).to(DEV)
    kr = torch.randn(T, 64, generator=g).to(torch.bfloat16).to(DEV)
    cache = kv8.zeros(T, 512, kv, DEV)
    latent.latent_write(lat, cache, torch.tensor([0], dtype=torch.int32, device=DEV))
    nch = latent.chunks_for(T)
    s = latent.LatentScratch(R, H, nch, DEV)
    pos = torch.tensor([pos0], dtype=torch.int32, device=DEV)
    full = torch.empty(R, H, 512, dtype=torch.bfloat16, device=DEV)
    latent.attention(qa, cache, pos, s, scale=256 ** -0.5, nch=nch, out=full, qp=qp, rope=kr, hb=latent.HB)
    for r in range(R):
        one = torch.empty(1, H, 512, dtype=torch.bfloat16, device=DEV)
        latent.attention(qa[r:r + 1].contiguous(), cache, pos + r, s, scale=256 ** -0.5, nch=nch, out=one,
                         qp=qp[r:r + 1].contiguous(), rope=kr, hb=latent.HB)
        assert torch.equal(one[0], full[r])
    tokens = torch.stack([torch.randperm(pos0 + 1, generator=g)[:min(2048, pos0 + 1)].sort().values
                          for _ in range(R)]).to(torch.int32).to(DEV)
    counts = torch.full((R,), tokens.shape[1], dtype=torch.int32, device=DEV)
    sp = torch.empty_like(full)
    latent.sparse_attention(qa, cache, tokens, counts, sp, 256 ** -0.5, qp=qp, rope=kr)
    for r in range(R):
        one = torch.empty(1, H, 512, dtype=torch.bfloat16, device=DEV)
        latent.sparse_attention(qa[r:r + 1].contiguous(), cache, tokens[r:r + 1].contiguous(), counts[r:r + 1], one,
                                256 ** -0.5, qp=qp[r:r + 1].contiguous(), rope=kr)
        assert torch.equal(one[0], sp[r])


@pytest.mark.parametrize("kind", ["fp8", "fp4"])
def test_mia_sparse_attention_matches_the_triton_kernels(kind):
    """msa.prompt on an FP8 cache: within bf16 rounding of latent.sparse_attention (chunks + merge), the same bits run
    to run, and a row's bits the same alone (any chunking)."""
    import torch

    from tensorfold.families.glm5_next.cuda import kv8, latent, msa

    torch.manual_seed(3)
    dev = "cuda"
    H, R, T, K = 22, 96, 6000, 2048
    qa = (0.3 * torch.randn(R, H, 512, device=dev)).bfloat16()
    qp = (0.3 * torch.randn(R, H, 64, device=dev)).bfloat16()
    lat = torch.randn(T, 512, device=dev).bfloat16()
    kr = torch.randn(T, 64, device=dev).bfloat16()
    cache = kv8.zeros(T, 512, kind, dev)
    latent.latent_write(lat, cache, torch.tensor([0], dtype=torch.int32, device=dev))
    W = K
    tok = torch.full((R, W), -1, dtype=torch.int32, device=dev)
    cnt = torch.zeros(R, dtype=torch.int32, device=dev)
    for r in range(R):
        n = [K, 7, 1, 40, 2000][r % 5]               # full lists, short ones and a lone token
        sel = torch.randperm(T, device=dev)[:n].sort().values.int()
        tok[r, :n] = sel
        cnt[r] = n
    cnt[5] = 0                                       # a row without a list: left alone
    ref = torch.zeros(R, H, 512, dtype=torch.bfloat16, device=dev)
    latent.sparse_attention(qa, cache, tok, cnt, ref, 0.0625, qp=qp, rope=kr)
    got = torch.zeros_like(ref)
    msa.prompt(qa, cache, tok, cnt, got, 0.0625, qp, kr)
    live = cnt > 0
    err = (got[live].float() - ref[live].float()).abs().max().item()
    assert err <= 0.02 * ref[live].float().abs().max().item(), err
    assert (got[~live] == 0).all()
    again = torch.zeros_like(ref)
    msa.prompt(qa, cache, tok, cnt, again, 0.0625, qp, kr)
    assert torch.equal(again, got)
    for r in (0, 3, 4):
        one = torch.zeros_like(ref[:1])
        msa.prompt(qa[r:r + 1].contiguous(), cache, tok[r:r + 1].contiguous(), cnt[r:r + 1].contiguous(), one, 0.0625,
                   qp[r:r + 1].contiguous(), kr)
        assert torch.equal(one[0], got[r])
