"""Context parallelism's faster prompt path (forward.cp_prompt, TF_GLM_CP_PIPE=1) on one GPU: the owner-merged
selection equals dcp.select_merge bit for bit at full GLM-5.3's sizes, the bf16 partial merge equals merge_ranks on
the widened partials, Mia's sparse attention on views of the gathered query block (the new layout) gives the bits of
0129's contiguous copies, and a timing of one 512-row part's local work (0129's copies + fp32 merge vs views + bf16
merge; the collectives left out) printed with -s."""

import time

import pytest
import torch

from tensorfold.families.glm5_next.cuda import dcp, kv8, latent

DEV = "cuda"
G = 3


def _cands(R, k, local, pos0, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    qi = torch.randn(R, 32 * 128, generator=g, device=DEV).bfloat16()
    wts = torch.randn(R, 32, generator=g, device=DEV).bfloat16()
    keys = torch.randn(local * G, 128, generator=g, device=DEV).bfloat16()
    keys[::7] = 0                                   # exact-zero scores: ties across ranks
    pos = torch.tensor([pos0], dtype=torch.int32, device=DEV)
    cols = dcp.visible_local(pos0 + R - 1, 0, G)
    return torch.stack([dcp.select_local(qi, wts, keys[r::G].contiguous(), r, G, pos, R, k=k, cols=cols)
                        for r in range(G)]).contiguous()


@pytest.mark.parametrize("R,pos0", [(2048, 30000), (2048, 1000), (1000, 50000)])
def test_owner_merge_equals_select_merge_gpu(R, pos0):
    k = 2048
    cands = _cands(R, k, (pos0 + R) // G + 2, pos0, R + pos0)
    M = dcp.owner_rows(R, G)
    padded = torch.zeros((G, G * M, k), dtype=torch.int64, device=DEV)
    padded[:, :R] = cands
    th_all = torch.empty((G * M,), dtype=torch.int64, device=DEV)
    for p in range(G):
        dcp.select_threshold(padded[:, p * M:(p + 1) * M].contiguous(), M, th_all[p * M:(p + 1) * M])
    for r in range(G):
        slots, counts = dcp.select_merge(cands, r, G, k)
        s2, c2 = dcp.select_keep(padded[r, :R].contiguous(), th_all, G)
        assert torch.equal(c2, counts) and torch.equal(s2, slots), r


@pytest.mark.parametrize("H,HP", [(22, 22), (21, 22)])
def test_merge_ranks_bf16_bits_gpu(H, HP):
    torch.manual_seed(H)
    n, LW = 512, 512
    o = torch.randn(G, n, HP, LW, device=DEV).bfloat16()
    lse = torch.randn(G, n, HP, device=DEV) * 4
    lse[1, :7] = float("-inf")
    o[1, :7] = 0
    want = dcp.merge_ranks(o.float().contiguous(), lse.contiguous())
    for rank in range(G):
        got_o = torch.stack([o[p] for p in range(G) if p != rank]).contiguous()
        got_l = torch.stack([lse[p] for p in range(G) if p != rank]).contiguous()
        out = torch.empty((n, H, LW), dtype=torch.bfloat16, device=DEV)
        dcp.merge_ranks_bf16(o[rank].contiguous(), lse[rank].contiguous(), got_o, got_l, rank, out)
        assert torch.equal(out, want[:, :H]), rank


def _attn_case(n, HP, T, K, kind="fp8"):
    torch.manual_seed(9)
    lat = torch.randn(T, 512, device=DEV).bfloat16()
    kr = torch.randn(T, 64, device=DEV).bfloat16()
    cache = kv8.zeros(T, 512, kind, DEV)
    latent.latent_write(lat, cache, torch.tensor([0], dtype=torch.int32, device=DEV))
    tok = torch.zeros((n, K), dtype=torch.int32, device=DEV)
    cnt = torch.zeros(n, dtype=torch.int32, device=DEV)
    for r in range(n):
        m = [K // 3, 9, 0, 700][r % 4]
        if m:
            tok[r, :m] = torch.randperm(T, device=DEV)[:m].sort().values.int()
        cnt[r] = m
    # the gathered block: per rank [qa (n, HP, 512), qp (n, HP, 64)]
    q = (0.3 * torch.randn(G * n * HP * 576, device=DEV)).bfloat16()
    return cache, kr, tok, cnt, q


@pytest.mark.parametrize("kind", ["fp8", "fp4"])
def test_msa_on_gathered_views_has_copy_bits(kind):
    from tensorfold.families.glm5_next.cuda import msa

    n, HP, LW = 96, 22, 512
    cache, kr, tok, cnt, q = _attn_case(n, HP, 9000, 2048, kind)
    allq = q.view(G, n * HP * 576)
    po = torch.empty((G, n, HP, LW), dtype=torch.bfloat16, device=DEV)
    pl = torch.empty((G, n, HP), dtype=torch.float32, device=DEV)
    for g in range(G):
        qa_g = allq[g][:n * HP * LW].view(n, HP, LW)
        qp_g = allq[g][n * HP * LW:].view(n, HP, 64)
        msa.prompt_lse(qa_g, cache, tok, cnt, po[g], pl[g], 0.0625, qp_g, kr)
        o2 = torch.empty((n, HP, LW), dtype=torch.bfloat16, device=DEV)
        l2 = torch.empty((n, HP), dtype=torch.float32, device=DEV)
        msa.prompt_lse(qa_g.clone(), cache, tok, cnt, o2, l2, 0.0625, qp_g.clone(), kr)
        assert torch.equal(po[g], o2) and torch.equal(pl[g], l2), g


def _time(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def test_part_local_work_timing():
    """One 512-row part at 22 heads a rank: 0129's local work (pad + copy, permute/split copies of the gathered
    queries, per-group copies, partial scatter, send copies, fp32 widening, merge, the ol copy) against the new path
    (copy into the send block, views, bf16 merge), the attention and the collectives left out (a local copy stands in
    for the wire in both). Prints both; the new path must be faster."""
    n, HP, HL, LW, RD = 512, 22, 22, 512, 64
    rank = 0
    qa = torch.randn(n, HL, LW, device=DEV).bfloat16()
    qp = torch.randn(n, HL, RD, device=DEV).bfloat16()
    og = torch.randn(n, HP, LW, device=DEV).bfloat16()
    lg = torch.randn(n, HP, device=DEV)
    ol = torch.empty(n, HL, LW, device=DEV, dtype=torch.bfloat16)

    def old():
        mine = torch.empty((n, HP, LW + RD), dtype=torch.bfloat16, device=DEV)
        mine.zero_()
        mine[:, :HL, :LW] = qa
        mine[:, :HL, LW:] = qp
        allq = mine.reshape(-1).repeat(G)                       # the all-gather
        allq = allq.view(G, n, HP, LW + RD).permute(1, 0, 2, 3).reshape(n, G * HP, LW + RD)
        qa_all = allq[:, :, :LW].contiguous()
        qp_all = allq[:, :, LW:].contiguous()
        qa_all[:, :HP].contiguous()
        ob = torch.empty((n, G * HP, LW), dtype=torch.bfloat16, device=DEV)
        lse = torch.empty((n, G * HP), dtype=torch.float32, device=DEV)
        for gq in range(G):
            hs = slice(gq * HP, (gq + 1) * HP)
            qa_all[:, hs].contiguous()
            qp_all[:, hs].contiguous()
            ob[:, hs].copy_(og)                                 # the attention's output
            lse[:, hs].copy_(lg)
        got_o = torch.empty((G, n, HP, LW), dtype=torch.bfloat16, device=DEV)
        got_l = torch.empty((G, n, HP), dtype=torch.float32, device=DEV)
        for p in range(1, G):
            got_o[p].copy_(ob[:, p * HP:(p + 1) * HP].contiguous())    # send copy + the wire
            got_l[p].copy_(lse[:, p * HP:(p + 1) * HP].contiguous())
        got_o[rank].copy_(ob[:, :HP])
        got_l[rank].copy_(lse[:, :HP])
        merged = dcp.merge_ranks(got_o.float(), got_l)
        ol.copy_(merged[:, :HL])

    q = torch.empty((G * n * HP * (LW + RD),), dtype=torch.bfloat16, device=DEV)
    mine_b = torch.empty((n * HP * (LW + RD),), dtype=torch.bfloat16, device=DEV)
    po = torch.empty((G, n, HP, LW), dtype=torch.bfloat16, device=DEV)
    pl = torch.empty((G, n, HP), dtype=torch.float32, device=DEV)
    got = torch.empty((G - 1, n, HP, LW), dtype=torch.bfloat16, device=DEV)
    gl = torch.empty((G - 1, n, HP), dtype=torch.float32, device=DEV)

    def new():
        mine_b[:n * HP * LW].view(n, HP, LW)[:, :HL].copy_(qa)
        mine_b[n * HP * LW:].view(n, HP, RD)[:, :HL].copy_(qp)
        q.view(G, -1).copy_(mine_b.view(1, -1).expand(G, -1))  # the all-gather
        for g in range(G):
            po[g].copy_(og)                                     # the attention's output
            pl[g].copy_(lg)
        got.copy_(po[1:])                                       # the wire
        gl.copy_(pl[1:])
        dcp.merge_ranks_bf16(po[rank], pl[rank], got, gl, rank, ol)

    t_old, t_new = _time(old), _time(new)
    print(f"\n[cp part local work] 0129 {t_old:.2f} ms, new {t_new:.2f} ms (the attention's own writes and the "
          "stand-in wire copies in both)")
    assert t_new < t_old
