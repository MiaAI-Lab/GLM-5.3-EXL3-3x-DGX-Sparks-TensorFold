# Changelog

## Unreleased (in development)

- Full GLM-5.3 (`glm_moe_dsa`) on TensorFold v0.6.0 across three DGX Sparks: the GLM-5.3-Flash recipe's v1.4 patches
  (0001-0068) and patches 0100-0104 (the 3-rank layout with expert blocks rotating by layer, the mixed-width EXL3
  loader, RoPE MLA and the per-token DSA indexer, the forward and MTP head, the family and its memory estimate).
- FP8 latent cache by default (`KV=fp8`; rotary and index keys stay bf16); the embedding split by vocabulary rows.
- Memory: `scripts/memplan.py` (every rank's plan from TensorFold's own estimate), checked by `start.sh` before
  anything starts; `scripts/memguard.sh` on every Spark while the server runs.
- Worker weights over NFS only (`prepare.sh` checks each worker's view file by file); the image built on each Spark
  from the GLM-5.3-Flash recipe's published image.
- Tests: `scripts/test-cpu.sh` (no GPU), `scripts/test-gpu.sh`, `tools/exact.py`.
- First boots on three Sparks (2026-10-03): exact (12/12 at 32k and 64k), needles correct to 58.2k tokens, decode
  28.3 / 33.0 tok/s (prose / code), prefill ~140 tok/s; lowest free memory at a 64k window 16.7 / 20.1 / 13.3 GiB.
- Prefill: prompt chunks of 2,048 rows through the sparse-attention kernels in 512-row blocks and the routed experts
  through a 2-tile kernel (0105); then drowzeys' EXL3 prompt kernels for the routed experts, in a deterministic mode
  (0106): 3.8x on a real layer's 2,048-row chunk (55.6 -> 14.6 ms, spark2, 2026-10-03).
- Our own prompt kernels: MoE experts (0108: one input rotation a layer, faster than drowzeys' kernel and ~6x closer
  to the reference arithmetic) and sparse attention (0110: 30% faster than the chunk programs); 4-bit decode tiles for
  full GLM's shapes (0111). Prefill 655 / 656 / 638 tok/s at 8k / 16k / 32k (boot p10, 2026-10-03; first boot 151 /
  140).
- Decode: RoCE one-shot all-gathers by default (COMM=roce, +10%), the DSpark speculator by default (DRAFTER=dspark,
  0114, MTP head off): code 39.5-40.4, prose 29.1-30.2 tok/s (boot p17, sampled, 3 runs each; MTP + NCCL: 33.0-34.5 /
  27.6-28.1), the same replies (exact checks 12/12).
- Context parallelism (CP=1, 0112-0113): each Spark keeps every third token's caches, exact top-k selection across
  the Sparks, attention partials merged by log-sum-exp in rank order. Boot p19 (450k window, fp8): exact 12/12,
  needles correct to 235,660 tokens, prefill ~250 tok/s; boot cp655 (655,360 window, fp8, eager decode): prose
  23.8-25.1, code 29.6-30.1 tok/s.
- 4-bit latent cache (KV=fp4, 0115): e2m1 codes, an e4m3 scale per 16 values and a power-of-two row scale, 304 bytes
  a row (fp8: 528). First quality probe against fp8 (greedy): 30 short questions 29/30 (fp8 27/30), 27/30 replies
  identical, 8-key recall 8/8 at 64k and 128k. Not the default yet: a larger evaluation is in progress.
- Draft costs measured on text, the verify profile fixed (0116, 0119); DSpark sampling filter (0117); static scratch
  for captured CP decode windows (0118); CP windows within the top-k skip selection (0120); a RoCE timeout inside a
  replayed graph is reported (0121).
- Captured decode windows under CP (TF_GLM_CP_GRAPHS=1, off by default) cost ~2.9 GiB on each Spark. At 655k with fp8
  that left spark3 under the memory guard's 3 GiB during a long prompt: the guard stopped it and the other ranks
  waited (the "hang" seen with graphs on). With fp4, boot fp4g655 (KV=fp4 CP=1 TF_GLM_CP_GRAPHS=1, 655,360 window,
  OVERHEAD_GIB=10): exact 12/12, needles correct at 9.9k / 94k / 314k tokens, decode prose 29.8-30.4, code 39.9-40.1
  tok/s (3 runs each), the same as without CP; but spark3 fell to 3.30 GiB during the 314k prefill: too thin a margin
  for a default until the prompt path's peak memory comes down.
- Fixes from those boots: the memory guard no longer ends its own ssh command or holds its parent's lock files;
  prompt chunks run the routed experts in 1024-row blocks (the universal kernels' grouping fits 48 KiB of shared
  memory).
