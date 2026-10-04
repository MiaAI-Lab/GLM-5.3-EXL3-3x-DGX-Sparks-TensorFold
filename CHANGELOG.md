# Changelog

## v1.0 (2026-10-04)

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
- Decode: RoCE one-shot all-gathers by default (COMM=roce, +10%), Red Hat AI's DSpark speculator by default
  (DRAFTER=dspark, 0114, MTP head off; the drafter adapted from vllm-project/speculators and vLLM): code 39.5-40.4,
  prose 29.1-30.2 tok/s (boot p17, sampled, 3 runs each; MTP + NCCL: 33.0-34.5 / 27.6-28.1), the same replies (exact
  checks 12/12).
- Context parallelism (CP=1, 0112-0113; the scheme after drowzeys' TensorFold fork, our own kernels): each Spark keeps
  every third token's caches, exact top-k selection across the Sparks, attention partials merged by log-sum-exp in
  rank order. Boot p19 (450k window, fp8): exact 12/12, needles correct to 235,660 tokens, prefill ~250 tok/s; boot
  cp655 (655,360 window, fp8, eager decode): prose 23.8-25.1, code 29.6-30.1 tok/s.
- 4-bit latent cache (KV=fp4, 0115): e2m1 codes, an e4m3 scale per 16 values and a power-of-two row scale, 304 bytes
  a row (fp8: 528). First quality probe against fp8 (greedy): 30 short questions 29/30 (fp8 27/30), 27/30 replies
  identical, 8-key recall 8/8 at 64k and 128k. The full paired evaluation (README "Quality") found no measurable difference from fp8; fp4 is the long-context setting.
- Draft costs measured on text, the verify profile fixed (0116, 0119); DSpark sampling filter (0117); static scratch
  for captured CP decode windows (0118); CP windows within the top-k skip selection (0120); a RoCE timeout inside a
  replayed graph is reported (0121).
- A stopped request (client gone, stop string) ends on every rank within a round instead of decoding to max_tokens
  (0122, the GLM-5.3-Flash recipe's fix for its issue #38); CPU test `tests/cpu/test_stop_vote.py`.
- `tools/quality.py`: a fixed quality suite with paired comparison; results of the 4-bit KV cache in the README.
- DSpark draft dumps and an offline simulator (0123): `TF_GLM_DRAFT_DUMP` records the drafter's view at every
  position, `dspark_sim` replays proposal policies on it exactly (same drafts as the engine) and scores them against
  the reply.
- `TF_GLM_MEM_TRACE=1` logs every rank's memory after each prompt chunk; the KV line names the actual format (0124).
- DSpark's draft selection ~8 ms faster a round with the same drafts (0125): its Markov rows are read through NumPy
  instead of a torch gather (7.8 -> 0.12 ms a greedy chain of 8 slots on the host).
- Sampled replies check the top_p nucleus on all ranks' candidates together by default (`TENSORFOLD_NUCLEUS_UNION=1`):
  the same tokens, ~1% faster.
- Captured decode windows under CP (TF_GLM_CP_GRAPHS=1, off by default) cost ~2.9 GiB on each Spark. At 655k with fp8
  that left spark3 under the memory guard's 3 GiB during a long prompt: the guard stopped it and the other ranks
  waited (the "hang" seen with graphs on). With fp4, boot fp4g655 (KV=fp4 CP=1 TF_GLM_CP_GRAPHS=1, 655,360 window,
  OVERHEAD_GIB=10): exact 12/12, needles correct at 9.9k / 94k / 314k tokens, decode prose 29.8-30.4, code 39.9-40.1
  tok/s (3 runs each), the same as without CP; but spark3 fell to 3.30 GiB during the 314k prefill: too thin a margin
  for a default until the prompt path's peak memory comes down.
- Fixes from those boots: the memory guard no longer ends its own ssh command or holds its parent's lock files;
  prompt chunks run the routed experts in 1024-row blocks (the universal kernels' grouping fits 48 KiB of shared
  memory).
- `KV=fp4x` (0133, 0137; opt-in): fp4's latent rows plus e4m3 rotary and indexer keys, 31,476 bytes a token a Spark
  against fp4's 39,072 (~24% more context in the same memory, ~618k instead of 499,712 under CP). Quality suite
  against fp4 (boot fp4x500): equal on short questions, arithmetic and Python tasks, ledger tracking 31/40 against
  35/40 (p 0.34, not significant). `fp4` stays the long-context setting.
- Concurrent requests with CUDA graphs (0138 segmented kernels a stream, 0139 graphs of the batched windows): boot
  par2g 37.8 tok/s for two requests together (eager 35.5-35.9), par4g 48.2 tok/s for four (12.1-14.0 each); exact
  12/12 and the same reply hashes alone and together. GPU tests 21/21 (multi kernels, multi graphs, fp4x).
- Copy drafts checked by DSpark (0140, `COPY_HYBRID=1` by default): +8.4% on prose edits, +3.6% on JSON edits, 0 on
  plain text (boots cpyon / cp500b, every reply identical with it on and off).
- Tried and not adopted: 12 variants of the decode expert kernels (fused epilogues, one prep launch, 4-tile CTAs,
  higher occupancy): all bit-identical, none faster than ~1% on 3-9-row windows (GPU bench, 2026-10-04).
