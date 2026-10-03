# GLM-5.3 EXL3 on three DGX Sparks with TensorFold

Serve **GLM-5.3** (zai-org's full model: 78 layers of MLA with DeepSeek sparse attention, 256 routed experts, an MTP
head) from three NVIDIA DGX Sparks with [TensorFold](https://github.com/ashhart/TensorFold), as an OpenAI-compatible
API on port 8888. The checkpoint is Mia's AI Lab's EXL3 quantization,
[`Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold`](https://huggingface.co/Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold):
routed experts at 2.75 bits a weight on average (each expert at its own width, 2 to 4 bits), everything else BF16,
~273 GiB in all, ~86 GiB on each Spark.

> **Status (2026-10-03):** on three DGX Sparks: drafted replies equal serial ones and concurrent ones equal serial
> ones (12/12 at 32k and 64k windows); long-prompt recall correct at 7.9k-58.2k tokens. Decode 28.3 tok/s (prose) and
> 33.0 (code) for one request. Prefill ~290 tok/s (a 15,763-token prompt in 55 s, boot p3; 131 tok/s before patch
> 0105). Default window 163,840 tokens. Numbers name their boots.

## What it runs

TensorFold v0.6.0 with the patches in `patches/`, applied with `patch -p0` in filename order:

- **0001-0068**: the [GLM-5.3-Flash recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)'s v1.4 patches, unchanged: TensorFold's
  GLM CUDA engine across N ranks (0066-0068: three Sparks), its latent MLA and DSA kernels, exact MTP drafting, tool
  calls, the server's features.
- **0100-0104**: full GLM-5.3 (`glm_moe_dsa`) on that engine:
  - `0100-glm-full-layout`: the 3-rank split: attention heads 22/21/21, every expert's 16 Hadamard blocks of 128
    columns 6/5/5, the extra block **rotating by layer** (each Spark holds a third of the 233 GiB of routed experts
    instead of rank 0 holding 6/16), the shared expert's extra block one rank further on.
  - `0101-glm-full-weights`: the loader: EXL3 experts of mixed widths through TensorFold's universal grouped kernels
    (one device buffer a layer), kv_b's key rows zero-padded over the rotary dims (so the latent absorb kernels apply
    unchanged, exactly), the indexer on the "full" layers only, the embedding split by vocabulary rows like the head.
  - `0102-glm-full-dsa`: interleaved RoPE on the MLA's 64 rotary dims (a separate bf16 key plane beside the latent
    cache, as DeepSeek's FP8 MLA cache keeps its rotary part), DeepSeek-V3.2's indexer with per-token keys (RoPE on
    their first 64 dims, LayerNorm), and the top-2048 token selection.
  - `0103-glm-full-forward`: plain pre-norm residuals, the indexer's top-k reused on "shared" layers, the MoE with
    mixed-width experts, the MTP head, CUDA graphs a token bucket, kept-prompt snapshots.
  - `0104-glm-full-engine`: the `glm_moe_dsa` family, the startup memory estimate of these caches and buffers.

Replies are exact in TensorFold's sense: a drafted reply (MTP and prompt-lookup drafts) equals the `"draft": false`
serial reply, and sending requests together does not change any reply. One request decodes at a time; others queue.

## Requirements

- Three DGX Sparks (GB10, 128 GB unified memory each), joined by a **triangle of direct CX7 cables** (one subnet
  per cable), and one network all three share for ssh and NCCL's bootstrap.
- Docker with the NVIDIA runtime on each; key-based ssh from the head (rank 0) to both workers.
- The head holds the checkpoint (~273 GiB in its Hugging Face cache, plus 100 GB left free) and **exports that cache
  read-only over NFS to both workers** (below). The workers need no copy of the weights, only ~25 GB for the image.

## Quick start

```bash
cp scripts/local.sh.example scripts/local.sh     # WORKER=user@<rank 1>, WORKER2=user@<rank 2>, NFS settings
./start.sh                                       # prepare, check the memory plan, start, smoke test
./stop.sh
DRY_RUN=1 ./start.sh                             # print the memory plan and every rank's docker command only
```

`./start.sh` runs `scripts/prepare.sh` when something is missing: it builds the image on each Spark (from the
GLM-5.3-Flash recipe's published image, adding only patches 0100 on), downloads the checkpoint when the head lacks it,
and checks that each worker sees every file of it over NFS. Then it checks every rank's memory plan, starts the memory
guard on every Spark, starts ranks 2 and 1 on the workers and rank 0 on the head, and waits for the API.

## Weights over NFS

The workers never hold a copy of the checkpoint: each rank reads its share of every tensor from the head's Hugging
Face cache over NFS (the head's NFS server on the cable to that worker). The head exports its cache read-only, e.g.
with an NFSv4 root (`/etc/exports`, one line per cable subnet):

```
/home/<user>/.cache/huggingface 10.0.22.0/24(ro,fsid=0,no_subtree_check) 10.0.23.0/24(ro,fsid=0,no_subtree_check)
```

and `NFS_PATH=/` in `scripts/local.sh` (the export's root as the workers mount it; default: the cache's own path).
`prepare.sh` creates a read-only docker volume (`NFS_VOLUME`) on each worker (no sudo there), mounted from the head's
address on that worker's cable (`NFS_SERVER` / `NFS_SERVER2` override it), and compares each worker's view of the
snapshot, file by file with sizes, with the head's. `start.sh` refuses to start when a worker's mount is missing or does
not show the checkpoint.

## Memory

On a DGX Spark the GPU and the CPU share one 128 GB pool (~121 GiB visible), and **a Spark that runs out of it
freezes instead of failing**. So the recipe plans memory before it starts anything, and watches it while it runs:

- **Per rank at TP=3** (TensorFold's own estimate, from the checkpoint's headers, `scripts/memplan.py`): weights
  ~85.3 GiB on rank 0, ~84.8 on ranks 1 and 2 (q4 dense weights, MTP head included); the decode, MTP and prompt
  buffers ~2.3 GiB; the caches **56.1 KiB a token with the default FP8 latent cache** (94.4 KiB in bf16: per layer
  a 512-wide latent row and a 64-wide rotary key, per indexer layer a 128-wide key; the same on every rank).
- **What TensorFold does not count** (the CUDA context, NCCL, compiled kernels, the Python process; `OVERHEAD_GIB`,
  10: ~7 GiB measured at idle and ~3 more during startup) and a **floor** of free memory every Spark keeps under the largest prompt (`FLOOR_GIB`,
  4 by default; at least 4 and above the guard's 3).
- `start.sh` reads every Spark's MemAvailable, runs `scripts/memplan.py` with them, and **refuses to start** when a
  rank would fall below the floor, naming the rank and what to lower. TensorFold's own admission uses the same margin
  (`TENSORFOLD_MEMORY_RESERVE_GIB` = floor + overhead).
- **The guard** (`GUARD=1`, default): `scripts/memguard.sh` on every Spark samples MemAvailable every 0.5 s while the
  server runs, keeps the low-water mark, and stops that Spark's rank below `GUARD_KILL_GIB` (3): a failed rank beats a
  frozen Spark.

Measured (MemAvailable in GiB on spark1 / spark2 / spark3, fp8 KV, q4 dense, MTP on; boots of 2026-10-03):

| window | at start | TensorFold estimate | idle | lowest under the longest prompt | used past the estimate |
|---|---|---|---|---|---|
| 32,768 (boot b1n) | 116.2 / 117.7 / 111.0 | 90.6 / 89.4 / 89.4 | 18.8 / 22.0 / 15.9 | 18.2 / 21.7 / 15.0 (28.3k-token prompt) | 6.2-8.5 |
| 65,536 (boot b2) | 115.4 / 117.8 / 111.0 | 91.1 / 90.7 / 90.7 | 16.7 / 20.1 / 13.4 | 16.7 / 20.1 / 13.3 (58.2k-token prompt) | |

| 163,840 (boot p3, defaults) | 115.2 / 117.2 / 111.8 | 96.5 / 96.1 / 96.1 | 11.6 / 14.0 / 7.4 | 11.5 / 14.0 / 3.9 (startup) | ~7 idle, ~3 more at startup |

spark3 (rank 2; it also runs other containers) is the tightest: at the default window it keeps ~7.4 GiB at idle and
dips to ~3.9 GiB during startup (graph capture and calibration), just above the guard. 196,608 tokens took it under the
guard at startup (the guard stopped that rank; nothing froze). `DRY_RUN=1 CONTEXT=<n> ./start.sh` prints the plan of
any window with the Sparks' memory of the moment.

## Settings

From the environment, `scripts/local.sh` or `./.env`; every default and its reason is in `scripts/config.sh`.

| Setting | Default | |
|---|---|---|
| `CONTEXT` | 163840 | prompt + reply window (boot p3; the memory plan checks every start) |
| `KV` | fp8 | the latent cache as e4m3 rows with a power-of-two scale each (rotary and index keys stay bf16); `bf16`: exact, 1.7x the bytes |
| `DENSE` | q4 | non-expert weights as 4-bit groups (head FP8, kv_b BF16); `fp8`, `bf16` take more memory |
| `MTP` | 1 | MTP drafts (exact); 0 saves ~1.8 GiB a Spark |
| `PREFILL_ROWS` | 2048 | prompt chunk rows; 1024 saves ~1 GiB a Spark |
| `COPY`, `COPY_MAX` | 1, 15 | prompt-lookup drafts for replies that repeat earlier text (exact) |
| `SHARED_PREFIX` | 1 | conversations sharing a system prompt reuse its state |
| `KV_POOL_GIB` | 1 | other conversations' kept prompt states |
| `THINKING`, `MAX_TOKENS` | 1, 32768 | thinking on; the reply budget of a request without max_tokens |
| `FLOOR_GIB`, `OVERHEAD_GIB`, `GUARD`, `GUARD_KILL_GIB` | 4, 10, 1, 3 | memory (above) |
| `COMM` | nccl | `roce`: the small all-gathers over RoCE (not yet measured with this model) |
| `PORT`, `SERVED_NAME` | 8888, GLM-5.3-EXL3 | the API |

Sampling defaults come from the checkpoint's `generation_config.json` (temperature 1.0, top_p 0.95), as in the
GLM-5.3-Flash recipe; a request's own values win.

## Tests

- `scripts/test-cpu.sh` (no GPU): the new kernels against transformers' formulas; the full forward of a tiny random
  GLM-5.3 checkpoint in the real layout (mixed-width EXL3 experts, indexer and shared layers, MTP) on one rank and on
  three against an independent reference written from transformers' modelling code, in bf16 and FP8 caches; drafted
  windows against serial steps bit for bit; the memory formulas against what the engine allocates, at the real
  dimensions on every rank (`CHECKPOINT=<snapshot>` adds the tests that read the real config and headers). Triton's
  interpreter runs the kernels on CPU tensors.
- `scripts/test-gpu.sh` (one Spark's GPU, a few GiB): the same kernels' rows keep their bits on the GPU, the tiny
  checkpoint through every real kernel, CUDA graphs against eager steps, one layer of the real checkpoint's experts
  against a float64 reference.
- With the server up: `tools/exact.py` (drafted == serial, concurrent == serial), `tools/needle.py`,
  `tools/toolcheck.py`.

## Maintainers

`tools/make-patches.sh <TensorFold checkout>` writes `patches/0100-*.patch` on from a development branch whose
commits are named after the patches, and checks that all of `patches/` on a fresh v0.6.0 gives that branch's tree.

## License and credits

Apache License 2.0 for this project's own work ([`LICENSE`](LICENSE)); [`NOTICE`](NOTICE) has the third-party
notices, [`CREDITS.md`](CREDITS.md) everyone this builds on. The checkpoint is under the GLM-5.3 License (Z.ai), which
ships with it. Made by Mia's AI Lab.
