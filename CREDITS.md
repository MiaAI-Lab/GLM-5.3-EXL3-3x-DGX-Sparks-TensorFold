# Credits

This repository is a thin layer of scripts and patches. Almost everything that makes it work was built by others.
Its own work is licensed under the Apache License 2.0 ([`LICENSE`](LICENSE)); [`NOTICE`](NOTICE) carries the
third-party notices that go with it.

## Model

- **[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)** by [Z.ai](https://huggingface.co/zai-org) (BF16 weights:
  [`zai-org/GLM-5.3-BF16`](https://huggingface.co/zai-org/GLM-5.3-BF16)): the model's design, training and
  evaluations. The GLM-5.3 License (in the checkpoint) governs any use of the weights. The weights are not part of
  this repository; `scripts/prepare.sh` downloads them from Hugging Face.
- **Mia's AI Lab**: the EXL3 checkpoint,
  [`Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold`](https://huggingface.co/Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold) (a
  derivative of GLM-5.3, under the GLM-5.3 License), made with a fork of
  [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT).
- **[Red Hat AI](https://huggingface.co/RedHatAI)**: the DSpark speculator,
  [`RedHatAI/GLM-5.3-speculator.dspark`](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark) (glm-5.3 license,
  as its model card states), the default drafter (`DRAFTER=dspark`). Downloaded from its source by
  `scripts/prepare.sh`, never redistributed here.

## Inference engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  TensorFold contributors (Apache 2.0 from v0.6.0; code written before v0.6.0 keeps its MIT notice): the engine that
  serves the model. This recipe extends its GLM-5.3-Flash CUDA engine (the two-rank engine, its latent MLA and DSA
  kernels, MTP drafting with exact verification, the OpenAI-compatible server). Every file in `patches/` modifies
  TensorFold v0.6.0.
- TensorFold's universal EXL3 expert kernels (`tensorfold/cuda/exl3/experts`, a bit width per expert) serve the
  checkpoint's mixed widths. Routing GLM's experts through them follows upstream commit 8613488 by Andrey Kolesnikov
  ([akol1](https://github.com/akol1)), "glm5_next: route EXL3 experts through the universal path for any bit width"
  (TensorFold v0.6.1, Apache 2.0); patch 0101 does the same for full GLM-5.3 on v0.6.0.
- TensorFold itself builds on, and credits in its
  [third-party notices](https://github.com/ashhart/TensorFold/blob/v0.6.0/THIRD_PARTY_NOTICES.md):
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp, MIT), whose EXL3 format the routed experts are
  stored in (TensorFold's EXL3 kernels, and the prompt kernels here built on them, read that format), the GLM
  modelling code in Hugging Face
  [transformers](https://github.com/huggingface/transformers) (Apache 2.0), and
  [z-lab/dflash](https://github.com/z-lab/dflash) (Z Lab, MIT), whose DFlash2 architecture its drafter ports (the
  DSpark drafter here reuses those DFlash2 kernels).

## Patches 0100 on (this recipe's: full GLM-5.3)

- `0100-glm-full-layout` to `0104-glm-full-engine`: full GLM-5.3 (`glm_moe_dsa`) on the GLM-5.3-Flash engine. The
  model's arithmetic (MLA with interleaved RoPE, the DSA indexer with its interleaved RoPE and LayerNorm'd keys, the
  "shared" indexer layers that reuse the last full layer's top-k, the sigmoid router, the MTP layer) follows the
  modelling code of Hugging Face [transformers](https://github.com/huggingface/transformers) (Apache 2.0:
  `GlmMoeDsa`, `DeepseekV32Indexer`, `DeepseekV3Attention`), which the CPU tests' reference forward re-implements; no
  code is copied from it. The sparse-attention design is DeepSeek-V3.2's
  ([DeepSeek-AI](https://github.com/deepseek-ai)); keeping the rotary keys in bf16 beside an FP8 latent cache follows
  its FP8 MLA cache. Design only: no DeepSeek code is used.
- The top-k selection (`dsa_full._select`) and score tiles adapt TensorFold's GLM-5.3-Flash pooled-key kernels
  (`sparse.py`, the radix select and its tie rule) to per-token keys.
- `0106-glm-full-prompt-experts`: the routed experts of prompt chunks run the EXL3 prompt kernels
  (`tensorfold/cuda/exl3/prompt_experts.*`) by **[drowzeys](https://github.com/drowzeys)**, from their TensorFold fork
  ([drowzeys/TensorFold](https://github.com/drowzeys/TensorFold), branch `glm-moe-dsa-tp4`, commit befd47d, Apache
  2.0), written for their [GLM-5.3 TP=4 recipe](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark)
  (its scripts MIT; none of them are used here). Their kernels' design is itself adapted from the GLM-5.3-Flash
  recipe's prompt kernels (patches 0001, 0004, 0020), as their header says. Added here: a deterministic "pairs" mode
  (each slot's output row stored, no atomics), so a prompt gives the same bits every run; the extension renamed, the
  trellis row stride taken from the expert width, an fp16 buffer that may be lent. `TF_GLM_PROMPT_EXPERTS=pe` selects
  them; the default since 0108 is our own kernel (below), which still falls back to theirs for layers it does not
  support.
- `0108-glm-full-mia-prompt-experts`, `0110-glm-full-mia-sparse-attention`: Mia's AI Lab's own prompt kernels (the
  routed experts with one input rotation a layer; sparse attention over the FP8 latent cache, one block a row), built
  on TensorFold's EXL3 decode primitives (after ExLlamaV3, MIT).
- `0112-glm-full-cp-caches`, `0113-glm-full-context-parallel` (`CP=1`): the context-parallel scheme (tokens
  interleaved over the ranks, local candidates merged into the global top-k, attention partials merged by
  log-sum-exp in rank order) follows drowzeys' TensorFold fork
  ([drowzeys/TensorFold](https://github.com/drowzeys/TensorFold), branch `glm-moe-dsa-tp4`, commit befd47d, Apache
  2.0: `glm_moe_dsa/cuda/fused.py`'s `select`, `_attn_dcp`, `_merge_lse`, `_dcp_combine`); the code and kernels (FP8
  rows, separate rotary plane, exact top-k merge, fp32 partials) are ours.
- `0114-glm-full-dspark` (`DRAFTER=dspark`, the default; the drafter, `dspark.py`, arrives with patch 0113, and 0117
  adds its sampling filter): drafts with Red Hat AI's DSpark speculator (above). The drafter's forward, Markov bias
  and confidence head are adapted from [vllm-project/speculators](https://github.com/vllm-project/speculators)
  (commit 36a19ca) and [vLLM](https://github.com/vllm-project/vllm) (commit 5f30fc7) (both Apache 2.0); its file
  header lists the files and functions. The kernels and their tensor-parallel split reuse TensorFold's DFlash2
  drafter (after z-lab/dflash, MIT); the drafter's Qwen3-style layers are written anew on them.
- `0122` (a stopped request ends on every rank within a round): the GLM-5.3-Flash recipe's `0070-glm-serial-stop`
  (Mia's AI Lab), ported to this recipe.
- The recipe's scripts (`start.sh`, `stop.sh`, `scripts/`) and tools are adapted from the GLM-5.3-Flash recipe's
  (Mia's AI Lab, Apache 2.0).

## Patches 0001-0068 (the GLM-5.3-Flash recipe's v1.4 patches, carried unchanged)

These come from Mia's AI Lab's
[GLM-5.3-Flash recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) (Apache 2.0); its
credits for them carry over here:

- `0003-glm-vision`: GLM's image and video processors (resize with pad, 2 fps frame choice, prompt layout) and vision
  tower, checked bit for bit against Hugging Face [transformers](https://github.com/huggingface/transformers) 5.17
  (Apache 2.0), the reference they follow; builds on TensorFold's Qwen image pipeline.
- `0006-cuda-roce-allgather`: the one-shot RoCE all-gather (`COMM=roce`) is the "RoCEnante" transport of
  **[b12x](https://github.com/local-inference-lab/b12x)** by local-inference-lab (Apache 2.0): its C proxy
  (`roce_proxy.c`), modified for more than two Sparks (per-peer routes over up to 4 network cards; its header lists the
  changes), and its CuTe all-gather kernel reimplemented in CUDA C++ (`roce.cu`).
- `0012-glm-kda-chunked`, `0014-glm-kda-chunked-gb10`, `0039-glm-kda-chunked-kernel` (`KDA_CHUNKED`, on by default):
  the chunked WY / UT form of the delta-rule recurrence follows the published chunkwise algorithm of gated delta
  networks and Kimi Delta Attention, as implemented in
  [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT); the kernels are written anew.
- `0013-glm-decode-rounds`: verify windows of up to 16 rows so copy drafts can run long, after the wider copy windows
  proposed in [TensorFold PR #115](https://github.com/ashhart/TensorFold/pull/115) by Ash Hart.
- `0036-glm-tool-calls` (tool calling for agent clients): the rule that tool calls written inside the think block
  count when they end the reply, the store that gives back a step's reasoning to clients that drop it, and null /
  `const` typing of tool arguments are adapted from patch 0620 of
  [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) (Apache 2.0); the rest of
  that patch's fixes are our own implementation.
- `0037-cuda-tokenize`: the `/tokenize` and `/detokenize` endpoints take and return what
  [vLLM](https://github.com/vllm-project/vllm)'s (Apache 2.0) do, so clients written for vLLM work unchanged;
  the code is our own.
- `0043-glm-visible-pools`: the idea of bounding decode token selection to the pools a row can see follows
  [TensorFold PR #140](https://github.com/ashhart/TensorFold/pull/140) by mikolaj92, which TensorFold v0.6.0 has for
  its selection kernel; this patch's bound on the decode scoring and the split selection is our own.
- `0044-cuda-context-errors`: TensorFold v0.6.0's context-window refusals (commit 68c6e35 by Ash Hart) extended: the
  `param` field, and the same wording and `context_length_exceeded` code for the GLM app's window check and image
  prompts.
- `0045-cuda-metrics`: TensorFold v0.6.0's Prometheus `/metrics` (commit 4447ac3, #110, by Ash Hart) extended with
  `/health`'s figures as `tensorfold_health:` metrics.
- `0046-glm-l2-prefetch` (L2 prefetch in decode, `TF_GLM_L2PF`): adapted from patch 0460 of
  [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) (Apache 2.0); changes listed
  in `NOTICE`.
- `0047-glm-exl3-decode-loads` (routed-expert decode loads, `TF_GLM_EXL3_LOADS`): adapted from patch 0580 of
  [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) (Apache 2.0); changes listed
  in `NOTICE`.
- `0054-glm-image-prompt-reuse` (image prompts resume from kept prompt states, the Flash recipe's issue #11): by
  [abhicnv007](https://github.com/abhicnv007), applied as contributed, with two small review changes.
- `0057-server-thinking-alias` (`chat_template_kwargs.thinking`): from [Alexbob0](https://github.com/Alexbob0)'s
  pull request #25 to the Flash recipe; the `{"type": ...}` forms and the refusal of other values were added here.
- `0058-server-client-gone-poll`: [TensorFold PR #218](https://github.com/ashhart/TensorFold/pull/218) by
  [jayleaton](https://github.com/jayleaton) (Apache 2.0), applied unchanged: the client-gone check sees descriptors
  past 1023.
- `0059-server-refused-bodies`: TensorFold v0.6.1's fix for #181 (commit 50dfe38a, by
  [SxMShaDoW](https://github.com/SxMShaDoW)), backported to v0.6.0.
- `0060-glm-keep-thinking` (earlier turns keep their reasoning, `TF_GLM_CLEAR_THINKING`): by
  [kky42](https://github.com/kky42), the Flash recipe's pull request #23.
- Every patch, except the parts credited above: by Mia's AI Lab, developed with
  [Claude Code](https://claude.com/claude-code), under the Apache License 2.0; the TensorFold code the patches modify or
  quote as context stays under TensorFold's licenses (Apache 2.0, and MIT for code written before v0.6.0; see
  `NOTICE`).

## Runtime stack

- **[NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)**
  (`nvcr.io/nvidia/pytorch:26.07-py3`), the base of the image, with NVIDIA's CUDA, cuDNN, cuBLAS, NCCL and related
  libraries. Governed by the NVIDIA Software License Agreement and the Product-Specific Terms for NVIDIA AI Products.
- **[PyTorch](https://pytorch.org/)** (BSD-3-Clause): tensors, CUDA streams and the C++ extension builder that compiles
  the patches' CUDA kernels.
- **[Triton](https://github.com/triton-lang/triton)** (MIT): the language many of TensorFold's CUDA kernels are
  written in.
- **[NCCL](https://github.com/NVIDIA/nccl)** (BSD-3-Clause) and **[rdma-core](https://github.com/linux-rdma/rdma-core)**
  (libibverbs, GPL-2.0 / BSD-2-Clause): the ranks' exchanges over the Sparks' ConnectX-7 RoCE links.
- **[PyAV](https://github.com/PyAV-Org/PyAV)** (BSD-3-Clause) and **[FFmpeg](https://ffmpeg.org/)** (LGPL): video
  decoding. **[Pillow](https://python-pillow.org/)** (MIT-CMU): image decoding.
- **[xgrammar](https://github.com/mlc-ai/xgrammar)** (Apache 2.0): structured outputs (`response_format` and the
  `guided_*` fields).
- **[Hugging Face Hub](https://huggingface.co/)**: model hosting, the `hf` CLI and `huggingface_hub` (Apache 2.0), and
  the [safetensors](https://github.com/huggingface/safetensors) format (Apache 2.0) the checkpoint ships in.
- **[Docker](https://www.docker.com/)** and the
  **[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** (Apache 2.0): running the server
  on the GPU in a container.

## Hardware

- **[NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** (GB10 Grace Blackwell,
  128 GB unified memory), three of them in a triangle of ConnectX-7 cables: every number in the README is measured there.

If you believe something here is missing or credited wrongly, please open an issue.
