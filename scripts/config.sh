# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `CONTEXT=32768 ./start.sh`, `PULL=0 scripts/prepare.sh`
#   2. scripts/local.sh (this setup's own values, above all WORKER and WORKER2; sourced as bash), then ./.env
#      (KEY=value lines, read, never run): both are yours, not the repository's; where both set a key, local.sh wins
#   3. the defaults below
_cfg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$_cfg_root/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _n; do _cfg_env[$_n]=${!_n}; done < <(compgen -e)
  source "$_cfg_root/scripts/local.sh"
  # the environment wins over local.sh: put back any variable it had that local.sh changed
  for _n in "${!_cfg_env[@]}"; do [[ "${!_n-}" == "${_cfg_env[$_n]}" ]] || export "$_n=${_cfg_env[$_n]}"; done
  unset _cfg_env
fi
if [[ -f "$_cfg_root/.env" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < "$_cfg_root/.env"
fi
unset _n _line _key _value

# The Sparks: this machine serves rank 0 and the API and exports the checkpoint over NFS; WORKER (rank 1) and WORKER2
# (rank 2) are ssh targets (key-based), each joined to the others by a direct CX7 cable (a triangle, one subnet per
# cable). Set them in scripts/local.sh (scripts/local.sh.example).
TP=3
WORKER="${WORKER:-}"                 # e.g. user@<rank 1's address>
WORKER2="${WORKER2:-}"               # e.g. user@<rank 2's address>
FABRIC_PEER="${FABRIC_PEER:-}"; FABRIC_PEER2="${FABRIC_PEER2:-}"   # a worker's CX7 address, when reached otherwise
WORKER_HF_CACHE="${WORKER_HF_CACHE:-}"; WORKER_HF_CACHE2="${WORKER_HF_CACHE2:-}"   # (unused: workers read over NFS)
MASTER_PORT="${MASTER_PORT:-29561}"  # TensorFold's rendezvous port between the ranks
# The rendezvous address (rank 0's, --master): this node's LAN address (what its hostname resolves to), which every
# worker reaches. SOCKET_IFNAME: the netdev of NCCL's bootstrap socket on every node (default: each node's
# default-route netdev).
_ma=$(getent ahostsv4 "$(hostname)" 2>/dev/null | awk '$1 !~ /^127\./ {print $1; exit}' || true)
[[ -n "$_ma" ]] || _ma=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
MASTER_ADDR="${MASTER_ADDR:-$_ma}"
SOCKET_IFNAME="${SOCKET_IFNAME:-}"

# The checkpoint: GLM-5.3 (zai-org's full model) with EXL3 routed experts of mixed widths (2.75 bits on average a
# weight) and BF16 elsewhere, ~273 GiB. MODEL_REVISION pins a Hugging Face commit (empty: the cache's refs/main, or
# the Hub's main when first downloaded). A checkpoint already in HF_CACHE at that revision is served as it is.
MODEL_ID="${MODEL_ID:-Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold}"
MODEL_REVISION="${MODEL_REVISION-}"
# DRAFTER: dspark (default: RedHatAI's DSpark speculator for GLM-5.3, glm-5.3 license, ~2.4 GiB download, ~0.3 GiB a
# Spark as 4-bit copies; up to 7 drafts a round cut by its learned confidence; code 39.5-40.4 tok/s vs 37.1 with the
# MTP head, prose the same, 3 runs each) or mtp (the checkpoint's MTP head drafts). Drafts only propose: replies are
# the same either way.
DRAFTER="${DRAFTER:-dspark}"
DSPARK_ID="${DSPARK_ID:-RedHatAI/GLM-5.3-speculator.dspark}"
DSPARK_REVISION="${DSPARK_REVISION-b374b95663447ea0e935151be4f3d6666e36e6d7}"
# DSpark drafts up to 8 a round: its verify window of 9 rows also replays a captured CUDA graph
[[ "$DRAFTER" == dspark ]] && export TF_GLM_WIDE_GRAPHS="${TF_GLM_WIDE_GRAPHS:-9}"

# The image: TensorFold v0.6.0 with patches/*.patch (0001-0068: the GLM-5.3-Flash recipe's v1.4 patches, the engine
# this recipe extends; 0100 on: full GLM-5.3). prepare.sh builds it on every Spark FROM the GLM-5.3-Flash recipe's
# published image (FLASH_IMAGE: TensorFold v0.6.0 with exactly patches 0001-0068, pulled by digest), applying the
# patches past them: a layer of a few hundred KB, nothing big crosses the Sparks' links. FLASH_IMAGE= (empty) builds
# from BASE_IMAGE instead (pip install of TensorFold, every patch).
TF_VERSION="${TF_VERSION:-v0.6.0}"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
FLASH_IMAGE="${FLASH_IMAGE-ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529}"
FLASH_PATCHES="5e01f1bb74d8"          # its tf.patches label: the hash of patches 0001-0068 and IMAGE_EXTRAS
IMAGE="${IMAGE:-tensorfold-glm53-full:${TF_VERSION}}"
IMAGE_EXTRAS="av==18.1.0 xgrammar>=0.2.8,<0.3"
image_hash() { (cat patches/*.patch 2>/dev/null; echo "$IMAGE_EXTRAS") | sha256sum | cut -c1-12; }
flash_hash() { (cat patches/00[0-6][0-9]-*.patch 2>/dev/null; echo "$IMAGE_EXTRAS") | sha256sum | cut -c1-12; }
CONTAINER_NAME="${CONTAINER_NAME:-glm53-full-tf}"   # the same name on every Spark

SERVED_NAME="${SERVED_NAME:-GLM-5.3-EXL3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# Prompt + reply window. Provisional default (see README "Memory"): start.sh asks scripts/memplan.py whether every
# rank keeps FLOOR_GIB free with it, and refuses to start otherwise, naming what fits.
CONTEXT="${CONTEXT:-163840}"
# Concurrent requests (patch 0134): 1 (default) decodes one request at a time, others queue; 2-4 decode that many
# together, each with its own window of CONTEXT tokens (so the KV pool holds PARALLEL x CONTEXT). Not with CP=1.
PARALLEL="${PARALLEL:-1}"
# The DSA latent cache: fp8 (default: e4m3 rows with a power-of-two scale each; the rotary keys and the indexer's keys
# stay bf16, as DeepSeek's FP8 MLA cache keeps its rotary part): 56.1 KiB a token on every rank instead of bf16's
# 94.4, so ~1.7x the window in the same memory. Lossy against bf16; drafted replies still equal serial ones under
# either. bf16: the exact cache. fp4: e2m1 codes with an e4m3 scale a block of 16 (~0.58x fp8's latent bytes; more
# lossy: see the README's quality numbers before using it).
KV="${KV:-fp8}"
export TF_GLM_KV="$KV"
# The non-expert BF16 weights: q4 (default: the projections as 4-bit groups of 64 with MSE-searched ranges, the head
# FP8, kv_b BF16; lossy), fp8 (~2.8 GiB more a rank) or bf16 (as stored; ~9 GiB more a rank: does not fit with a
# useful window).
DENSE="${DENSE:-q4}"
export TF_GLM_DENSE="$DENSE"
# The checkpoint's MTP head drafts (exact: drafted replies equal "draft": false serial ones). 0: leave it out (~1.8
# GiB less a rank). Default: off with DRAFTER=dspark (the speculator drafts every round: as fast with or without the
# MTP head beside it, 3 runs each, 2026-10-03), on with DRAFTER=mtp.
MTP="${MTP:-$([[ "${DRAFTER:-dspark}" == dspark ]] && echo 0 || echo 1)}"
export TF_GLM_MTP="$MTP"
# Rows a prompt chunk runs at once: 2048 (default) or 1024 (~1 GiB less a rank, slower prompts).
PREFILL_ROWS="${PREFILL_ROWS:-2048}"
export TF_GLM_PREFILL_ROWS="$PREFILL_ROWS"
# Prompt chunks' rows split between the ranks (exact): each rank adds the residual and runs the next norm on its own
# third of the rows, so a link carries a third of a partial and of the normed rows instead of whole partials.
# PREFILL_SPLIT: 1 (default) or 0 (whole-partial all-gathers); PREFILL_OVERLAP: 1 (default; the exchanges on a second
# CUDA stream in row pieces) or 0.
# CP: context parallelism (1: each Spark keeps every third token's caches: ~3x the window; exact; one request at a
# time, no kept prompt states, eager decode windows for now) or 0 (default: every Spark keeps every token).
CP="${CP:-0}"
export TF_GLM_CP="$CP"
PREFILL_SPLIT="${PREFILL_SPLIT:-1}"
PREFILL_OVERLAP="${PREFILL_OVERLAP:-1}"
[[ "$PREFILL_SPLIT" == 1 ]] || PREFILL_OVERLAP=0
export TF_GLM_HC_SPLIT="$PREFILL_SPLIT" TF_GLM_PREFILL_OVERLAP="$PREFILL_OVERLAP"
THINKING="${THINKING:-1}"
# The reply budget of a request that sets no max_tokens (reasoning and answer together); cut to what the window has
# left, never refused.
MAX_TOKENS="${MAX_TOKENS:-32768}"
# The ranks' all-gathers: roce (default: the small ones, up to TF_ROCE_MAX_KB, as one-shot RDMA writes over the
# Sparks' cables, the Flash recipe's patch 0006 with its multi-Spark routes; decode +10% over nccl at 3 runs each,
# prose 30.8 vs 27.6-28.1, code 37.1 vs 33.0-34.5 tok/s, the same replies, 2026-10-03) or nccl (every gather).
COMM="${COMM:-roce}"
export TF_GLM_COMM="$COMM"
export TF_ROCE_MAX_KB="${TF_ROCE_MAX_KB:-512}"
# Prompt-lookup ("copy") drafts verified ahead of the MTP head's when the reply repeats earlier text (exact).
COPY="${COPY:-1}"
export TF_GLM_COPY_DRAFTS="$COPY"
COPY_MAX="${COPY_MAX:-15}"
export TF_GLM_COPY_MAX="$COPY_MAX"
# Conversations sharing a system prompt reuse its prompt state (same replies).
SHARED_PREFIX="${SHARED_PREFIX:-1}"
export TF_GLM_SHARED_PREFIX="$SHARED_PREFIX"
# Earlier turns keep their reasoning in the prompt, as in zai-org's current template (1: drop it).
export TF_GLM_CLEAR_THINKING="${TF_GLM_CLEAR_THINKING:-0}"
# Smooth streaming (the Flash recipe's patch 0061): tokens leave one event each at a steady pace from a playout
# buffer of STREAM_SMOOTH_MS; 0: one event a round.
STREAM_SMOOTH="${STREAM_SMOOTH:-1}"
STREAM_SMOOTH_MS="${STREAM_SMOOTH_MS:-400}"
export TF_GLM_STREAM_SMOOTH="$STREAM_SMOOTH" TF_GLM_STREAM_SMOOTH_MS="$STREAM_SMOOTH_MS"
# Sampled replies: the top_p nucleus is checked on every rank's candidates together (TENSORFOLD_NUCLEUS_UNION=1),
# the same draws as reading whole vocabulary shards (reply hashes equal on boots fp4g655nu vs fp4g655b), ~1% faster.
export TENSORFOLD_NUCLEUS_UNION="${TENSORFOLD_NUCLEUS_UNION:-1}"
# Other conversations' kept prompt states (the snapshots' saved rows): at most KV_POOL_GIB (TF_GLM_CACHE_GIB) and
# TF_GLM_CACHE_ENTRIES of them.
KV_POOL_GIB="${KV_POOL_GIB:-1}"
export TF_GLM_CACHE_GIB="$KV_POOL_GIB"
export TF_GLM_CACHE_ENTRIES="${TF_GLM_CACHE_ENTRIES:-8}"
# Memory (README "Memory"). A GB10 that runs out of memory freezes instead of failing, so:
#   FLOOR_GIB     the least MemAvailable any Spark may be left with under the largest prompt (10; at least 4, and
#                 above GUARD_KILL_GIB: under 8 a spike may reach the guard, which stops that rank)
#   OVERHEAD_GIB  what a rank uses past TensorFold's own startup estimate (CUDA context, NCCL, compiled kernels, the
#                 Python process: ~7 GiB at idle, ~3 more during startup's graph capture and calibration; measured
#                 2026-10-03); scripts/memplan.py adds it
#   MEMORY_RESERVE_GIB  TensorFold's own startup reserve (TENSORFOLD_MEMORY_RESERVE_GIB): its admission refuses a
#                 window that leaves less; FLOOR_GIB + OVERHEAD_GIB by default, so the two checks agree
#   GUARD         1 (default): scripts/memguard.sh watches MemAvailable on every Spark every 0.5 s while the server
#                 runs, logs its low-water mark, and stops the rank there below GUARD_KILL_GIB (3) (a failed rank
#                 beats a frozen Spark)
FLOOR_GIB="${FLOOR_GIB:-4}"
OVERHEAD_GIB="${OVERHEAD_GIB:-10}"
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-$(awk -v f="$FLOOR_GIB" -v o="$OVERHEAD_GIB" 'BEGIN { print f + o }')}"
export TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB"
GUARD="${GUARD:-1}"
GUARD_KILL_GIB="${GUARD_KILL_GIB:-3}"

export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
# Every worker reads the head's HF_CACHE over NFS (required: the checkpoint is never copied to a worker). The head
# exports NFS_PATH (default: HF_CACHE) read-only to each worker's address on its cable; prepare.sh creates the
# read-only docker volume NFS_VOLUME on each worker (no sudo there), mounted from NFS_SERVER (rank 1) / NFS_SERVER2
# (rank 2): by default the head's address on that worker's cable (each worker its own subnet).
WORKER_WEIGHTS=nfs
NFS_PATH="${NFS_PATH:-$HF_CACHE}"
NFS_SERVER="${NFS_SERVER:-}"; NFS_SERVER2="${NFS_SERVER2:-}"
NFS_VOLUME="${NFS_VOLUME:-glm53-full-hf}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-glm53-full}"   # compiled CUDA kernels, a folder per image hash
STATE_DIR="${STATE_DIR:-$HOME/.local/state/glm53-full-tensorfold}"   # this recipe's locks, setup marker, guard logs
# Server logs: each rank's container log, gzipped, as <date>-<time>-rank<N>.log.gz in LOG_DIR here and in
# ~/.cache/tensorfold-glm53-full/logs on each worker, the newest LOG_KEEP kept (0: none).
LOG_DIR="${LOG_DIR:-$HOME/.cache/tensorfold-glm53-full/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
# Disk: prepare.sh keeps KEEP_FREE_GB free under HF_CACHE after a download (~280 GB for this checkpoint) and asks
# IMAGE_FREE_GB under Docker's root for an image build on each Spark.
KEEP_FREE_GB="${KEEP_FREE_GB:-100}"
IMAGE_FREE_GB="${IMAGE_FREE_GB:-10}"

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

model_cache_dir() { local id=${1:-$MODEL_ID}; echo "$HF_CACHE/hub/models--${id//\//--}"; }
model_revision() { [[ "$1" == "$MODEL_ID" ]] && echo "$MODEL_REVISION" || true; }
# snapshot_rev <id>: the snapshot this setup serves: the pin, else what refs/main names on this Spark
snapshot_rev() { local rev; rev=$(model_revision "$1"); [[ -n "$rev" ]] || rev=$(cat "$(model_cache_dir "$1")/refs/main" 2>/dev/null); echo "$rev"; }

# What scripts/prepare.sh last left ready on every Spark (it writes this line to PREPARED_MARKER when it succeeds);
# start.sh runs prepare.sh again whenever the current line differs: an image missing or different on any Spark, new
# patches, another checkpoint or revision, other workers or NFS servers.
PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local line i
  line="model=$MODEL_ID@$(snapshot_rev "$MODEL_ID") patches=$(image_hash) image=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)"
  for i in 1 2; do
    line+=" worker$i=$(worker_host "$i") image$i=$(worker "$i" docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing) nfs$i=$(wval NFS_SERVER "$i")"
  done
  echo "$line"
}
