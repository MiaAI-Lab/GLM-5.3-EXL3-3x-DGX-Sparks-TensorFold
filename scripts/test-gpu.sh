#!/usr/bin/env bash
# The GPU tests (tests/gpu) on this Spark's GPU: the new kernels' rows keep their bits on the hardware, the tiny
# checkpoint through every real kernel (the universal EXL3 experts included) against the reference, drafted windows
# against serial steps, CUDA graphs against eager steps, and one layer of the real checkpoint's mixed-width experts.
# Small (a few GiB); never while the server runs on this Spark: it refuses then.
#   scripts/test-gpu.sh [pytest args]     CHECKPOINT=<snapshot folder> (default: the configured snapshot)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing: run scripts/prepare.sh first"
[[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != true ]] ||
  die "$CONTAINER_NAME runs on this Spark: stop it first (./stop.sh)"
ck=${CHECKPOINT:-$(model_cache_dir "$MODEL_ID")/snapshots/$(snapshot_rev "$MODEL_ID")}
KCACHE=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE")
mkdir -p "$KERNEL_CACHE/$KCACHE"
args=(--rm --gpus all --ipc=host --memory "${MEM:-24g}" -v "$PWD/tests:/tests:ro" -w /tests
      -v "$KERNEL_CACHE/$KCACHE:/cache")
[[ -d "$ck" ]] && args+=(-v "$ck:/ckpt:ro" -e CHECKPOINT=/ckpt)
exec docker run "${args[@]}" --entrypoint bash "$IMAGE" -c \
  'pip install -q --no-cache-dir pytest==8.3.5 >/dev/null 2>&1 || { echo "cannot install pytest (no network?)"; exit 2; }; exec python -m pytest -q -p no:cacheprovider gpu "$@"' _ "$@"
