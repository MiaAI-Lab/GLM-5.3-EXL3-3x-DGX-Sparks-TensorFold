#!/usr/bin/env bash
# The CPU tests (tests/cpu): the patched engine's new kernels and its full GLM-5.3 forward on a tiny checkpoint, on
# one, three ranks, against a reference written from transformers' modelling code; drafted windows against serial
# steps bit for bit; the memory estimate against what the engine allocates. Triton's interpreter runs the kernels on
# CPU tensors: no GPU, in a memory-capped container of the image (MEM, default 6g; pytest is pip-installed into it).
#   scripts/test-cpu.sh [pytest args]       CHECKPOINT=<snapshot folder> adds the tests on its config and headers
#   TF_SRC=<patched TensorFold src/> scripts/test-cpu.sh   tests that tree instead of the image's
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing: run scripts/prepare.sh first"
args=(--rm --memory "${MEM:-6g}" --cpus "${CPUS:-6}" -v "$PWD/tests:/tests:ro" -w /tests)
[[ -z "${CHECKPOINT:-}" ]] || args+=(-v "$CHECKPOINT:/ckpt:ro" -e CHECKPOINT=/ckpt)
[[ -z "${TF_SRC:-}" ]] || args+=(-v "$TF_SRC:/tf-src:ro" -e TF_SRC=/tf-src)
# pytest is not in the image: a throwaway layer adds it
exec docker run "${args[@]}" --entrypoint bash "$IMAGE" -c \
  'pip install -q --no-cache-dir pytest==8.3.5 >/dev/null 2>&1 || { echo "cannot install pytest (no network?)"; exit 2; }; exec python -m pytest -q -p no:cacheprovider cpu "$@"' _ "$@"
