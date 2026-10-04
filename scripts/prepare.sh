#!/usr/bin/env bash
# Prepare the three Sparks to serve GLM-5.3 (MODEL_ID) with TensorFold:
#   1. preflight: docker and the GPU on every Spark, key-based ssh to both workers, the CX7 links, disk space
#   2. the image on every Spark: this release's prebuilt image pulled from GHCR_IMAGE (PULL=0 skips that), else built
#      there FROM the GLM-5.3-Flash recipe's published image (TensorFold v0.6.0 with
#      patches 0001-0068, pulled by digest when a Spark lacks it) plus this recipe's later patches, a layer of a few
#      hundred KB; or, with FLASH_IMAGE= (empty), from BASE_IMAGE with every patch (pip install of TensorFold)
#   3. the checkpoint on the head (HF_CACHE): served as it is when its snapshot is complete there, else downloaded
#      (~273 GiB, resumable; with the hf CLI, else from inside the image), keeping KEEP_FREE_GB free
#   4. the checkpoint checked: its config read by the patched engine, every file of its index present
#   5. each worker reads the head's HF_CACHE over NFS (required; nothing is copied): a read-only docker volume
#      NFS_VOLUME on the worker, mounted from its NFS server (the head's address on that worker's cable), checked file
#      by file (names and sizes) against the head's snapshot
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# --rebuild rebuilds the image on every Spark. DRY_RUN is start.sh's (prepare.sh changes things: it has none).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."     # the repository root
source ./scripts/config.sh
source ./scripts/nodes.sh

REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

# ---------------------------------------------------------------- 1. preflight
mkdir -p "$KERNEL_CACHE" "$STATE_DIR" "$HF_CACHE/hub"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running; wait for it: pgrep -af prepare.sh"
log "Preflight checks on all 3 Sparks"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
nvidia-smi -L >/dev/null 2>&1 || warn "nvidia-smi failed on this node: is the NVIDIA driver working?"
check_workers
for i in $(worker_ids); do
  need_worker "$i"
  h=$(worker_host "$i")
  worker "$i" 'docker info >/dev/null 2>&1' || die "worker $i ($h) cannot talk to its docker daemon (docker group?)"
  worker "$i" 'nvidia-smi -L >/dev/null 2>&1' || warn "nvidia-smi failed on worker $i ($h)"
done
detect_links
for i in $(worker_ids); do
  log "Link to worker $i ($(worker_host "$i")): head ${LINK_HEAD_ADDR[i]:-?} <-> ${LINK_WORKER_ADDR[i]:-?}; NFS from $(nfs_server "$i")"
done
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
worker_free_gb() { worker "$1" "df -BG --output=avail '$2' | tail -1 | tr -dc '0-9'"; }

# ---------------------------------------------------------------- 2. the image, built on each Spark
HASH=$(image_hash)
FAST=0
if [[ -n "$FLASH_IMAGE" ]]; then
  [[ "$(flash_hash)" == "$FLASH_PATCHES" ]] ||
    die "patches 0001-0068 are not the GLM-5.3-Flash recipe's v1.4 patches (hash $(flash_hash), its image's $FLASH_PATCHES): build with FLASH_IMAGE= (empty) instead"
  FAST=1
fi
CTX=$(mktemp -d); trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/patches"
if (( FAST )); then
  for p in patches/*.patch; do [[ "$(basename "$p")" < "0100" ]] || cp "$p" "$CTX/patches/"; done
  cat > "$CTX/Dockerfile" <<DOCKERFILE
FROM $FLASH_IMAGE
COPY patches /opt/tf-patches-full
RUN cd "\$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \\
    for p in /opt/tf-patches-full/*.patch; do echo "applying \$p"; patch -p0 --forward < "\$p" || exit 1; done && \\
    python -c "import tensorfold.cuda.server, tensorfold.families.glm_moe_dsa, tensorfold.families.glm5_next.cuda.engine"
LABEL tf.patches=$HASH tf.base=flash-$FLASH_PATCHES
DOCKERFILE
else
  cp patches/*.patch "$CTX/patches/"
  cat > "$CTX/Dockerfile" <<DOCKERFILE
FROM $BASE_IMAGE
RUN pip install --no-cache-dir --upgrade "git+${TF_REPO}@${TF_VERSION}" && pip install --no-cache-dir $IMAGE_EXTRAS && tensorfold --version
COPY patches /opt/tf-patches
RUN cd "\$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \\
    for p in /opt/tf-patches/*.patch; do echo "applying \$p"; patch -p0 --forward < "\$p" || exit 1; done && \\
    python -c "import tensorfold.cuda.server, tensorfold.families.glm_moe_dsa, tensorfold.families.glm5_next.cuda.engine"
ENV HF_HOME=/root/.cache/huggingface TORCH_EXTENSIONS_DIR=/cache/torch_extensions TRITON_CACHE_DIR=/cache/triton
WORKDIR /workspace
LABEL tf.patches=$HASH tf.base=$BASE_IMAGE
DOCKERFILE
fi
nocache=(); (( REBUILD )) && nocache=(--no-cache)
built_here() { [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null)" == "$HASH" ]]; }
built_there() { [[ "$(worker "$1" docker image inspect -f "'{{index .Config.Labels \"tf.patches\"}}'" "$IMAGE" 2>/dev/null)" == "$HASH" ]]; }
label_of() { docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$1" 2>/dev/null; }
prebuilt=$(prebuilt_image)            # the pinned digest (config.sh's IMAGE_TAG / IMAGE_DIGEST), else the hash's tag
if (( ! REBUILD )) && [[ "${PULL:-1}" == 1 ]] && ! built_here; then
  log "Pulling the prebuilt image $prebuilt (~25 GB; PULL=0 builds instead)"
  if docker pull "$prebuilt" >/dev/null && [[ "$(label_of "$prebuilt")" == "$HASH" ]]; then
    docker tag "$prebuilt" "$IMAGE"; log "Using $prebuilt as $IMAGE"
  else
    warn "could not pull $prebuilt (no image for these patches, the package is not public, or no network): building it"
  fi
fi
if (( REBUILD )) || ! built_here; then
  if (( FAST )); then
    docker image inspect "$FLASH_IMAGE" >/dev/null 2>&1 || { log "Pulling $FLASH_IMAGE (the GLM-5.3-Flash recipe's image, ~25 GB)"; docker pull "$FLASH_IMAGE"; }
  else
    (( $(free_gb "$(docker info -f '{{.DockerRootDir}}')") >= 35 )) || die "an image build from $BASE_IMAGE needs ~35 GB under Docker's root"
  fi
  log "Building $IMAGE here (patches $HASH$( (( FAST )) && echo ", on the Flash recipe's image"))"
  tar -C "$CTX" -cf - . | docker build "${nocache[@]}" -q -t "$IMAGE" - >/dev/null
fi
log "Image $IMAGE (patches $HASH) here"
for i in $(worker_ids); do
  h=$(worker_host "$i")
  if (( ! REBUILD )) && built_there "$i"; then log "Image $IMAGE (patches $HASH) on worker $i"; continue; fi
  if (( ! REBUILD )) && [[ "${PULL:-1}" == 1 ]] && worker "$i" docker pull "$prebuilt" >/dev/null 2>&1 &&
     [[ "$(worker "$i" docker image inspect -f "'{{index .Config.Labels \"tf.patches\"}}'" "$prebuilt" 2>/dev/null)" == "$HASH" ]]; then
    worker "$i" docker tag "$prebuilt" "$IMAGE"
    built_there "$i" && { log "Using $prebuilt as $IMAGE on worker $i ($h)"; continue; }
  fi
  if (( FAST )); then
    if ! worker "$i" docker image inspect "$FLASH_IMAGE" >/dev/null 2>&1; then
      root=$(worker "$i" "docker info -f '{{.DockerRootDir}}'" 2>/dev/null || echo /var/lib/docker)
      (( $(worker_free_gb "$i" "$root") >= 35 )) || die "worker $i ($h) needs ~35 GB under $root to pull $FLASH_IMAGE"
      log "Pulling $FLASH_IMAGE on worker $i ($h) from the registry (~25 GB, over its own network, not the CX7 links)"
      worker "$i" docker pull "$FLASH_IMAGE" >/dev/null || die "worker $i ($h) could not pull $FLASH_IMAGE"
    fi
  fi
  log "Building $IMAGE on worker $i ($h)"
  tar -C "$CTX" -cf - . | worker "$i" docker build "${nocache[@]}" -q -t "$IMAGE" - >/dev/null ||
    die "the image build failed on worker $i ($h)"
  built_there "$i" || die "worker $i's $IMAGE is not labelled with patches $HASH after the build"
done

# ---------------------------------------------------------------- 3. the checkpoint on the head
dir=$(model_cache_dir "$MODEL_ID")
complete() {  # <snapshot folder>: every file the index names is there, and the config
  local snap=$1
  [[ -f "$snap/config.json" && -f "$snap/model.safetensors.index.json" ]] || return 1
  python3 - "$snap" <<'PY'
import json, os, sys
snap = sys.argv[1]
files = set(json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"].values())
sys.exit(0 if all(os.path.isfile(os.path.join(snap, f)) for f in files) else 1)
PY
}
command -v hf >/dev/null || warn "host 'hf' CLI not found: downloads run from inside the image"
download() {  # <repo id> <revision or empty>: into HF_CACHE/hub, the standard cache layout, resumable
  if command -v hf >/dev/null; then
    hf download "$1" ${2:+--revision "$2"} --cache-dir "$HF_CACHE/hub" >/dev/null
  else                                 # owned by this user, as the host CLI's files would be
    docker run --rm --user "$(id -u):$(id -g)" --network host --entrypoint python ${HF_TOKEN:+-e HF_TOKEN} \
      -v "$HF_CACHE":/hf -e HF_HOME=/hf -e HOME=/tmp "$IMAGE" -c \
      'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2] or None)' "$1" "$2"
  fi
}
rev=$(snapshot_rev "$MODEL_ID")
if [[ -n "$rev" ]] && complete "$dir/snapshots/$rev"; then
  log "Checkpoint: $MODEL_ID @ ${rev:0:12} complete in $HF_CACHE (nothing to download)"
else
  have=$(free_gb "$HF_CACHE")
  (( have >= 280 + KEEP_FREE_GB )) ||
    die "only ${have} GB free under $HF_CACHE: the download needs ~280 GB and leaves KEEP_FREE_GB=$KEEP_FREE_GB free"
  log "Downloading $MODEL_ID${MODEL_REVISION:+ @ ${MODEL_REVISION:0:12}} into $HF_CACHE/hub (~273 GiB, resumes if interrupted)"
  download "$MODEL_ID" "$MODEL_REVISION" ||
    die "$MODEL_ID: the download failed (a gated repository: accept its terms on Hugging Face and log in with hf auth login)"
  [[ -z "$MODEL_REVISION" || -f "$dir/refs/main" ]] || { mkdir -p "$dir/refs"; printf %s "$MODEL_REVISION" > "$dir/refs/main"; }
  rev=$(snapshot_rev "$MODEL_ID")
  complete "$dir/snapshots/$rev" || die "$MODEL_ID: the snapshot is incomplete after the download"
fi
SNAP="$dir/snapshots/$rev"
if [[ "$DRAFTER" == dspark ]]; then              # the DSpark speculator, in the same cache (the workers read it over NFS)
  ddir=$(model_cache_dir "$DSPARK_ID")
  drev=${DSPARK_REVISION:-$(cat "$ddir/refs/main" 2>/dev/null)}
  if [[ -n "$drev" && -f "$ddir/snapshots/$drev/model.safetensors" ]]; then
    log "Drafter: $DSPARK_ID @ ${drev:0:12} in $HF_CACHE (nothing to download)"
  else
    log "Downloading the drafter $DSPARK_ID${DSPARK_REVISION:+ @ ${DSPARK_REVISION:0:12}} (~2.4 GiB)"
    download "$DSPARK_ID" "$DSPARK_REVISION" ||
      die "$DSPARK_ID: the download failed"
  fi
fi

# ---------------------------------------------------------------- 4. check it (CPU only, no GPU)
log "Checking the checkpoint with the patched engine's config reader"
docker run --rm --network none --memory 4g --entrypoint python -v "$SNAP:/ckpt:ro" "$IMAGE" -c '
import sys
from tensorfold.families import glm_moe_dsa
from tensorfold.families.glm5_next.cuda.weights import Config
glm_moe_dsa.check("/ckpt")
c = Config.read("/ckpt")
assert c.full and c.layers == 78 and c.quant == "exl3", (c.full, c.layers, c.quant)
n = sum(k == "full" for k in c.index_kinds)
print(f"GLM-5.3: {c.layers} layers + MTP, {c.experts} routed experts, {n} indexer layers")
' > "$STATE_DIR/check.log" 2>&1 || { tail -5 "$STATE_DIR/check.log" >&2; die "the patched engine cannot read $SNAP"; }
grep -E '^GLM-5.3:' "$STATE_DIR/check.log" | sed 's/^/  /' 

# ---------------------------------------------------------------- 5. NFS on each worker
manifest=$(cd "$SNAP" && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort)
sub=${SNAP#"$HF_CACHE"/}
[[ "$sub" != "$SNAP" ]] || die "the snapshot $SNAP is not under HF_CACHE ($HF_CACHE), which the workers mount"
for i in $(worker_ids); do
  h=$(worker_host "$i"); server=$(nfs_server "$i")
  ensure_nfs_volume "$i"
  if ! out=$(worker_nfs "$i" true 2>&1); then
    die "worker $i ($h) cannot mount :$NFS_PATH from $server over NFS ($NFS_VOLUME): $(tail -1 <<<"$out")
    The head must export $NFS_PATH (HF_CACHE: $HF_CACHE; with NFS_PATH=/ the export's fsid=0 root) read-only to that worker's address on its cable (README: Weights over NFS)"
  fi
  # NFS_PATH is the export's path as the client sees it: / for an fsid=0 root that is HF_CACHE, else HF_CACHE itself
  rel=$sub; [[ "$NFS_PATH" == / ]] || rel=${SNAP#"$NFS_PATH"/}
  have=$(worker_nfs "$i" find -L "/hf/$rel" -type f -printf '%P %s\n' 2>/dev/null | LC_ALL=C sort || true)
  if [[ "$have" != "$manifest" ]]; then
    warn "first differences (< head, > worker $i):"; diff <(printf '%s\n' "$manifest") <(printf '%s\n' "$have") | head -10 >&2 || true
    die "worker $i ($h) does not see $MODEL_ID @ ${rev:0:12} over NFS ($NFS_VOLUME: :$NFS_PATH from $server)"
  fi
  log "Worker $i ($h) reads $MODEL_ID @ ${rev:0:12} from the head over NFS ($NFS_VOLUME from $server): $(wc -l <<<"$manifest") files, identical"
done

prepared_state > "$PREPARED_MARKER"
# ---------------------------------------------------------------- 6. the CUDA kernels, built now on every Spark
# A first start would otherwise compile them while the weights load: the compilers' few GB at the tightest moment
# (a smaller admitted window, or a rank under the memory guard). Skipped on a Spark where the server runs.
KCACHE=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
[[ "$KCACHE" =~ ^[0-9a-f]{12}$ ]] || KCACHE=$(image_hash)
PREBUILD_PY='
import importlib, sys, time
mods = ["tensorfold.cuda.kernels.qmm", "tensorfold.cuda.exl3.experts", "tensorfold.cuda.exl3.mpe",
        "tensorfold.cuda.exl3.prompt_experts", "tensorfold.cuda.roce", "tensorfold.families.glm5_next.cuda.msa",
        "tensorfold.families.glm5_next.cuda.glue", "tensorfold.families.glm5_next.cuda.l2pf",
        "tensorfold.families.glm5_next.cuda.exl3_mm"]
t = time.time(); built = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        if hasattr(mod, "_ext"):
            mod._ext(); built.append(m.rsplit(".", 1)[1])
    except Exception as e:
        print(f"prebuild: {m}: {type(e).__name__}: {e}", file=sys.stderr)
print(f"kernels ready ({", ".join(built)}) in {time.time() - t:.0f}s")
'
prebuild() {  # <node> (0: here, else a worker)
  local n=$1 out
  if [[ "$n" == 0 ]]; then
    if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then log "Kernels here: the server runs; built at its next start"; return; fi
    mkdir -p "$KERNEL_CACHE/$KCACHE"
    out=$(docker run --rm --gpus all -e MAX_JOBS=4 -v "$KERNEL_CACHE/$KCACHE":/cache --entrypoint python "$IMAGE" -c "$PREBUILD_PY" 2>&1 | tail -1)
  else
    if worker "$n" "docker ps --format '{{.Names}}'" | grep -qx "$CONTAINER_NAME"; then log "Kernels on worker $n: the server runs; built at its next start"; return; fi
    out=$(worker "$n" "mkdir -p \$HOME/.cache/tensorfold-glm53-full/$KCACHE && docker run --rm --gpus all -e MAX_JOBS=4 -v \$HOME/.cache/tensorfold-glm53-full/$KCACHE:/cache --entrypoint python $IMAGE -c $(printf '%q' "$PREBUILD_PY")" 2>&1 | tail -1)
  fi
  log "Kernels on $( [[ "$n" == 0 ]] && echo here || echo "worker $n"): $out"
}
# every Spark at once (each builds its own kernels: ~3.5 min each, not ~10 one after another)
prebuild 0 &
for i in $(worker_ids); do prebuild "$i" & done
wait

log "Done: all 3 Sparks are ready. Start the server with ./start.sh (port $PORT)."
log "The first start compiles CUDA kernels for GB10 (a few minutes); they are cached in $KERNEL_CACHE/<image hash> on each Spark."
