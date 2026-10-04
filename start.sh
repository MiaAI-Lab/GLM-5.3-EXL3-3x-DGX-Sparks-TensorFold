#!/usr/bin/env bash
# Serve GLM-5.3 EXL3 (MODEL_ID) with TensorFold on three DGX Sparks, end to end: runs scripts/prepare.sh when the
# image or the checkpoint is not ready on every Spark, checks every rank's memory plan (scripts/memplan.py) and refuses
# to start when a Spark would be left with less than FLOOR_GIB, starts the memory guard on every Spark, then rank 2
# and rank 1 on the workers and rank 0 here (the API on port 8888), waits until the API answers and runs a smoke test.
# Stop it with ./stop.sh.
#
# Usage: ./start.sh [restart] [extra tensorfold serve args]
#   ./start.sh                         # scripts/config.sh's defaults: the long-context mode, a 499,712-token window
#                                      # (context parallelism, fp4 KV cache), DSpark drafts, one request at a time
#                                      # (if the server already runs, says so and leaves it alone)
#   ./start.sh restart                 # stop all three ranks (./stop.sh), then start them again, e.g. to apply changed
#                                      # settings or patches; the new arguments are checked before stopping
#   CP=0 ./start.sh restart            # without context parallelism: a 163,840-token window, FP8 KV cache
#   KV=fp4x CONTEXT=618496 ./start.sh restart
#                                      # opt-in: ~24% more context than fp4 (README "More context")
#   CP=0 PARALLEL=2 CONTEXT=65536 ./start.sh restart   # two requests decoded together (up to 4; needs CP=0)
#   CONTEXT=32768 ./start.sh restart   # another window (the memory plan checks it first)
#   DRY_RUN=1 ./start.sh               # print the memory plan and every rank's docker command; change nothing
# Extra arguments go to every rank after the defaults, so they win (the last value of a flag counts).
# Setup: WORKER and WORKER2 in scripts/local.sh (scripts/local.sh.example), key-based ssh; the head exports its
# Hugging Face cache over NFS to both workers (README "Weights over NFS").
# Settings, from the environment, scripts/local.sh or ./.env (defaults and their reasons in scripts/config.sh):
#   serving  CONTEXT, PARALLEL, KV, DENSE, DRAFTER, MTP, CP, PREFILL_ROWS, PREFILL_SPLIT, PREFILL_OVERLAP, COPY, COPY_MAX, COPY_HYBRID, DISK_CACHE, DISK_CACHE_GIB,
#            SHARED_PREFIX, STREAM_SMOOTH, STREAM_SMOOTH_MS, KV_POOL_GIB, MAX_TOKENS, THINKING, COMM, SERVED_NAME,
#            HOST, PORT
#   memory   FLOOR_GIB, OVERHEAD_GIB, MEMORY_RESERVE_GIB, GUARD, GUARD_KILL_GIB
#   nodes    WORKER, WORKER2, FABRIC_PEER, FABRIC_PEER2, MASTER_ADDR, MASTER_PORT, SOCKET_IFNAME, NCCL_CHANNELS,
#            NCCL_DEBUG
#   files    MODEL_ID, MODEL_REVISION, HF_CACHE, NFS_PATH, NFS_SERVER, NFS_SERVER2, NFS_VOLUME, KERNEL_CACHE,
#            STATE_DIR, LOG_DIR, LOG_KEEP, HF_HUB_OFFLINE=0 (let TensorFold reach the Hub; default: the local cache)
#   image    IMAGE, TF_VERSION, TF_REPO, BASE_IMAGE, FLASH_IMAGE, CONTAINER_NAME
#   setup    PREPARE (auto | 1 | 0), KEEP_FREE_GB, IMAGE_FREE_GB; WAIT_TIMEOUT (seconds, 2400); DRY_RUN=1;
#            FOREGROUND=1 (stay attached to rank 0's log, exit with its code); STOP_TIMEOUT (stop.sh)
#   ranks    every TENSORFOLD_*, TF_GLM_* and TF_ROCE_* variable goes to every rank
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh
source ./scripts/nodes.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2400}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help) usage; exit 0 ;;
esac
for arg in "$@"; do [[ "$arg" == -h || "$arg" == --help ]] && { usage; exit 0; }; done

DRAFT_ARG=none
if [[ "$DRAFTER" == dspark ]]; then              # DSpark: its snapshot, the pin or the cache's refs/main
  DSPARK_REV=${DSPARK_REVISION:-$(cat "$(model_cache_dir "$DSPARK_ID")/refs/main" 2>/dev/null)}
  DSPARK_SUB="hub/models--${DSPARK_ID//\//--}/snapshots/$DSPARK_REV"
  DRAFT_ARG="/root/.cache/huggingface/$DSPARK_SUB"
fi
SERVE_ARGS=(--context "$CONTEXT" --max-tokens "$MAX_TOKENS" --drafter "$DRAFT_ARG")
[[ "$PARALLEL" =~ ^[1-4]$ ]] || die "PARALLEL is 1 to 4, not $PARALLEL"
(( PARALLEL > 1 && CP == 1 )) && die "PARALLEL=$PARALLEL needs CP=0 (context parallelism, the default, serves one request at a time): e.g. CP=0 PARALLEL=$PARALLEL CONTEXT=65536 ./start.sh restart"
(( PARALLEL > 1 )) && SERVE_ARGS+=(--parallel "$PARALLEL")
[[ "$DENSE" =~ ^(bf16|fp8|q4)$ ]] || die "DENSE is bf16, fp8 or q4, not $DENSE"
[[ "$COMM" =~ ^(nccl|roce)$ ]] || die "COMM is nccl or roce, not $COMM"
[[ "$KV" =~ ^(bf16|fp8|fp4|fp4x)$ ]] || die "KV is bf16, fp8, fp4 or fp4x, not $KV"
[[ "$MTP" =~ ^[01]$ ]] || die "MTP is 0 or 1, not $MTP"
[[ "$CONTEXT" =~ ^[0-9]+$ && "$CONTEXT" -ge 4096 && "$CONTEXT" -le 1048576 ]] || die "CONTEXT is a token count from 4096 to 1048576, not $CONTEXT"
[[ "$PREFILL_ROWS" =~ ^(1024|2048|3072)$ ]] || die "PREFILL_ROWS is 1024, 2048 or 3072, not $PREFILL_ROWS"
[[ "$PREFILL_SPLIT" =~ ^[01]$ && "$PREFILL_OVERLAP" =~ ^[01]$ ]] || die "PREFILL_SPLIT and PREFILL_OVERLAP are 0 or 1"
[[ "$CP" =~ ^[01]$ ]] || die "CP is 0 or 1"
[[ "$DRAFTER" =~ ^(mtp|dspark)$ ]] || die "DRAFTER is mtp or dspark, not $DRAFTER"
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "MAX_TOKENS is a token count, not $MAX_TOKENS"
for v in COPY COPY_HYBRID DISK_CACHE SHARED_PREFIX STREAM_SMOOTH GUARD; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$COPY_MAX" =~ ^([1-9]|1[0-5])$ ]] || die "COPY_MAX is 1 to 15, not $COPY_MAX"
for v in KV_POOL_GIB DISK_CACHE_GIB FLOOR_GIB OVERHEAD_GIB MEMORY_RESERVE_GIB GUARD_KILL_GIB; do
  [[ "${!v}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "$v is a number of GiB, not ${!v}"
done
awk -v f="$FLOOR_GIB" -v k="$GUARD_KILL_GIB" 'BEGIN { exit !(f >= 4 && f > k) }' ||
  die "FLOOR_GIB is at least 4 and above GUARD_KILL_GIB ($GUARD_KILL_GIB) (a GB10 that runs out of memory freezes), not $FLOOR_GIB"
check_workers
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
SERVE_ARGS+=("$@")
arg_value() {  # the effective value of a flag (its last occurrence, as --flag value or --flag=value)
  local flag=$1 value="" i
  for (( i = 0; i < ${#SERVE_ARGS[@]}; i++ )); do
    case "${SERVE_ARGS[i]}" in
      "$flag") value="${SERVE_ARGS[i + 1]:-}" ;;
      "$flag="*) value="${SERVE_ARGS[i]#*=}" ;;
    esac
  done
  echo "$value"
}
API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" == *:* ]] && API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"

running_here()   { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
running_worker() { [[ "$(worker "$1" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
wname() { echo "rank $1 ($(worker_host "$1"))"; }
served_name() {
  curl -s --max-time 5 "$URL/v1/models" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null
}

# ---------------------------------------------------------------- banner and progress
B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
echo
banner
printf '\n%s  Mia'"'"'s TensorFold Start Script%s\n' "$M" "$R"
printf '%s  %s · 3 x DGX Spark · %s at once · %s-token window · %s KV%s · %s drafts · port %s%s\n\n' "$D" "$MODEL_ID" \
  "$PARALLEL" "$(arg_value --context)" "$KV" "$( [[ "$CP" == 1 ]] && echo " · context parallel")" "$DRAFTER" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$KERNEL_CACHE" "$STATE_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
need_workers
(( DRY )) && log "DRY_RUN=1: printing the memory plan and the docker commands; nothing is stopped, started or prepared"

all_running() { local i; running_here || return 1; for i in $(worker_ids); do running_worker "$i" || return 1; done; }
if (( ! DRY )) && [[ "$MODE" == start ]] && all_running; then
  log "$CONTAINER_NAME is already running on all 3 Sparks (model: $(served_name || echo "not answering yet"), port $PORT): nothing to do."
  log "Use ./start.sh restart to restart it (e.g. with new settings), or ./stop.sh to stop it."
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
step 1 "Setup: image and checkpoint on all 3 Sparks"
if [[ "${PREPARE:-auto}" == 1 || ( "${PREPARE:-auto}" != 0 && "$(prepared_state 2>/dev/null)" != "$(cat "$PREPARED_MARKER" 2>/dev/null)" ) ]]; then
  if (( DRY )); then log "DRY_RUN: scripts/prepare.sh would run now (not ready yet)"
  else
    log "Not ready yet: running scripts/prepare.sh (the first time this builds the image on every Spark, downloads ~273 GiB and sets up NFS for the workers)"
    ./scripts/prepare.sh
  fi
elif [[ "${PREPARE:-auto}" == 0 ]]; then
  log "PREPARE=0: setup skipped (the image and the checkpoint are checked below)"
else
  log "Ready: $IMAGE (patches $(image_hash)) and $MODEL_ID on all 3 Sparks"
fi
why="scripts/prepare.sh did not"; [[ "${PREPARE:-auto}" == 0 ]] && why="PREPARE=0 skipped scripts/prepare.sh, which would"
(( DRY )) && why="DRY_RUN: scripts/prepare.sh would"
image_ok=1
docker image inspect "$IMAGE" >/dev/null 2>&1 || { (( DRY )) && { warn "image $IMAGE missing: $why build it"; image_ok=0; } || die "image $IMAGE missing: $why build it"; }
for i in $(worker_ids); do
  [[ -z "${WORKER_DOWN[$i]:-}" ]] || continue
  worker "$i" docker image inspect "$IMAGE" >/dev/null 2>&1 || { (( DRY )) && warn "image $IMAGE missing on $(wname "$i"): $why build it there"; } ||
    die "image $IMAGE missing on $(wname "$i"): $why build it there"
done
KCACHE=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
[[ "$KCACHE" =~ ^[0-9a-f]{12}$ ]] || KCACHE=$(image_hash)
REV=$(snapshot_rev "$MODEL_ID")
[[ -n "$REV" ]] || die "$MODEL_ID not in $HF_CACHE: $why download it"
SUB="hub/models--${MODEL_ID//\//--}/snapshots/$REV"
SNAP="$HF_CACHE/$SUB"
[[ -f "$SNAP/config.json" ]] || die "$MODEL_ID @ ${REV:0:12} not in $HF_CACHE: $why download it"
MODEL_ARG="/root/.cache/huggingface/$SUB"
if [[ "$DRAFTER" == dspark ]]; then              # the speculator from the same cache (the workers over NFS)
  [[ -f "$HF_CACHE/$DSPARK_SUB/config.json" ]] || die "$DSPARK_ID @ ${DSPARK_REV:0:12} not in $HF_CACHE: $why download it"
fi
# rank i's view of the head's cache: the NFS volume (read-only); with NFS_PATH=/ the export root is HF_CACHE
WORKER_MOUNT="$NFS_VOLUME:/root/.cache/huggingface:ro"

# ---------------------------------------------------------------- 2. checks
step 2 "Checks: arguments, links, NFS, previous server, port, memory plan"
if (( image_ok )); then
  docker run --rm --network none --entrypoint python "$IMAGE" -c \
    'import sys; from tensorfold.cli import build_parser; build_parser().parse_args(sys.argv[1:])' \
    serve "$MODEL_ARG" --tp 3 --rank 0 --master 127.0.0.1 --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}" >/dev/null 2>"$STATE_DIR/args.err" ||
    if (( DRY )); then warn "DRY_RUN: tensorfold serve in $IMAGE rejects these arguments: $(tail -1 "$STATE_DIR/args.err")"
    else cat "$STATE_DIR/args.err" >&2; die "tensorfold serve rejects these arguments (see above); nothing was changed"; fi
fi
detect_links
log "Rendezvous: $MASTER_ADDR:$MASTER_PORT; NCCL bootstrap over ${NODE_DEV[*]} (rank 0 to 2)"
for r in 0 $(worker_ids); do
  log "  rank $r: RoCE ${NODE_HCAS[r]} (GID ${NODE_GID[r]:-per device})$( (( r == 0 )) || echo ", cable to the head ${LINK_WORKER_ADDR[r]:-?} <-> ${LINK_HEAD_ADDR[r]:-?}, NFS from $(nfs_server "$r")")"
done
# the workers must see the checkpoint over NFS (never a copy): its config at least, the rest prepare.sh checked
for i in $(worker_ids); do
  (( DRY )) && break
  rel=$SUB; [[ "$NFS_PATH" == / ]] || rel=${SNAP#"$NFS_PATH"/}
  worker_nfs "$i" test -f "/hf/$rel/config.json" ||
    die "$(wname "$i") does not see $MODEL_ID @ ${REV:0:12} over NFS ($NFS_VOLUME); run scripts/prepare.sh (README: Weights over NFS)"
done
here_up=0; running_here && here_up=1
any_worker_up=0
for i in $(worker_ids); do running_worker "$i" && any_worker_up=1; done
# memory: every Spark's MemAvailable now (with this server stopped: a restart's figures come after ./stop.sh below)
declare -a AVAIL=()
mem_now() { awk '/^MemAvailable:/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo; }
memplan() {  # <rank=GiB,...>: the plan's table; non-zero when a rank would fall under FLOOR_GIB
  local snapdir
  snapdir=$SNAP
  docker run --rm --network none --memory 3g --entrypoint python -v "$snapdir:/ckpt:ro" -v "$PWD/scripts:/recipe:ro" "$IMAGE" \
    /recipe/memplan.py /ckpt --tp 3 --context "$(arg_value --context)" --kv "$KV" --dense "$DENSE" --mtp "$MTP" \
    --prefill-rows "$PREFILL_ROWS" --split "$PREFILL_SPLIT" --cp "$([[ "$CP" == 1 ]] && echo 3 || echo 1)" --kept-gib "$KV_POOL_GIB" --floor "$FLOOR_GIB" \
    --overhead "$OVERHEAD_GIB" --reserve "$MEMORY_RESERVE_GIB" --avail "$1" 2>&1 | grep -v -E '^$|^=+$|PyTorch|Copyright|rights reserved|NVIDIA|found at|CUDA|SHMEM|docker run|insufficient'
  return "${PIPESTATUS[0]}"
}
if (( DRY )); then
  (( here_up || any_worker_up )) && log "DRY_RUN: $CONTAINER_NAME is running; a real start would stop it first (./stop.sh)"
elif (( here_up || any_worker_up )); then
  [[ "$MODE" == start ]] && log "Only some ranks are running: stopping them, then starting all 3 ranks"
  ./stop.sh
fi
AVAIL[0]=$(mem_now)
for i in $(worker_ids); do AVAIL[i]=$(worker "$i" "$(declare -f mem_now); mem_now" 2>/dev/null || echo 0); done
log "MemAvailable now: ${AVAIL[0]} GiB here, ${AVAIL[1]} on rank 1, ${AVAIL[2]} on rank 2"
if (( image_ok )); then
  plan_rc=0
  memplan "0=${AVAIL[0]},1=${AVAIL[1]},2=${AVAIL[2]}" | sed 's/^/  /' || plan_rc=$?
  if (( plan_rc == 1 )); then
    msg="the memory plan refuses this start (above): with a ${CONTEXT}-token window a Spark would be left with less than FLOOR_GIB=$FLOOR_GIB GiB, and a GB10 that runs out of memory freezes. Lower CONTEXT, use KV=fp4 (or fp4x), PREFILL_ROWS=1024 or MTP=0, or free memory on the Spark it names"
    (( DRY )) && warn "DRY_RUN: $msg" || die "$msg; nothing was started"
  elif (( plan_rc != 0 )); then
    msg="the memory plan could not be computed (exit $plan_rc, above): the image and scripts/memplan.py disagree; rebuild the image (scripts/prepare.sh)"
    (( DRY )) && warn "DRY_RUN: $msg" || die "$msg; nothing was started"
  fi
fi
left_here=0; docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" && left_here=1
declare -a left_worker=()
any_left=0
for i in $(worker_ids); do
  left_worker[i]=0
  worker "$i" "docker ps -a --format '{{.Names}}' | grep -qx '$CONTAINER_NAME'" 2>/dev/null && { left_worker[i]=1; any_left=1; }
done
if (( ! DRY && ( left_here || any_left ) )); then
  log "Removing the previous (stopped) container $CONTAINER_NAME, its logs saved first"
  if (( left_here )); then
    saved=$(save_log "$LOG_DIR" 0 "$CONTAINER_NAME" "$LOG_KEEP") || warn "could not save rank 0's log to $LOG_DIR"
    [[ -z "${saved:-}" ]] || log "Rank 0's log: $saved"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi
  for i in $(worker_ids); do
    (( left_worker[i] )) || continue
    saved=$(worker_save_log "$i") || warn "could not save rank $i's log on $(worker_host "$i")"
    [[ -z "${saved:-}" ]] || log "Rank $i's log (on $(worker_host "$i")): $saved"
    worker "$i" "docker rm -f '$CONTAINER_NAME' >/dev/null"
  done
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  (( DRY )) && log "DRY_RUN: port $PORT is in use now" ||
    die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"
fi

# TensorFold's own switches (TENSORFOLD_*, TF_GLM_*, TF_ROCE_*) reach every rank with the same values (TF_ROCE_HCA is
# set per node by rank_nccl_env). None of them is a secret.
ENV_ARGS=(-e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}")
while IFS='=' read -r name _; do ENV_ARGS+=(-e "$name=${!name}"); done < <(env | grep -E '^(TENSORFOLD|TF_GLM|TF_ROCE)_[A-Z0-9_]+=' | grep -v '^TF_ROCE_HCA=' || true)
RUN_ARGS=(--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK
          --ulimit memlock=-1 --ulimit stack=67108864)

# ---------------------------------------------------------------- 3. guard, launch; 4. load
# The guard (GUARD=1): scripts/memguard.sh on every Spark from before its rank starts until its container is gone
# (on a worker: copied to ~/.cache/tensorfold-glm53-full/ there and run detached).
GUARD_DIR="$STATE_DIR/guard"
# The prompt cache (DISK_CACHE) lives in each image's own folder, KERNEL_CACHE/<image hash>/pcache: an older image's
# saved states cannot be loaded by this one, so before a start they are deleted on every Spark (the compiled kernels
# stay). They are root's files (the containers write them), so a throwaway container removes them.
PCACHE_CLEAN='for d in /k/*/pcache; do [ -d "$d" ] || continue; [ "$d" = "/k/$0/pcache" ] && continue; du -sm "$d" | cut -f1; rm -rf "$d"; done'
clean_pcaches() {
  local mb i
  (( DRY )) && { printf '[dry-run] prompt caches of other images removed on every Spark (KERNEL_CACHE/<other hash>/pcache)\n'; return 0; }
  mb=$(docker run --rm --network none -v "$KERNEL_CACHE":/k --entrypoint sh "$IMAGE" -c "$PCACHE_CLEAN" "$KCACHE" 2>/dev/null | awk '{ s += $1 } END { print s + 0 }')
  for i in $(worker_ids); do
    mb=$(( mb + $(worker "$i" "docker run --rm --network none -v \$HOME/.cache/tensorfold-glm53-full:/k --entrypoint sh '$IMAGE' -c $(printf '%q' "$PCACHE_CLEAN") '$KCACHE' 2>/dev/null" | awk '{ s += $1 } END { print s + 0 }') ))
  done
  (( mb == 0 )) || log "Removed older images' prompt caches: $(( mb / 1024 )).$(( mb % 1024 * 10 / 1024 )) GiB across the Sparks"
}
start_guards() {
  local i
  (( GUARD )) || return 0
  if (( DRY )); then
    printf '[dry-run] memory guard on every Spark: scripts/memguard.sh %s %s <state dir>/guard (kills the rank below %s GiB)\n' \
      "$CONTAINER_NAME" "$GUARD_KILL_GIB" "$GUARD_KILL_GIB"
    return 0
  fi
  pkill -f "[m]emguard.sh $CONTAINER_NAME" 2>/dev/null || true
  setsid nohup ./scripts/memguard.sh "$CONTAINER_NAME" "$GUARD_KILL_GIB" "$GUARD_DIR" >/dev/null 2>&1 < /dev/null &
  for i in $(worker_ids); do
    worker "$i" "mkdir -p \$HOME/.cache/tensorfold-glm53-full && cat > \$HOME/.cache/tensorfold-glm53-full/memguard.sh && chmod +x \$HOME/.cache/tensorfold-glm53-full/memguard.sh" < scripts/memguard.sh
    # ([m]emguard: a pattern that does not match this very ssh command, which pkill would otherwise end)
    worker "$i" "pkill -f '[m]emguard.sh $CONTAINER_NAME' 2>/dev/null; setsid nohup \$HOME/.cache/tensorfold-glm53-full/memguard.sh '$CONTAINER_NAME' '$GUARD_KILL_GIB' \$HOME/.cache/tensorfold-glm53-full/guard >/dev/null 2>&1 < /dev/null & true" ||
      die "could not start the memory guard on $(wname "$i")"
  done
  log "Memory guard on all 3 Sparks: rank stopped below $GUARD_KILL_GIB GiB; low-water marks in $GUARD_DIR/memguard.low and ~/.cache/tensorfold-glm53-full/guard on the workers"
}
launch() {
  local rank0 rankw worker_cmd remote a i
  local -a here_cmd
  for i in 2 1; do
    rankw=(tensorfold serve "$MODEL_ARG" --tp 3 --rank "$i" --master "$MASTER_ADDR" --master-port "$MASTER_PORT" "${SERVE_ARGS[@]}")
    log "Rank $i on $(worker_host "$i"): ${rankw[*]}"
    worker_cmd=(docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" "${ENV_ARGS[@]}"
                $(rank_nccl_env "$i")
                -v "$WORKER_MOUNT" -v "\$HOME/.cache/tensorfold-glm53-full/$KCACHE:/cache"
                "$IMAGE" "${rankw[@]}")
    remote=""; for a in "${worker_cmd[@]}"; do
      case "$a" in '$HOME'*) remote+=" \"$a\"" ;; *) remote+=" $(printf '%q' "$a")" ;; esac
    done
    if (( DRY )); then
      printf '[dry-run] rank %s on %s:\n  %s\n' "$i" "$(worker_host "$i")" "mkdir -p \$HOME/.cache/tensorfold-glm53-full &&$remote"
      continue
    fi
    worker "$i" "mkdir -p \$HOME/.cache/tensorfold-glm53-full &&$remote" >/dev/null || die "could not start rank $i on $(worker_host "$i")"
  done
  rank0=(tensorfold serve "$MODEL_ARG" --tp 3 --rank 0 --master "$MASTER_ADDR" --master-port "$MASTER_PORT"
         --name "$SERVED_NAME" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}")
  log "Rank 0 here: ${rank0[*]}"
  here_cmd=(docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" "${ENV_ARGS[@]}"
            $(rank_nccl_env 0)
            -v "$HF_CACHE":/root/.cache/huggingface:ro -v "$KERNEL_CACHE/$KCACHE":/cache
            "$IMAGE" "${rank0[@]}")
  if (( DRY )); then
    printf '[dry-run] rank 0 here:\n  %s\n' "$(printf '%q ' "${here_cmd[@]}" | sed 's/ $//')"
    exit 0
  fi
  "${here_cmd[@]}" >/dev/null
}
foreground() {
  local watch w code i
  trap './stop.sh; exit 130' INT TERM
  ( exec 8>&-
    while sleep 30; do
      running_here || exit 0
      for i in $(worker_ids); do
        worker "$i" true 2>/dev/null || continue
        running_worker "$i" && continue
        warn "rank $i on $(worker_host "$i") exited: stopping rank 0"
        docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null 2>&1
        exit 1
      done
    done ) &
  watch=$!
  docker logs -f "$CONTAINER_NAME" || true
  kill "$watch" 2>/dev/null || true
  w=0; wait "$watch" || w=$?
  code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo 1)
  [[ "$w" != 1 || "$code" != 0 ]] || code=1
  for i in $(worker_ids); do worker "$i" "docker stop -t ${STOP_TIMEOUT:-30} '$CONTAINER_NAME'" >/dev/null 2>&1 || true; done
  exit "$code"
}
NOISE='^\s*$|EXL3 support is experimental|^=+$|^== PyTorch ==|^NVIDIA Release|Copyright|All rights reserved|PyTorch Version|Various files include|NOTE: CUDA Forward|Using CUDA|cuda-compatibility|Container image|torch/utils/_pytree\.py.*register_constant'
LOGS_PID=""
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT
fail() {
  kill $LOGS_PID 2>/dev/null || true
  sleep 0.5
  printf '\n%s── rank 0 (here): last server log lines ──%s\n' "$D" "$R"
  docker logs --tail 25 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  for i in $(worker_ids); do
    printf '%s── rank %s (%s): last server log lines ──%s\n' "$D" "$i" "$(worker_host "$i")" "$R"
    worker "$i" docker logs --tail 25 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  done
  [[ -f "$GUARD_DIR/memguard.low" ]] && log "lowest MemAvailable here: $(cat "$GUARD_DIR/memguard.low")"
  die "$1"
}
for attempt in 1 2; do
step 3 "Launch: memory guard, container $CONTAINER_NAME, ranks 2 and 1 on the workers, then rank 0 here"
start_guards
(( attempt == 1 )) && clean_pcaches
launch
[[ "${FOREGROUND:-0}" == 1 ]] && foreground
step 4 "Loading: ~86 GiB of weights on each Spark, the workers over NFS (3-5 min; the very first start also compiles CUDA kernels)"
docker logs -f "$CONTAINER_NAME" > >(grep --line-buffered -v -E "$NOISE" | sed -u "s/^/  ${D}│${R} /") 2>&1 &
LOGS_PID=$!
start=$SECONDS; next_beat=30; refit=""
until curl -sf --max-time 5 "$URL/v1/models" >/dev/null 2>&1; do
  if ! running_here; then
    # the memory at this start holds a smaller window than asked (a Spark's free memory drifts): once, start again
    # with the largest one TensorFold names
    refit=$(docker logs "$CONTAINER_NAME" 2>&1 | sed -n 's/.*largest fitting prompt-plus-reply window: \([0-9]*\) tokens.*/\1/p' | tail -1)
    # 97% of it, a multiple of 2,048: free memory keeps drifting between the two starts (by ~0.3 GiB on a busy Spark)
    [[ -n "$refit" ]] && refit=$(( refit * 97 / 100 / 2048 * 2048 ))
    [[ -n "$refit" && $attempt == 1 && "$refit" -ge 4096 ]] && break
    fail "rank 0 exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
  fi
  (( SECONDS - start < WAIT_TIMEOUT )) ||
    fail "not ready after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT); the ranks are still running: docker logs -f $CONTAINER_NAME"
  if (( SECONDS - start >= next_beat )); then
    for i in $(worker_ids); do
      running_worker "$i" || fail "rank $i on $(worker_host "$i") exited before the server was ready"
    done
    lows="here $(mem_now)"
    for i in $(worker_ids); do lows+=", rank $i $(worker "$i" "$(declare -f mem_now); mem_now" 2>/dev/null || echo '?')"; done
    printf '  %s⋯ %ss elapsed; MemAvailable GiB: %s%s\n' "$D" "$((SECONDS - start))" "$lows" "$R"
    next_beat=$((next_beat + 30))
  fi
  sleep 3
done
kill $LOGS_PID 2>/dev/null || true
[[ -z "$refit" ]] && break
warn "this start's memory holds a ${refit}-token window, not ${CONTEXT}: starting again with CONTEXT=$refit"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
for i in $(worker_ids); do worker "$i" "docker rm -f '$CONTAINER_NAME'" >/dev/null 2>&1 || true; done
CONTEXT=$refit
for i in "${!SERVE_ARGS[@]}"; do [[ "${SERVE_ARGS[$i]}" == --context ]] && SERVE_ARGS[$((i + 1))]=$refit; done
done
sleep 0.3
log "Server answered after $((SECONDS - start))s"

# ---------------------------------------------------------------- 5. smoke test
step 5 "Smoke test: one chat completion through all 3 ranks"
SERVED=$(served_name || echo "$SERVED_NAME")
if smoke=$(curl -s --max-time 300 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
             -d "{\"model\": \"$SERVED\", \"max_tokens\": 32, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" |
           python3 -c 'import json,sys; r = json.load(sys.stdin); c = r["choices"][0]["message"].get("content") or ""; assert c.strip(); print(repr(c.strip()[:40]) + ",", r["usage"]["completion_tokens"], "tokens,", r.get("tensorfold", {}).get("decode_s"), "s")' 2>/dev/null); then
  log "OK: $smoke"
else
  fail "the smoke test request failed (no reply text); the ranks are still running"
fi
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED)
    Window   $(arg_value --context) tokens · 3 Sparks · $( [[ "$PARALLEL" == 1 ]] && echo "one request at a time" || echo "$PARALLEL at once") · $KV KV$( [[ "$CP" == 1 ]] && echo " · context parallel")$( [[ -n "${TF_GLM_DISK_CACHE:-}" && "$PARALLEL" == 1 ]] && echo " · prompt cache on NVMe (up to ${TF_GLM_DISK_CACHE_GIB:-64} GiB a Spark)") · $DENSE dense weights$( [[ "$COMM" == roce ]] && echo " · RoCE all-gathers")
    Drafts   $DRAFTER$( [[ "$MTP" == 1 && "$DRAFTER" != mtp ]] && echo " + MTP") · copy drafts $( [[ "$COPY" == 1 ]] && echo "up to $COPY_MAX$( [[ "$COPY_HYBRID" == 1 ]] && echo ", checked by DSpark")" || echo off) · shared system prompts $( [[ "$SHARED_PREFIX" == 1 ]] && echo on || echo off)
    Memory   guard $( [[ "$GUARD" == 1 ]] && echo "on (stops a rank below $GUARD_KILL_GIB GiB; lowest so far: $STATE_DIR/guard/memguard.low)" || echo off)
    Logs     docker logs -f $CONTAINER_NAME$(for i in $(worker_ids); do printf '   (rank %s: ssh %s docker logs -f %s)' "$i" "$(worker_host "$i")" "$CONTAINER_NAME"; done)
    Restart  ./start.sh restart
    Stop     ./stop.sh

EOF
