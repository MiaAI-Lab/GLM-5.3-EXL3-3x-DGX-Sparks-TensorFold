#!/usr/bin/env bash
# The memory guard: a GB10 that runs out of memory freezes instead of failing, so while <container> runs on this Spark
# this samples MemAvailable every 0.5 s, keeps its low-water mark in <dir>/memguard.low ("<GiB> <time>"), and stops the
# container (docker kill) the moment MemAvailable drops below <kill GiB>, logging that in <dir>/memguard.log. It ends
# when the container is gone. start.sh starts it on every Spark before the ranks (GUARD=1, the default).
#   scripts/memguard.sh <container> <kill GiB> <dir>
set -u
# a detached guard must not hold its parent's open files (start.sh's lock, a caller's flock): close every inherited one
for fd in /proc/$$/fd/*; do fd=${fd##*/}; (( fd > 2 )) && eval "exec $fd>&-" 2>/dev/null; done
c=${1:?container}; kill_kib=$(awk -v g="${2:?kill GiB}" 'BEGIN { printf "%d", g * 1048576 }'); dir=${3:?dir}
mkdir -p "$dir"; low=999999999; seen=0; ticks=0
echo "$(date '+%F %T') guard for $c on $(hostname): kill below $2 GiB" >> "$dir/memguard.log"
while :; do
  a=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  if (( a < low )); then low=$a; printf '%s %s\n' "$(awk -v a="$a" 'BEGIN { printf "%.2f", a / 1048576 }')" "$(date +%T)" > "$dir/memguard.low"; fi
  if (( a < kill_kib )); then
    echo "$(date '+%F %T') MemAvailable $(awk -v a="$a" 'BEGIN { printf "%.2f", a / 1048576 }') GiB < $2: docker kill $c" >> "$dir/memguard.log"
    docker kill "$c" >/dev/null 2>&1
  fi
  if (( ticks % 10 == 0 )); then              # every 5 s: is the container (still) there?
    if [[ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" == true ]]; then seen=1
    elif (( seen || ticks > 1200 )); then break; fi     # gone, or never started within 10 minutes
  fi
  ticks=$((ticks + 1)); sleep 0.5
done
echo "$(date '+%F %T') guard ends ($c gone); lowest MemAvailable $(cat "$dir/memguard.low")" >> "$dir/memguard.log"
