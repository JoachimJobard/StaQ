#!/usr/bin/env bash
# Re-sync offline wandb run dirs, so media lands in the RUN THAT ALREADY EXISTS.
#
#   run/resync_offline_wandb.sh <wandb_dir> <project> [--dry-run] [--limit N]
#
# Each offline-run-<ts>-<id> dir carries its own run id, so syncing it UPDATES
# that run rather than creating a new one -- unlike syncing a tfevents directory,
# which always makes a fresh run.
#
# Two flags matter and are easy to miss:
#   --legacy          without it the call is rerouted to `wandb beta sync`, which
#                     only uploads .wandb files and ignores the tensorboard/offline
#                     options, leaving a run with nothing parsed into it.
#   --include-synced  these runs are already marked synced, so a plain re-sync
#                     skips them silently.

set -uo pipefail   # not -e: one bad run must not abort the batch

if [ $# -lt 2 ]; then
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

ROOT=$1; PROJECT=$2; shift 2
DRY_RUN=0; LIMIT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --limit)   LIMIT=$2; shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

[ -d "$ROOT" ] || { echo "no such directory: $ROOT" >&2; exit 1; }

mapfile -t DIRS < <(find "$ROOT" -maxdepth 1 -type d -name 'offline-run-*' | sort)
[ ${#DIRS[@]} -gt 0 ] || { echo "no offline-run-* dirs under $ROOT" >&2; exit 1; }
[ "$LIMIT" -gt 0 ] && DIRS=("${DIRS[@]:0:$LIMIT}")

echo "found ${#DIRS[@]} offline run(s) under $ROOT"
echo

ok=0; failed=0; failed_dirs=()
for d in "${DIRS[@]}"; do
    name=$(basename "$d")
    imgs=$(ls "$d"/files/media/images/ 2>/dev/null | wc -l)
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] $name  (${imgs} media file(s))"
        continue
    fi
    echo "=== $name  (${imgs} media file(s))"
    if [ "$imgs" -eq 0 ]; then
        echo "    no media on disk -- nothing to recover from this dir, skipping"
        continue
    fi
    if fuv run wandb sync --legacy --include-offline --include-synced -p "$PROJECT" "$d"; then
        ok=$((ok + 1))
    else
        failed=$((failed + 1)); failed_dirs+=("$name")
    fi
done

[ "$DRY_RUN" -eq 1 ] && exit 0
echo
echo "synced $ok, failed $failed"
for f in "${failed_dirs[@]:-}"; do [ -n "$f" ] && echo "  FAILED: $f"; done
