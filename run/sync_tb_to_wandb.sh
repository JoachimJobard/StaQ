#!/usr/bin/env bash
# Sync every tensorboard run under a multirun tree into wandb, images included.
#
#   run/sync_tb_to_wandb.sh <multirun_dir> <project> [--dry-run] [--limit N]
#
# Why tfevents and not the offline wandb dirs: `wandb sync` reads tfevent files
# directly, so this works even when an offline run's transaction log is truncated
# (the "unexpected EOF" case). Each leaf directory holding events.out.tfevents*
# becomes one wandb run, named after that directory -- which hydra already names
# by run_name, so the naming comes out right for free.
#
# NOTE: EventAccumulator and wandb both take a SINGLE run dir, neither recurses.
# Hence the find: with env_name containing a '/', runs sit two levels deep
# (<date>/MinAtar/<rest>/), which is easy to miss by hand.

set -uo pipefail   # deliberately not -e: one failed run must not abort the batch

if [ $# -lt 2 ]; then
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
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

mapfile -t DIRS < <(find "$ROOT" -name 'events.out.tfevents*' -printf '%h\n' | sort -u)
[ ${#DIRS[@]} -gt 0 ] || { echo "no tensorboard event files under $ROOT" >&2; exit 1; }

[ "$LIMIT" -gt 0 ] && DIRS=("${DIRS[@]:0:$LIMIT}")
echo "found ${#DIRS[@]} run dir(s) under $ROOT"
echo

ok=0; failed=0; failed_dirs=()
for d in "${DIRS[@]}"; do
    name=${d#"$ROOT"/}
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] would sync $name"
        continue
    fi
    echo "=== syncing $name"
    # --legacy is REQUIRED: without it the call is rerouted to `wandb beta sync`,
    # which only uploads .wandb files and ignores --sync-tensorboard (legacy-only),
    # producing a run with the raw event file attached and nothing parsed out of it.
    if fuv run wandb sync --legacy --sync-tensorboard -p "$PROJECT" "$d"; then
        ok=$((ok + 1))
    else
        failed=$((failed + 1)); failed_dirs+=("$name")
    fi
done

[ "$DRY_RUN" -eq 1 ] && exit 0
echo
echo "synced $ok, failed $failed"
for f in "${failed_dirs[@]:-}"; do [ -n "$f" ] && echo "  FAILED: $f"; done
