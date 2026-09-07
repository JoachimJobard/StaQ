"""Push policy heatmaps from tensorboard events into the EXISTING wandb runs.

    fuv run python run/push_heatmaps_wandb.py <multirun_dir> <project> [--dry-run] [--limit N]

<project> is either "entity/project" or just "project" (the entity then comes
from the logged-in account). The entity is the first segment of a run URL:
https://wandb.ai/<entity>/<project>/runs/<run_id>

Why this exists: syncing a tfevents directory always creates a NEW wandb run
(the directory carries no run identity), and re-syncing an offline dir only works
if those dirs still exist. This instead looks each run up BY NAME in the project
-- hydra names the tb directory with run_name, which is exactly the wandb run
name -- and resumes that run to log the images into it.

Images are logged against a custom `heatmap_step` axis rather than the global
step. Resuming a finished run and calling wandb.log(step=...) for steps that
already have history runs into wandb's monotonic-step rule; a custom axis
sidesteps it entirely. In the UI, set the media panel's x-axis to heatmap_step.
"""
import os
import sys
import tempfile

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

root, project = sys.argv[1], sys.argv[2]
dry_run = "--dry-run" in sys.argv
limit = 0
if "--limit" in sys.argv:
    limit = int(sys.argv[sys.argv.index("--limit") + 1])

run_dirs = sorted({d for d, _, fs in os.walk(root)
                   if any(f.startswith("events.out.tfevents") for f in fs)})
if not run_dirs:
    sys.exit(f"no tensorboard event files under {root}")
if limit:
    run_dirs = run_dirs[:limit]

import wandb  # noqa: E402  (after arg parsing so --help style errors are fast)

api = wandb.Api()
# `project` may be "entity/project" or just "project"; wandb needs the entity, so
# fall back to the logged-in account's default when it is not spelled out.
if "/" in project:
    entity, project_name = project.split("/", 1)
else:
    entity, project_name = api.default_entity, project
    print(f"no entity given, using default entity {entity!r}")
path = f"{entity}/{project_name}"

by_name, dupes = {}, {}
n_runs = 0
for r in api.runs(path):
    n_runs += 1
    if r.name in by_name:
        dupes.setdefault(r.name, 1)
        dupes[r.name] += 1
    by_name.setdefault(r.name, r.id)
print(f"{n_runs} run(s) in {path} ({len(by_name)} distinct names), "
      f"{len(run_dirs)} tb dir(s) under {root}")
if dupes:
    print("WARNING: several wandb runs share a name; only the first id is used for each:")
    for k, v in sorted(dupes.items()):
        print(f"    {v}x  {k}")
print()

used_ids = {}  # run_id -> tb dir already pushed into it, to catch many-to-one mappings

def match_run(rel, table):
    """Map a tb dir to a wandb run name.

    hydra names the tb dir with run_name, so relpath == run name when `root` is the
    sweep dir. Point `root` one level higher (at multirun/) and every path gains a
    date prefix and nothing matches -- so also try progressively shorter tails.
    """
    parts = rel.split(os.sep)
    for i in range(len(parts)):
        cand = os.sep.join(parts[i:])
        if cand in table:
            return cand
    return None


pushed = missing = 0
for d in run_dirs:
    rel = os.path.relpath(d, root)
    name = match_run(rel, by_name) or rel
    ea = EventAccumulator(d, size_guidance={"images": 0})
    ea.Reload()
    tags = [t for t in ea.Tags()["images"] if t.startswith("policy/")]
    run_id = by_name.get(name)

    if not tags:
        print(f"{name}\n    no policy/* images in events -- skipping")
        continue
    if run_id is None:
        print(f"{name}\n    NO MATCHING WANDB RUN by that name -- skipping")
        missing += 1
        continue
    n_img = sum(len(ea.Images(t)) for t in tags)
    print(f"{name}\n    run id {run_id} | {len(tags)} tags | {n_img} images")
    if run_id in used_ids:
        print(f"    REFUSING: run id {run_id} was already used by {used_ids[run_id]!r}.\n"
              f"    Two tb dirs map to one wandb run -- pushing both would pile every\n"
              f"    seed into the same run. Fix the names before continuing.")
        continue
    used_ids[run_id] = name
    if dry_run:
        continue

    run = wandb.init(project=project_name, entity=entity, id=run_id, resume="must")
    wandb.define_metric("heatmap_step")
    for t in tags:
        wandb.define_metric(t, step_metric="heatmap_step")
    with tempfile.TemporaryDirectory() as tmp:
        # group by step so each step becomes one log call carrying all four panels
        steps = sorted({ev.step for t in tags for ev in ea.Images(t)})
        for s in steps:
            payload = {"heatmap_step": s}
            for t in tags:
                for ev in ea.Images(t):
                    if ev.step != s:
                        continue
                    fn = os.path.join(tmp, f"{t.replace('/', '_')}_{s}.png")
                    with open(fn, "wb") as fh:
                        fh.write(ev.encoded_image_string)
                    payload[t] = wandb.Image(fn)
            wandb.log(payload)
    run.finish()
    pushed += 1

print(f"\npushed {pushed} run(s); {missing} had no matching wandb run")
