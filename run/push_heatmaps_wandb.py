"""Push policy heatmaps from tensorboard events into the EXISTING wandb runs.

    fuv run python run/push_heatmaps_wandb.py <multirun_dir> <entity/project> [--dry-run] [--limit N]

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
by_name = {}
for r in api.runs(project):
    by_name.setdefault(r.name, r.id)
print(f"{len(by_name)} run(s) in {project}, {len(run_dirs)} tb dir(s) under {root}\n")

pushed = missing = 0
for d in run_dirs:
    name = os.path.relpath(d, root)
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
    if dry_run:
        continue

    run = wandb.init(project=project.split("/")[-1], entity=(project.split("/")[0] if "/" in project else None),
                     id=run_id, resume="must")
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
