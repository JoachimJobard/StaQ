"""Extract policy heatmaps from tensorboard events, recursively.

Usage:  uv run python dump_heatmaps.py <dir> [out_dir]

<dir> may be a single run dir OR any parent (a whole multirun tree) -- every
event file underneath is found. EventAccumulator itself does not recurse, which
is why pointing it straight at a multirun dir silently yields nothing.
"""
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "heatmaps_out"

run_dirs = sorted({d for d, _, fs in os.walk(root)
                   if any(f.startswith("events.out.tfevents") for f in fs)})
if not run_dirs:
    print(f"no tensorboard event files anywhere under {root}")
    sys.exit(1)
print(f"found {len(run_dirs)} run dir(s)\n")

total = 0
for d in run_dirs:
    ea = EventAccumulator(d, size_guidance={"images": 0})
    ea.Reload()
    reached = max((ea.Scalars(t)[-1].step for t in ea.Tags()["scalars"]), default=0)
    tags = ea.Tags()["images"]
    name = os.path.relpath(d, root)
    print(f"{name}\n    reached step {reached:,} | image tags: {tags or 'NONE'}")

    sub = os.path.join(out_dir, name.replace(os.sep, "_"))
    for t in tags:
        os.makedirs(sub, exist_ok=True)
        for ev in ea.Images(t):
            fn = os.path.join(sub, f"{t.replace('/', '_')}_step{ev.step}.png")
            with open(fn, "wb") as fh:
                fh.write(ev.encoded_image_string)
            total += 1
    if tags:
        print(f"    -> wrote {sum(len(ea.Images(t)) for t in tags)} png into {sub}")
print(f"\n{total} images written")
