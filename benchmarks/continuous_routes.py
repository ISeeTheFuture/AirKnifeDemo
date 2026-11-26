"""Three consecutive physical racks, paired across start poses, end actions and routes."""

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from continuous_report import LABELS
from planner import END_ACTIONS, Motion, make_run, random_boards
from simulator import Cell

# Absolute heights used in the published three-board experiment.
FIXED_HEIGHTS_MM = {"상": 1150, "중": 840, "하": 530}


def initialize(scan):
    global SCAN
    SCAN = scan


def trial(n, repeat, height, action):
    seed = 20261124 + 100 * n + repeat
    # The first rack matches the old benchmark. Distinct, reproducible following racks.
    layouts = random_boards(SCAN, n, 3, seed)
    motion = Motion((0, 160, FIXED_HEIGHTS_MM[height]), action)
    result = {}
    cache = {}
    for (
        mode
    ) in LABELS:  # Reproduce the six routes in the historical three-board report.
        run = make_run(SCAN, layouts, mode=mode, motion=motion)
        signature = tuple(
            tuple((s.spraying, s.a.tobytes(), s.b.tobytes(), s.t1) for s in b.segments)
            for b in run.boards
        )
        if signature not in cache:
            sim = Cell(SCAN, run, visuals=False)
            try:
                while not sim.finished:
                    sim.step()
                cache[signature] = {
                    "elapsed": float(sim.elapsed),
                    "boards": sim.board_results(),
                }
            finally:
                sim.close()
        result[mode] = cache[signature]
    return {
        "n": n,
        "repeat": repeat,
        "seed": seed,
        "height": height,
        "action": action,
        "rows": [layout.rows for layout in layouts],
        "methods": result,
    }


def worker_geometry(archive):
    # Children only need the selected geometry, not the three full sensor images.
    from scan import load_scan

    s = load_scan(archive, 2)
    return SimpleNamespace(
        sample=s.sample,
        xyz=s.xyz[s.selected],
        selected=slice(None),
        center=s.center,
        surface=s.surface,
        vertices=s.vertices,
        triangles=s.triangles,
    )


def main():
    from scan import locate_tar

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--counts", nargs="+", type=int, default=range(1, 31))
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.workers < 1
        or not all(1 <= n <= 30 for n in args.counts)
    ):
        parser.error("Use positive workers/repeats and counts from 1 to 30.")
    archive = args.tar or locate_tar()
    if archive is None:
        parser.error("Specify the original capture archive with --tar.")
    scan = worker_geometry(archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for repeat, n, height, action in itertools.product(
        range(args.repeats), args.counts, FIXED_HEIGHTS_MM, END_ACTIONS
    ):
        dest = (
            args.output_dir
            / f"{n:02}-{repeat}-{FIXED_HEIGHTS_MM[height]}-{action}.json"
        )
        if not dest.exists():
            pending.append(((n, repeat, height, action), dest))
    # Releasing the physics workers between batches bounds native allocation growth.
    batch_size = args.workers * 8
    for start in range(0, len(pending), batch_size):
        with ProcessPoolExecutor(
            args.workers, initializer=initialize, initargs=(scan,)
        ) as pool:
            jobs = {
                pool.submit(trial, *setting): dest
                for setting, dest in pending[start : start + batch_size]
            }
            for i, job in enumerate(as_completed(jobs), start + 1):
                dest = jobs[job]
                tmp = dest.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(job.result(), ensure_ascii=False), encoding="utf-8"
                )
                os.replace(tmp, dest)
                if i % 10 == 0 or i == len(pending):
                    print(f"{i}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
