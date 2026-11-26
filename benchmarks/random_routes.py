"""Compare moving-rack routes on paired random layouts from capture 2."""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from planner import MIN_DWELL, ROUTES, make_run, random_layout
from scan import load_scan, locate_tar
from simulator import Cell

SCAN = None


def load_capture(archive):
    global SCAN
    SCAN = load_scan(archive, 2)


def simulate(plan):
    sim = Cell(SCAN, plan, visuals=False)
    try:
        while not sim.finished:
            sim.step()
        covered = (sim.dwell >= MIN_DWELL).reshape(2, len(plan.layout.positions), -1)
        result = sim.result()
        return {
            "seconds": float(plan.duration),
            "coverage": result["coverage_pct"],
            "worst": float(100 * covered.mean(axis=2).min()),
            "contact": sim.contact_frames,
            "ik_ok": sim.ik_ok == sim.ik_total,
            "ik_pct": result["ik_ok_pct"],
            "p95": result["tcp_p95_mm"],
            "clearance": result["rack_or_peer_clearance_mm_capped_at_150"],
        }
    finally:
        sim.close()


def trial(n, seed):
    layout = random_layout(SCAN, n, np.random.default_rng(seed))
    plans = {
        "current" if mode == "target" else mode: make_run(
            SCAN, (layout,), mode=mode
        ).boards[0]
        for mode in ROUTES
    }
    results, cache = {}, {}
    for name, plan in plans.items():
        key = tuple(
            (s.spraying, *np.round(s.a, 10), *np.round(s.b, 10), round(s.t1, 10))
            for s in plan.segments
        )
        if key not in cache:
            cache[key] = simulate(plan)
        results[name] = dict(cache[key], passes=sum(s.spraying for s in plan.segments))
    return {
        "n": n,
        "seed": seed,
        "rows": layout.rows,
        "positions": layout.positions,
        "slope": plans["oblique"].slope,
        "methods": results,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tar", type=Path)
    ap.add_argument("--counts", type=int, nargs="+", default=list(range(1, 31)))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    archive = args.tar or locate_tar()
    if archive is None:
        ap.error("Original capture TAR required.")
    args.output_dir.mkdir(exist_ok=True, parents=True)
    with ProcessPoolExecutor(
        args.workers, initializer=load_capture, initargs=(archive,)
    ) as pool:
        jobs = {}
        for n in args.counts:
            for rep in range(args.repeats):
                seed = 20261124 + 100 * n + rep
                dest = args.output_dir / f"{n:02}-{seed}.json"
                jobs[pool.submit(trial, n, seed)] = dest
        for job in as_completed(jobs):
            dest = jobs[job]
            dest.write_text(json.dumps(job.result()), encoding="utf-8")
            print(dest.name, flush=True)


if __name__ == "__main__":
    main()
