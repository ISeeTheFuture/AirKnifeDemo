"""Paired 5-rack trials. Each checkpoint contains one layout group and one route."""

import argparse
import hashlib
import json
import os
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import continuous_routes as shared
from planner import ADVANCE_OPTIONS, FULL_AUTO, ROUTES, Motion, make_run, random_boards
from scan import locate_tar
from simulator import Cell


def settings(route):
    if route == FULL_AUTO:
        return [(FULL_AUTO, "", "")]
    return [(route, "home", ""), (route, "point", "")] + [
        (route, "continue", advance) for advance in ADVANCE_OPTIONS
    ]


def command_hash(run):
    """Only compare within the same five layouts, robot setup and tool geometry."""
    h = hashlib.sha256()
    for i, b in enumerate(run.boards):
        h.update(
            struct.pack(
                "<dddI", run.starts[i], run.ends[i], b.origin_x, len(b.segments)
            )
        )
        idle = run.idle_moves.get(i)
        h.update(bytes([idle is not None]))
        for s in [*b.segments, *([idle] if idle else [])]:
            h.update(struct.pack("<dd?6d", s.t0, s.t1, s.spraying, *s.a, *s.b))
    return h.hexdigest()


def trial(n, rep, route, out):
    group_file = out / f"{n:02}-{rep}-{route}.json"
    if group_file.exists():
        return
    scan = shared.SCAN
    seed = 20261124 + 100 * n + rep
    layouts = random_boards(scan, n, 5, seed)
    rows = []
    for _, action, advance in settings(route):
        run = make_run(
            scan,
            layouts,
            mode=route,
            motion=Motion(action=action or "home", advance=advance),
        )
        key = command_hash(run)
        # Identical complete TCP commands on the same physical scene share a measurement.
        dest = out / "physics" / f"{n:02}-{rep}-{key}.json"
        if not dest.exists():
            cell = Cell(scan, run, measure_clearance=False, visuals=False)
            try:
                while not cell.finished:
                    cell.step()
                record = {
                    "elapsed": float(cell.elapsed),
                    "boards": cell.board_results(),
                }
            finally:
                cell.close()
            write_atomic(dest, record)
        rows.append(
            {
                "setting": [route, action, advance],
                "physics": dest.name,
                "fastest_fallback": run.fastest_fallback,
                "decisions": [run.decision(i) for i in range(5)],
                "traversals": [board.traversal for board in run.boards],
            }
        )
    result = {"n": n, "repeat": rep, "seed": seed, "rows": rows}
    write_atomic(group_file, result)


def write_atomic(dest, record):
    temp = dest.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tar", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--phase", choices=("fixed", "auto", "all"), default="all")
    parser.add_argument("--counts", nargs="+", type=int, default=range(1, 31))
    args = parser.parse_args()
    if args.workers < 1 or any(n not in range(1, 31) for n in args.counts):
        parser.error("workers >= 1, counts 1..30")
    out = args.output_dir.resolve()
    (out / "physics").mkdir(parents=True, exist_ok=True)
    modes = [*ROUTES, FULL_AUTO]
    if args.phase == "fixed":
        modes = list(ROUTES)
    elif args.phase == "auto":
        modes = [FULL_AUTO]
    jobs = [
        (n, rep, route, out)
        for n in args.counts
        for rep in range(5)
        for route in modes
        if not (out / f"{n:02}-{rep}-{route}.json").exists()
    ]
    scan = shared.worker_geometry(args.tar or locate_tar())
    workers = args.workers
    print(f"{len(jobs)} pending route groups, {workers} workers", flush=True)
    offset = 0
    while offset < len(jobs):
        # Dense layouts allocate more native PyBullet objects. Retire each worker
        # after one dense group and use the on-disk commands across restarts.
        batch = workers if jobs[offset][0] >= 25 else 5 * workers
        chunk = jobs[offset : offset + batch]
        try:
            with ProcessPoolExecutor(
                workers, initializer=shared.initialize, initargs=(scan,)
            ) as pool:
                futures = [pool.submit(trial, *job) for job in chunk]
                for future in as_completed(futures):
                    future.result()
        except BrokenProcessPool:
            if workers == 1:
                raise
            workers = max(1, workers // 2)
            print(
                f"Worker exited; resuming checkpointed group {offset + 1} with {workers} workers",
                flush=True,
            )
            continue
        offset += len(chunk)
        print(f"{offset}/{len(jobs)} route groups", flush=True)


if __name__ == "__main__":
    main()
