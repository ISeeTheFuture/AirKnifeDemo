"""Regressions for preview alignment and board-by-board automatic decisions."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from automatic import automatic_boards, automatic_run
from planner import (
    END_ACTIONS,
    FULL_AUTO,
    LEADING,
    PITCH,
    ROUTES,
    START_X,
    TRANSFER_SPEED,
    Motion,
    RackLayout,
    _idle_move,
    grid_layout,
    layout_targets,
    make_run,
    route_candidates,
)
from simulator import BoardFailure


class HandoffTest(unittest.TestCase):
    layout = RackLayout(None, ((-0.2, 1.22), (0.2, 1.10), (0.2, 0.85), (-0.1, 0.61)))
    start = np.array([0.12, 0.14, 0.8])

    def test_high_low_and_leading_tie(self):
        for option, x, z in [
            ("top", 0.12, 1.22),
            ("bottom", 0.12, 0.61),
            ("leading_top", START_X + 0.2, 1.10),
            ("leading_bottom", START_X + 0.2, 0.85),
        ]:
            with self.subTest(option=option):
                move = _idle_move(self.start, 1.0, 10.0, self.layout, option)
                np.testing.assert_allclose(move.b, [x, 0.14, z])
                self.assertFalse(move.spraying)
                self.assertAlmostEqual(
                    np.linalg.norm(move.b - move.a) / (move.t1 - move.t0),
                    TRANSFER_SPEED,
                )

    def test_short_gap_and_unique_leader(self):
        unique = RackLayout(None, ((0.1, 1.20), (0.2, 0.62)))
        high, low = [
            _idle_move(self.start, 1.0, 1.01, unique, opt)
            for opt in ("leading_top", "leading_bottom")
        ]
        np.testing.assert_array_equal(high.b, low.b)
        self.assertEqual(high.t1, 1.01)
        self.assertAlmostEqual(np.linalg.norm(high.b - high.a), 0.01 * TRANSFER_SPEED)
        for option, arrival in [("", 10), ("leading_top", 1), ("bottom", 0.99)]:
            self.assertIsNone(_idle_move(self.start, 1, arrival, unique, option))
        with self.assertRaises(ValueError):
            Motion(action="continue", advance="auto")

    def test_automatic_handoff(self):
        scan = SimpleNamespace(
            surface=np.array(
                [[-0.02, -0.01, 0.002], [0.02, 0.015, 0.006], [0.01, 0.01, 0.003]]
            )
        )
        layouts = (self.layout, RackLayout(None, ((0.14, 0.58),)), self.layout)
        motion = Motion(start_level="middle", point_mm=(55, 240, 860))
        # Exercise scheduling with a deterministic verifier. A separate test replays
        # the selected commands in the physical cell.
        run = automatic_run(scan, layouts, 65, 250, motion, 820, verify=lambda *_: None)
        self.assertEqual(len(run.decisions), len(layouts))
        np.testing.assert_allclose(
            run.sample(0, 0)[0],
            np.asarray(motion.for_layout(layouts[0]).start_mm) / 1000,
        )
        for i, (route, action, advance) in enumerate(run.decisions):
            self.assertIn(route, ROUTES)
            self.assertEqual(route, run.boards[i].mode)
            self.assertFalse(advance and action != "continue")
            self.assertEqual(
                run.starts[i], max(i * PITCH / 0.065, run.ends[i - 1] if i else 0)
            )
            if i:
                np.testing.assert_allclose(
                    run.sample(run.starts[i], i - 1)[0],
                    run.sample(run.starts[i], i)[0],
                    atol=1e-9,
                )
        self.assertEqual(run.decisions[-1][2], "")
        self.assertNotIn(len(layouts) - 1, run.idle_moves)

    def test_fastest_passing_candidate_and_no_passing_route(self):
        scan = SimpleNamespace(
            surface=np.array([[-0.02, -0.01, 0.002], [0.02, 0.015, 0.006]])
        )
        durations = set()
        for route in ROUTES:
            for action in END_ACTIONS:
                motion = Motion(action=action)
                durations.update(
                    float(b.duration)
                    for b in route_candidates(
                        layout_targets(scan, self.layout),
                        65,
                        250,
                        route,
                        self.layout,
                        start=np.asarray(motion.start_mm) / 1000,
                        finish=motion.finish(),
                        origin_x=START_X,
                        base_mm=820,
                    )
                )
        durations = sorted(durations)
        threshold = durations[len(durations) // 2]
        attempted = []

        def verify(scan, run, base, layouts, cancel):
            attempted.append(run.duration)
            return None if run.duration >= threshold else BoardFailure(0)

        run = automatic_run(scan, (self.layout,), 65, 250, Motion(), 820, verify=verify)
        self.assertEqual(run.duration, threshold)
        self.assertFalse(run.fastest_fallback)
        self.assertEqual(attempted, sorted(attempted))
        self.assertLess(attempted[0], run.duration)
        fallback = automatic_run(
            scan,
            (self.layout,),
            65,
            250,
            Motion(),
            820,
            verify=lambda *_: BoardFailure(0),
        )
        self.assertTrue(fallback.fastest_fallback)
        self.assertEqual(fallback.duration, durations[0])

    def test_impossible_first_board_does_not_prevent_second_board_validation(self):
        scan = SimpleNamespace(
            surface=np.array([[-0.02, -0.01, 0.002], [0.02, 0.015, 0.006]])
        )
        layouts = (RackLayout(None, ((-0.10, 0.9),)), RackLayout(None, ((0.10, 0.85),)))
        args = (scan, layouts, 65, 250, Motion(), 820)
        fastest = automatic_run(*args, verify=lambda *_: None)
        trials = []

        def fail_at_start(scan, run, base, scene, cancel):
            trials.append(len(run.boards))
            return BoardFailure(0, command_time=0.0) if len(run.boards) == 1 else None

        fallback = automatic_run(*args, verify=fail_at_start)
        self.assertEqual(trials, [1, 2])
        self.assertTrue(fallback.fastest_fallback)
        self.assertEqual(fallback.fallback_boards, (0,))
        self.assertEqual(fallback.decisions, fastest.decisions)
        np.testing.assert_array_equal(fallback.ends, fastest.ends)
        self.assertEqual(tuple(b.layout for b in fallback.boards), layouts)

    def test_handoff_failure_changes_alignment_only(self):
        scan = SimpleNamespace(
            surface=np.array([[-0.02, -0.01, 0.002], [0.02, 0.015, 0.006]])
        )
        layouts = (RackLayout(None, ((0.03, 0.85),)),) * 2
        rejected = set()

        def verify(scan, run, base, scene, cancel):
            self.assertEqual(scene, layouts)
            if len(run.boards) == 2 and not rejected:
                rejected.add(run.decisions[0][2])
                return BoardFailure(0, during_idle=True)
            return None

        with self.subTest("committed work"):
            stages = automatic_boards(
                scan, layouts, 65, 250, Motion(), 820, verify=verify
            )
            first, run = stages
            self.assertIs(first.boards[0], run.boards[0])
            self.assertEqual(first.decisions[0][:2], run.decisions[0][:2])
            self.assertTrue(rejected)
            self.assertNotIn(run.decisions[0][2], rejected)
            self.assertFalse(run.fastest_fallback)

    def test_verified_run_passes_on_normal_replay(self):
        from simulator import Cell

        points = np.array(
            [[-0.02, -0.01, 0.002], [0.02, -0.01, 0.002], [0, 0.01, 0.003]]
        )
        scan = SimpleNamespace(
            surface=points, vertices=points, triangles=np.array([[0, 1, 2]])
        )
        layouts = (RackLayout(None, ((0.02, 0.85),)), RackLayout(None, ((0.04, 0.88),)))
        run = make_run(scan, layouts, mode=FULL_AUTO)
        cell = Cell(scan, run)
        try:
            while not cell.finished:
                cell.step()
            self.assertEqual(
                [board["passed"] for board in cell.board_results()], [True, True]
            )
        finally:
            cell.close()

    def test_no_later_candidate_can_change_started_work(self):
        points = np.array([[-0.02, -0.01, 0.002], [0.02, 0.015, 0.006]])
        scan = SimpleNamespace(surface=points)
        layouts = (RackLayout(None, ((0.03, 0.85),)),) * 2
        attempts = []

        def verify(scan, run, base, scene, cancel):
            attempts.append(run)
            return BoardFailure(1) if len(run.boards) == 2 else None

        stages = automatic_boards(scan, layouts, 65, 250, Motion(), 820, verify=verify)
        first = next(stages)
        self.assertEqual(len(attempts), 1)
        final = next(stages)
        self.assertEqual(final.fallback_boards, (1,))
        self.assertGreater(len(attempts), 2)
        self.assertTrue(all(r.boards[0] is first.boards[0] for r in attempts))
        self.assertEqual(final.duration, min(r.duration for r in attempts[1:]))
        stages.close()

    def test_capture_two_four_by_two_grid(self):
        from scan import load_scan, locate_tar
        from simulator import Cell

        archive = locate_tar()
        if archive is None:
            self.skipTest("Original ToF capture archive is required")
        scan = load_scan(archive, 2)
        layout = grid_layout(scan, 4, 2)
        for level in ("", "top"):
            with self.subTest(level=level):
                run = make_run(
                    scan, (layout,), mode=FULL_AUTO, motion=Motion(start_level=level)
                )
                self.assertFalse(run.fastest_fallback)
                cell = Cell(scan, run)
                try:
                    while not cell.finished:
                        cell.step()
                    self.assertTrue(cell.board_results()[0]["passed"])
                finally:
                    cell.close()

    def test_leading_route_on_unsorted_uneven_rows_respects_manual_settings(self):
        # Three products on one row (positive X first), and one on another.
        scan = SimpleNamespace(
            surface=np.array([[-0.01, -0.01, 0.01], [0.01, 0.01, 0.01]])
        )
        layout = RackLayout(None, ((-0.2, 0.8), (0.03, 1.0), (0.2, 0.8), (0.0, 0.8)))
        layouts = (layout, layout)
        for action, advance in (("point", ""), ("continue", "top")):
            motion = Motion(action=action, advance=advance)
            run = automatic_run(
                scan,
                layouts,
                65,
                250,
                motion,
                820,
                mode=LEADING,
                verify=lambda *_: None,
            )
            self.assertFalse(run.decisions)
            self.assertEqual(run.decision(0), (LEADING, action, advance))
            board = run.boards[0]
            passes = [s for s in board.segments if s.spraying]
            self.assertEqual(len(passes), 3)
            lx = sorted((passes[-1].a[0], passes[-1].b[0]))
            np.testing.assert_allclose(lx, [-0.22, -0.18])
            self.assertEqual(len(board.targets), 4 * len(scan.surface))
            if action == "point":
                np.testing.assert_allclose(
                    board.sample(board.duration)[0], np.asarray(motion.point_mm) / 1000
                )
            else:
                self.assertIn(0, run.idle_moves)

    def test_capture_two_four_by_four_chooses_a_passing_leading_route(self):
        from scan import load_scan, locate_tar
        from simulator import Cell

        archive = locate_tar()
        if archive is None:
            self.skipTest("Original ToF capture archive is required")
        scan = load_scan(archive, 2)
        layout = grid_layout(scan, 4, 4)
        for mode in (LEADING, FULL_AUTO):
            with self.subTest(mode=mode):
                run = make_run(
                    scan, (layout,), mode=mode, motion=Motion(start_level="top")
                )
                self.assertEqual(run.decision(0)[0], LEADING)
                self.assertEqual(bool(run.decisions), mode == FULL_AUTO)
                self.assertFalse(run.fastest_fallback)
                cell = Cell(scan, run, visuals=False)
                try:
                    while not cell.finished:
                        cell.step()
                    self.assertTrue(cell.board_result(0)["passed"])
                finally:
                    cell.close()


if __name__ == "__main__":
    unittest.main()
