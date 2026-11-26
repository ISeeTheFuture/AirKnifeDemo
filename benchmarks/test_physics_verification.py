"""Compare lookahead checks with complete, frame-by-frame physical playback."""

import sys
import unittest
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import verification
from planner import Motion, RackLayout, make_run
from simulator import DT, JOINTS, BoardFailure, Cell, first_board_failure


class PhysicsVerificationTest(unittest.TestCase):
    points = np.array([[-0.02, -0.01, 0.002], [0.02, -0.01, 0.002], [0, 0.01, 0.003]])
    scan = SimpleNamespace(
        surface=points, vertices=points, triangles=np.array([[0, 1, 2]])
    )
    layouts = (RackLayout(None, ((0.02, 0.85),)), RackLayout(None, ((0.04, 0.88),)))

    def setUp(self):
        self.run = make_run(
            self.scan,
            self.layouts,
            mode="horizontal",
            motion=Motion(action="continue", advance="top"),
        )

    def replay(self, scan, run, *, visuals=True):
        cell = Cell(scan, run, visuals=visuals)
        try:
            while not cell.finished:
                cell.step()
            return cell.result(), cell.board_results()
        finally:
            cell.close()

    def test_hidden_replay_matches_visible_replay_through_handoff(self):
        self.assertEqual(
            self.replay(self.scan, self.run),
            self.replay(self.scan, self.run, visuals=False),
        )
        self.assertIsNone(
            first_board_failure(self.scan, self.run, 820, self.layouts, lambda: None)
        )

    def test_ik_pass_alone_cannot_accept_contact_or_low_coverage(self):
        obstructed = SimpleNamespace(
            surface=self.points,
            vertices=self.points + [0, 0, 0.30],
            triangles=self.scan.triangles,
        )
        first, second = self.run.boards
        missed = replace(second, targets=second.targets + [2.0, 0, 0])
        for scan, run, index in (
            (obstructed, self.run, 0),
            (self.scan, replace(self.run, boards=(first, missed)), 1),
        ):
            with self.subTest(index=index):
                result, boards = self.replay(scan, run)
                self.assertTrue(all(row["ik_misses"] == 0 for row in boards))
                self.assertFalse(boards[index]["passed"])
                failure = first_board_failure(
                    scan, run, 820, self.layouts, lambda: None
                )
                self.assertEqual(failure.index, index)
                if index == 0:
                    self.assertAlmostEqual(
                        failure.command_time, result["first_contact_s"], places=3
                    )
                else:
                    self.assertIsNone(failure.command_time)
                    self.assertEqual(boards[index]["worst_pct"], 0)

    def test_unreachable_start_never_builds_dynamic_scene(self):
        run = make_run(self.scan, self.layouts, motion=Motion(start_mm=(0, 160, 2000)))
        with patch.object(
            Cell, "_build_world", side_effect=AssertionError("unnecessary dynamics")
        ):
            failure = first_board_failure(
                self.scan, run, 820, self.layouts, lambda: None
            )
        self.assertEqual((failure.index, failure.command_time), (0, 0.0))

    def test_first_board_reuses_only_proven_exposure_failures(self):
        layouts = self.layouts[:1]
        runs = {
            action: make_run(
                self.scan, layouts, mode="target", motion=Motion(action=action)
            )
            for action in ("continue", "home", "point")
        }
        missed = {
            action: replace(
                run,
                boards=(
                    replace(run.boards[0], targets=run.boards[0].targets + [2, 0, 0]),
                ),
            )
            for action, run in runs.items()
        }
        for first, second, expected_replays in (
            ("continue", "home", 1),  # A dry return cannot add exposure.
            ("home", "point", 1),
            ("home", "continue", 2),  # Stopping can add a fractional spray frame.
        ):
            with (
                self.subTest(first=first, second=second),
                patch.object(verification, "_verdicts", OrderedDict()),
                verification.rolling_verification(
                    self.scan, 820, layouts, lambda: None
                ) as verifier,
                patch.object(verifier, "_start", wraps=verifier._start) as replay,
            ):
                self.assertEqual(verifier.validate(missed[first]), BoardFailure(0))
                self.assertEqual(verifier.validate(missed[second]), BoardFailure(0))
                self.assertEqual(replay.call_count, expected_replays)
                self.assertIsNone(verifier.validate(runs[second]))
                self.assertEqual(replay.call_count, expected_replays + 1)
                with (
                    verification.rolling_verification(
                        self.scan, 820, layouts, lambda: None
                    ) as repeat,
                    patch.object(
                        repeat,
                        "_start",
                        side_effect=AssertionError("replayed cached result"),
                    ),
                ):
                    self.assertEqual(repeat.validate(missed[second]), BoardFailure(0))

    def test_exposure_shortcut_excludes_snapped_ticks_and_timed_failures(self):
        layouts = self.layouts[:1]
        for snap, failure in (
            (True, BoardFailure(0)),
            (False, BoardFailure(0, command_time=0.5)),
        ):
            with (
                self.subTest(snap=snap),
                patch.object(verification, "_verdicts", OrderedDict()),
                verification.rolling_verification(
                    self.scan, 820, layouts, lambda: None
                ) as verifier,
                patch.object(Cell, "verify", side_effect=[failure, None]) as verify,
            ):
                for action, expected in (("continue", failure), ("home", None)):
                    run = make_run(
                        self.scan, layouts, mode="target", motion=Motion(action=action)
                    )
                    if snap:
                        board = run.boards[0]
                        end = next(s.t1 for s in reversed(board.segments) if s.spraying)
                        scale = (round(end / DT) * DT) / end
                        segments = [
                            replace(s, t0=s.t0 * scale, t1=s.t1 * scale)
                            for s in board.segments
                        ]
                        board = replace(
                            board, segments=segments, ends=board.ends * scale
                        )
                        run = replace(run, boards=(board,), ends=run.ends * scale)
                    self.assertEqual(verifier.validate(run), expected)
                self.assertEqual(verify.call_count, 2)

    def test_verified_work_and_idle_resume_with_identical_frame_states(self):
        def state(cell):
            joints = tuple(
                np.array(
                    [s[:2] for s in cell.world.getJointStates(arm, JOINTS)]
                ).tobytes()
                for arm in cell.arms
            )
            exposure = tuple(d.tobytes() for d in cell.dwells[: cell.board_index + 1])
            return cell.elapsed, joints, exposure, cell.had_contact

        reference = []
        cell = Cell(self.scan, self.run)
        try:
            while not cell.finished:
                cell.step()
                reference.append(state(cell))
        finally:
            cell.close()

        first, second = self.run.boards
        prefix = replace(
            self.run,
            boards=(first,),
            starts=self.run.starts[:1],
            ends=self.run.ends[:1],
            idle_moves={},
        )
        unreachable = [
            replace(s, a=s.a + [0, 0, 2], b=s.b + [0, 0, 2]) for s in second.segments
        ]
        bad = replace(self.run, boards=(first, replace(second, segments=unreachable)))
        observed = []
        step = Cell._apply_frame

        def record(cell, command, **kwargs):
            step(cell, command, **kwargs)
            observed.append(state(cell))

        with (
            patch.object(verification, "_verdicts", OrderedDict()),
            patch.object(Cell, "_apply_frame", record),
        ):
            with verification.verification_session(
                self.scan, 820, self.layouts, lambda: None
            ) as validate:
                self.assertIsNone(validate(prefix))
                self.assertEqual(validate(bad).index, 1)
                self.assertIsNone(validate(self.run))
            with (
                patch.object(
                    Cell, "verify", side_effect=AssertionError("replayed cached result")
                ),
                verification.verification_session(
                    self.scan, 820, self.layouts, lambda: None
                ) as validate,
            ):
                self.assertEqual(validate(bad).index, 1)
                self.assertIsNone(validate(self.run))
        self.assertEqual(observed, reference)

    def test_cached_pass_does_not_mask_changed_exposure_or_collision_geometry(self):
        first, second = self.run.boards
        changed_targets = replace(
            self.run,
            boards=(first, replace(second, targets=second.targets + [2, 0, 0])),
        )
        obstructed = SimpleNamespace(
            surface=self.points,
            vertices=self.points + [0, 0, 0.3],
            triangles=self.scan.triangles,
        )
        with patch.object(verification, "_verdicts", OrderedDict()):
            with verification.verification_session(
                self.scan, 820, self.layouts, lambda: None
            ) as validate:
                self.assertIsNone(validate(self.run))
                self.assertEqual(validate(changed_targets).index, 1)
            with verification.verification_session(
                obstructed, 820, self.layouts, lambda: None
            ) as validate:
                self.assertEqual(
                    validate(self.run),
                    first_board_failure(
                        obstructed, self.run, 820, self.layouts, lambda: None
                    ),
                )
                self.assertIsNotNone(validate(self.run))

    def test_ik_failure_during_advance_belongs_to_previous_board(self):
        layouts = (self.layouts[0], RackLayout(None, ((0.04, 1.80),)))
        run = make_run(
            self.scan,
            layouts,
            mode="horizontal",
            motion=Motion(action="continue", advance="top"),
        )
        failure = first_board_failure(self.scan, run, 820, layouts, lambda: None)
        self.assertEqual((failure.index, failure.during_idle), (0, True))
        self.assertLess(run.ends[0], failure.command_time)
        self.assertLess(failure.command_time, run.starts[1])
        cell = Cell(self.scan, run)
        try:
            while not cell.finished and not cell.board_ik_misses[0]:
                cell.step()
            self.assertEqual(cell.elapsed, failure.command_time)
        finally:
            cell.close()


if __name__ == "__main__":
    unittest.main()
