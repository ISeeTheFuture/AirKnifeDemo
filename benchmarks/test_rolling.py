"""Rolling validation, late handoffs and deterministic continuous playback."""

import os
import sys
import unittest
from collections import OrderedDict
from dataclasses import replace
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from automatic import automatic_boards, automatic_run
from planner import Motion, RackLayout, make_run
from playback import AutomaticPlayback, VerificationWorker
from simulator import JOINTS, Cell
from verification import rolling_verification


def gated_boards(requested, release, stopped, pids, *args, **kwargs):
    # A top-level factory is importable by Windows spawn. The gates are shared
    # with the test so it can deliberately stall the second board in the child.
    plans = automatic_boards(*args, verify=lambda *_: None, **kwargs)
    try:
        for i, event in enumerate(requested):
            pids.append(os.getpid())
            event.set()
            if i == 1:
                while not release.wait(0.01):
                    kwargs["check_cancel"]()
            yield next(plans)
    finally:
        plans.close()
        stopped.set()


def frame_state(cell):
    joints = tuple(
        np.array([q[:2] for q in cell.world.getJointStates(arm, JOINTS)]).tobytes()
        for arm in cell.arms
    )
    dwell = tuple(d.tobytes() for d in cell.dwells[: cell.board_index + 1])
    return cell.elapsed, joints, dwell, cell.had_contact, tuple(cell.last_ik)


class RollingTest(unittest.TestCase):
    points = np.array([[-0.02, -0.01, 0.002], [0.02, -0.01, 0.002], [0, 0.01, 0.003]])
    scan = SimpleNamespace(
        surface=points, vertices=points, triangles=np.array([[0, 1, 2]])
    )
    layouts = tuple(
        RackLayout(None, ((x, z),))
        for x, z in ((0.02, 0.85), (0.04, 0.88), (0.03, 0.86))
    )

    def test_process_selected_boards_and_preview_match_offline_physics(self):
        layouts = self.layouts[:2]
        args = (self.scan, layouts, 65, 250, Motion(), 820)
        expected = automatic_run(*args)
        reference = Cell(self.scan, expected, visuals=False)
        playback = AutomaticPlayback(*args)
        try:
            while not playback.finished:
                self.assertTrue(playback.step(block=True))
                cell = playback.cell
                if cell.frame in (1, 2, 900):
                    # Render both the moving preview and the paused, shadowed view.
                    image, _ = cell.render(120, 80, shadows=cell.frame == 2)
                    self.assertEqual(image.shape, (80, 120, 3))
                reference.step()
                self.assertEqual(frame_state(cell), frame_state(reference))
            self.assertEqual(cell.run.decisions, expected.decisions)
            self.assertEqual(cell.run.fallback_boards, expected.fallback_boards)
            np.testing.assert_array_equal(cell.run.ends, expected.ends)
            self.assertEqual(cell.board_results(), reference.board_results())
        finally:
            playback.close()
            reference.close()

    def test_late_worker_allows_playback_and_only_one_board_ahead(self):
        manager = get_context("spawn").Manager()
        self.addCleanup(manager.shutdown)
        requested = [manager.Event() for _ in self.layouts]
        release, stopped, pids = manager.Event(), manager.Event(), manager.list()
        with patch(
            "playback.automatic_boards",
            partial(gated_boards, requested, release, stopped, pids),
        ):
            playback = AutomaticPlayback(
                self.scan, self.layouts, 65, 250, Motion(), 820, visuals=False
            )
        observed = []
        try:
            self.assertTrue(playback.prepare(block=True))
            cell = playback.cell
            self.assertFalse(requested[1].is_set())
            while not cell.finished:
                self.assertTrue(playback.step())
                observed.append(frame_state(cell))
            self.assertTrue(requested[1].wait(2))
            self.assertFalse(playback.finished)
            self.assertFalse(playback.step())
            self.assertEqual(cell.elapsed, cell.run.ends[0])
            self.assertFalse(requested[2].is_set())
            release.set()
            while not playback.finished:
                if len(cell.run.boards) == 1 or (
                    len(cell.run.boards) == 2 and cell.elapsed < cell.run.starts[1]
                ):
                    self.assertFalse(requested[2].is_set())
                self.assertTrue(playback.step(block=True))
                observed.append(frame_state(cell))
            self.assertTrue(requested[2].is_set())
            self.assertEqual(len(set(pids)), 1)
            self.assertNotIn(os.getpid(), pids)
            reference = Cell(self.scan, cell.run, visuals=False)
            try:
                for frame in observed:
                    reference.step()
                    self.assertEqual(frame, frame_state(reference))
                self.assertTrue(reference.finished)
                self.assertEqual(reference.board_results(), cell.board_results())
                self.assertEqual(reference.result(), cell.result())
            finally:
                reference.close()
        finally:
            release.set()
            playback.close()

    def test_cancel_while_next_board_is_in_flight(self):
        manager = get_context("spawn").Manager()
        self.addCleanup(manager.shutdown)
        requested = [manager.Event() for _ in self.layouts]
        release, stopped, pids = manager.Event(), manager.Event(), manager.list()
        worker = VerificationWorker()
        self.addCleanup(worker.close)
        with patch(
            "playback.automatic_boards",
            partial(gated_boards, requested, release, stopped, pids),
        ):
            playback = AutomaticPlayback(
                self.scan,
                self.layouts,
                65,
                250,
                Motion(),
                820,
                visuals=False,
                worker=worker,
            )
        try:
            self.assertTrue(playback.step(block=True))
            self.assertTrue(requested[1].wait(5))
        finally:
            playback.close()
        self.assertTrue(stopped.wait(5))
        # Fresh physics in the same live process after cancelling a partial scenario.
        again = AutomaticPlayback(
            self.scan,
            self.layouts[:1],
            65,
            250,
            Motion(),
            820,
            visuals=False,
            worker=worker,
        )
        try:
            self.assertTrue(again.prepare(block=True))
            self.assertEqual(again.cell.elapsed, 0)
            self.assertEqual(worker.pool.submit(os.getpid).result(timeout=5), pids[0])
        finally:
            again.close()

    def test_restored_handoff_matches_every_physics_frame_after_retry(self):
        layouts = self.layouts[:2]
        run = make_run(
            self.scan,
            layouts,
            mode="horizontal",
            motion=Motion(action="continue", advance="top"),
        )
        first, second = run.boards
        prefix = replace(
            run,
            boards=(first,),
            starts=run.starts[:1],
            ends=run.ends[:1],
            idle_moves={},
        )
        missed = replace(
            run, boards=(first, replace(second, targets=second.targets + [2.0, 0, 0]))
        )
        reference = Cell(self.scan, run, visuals=False)
        try:
            expected = []
            while not reference.finished:
                reference.step()
                if reference.elapsed > run.starts[1]:
                    expected.append(frame_state(reference))
        finally:
            reference.close()
        observed = []
        apply_frame = Cell._apply_frame

        def record(cell, frame, **kwargs):
            apply_frame(cell, frame, **kwargs)
            observed.append(frame_state(cell))

        with (
            patch("verification._verdicts", OrderedDict()),
            rolling_verification(self.scan, 820, layouts, lambda: None) as verifier,
        ):
            self.assertIsNone(verifier.validate(prefix))
            verifier.commit(prefix)
            self.assertEqual(verifier.validate(missed).index, 1)
            with patch.object(Cell, "_apply_frame", record):
                self.assertIsNone(verifier.validate(run))
        self.assertEqual(len(observed), len(expected))
        for actual, target in zip(observed, expected, strict=True):
            self.assertEqual(actual, target)

    def test_failed_board_endpoint_carries_real_motion_and_contact_history(self):
        layouts = self.layouts[:2]
        scan = SimpleNamespace(
            surface=self.points,
            vertices=self.points + [0, 0, 0.30],
            triangles=self.scan.triangles,
        )
        run = make_run(
            scan, layouts, mode="horizontal", motion=Motion(action="continue")
        )
        prefix = replace(
            run,
            boards=run.boards[:1],
            starts=run.starts[:1],
            ends=run.ends[:1],
            idle_moves={},
        )
        reference = Cell(scan, run, visuals=False)
        try:
            while reference.elapsed < prefix.duration:
                reference.step()
            self.assertGreater(reference.board_contacts[0], 0)
            with (
                patch("verification._verdicts", OrderedDict()),
                rolling_verification(scan, 820, layouts, lambda: None) as verifier,
            ):
                self.assertIsNotNone(verifier.validate(prefix))
                verifier.commit(prefix)
                self.assertEqual(frame_state(verifier.cell), frame_state(reference))
                verifier.validate(run)
                verifier.commit(run)
                while not reference.finished:
                    reference.step()
                self.assertEqual(frame_state(verifier.cell), frame_state(reference))
                self.assertEqual(
                    verifier.cell.board_results(), reference.board_results()
                )
        finally:
            reference.close()
