"""Check the actual iiwa link positions and knife orientation at initialization."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as pb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from planner import DEFAULT_LAYOUT, Motion, make_run
from simulator import JOINTS, TCP, Cell, mirrored_start_joints


class MirroredStartTest(unittest.TestCase):
    surface = np.array([[-0.02, -0.01, 0.002], [0.02, -0.01, 0.002], [0, 0.01, 0.003]])
    scan = SimpleNamespace(
        surface=surface, vertices=surface, triangles=np.array([[0, 1, 2]])
    )

    def assert_mirror(self, client, arms):
        for j in (*JOINTS, TCP):
            a, b = [
                client.getLinkState(arm, j, computeForwardKinematics=True)
                for arm in arms
            ]
            np.testing.assert_allclose(np.array(a[4]) * [1, -1, 1], b[4], atol=5e-7)
        a_rot, b_rot = [
            np.array(pb.getMatrixFromQuaternion(p[5])).reshape(3, 3) for p in (a, b)
        ]
        np.testing.assert_allclose(
            np.diag([1, -1, 1]) @ a_rot @ np.diag([-1, 1, 1]), b_rot, atol=1e-6
        )

    def test_simulator_starts_with_matching_physics_and_solver_poses(self):
        for motion in (Motion(start_level="top"), Motion(start_mm=(-500, 160, 1125))):
            with self.subTest(motion=motion):
                cell = Cell(
                    self.scan, make_run(self.scan, (DEFAULT_LAYOUT,), motion=motion)
                )
                try:
                    self.assertTrue(all(cell.last_ik))
                    self.assert_mirror(cell.world, cell.arms)
                    for arm, solver in zip(cell.arms, cell.solvers):
                        q = np.array(
                            [s[0] for s in cell.world.getJointStates(arm, JOINTS)]
                        )
                        np.testing.assert_array_equal(
                            q, [s[0] for s in cell.kin.getJointStates(solver, JOINTS)]
                        )
                        self.assertTrue(np.all((q >= cell.lo) & (q <= cell.hi)))
                finally:
                    cell.close()

    def test_equivalent_rolls_preserve_the_mirror_at_joint_stops(self):
        cell = Cell(self.scan, make_run(self.scan, (DEFAULT_LAYOUT,)))
        try:
            for degrees in (
                [30, 45, 30, -60, 45, 17, 0],
                [30, 45, 30, -60, 0, 17, 0],
                [30, 45, 0, -60, 0, 17, 0],
            ):
                with self.subTest(degrees=degrees):
                    orange = np.deg2rad(degrees)
                    blue = mirrored_start_joints(orange, cell.lo, cell.hi)
                    self.assertTrue(np.all((blue >= cell.lo) & (blue <= cell.hi)))
                    for solver, q in zip(cell.solvers, (orange, blue)):
                        for j, angle in enumerate(q):
                            cell.kin.resetJointState(solver, j, angle)
                    self.assert_mirror(cell.kin, cell.solvers)
        finally:
            cell.close()


if __name__ == "__main__":
    unittest.main()
