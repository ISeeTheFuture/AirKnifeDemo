"""PyBullet execution adapter. No robot SDK or field-control interface."""

import math
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as pb
import pybullet_data
from planner import (
    BEAM_WIDTH,
    FACE,
    MIN_DWELL,
    PITCH,
    RACK_BOTTOM,
    RACK_HEIGHT,
    START_X,
    Motion,
    RunPlan,
)
from pybullet_utils.bullet_client import BulletClient

JOINTS = list(range(7))
TCP = 7
DT = 1 / 60
SUBSTEPS = 4
MAX_PREFLIGHT_FRAMES = round(10 / DT)
FIRST_BOARD_PREFLIGHT_FRAMES = round(20 / DT)
SIDES = (-1, 1)
BASE_Z = 0.47


@dataclass(slots=True)
class ControlFrame:
    board: int
    time: float
    next_tick: int
    goal: np.ndarray
    spraying: bool
    phase: str
    joints: list[np.ndarray]
    ik_ok: list[bool]


def tool_urdf(folder, length):
    original = Path(pybullet_data.getDataPath()) / "kuka_iiwa" / "model.urdf"
    tree = ET.parse(original)
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        mesh.set(
            "filename", (original.parent / mesh.get("filename")).resolve().as_posix()
        )
    # Fixed link origin is the centre of the discharge slit.
    root.append(
        ET.fromstring(f"""<link name="airknife_tcp">
      <inertial><origin xyz="0 0 -0.045"/><mass value="0.8"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.005"/></inertial>
      <visual><origin xyz="0 0 -0.035"/><geometry><box size="0.040 {length} 0.070"/></geometry>
      <material name="knife"><color rgba="0.10 0.70 0.72 1"/></material></visual>
      <collision><origin xyz="0 0 -0.035"/><geometry><box size="0.040 {length} 0.070"/></geometry></collision>
      </link>""")
    )
    root.append(
        ET.fromstring("""<joint name="knife_mount" type="fixed">
      <parent link="lbr_iiwa_link_7"/><child link="airknife_tcp"/><origin xyz="0 0 0.09"/></joint>""")
    )
    dest = Path(folder) / "robot.urdf"
    tree.write(dest, encoding="utf-8", xml_declaration=True)
    return str(dest)


def angle_error(q, r):
    return float(np.rad2deg(2 * np.arccos(np.clip(abs(np.dot(q, r)), 0, 1))))


def mirrored_start_joints(orange, lo, hi):
    """Reflect the iiwa arm across world Y=0, with the knife facing the other side."""
    blue = orange.copy()
    blue[::2] *= -1  # Reverse J1, J3, J5 and J7 in the opposing base frame.
    # The opposite knife frame needs a half turn. Prefer wrist roll (J7).
    # If it hits a stop, an upstream roll and reversed downstream bends give
    # the same mirrored joint centres and tool frame on this iiwa chain.
    for roll in (6, 4, 2, 0):
        turned = math.remainder(blue[roll] + math.pi, math.tau)
        if lo[roll] <= turned <= hi[roll]:
            blue[roll] = turned
            blue[roll + 1 :: 2] *= -1
            return blue
    # No legal half-turn representation. Keep the arm reflected and assess
    # the actual tool pose at the wrist stop with the usual FK tolerances.
    blue[6] = np.clip(math.remainder(blue[6] + math.pi, math.tau), lo[6], hi[6])
    return blue


class Cell:
    def __init__(
        self,
        scan,
        plan,
        base_mm=820,
        *,
        measure_clearance=True,
        scene_layouts=None,
        visuals=True,
        reuse_ik=False,
    ):
        self.run = (
            plan
            if isinstance(plan, RunPlan)
            else RunPlan((plan,), np.array([0.0]), np.array([plan.duration]), Motion())
        )
        self.board_index = 0
        self.plan = self.run.boards[0]
        self.base_y = base_mm / 1000
        self._ik_cache = OrderedDict() if reuse_ik else None
        self.measure_clearance = measure_clearance
        self.visuals = visuals
        self.scene_layouts = scene_layouts
        self.scene_target_count = (
            sum(len(layout.positions) for layout in scene_layouts) * len(scan.surface)
            if scene_layouts is not None
            else None
        )
        self.scan = scan
        self.world = None
        self.kin = BulletClient(connection_mode=pb.DIRECT)
        self.files = tempfile.TemporaryDirectory(prefix="airknife-model-")
        try:
            self.robot_file = tool_urdf(self.files.name, self.plan.length)
            self._build_kinematics()
            self.start_run(self.run)
        except Exception:
            self.close()
            raise

    def close(self):
        if hasattr(self, "commands"):
            self.commands.close()
        for client in (self.world, self.kin):
            if client is not None and client.isConnected():
                client.disconnect()
        self.files.cleanup()

    def _box_shapes(self, size, color):
        half = (np.array(size) / 2).tolist()
        return (
            self.world.createCollisionShape(pb.GEOM_BOX, halfExtents=half),
            -1
            if not self.visuals
            else self.world.createVisualShape(
                pb.GEOM_BOX, halfExtents=half, rgbaColor=color
            ),
        )

    def _box(self, pos, size, color):
        co, vi = self._box_shapes(size, color)
        return self.world.createMultiBody(0, co, vi, pos)

    def _rack_parts(self, scan, layout):
        parts = []
        metal = (0.40, 0.55, 0.60, 1)

        def bar(pos, size):
            co, vi = self._box_shapes(size, metal)
            parts.append((pos, co, vi))

        mid = RACK_BOTTOM + RACK_HEIGHT / 2
        for x in (-0.344, 0.344):
            bar((x, 0, mid), (0.012, 0.030, RACK_HEIGHT))
        for z in (
            RACK_BOTTOM,
            RACK_BOTTOM + RACK_HEIGHT,
            *reversed(layout.support_heights),
        ):
            bar((0, 0, z), (0.70, 0.028, 0.015))
        bar((0, 0, 1.452), (0.020, 0.025, 0.124))
        for side in SIDES:
            v = scan.vertices
            coords = np.column_stack((v[:, 0], side * (FACE + v[:, 2]), v[:, 1]))
            vi = -1
            if self.visuals:
                faces = scan.triangles if side == -1 else scan.triangles[:, ::-1]
                # Render both sides of the open measured surface.
                vi = self.world.createVisualShape(
                    pb.GEOM_MESH,
                    vertices=coords.tolist(),
                    indices=np.concatenate((faces, faces[:, ::-1])).ravel().tolist(),
                    rgbaColor=(0.88, 0.60, 0.20, 1),
                )
            lo, hi = coords.min(axis=0), coords.max(axis=0)
            # Conservative envelope down to jig face, even for ring-shaped parts.
            lo[1], hi[1] = min(0, lo[1]), max(0, hi[1])
            co = self.world.createCollisionShape(
                pb.GEOM_BOX,
                halfExtents=np.maximum((hi - lo) / 2, 0.001).tolist(),
                collisionFramePosition=((hi + lo) / 2).tolist(),
            )
            for u, z in layout.positions:
                parts.append(((u, 0, z), co, vi))
        return parts

    def _rack_body(self, parts):
        n = len(parts)
        return self.world.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            linkMasses=[0.0] * n,
            linkCollisionShapeIndices=[p[1] for p in parts],
            linkVisualShapeIndices=[p[2] for p in parts],
            linkPositions=[p[0] for p in parts],
            linkOrientations=[[0, 0, 0, 1]] * n,
            linkInertialFramePositions=[[0, 0, 0]] * n,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
            linkParentIndices=[0] * n,
            linkJointTypes=[pb.JOINT_FIXED] * n,
            linkJointAxis=[[0, 0, 1]] * n,
        )

    def _build_world(self):
        self.world = w = BulletClient(connection_mode=pb.DIRECT)
        w.setRealTimeSimulation(0)
        w.setGravity(0, 0, -9.81)
        self.substep_time = DT / SUBSTEPS
        w.setTimeStep(self.substep_time)
        w.setPhysicsEngineParameter(numSolverIterations=50)
        self.names = {}
        self.names[self._box((0, 0, -0.045), (6, 4, 0.08), (0.62, 0.69, 0.71, 1))] = (
            "floor"
        )
        self.names[
            self._box((0, 0, 1.54), (5, 0.065, 0.065), (0.24, 0.34, 0.39, 1))
        ] = "rail"
        self.racks = {}
        layouts = (
            self.scene_layouts
            if self.scene_layouts is not None
            else [b.layout for b in self.run.boards]
        )
        parts_by_layout = {}
        # Keep each multibody below PyBullet's link limit even on a full rack.
        for i, layout in enumerate([*layouts, layouts[-1], layouts[0]]):
            offset = PITCH if i == len(layouts) + 1 else -i * PITCH
            label = "rack" if i == 0 else f"rack_{i + 1}"
            if layout not in parts_by_layout:
                parts_by_layout[layout] = self._rack_parts(self.scan, layout)
            parts = parts_by_layout[layout]
            for start in range(0, len(parts), 100):
                body = self._rack_body(parts[start : start + 100])
                self.racks[body] = offset
                self.names[body] = (
                    label if start == 0 else f"{label}_{start // 100 + 1}"
                )
        self._move_racks(0)
        self.arms = []
        flags = pb.URDF_USE_SELF_COLLISION | pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        if not self.visuals:
            flags |= pb.URDF_IGNORE_VISUAL_SHAPES
        for side, (base, bq), q in zip(SIDES, self.bases, self.start_joints):
            self.names[
                self._box(
                    (0, side * self.base_y, BASE_Z / 2),
                    (0.29, 0.29, BASE_Z),
                    (0.19, 0.28, 0.31, 1),
                )
            ] = "pedestal"
            arm = w.loadURDF(
                self.robot_file,
                base,
                bq,
                useFixedBase=True,
                flags=flags,
            )
            self.names[arm] = "arm_A" if side == -1 else "arm_B"
            self.arms.append(arm)
            for j, v in zip(JOINTS, q):
                w.resetJointState(arm, j, v)
                if self.visuals:
                    color = (
                        (0.95, 0.48, 0.16, 1) if side == -1 else (0.28, 0.52, 0.72, 1)
                    )
                    if j % 2:
                        color = (0.35, 0.41, 0.45, 1)
                    w.changeVisualShape(arm, j, rgbaColor=color)
            self._drive(arm, q)
        self.poses = [self._pose(w, a) for a in self.arms]

    def _build_kinematics(self):
        self.bases = [
            (
                (0, side * self.base_y, BASE_Z),
                pb.getQuaternionFromEuler((0, 0, -side * math.pi / 2)),
            )
            for side in SIDES
        ]
        self.solvers = [
            self.kin.loadURDF(
                self.robot_file,
                base,
                bq,
                useFixedBase=True,
                flags=pb.URDF_IGNORE_VISUAL_SHAPES | pb.URDF_IGNORE_COLLISION_SHAPES,
            )
            for base, bq in self.bases
        ]
        self.rot = [
            pb.getQuaternionFromEuler((math.pi / 2, 0, math.pi if side == -1 else 0))
            for side in SIDES
        ]
        info = [self.kin.getJointInfo(self.solvers[0], j) for j in JOINTS]
        self.lo = np.array([p[8] for p in info])
        self.hi = np.array([p[9] for p in info])
        self.forces = [min(220, float(p[10])) for p in info]

    def start_run(self, run):
        """Start from frame zero using the same cell geometry and mounted tool."""
        if run.boards[0].length != self.run.boards[0].length:
            raise ValueError("공구 길이가 바뀌면 새 작업 셀이 필요합니다.")
        # Dynamics (including solver contacts and motor state) always start fresh.
        # The kinematic client has no time integration. Reset its joint seeds
        # before calculating the mirrored starting poses for the next candidate.
        if self.world is not None:
            self.world.disconnect()
            self.world = None
        if hasattr(self, "commands"):
            self.commands.close()
        self.run = run
        self.board_index = 0
        self.plan = run.boards[0]
        start, _, _ = self.plan.sample(0)
        for j, v in enumerate((0, 0.45, 0, -1.45, 0, 1.1, 0)):
            self.kin.resetJointState(self.solvers[0], j, v)
        orange_target = start * (1, SIDES[0], 1)
        for _ in range(4):
            orange, _, _ = self._solve(0, orange_target)
        blue = mirrored_start_joints(orange, self.lo, self.hi)
        self.start_joints = orange, blue
        self.board_ik_misses = [0] * len(self.run.boards)
        self.board_contacts = [0] * len(self.run.boards)
        self.last_ik = []
        for k, (ghost, q) in enumerate(zip(self.solvers, self.start_joints)):
            self.kin.resetJointStatesMultiDof(ghost, JOINTS, q[:, None].tolist())
            p, r = self._pose(self.kin, ghost)
            ep = np.linalg.norm(p - start * (1, SIDES[k], 1))
            self.last_ik.append(
                bool(ep <= 0.004 and angle_error(r, self.rot[k]) <= 3.0)
            )
            self.board_ik_misses[0] += int(not self.last_ik[-1])
        self.frame = 0
        self.elapsed = 0.0
        self.phase = "대기"
        self.spraying = False
        self.dwells = [np.zeros((2, len(b.targets))) for b in self.run.boards]
        self.dwell = self.dwells[0]
        self.errors = []
        self.angle_errors = []
        self.ik_ok = 0
        self.ik_total = 0
        self.contact_frames = 0
        self.pairs = set()
        self.first_contact = None
        self.min_clearance = 0.15
        self.had_contact = False
        self.verified_boundary = None
        self.last_command = None
        self.commands = self._control_frames()
        if self.visuals:
            self._build_world()

    def continue_run(self, run):
        """Continue an identical verified prefix in its original dynamics world."""
        index, _ = self.verified_boundary
        self.extend_run(run, index + 1)

    def extend_run(self, run, keep=None):
        """Attach new work without resetting the clock, dynamics or applied frames."""
        keep = len(self.run.boards) if keep is None else keep
        self.run = run
        self.plan = run.boards[self.board_index]
        self.dwells = self.dwells[:keep] + [
            np.zeros((2, len(b.targets))) for b in run.boards[keep:]
        ]
        self.dwell = self.dwells[self.board_index]
        self.board_contacts = self.board_contacts[:keep] + [0] * (
            len(run.boards) - keep
        )
        self.board_ik_misses = self.board_ik_misses[:keep] + [0] * (
            len(run.boards) - keep
        )
        # IK lookahead may already have tested a different suffix. Rewind only
        # the two feedback-free IK models to the last applied motor commands.
        command = self.last_command
        if command:
            for ghost, q in zip(self.solvers, command.joints):
                self.kin.resetJointStatesMultiDof(ghost, JOINTS, q[:, None].tolist())
        self.commands.close()
        self.commands = self._control_frames(
            self.elapsed, command.next_tick if command else 1
        )

    def _drive(self, arm, q):
        set_motor = self.world.setJointMotorControl2
        for j, v, force in zip(JOINTS, q, self.forces):
            set_motor(
                arm,
                j,
                pb.POSITION_CONTROL,
                targetPosition=v,
                force=force,
                maxVelocity=2.2,
                positionGain=0.35,
                velocityGain=1.0,
            )

    @staticmethod
    def _pose(client, arm):
        state = client.getLinkState(arm, TCP, computeForwardKinematics=True)
        return np.array(state[4]), np.array(state[5])

    def _move_racks(self, t):
        rack_x = START_X + self.plan.speed * t
        set_pose = self.world.resetBasePositionAndOrientation
        for body, offset in self.racks.items():
            set_pose(body, (rack_x + offset, 0, 0), (0, 0, 0, 1))

    def _solve(self, k, pos):
        ghost = self.solvers[k]
        qq = self.rot[k]
        cur = np.array([st[0] for st in self.kin.getJointStates(ghost, JOINTS)])
        cache = self._ik_cache
        if cache is not None:
            key = k, cur.tobytes(), pos.tobytes()
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                self.kin.resetJointStatesMultiDof(
                    ghost, JOINTS, cached[0][:, None].tolist()
                )
                return cached
        sol = self.kin.calculateInverseKinematics(
            ghost,
            TCP,
            pos,
            qq,
            jointDamping=[0.04] * 7,
            maxNumIterations=55,
            residualThreshold=0.0002,
        )
        near = cur + (np.asarray(sol) - cur + np.pi) % (2 * np.pi) - np.pi
        q = np.clip(near, self.lo, self.hi)
        self.kin.resetJointStatesMultiDof(ghost, JOINTS, q[:, None].tolist())
        p, r = self._pose(self.kin, ghost)
        solved = q, np.linalg.norm(p - pos), angle_error(r, qq)
        if cache is not None:
            if len(cache) >= 8192:
                cache.popitem(last=False)
            cache[key] = solved
        return solved

    def _contacts(self):
        hits = set()
        for arm in self.arms:
            for c in self.world.getContactPoints(bodyA=arm):
                a, b, la, lb = c[1:5]
                if c[8] > 0:
                    continue
                if self.names.get(b) == "pedestal" and la == -1:
                    continue
                if b in self.racks or (a != b and b in self.arms):
                    self.min_clearance = min(self.min_clearance, c[8])
                hits.add(
                    tuple(
                        sorted(
                            (
                                f"{self.names.get(a, a)}:{la}",
                                f"{self.names.get(b, b)}:{lb}",
                            )
                        )
                    )
                )
        return hits

    def _measure_clearance(self):
        pairs = [(arm, rack) for arm in self.arms for rack in self.racks] + [
            tuple(self.arms)
        ]
        for a, b in pairs:
            near = self.world.getClosestPoints(a, b, 0.15)
            if near:
                self.min_clearance = min(self.min_clearance, min(c[8] for c in near))

    @property
    def finished(self):
        return self.elapsed >= self.run.duration

    def _control_frames(self, elapsed=0.0, next_tick=1):
        # The IK client is independent of rigid-body feedback. Both preflight and
        # playback use this exact sequence, including fractional handoff frames.
        while elapsed < self.run.duration:
            index = self.run.index_at(elapsed)
            boundary = self.run.next_boundary(elapsed, index)
            tick = next_tick * DT
            t_cmd = boundary if boundary <= tick + 1e-12 else tick
            if t_cmd >= tick - 1e-12:
                next_tick += 1
            goal, spraying, phase = self.run.sample(t_cmd, index)
            joints, ik_ok = [], []
            for k, side in enumerate(SIDES):
                pos = goal.copy()
                pos[1] *= side
                q, ep, ea = self._solve(k, pos)
                joints.append(q)
                ik_ok.append(bool(ep <= 0.004 and ea <= 3.0))
            yield ControlFrame(
                index, t_cmd, next_tick, goal, spraying, phase, joints, ik_ok
            )
            elapsed = t_cmd

    def step(self):
        if not self.finished:
            self._apply_frame(next(self.commands))

    def _apply_frame(self, command, *, diagnostics=True):
        self.verified_boundary = None
        self.last_command = command
        if self.world is None:
            self._build_world()
        index = self.board_index = command.board
        self.plan = self.run.boards[index]
        self.dwell = self.dwells[index]
        t_cmd = command.time
        step_time = t_cmd - self.elapsed
        substep_time = (DT if abs(step_time - DT) < 1e-10 else step_time) / SUBSTEPS
        if substep_time != self.substep_time:
            self.world.setTimeStep(substep_time)
            self.substep_time = substep_time
        previous = self.elapsed
        goal, self.spraying, self.phase = command.goal, command.spraying, command.phase
        self.last_ik = command.ik_ok
        ok = sum(self.last_ik)
        self.board_ik_misses[index] += len(SIDES) - ok
        self.ik_ok += ok
        self.ik_total += len(SIDES)
        for arm, q in zip(self.arms, command.joints):
            self._drive(arm, q)
        hits = set()
        step_simulation = self.world.stepSimulation
        for i in range(SUBSTEPS):
            tick = previous + (i + 1) * step_time / SUBSTEPS
            self._move_racks(tick)
            step_simulation()
            hits.update(self._contacts())
        self.frame += 1
        self.elapsed = t_cmd
        self.poses = [self._pose(self.world, a) for a in self.arms]
        self.had_contact = bool(hits)
        if hits:
            self.board_contacts[index] += 1
            self.contact_frames += 1
            self.pairs.update(hits)
            if self.first_contact is None:
                self.first_contact = self.elapsed
        last_board = self.scene_layouts is None or index + 1 == len(self.scene_layouts)
        if self.measure_clearance and (
            self.frame % 6 == 0 or (last_board and self.finished)
        ):
            self._measure_clearance()
        for k, (tcp, rot) in enumerate(self.poses):
            side = SIDES[k]
            if not diagnostics and (not self.spraying or hits):
                continue
            ae = angle_error(rot, self.rot[k])
            if diagnostics:
                pos = goal.copy()
                pos[1] *= side
                self.errors.append(float(np.linalg.norm(tcp - pos)))
                self.angle_errors.append(ae)
            if self.spraying and ae <= 10.0 and not hits:
                surface = self.plan.targets.copy()
                surface[:, 0] += self.active_rack_x
                surface[:, 1] *= side
                R = np.array(pb.getMatrixFromQuaternion(rot)).reshape(3, 3)
                local = (surface - tcp) @ R
                inside = (
                    (np.abs(local[:, 0]) <= BEAM_WIDTH / 2)
                    & (np.abs(local[:, 1]) <= self.plan.effective / 2)
                    & (local[:, 2] >= 0.025)
                    & (local[:, 2] <= 0.085)
                )
                self.dwell[k, inside] += step_time
        if self.elapsed == self.run.ends[index]:
            self.spraying = False
            self.phase = "완료" if last_board and self.finished else "판 작업 완료"

    @property
    def active_rack_x(self):
        return START_X + self.plan.speed * self.elapsed - self.board_index * PITCH

    def board_result(self, i):
        b = self.run.boards[i]
        hits = self.dwells[i] >= MIN_DWELL
        per_face = hits.reshape(2, len(b.layout.positions), -1).mean(axis=2)
        worst = float(100 * per_face.min())
        return {
            "start": float(self.run.starts[i]),
            "end": float(self.run.ends[i]),
            "work_s": float(b.duration),
            "worst_pct": worst,
            "contacts": self.board_contacts[i],
            "ik_misses": self.board_ik_misses[i],
            "passed": self.elapsed >= self.run.ends[i]
            and worst >= 95
            and self.board_contacts[i] == 0
            and self.board_ik_misses[i] == 0,
        }

    def board_results(self):
        return [self.board_result(i) for i in range(len(self.run.boards))]

    def result(self):
        return {
            "elapsed_s": round(float(self.elapsed), 3),
            "plan_s": round(float(self.run.duration), 3),
            "arrival_interval_s": round(PITCH / self.plan.speed, 3),
            "margin_s": round(
                len(self.run.boards) * PITCH / self.plan.speed - self.run.duration, 3
            ),
            "coverage_pct": [
                round(
                    100
                    * sum(np.count_nonzero(d[k] >= MIN_DWELL) for d in self.dwells)
                    / (self.scene_target_count or sum(d.shape[1] for d in self.dwells)),
                    2,
                )
                for k in range(2)
            ],
            "ik_ok_pct": round(100 * self.ik_ok / self.ik_total, 2)
            if self.ik_total
            else None,
            "tcp_p95_mm": round(float(np.percentile(self.errors, 95)) * 1000, 2)
            if self.errors
            else None,
            "tcp_max_mm": round(max(self.errors) * 1000, 2) if self.errors else None,
            "angle_max_deg": round(max(self.angle_errors), 2)
            if self.angle_errors
            else None,
            "contact_frames": self.contact_frames,
            "first_contact_s": self.first_contact,
            "contact_pairs": [list(pair) for pair in sorted(self.pairs)],
            "rack_or_peer_clearance_mm_capped_at_150": round(
                1000 * self.min_clearance, 2
            ),
        }

    def verify(
        self, check_cancel, *, on_handoff=None, lookahead_frames=MAX_PREFLIGHT_FRAMES
    ):
        """Check the current run at full physics resolution, without playback pacing."""
        run = self.run
        if self.frame == 0 and self.board_ik_misses[0]:
            return BoardFailure(0, command_time=0.0)
        pending = []
        for command in self.commands:
            check_cancel()
            i = command.board
            if not all(command.ik_ok):
                return BoardFailure(i, command.time > run.ends[i], command.time)
            pending.append(command)
            work_end = command.time == run.ends[i]
            handoff = i + 1 < len(run.boards) and command.time == run.starts[i + 1]
            if not (work_end or handoff or len(pending) >= lookahead_frames):
                continue
            # Replay every checked frame exactly once before accepting this board
            # or its handoff. These very joint targets drive the dynamic arms.
            for frame in pending:
                check_cancel()
                self._apply_frame(frame, diagnostics=False)
                if self.had_contact:
                    return BoardFailure(i, self.elapsed > run.ends[i], self.elapsed)
            pending.clear()
            if work_end and not self.board_result(i)["passed"]:
                return BoardFailure(i)
            if work_end or handoff:
                self.verified_boundary = (i, handoff and not work_end)
            if handoff and not work_end and on_handoff is not None:
                on_handoff()
        return None

    def render(self, width, height, yaw=36, pitch=-18, distance=2.55, *, shadows=True):
        cell_center = (0.0, 0.0, 0.82)
        view = pb.computeViewMatrixFromYawPitchRoll(
            cell_center, distance, yaw, pitch, 0, 2
        )
        proj = pb.computeProjectionMatrixFOV(40, width / height, 0.035, 10)
        im = self.world.getCameraImage(
            width,
            height,
            view,
            proj,
            lightDirection=[-2, -4, 6],
            shadow=int(shadows),
            renderer=pb.ER_TINY_RENDERER,
            flags=pb.ER_NO_SEGMENTATION_MASK,
        )
        rgb = np.array(im[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()
        matrix = np.reshape(proj, (4, 4), order="F") @ np.reshape(
            view, (4, 4), order="F"
        )
        return rgb, matrix


@dataclass(frozen=True)
class BoardFailure:
    index: int
    during_idle: bool = False
    # Only IK/contact can eliminate continuations before the end of the work.
    command_time: float | None = None


def first_board_failure(scan, run, base_mm, layouts, check_cancel):
    """Reject definite IK failures before paying for the same interval's dynamics."""
    check_cancel()
    cell = Cell(
        scan,
        run,
        base_mm,
        measure_clearance=False,
        scene_layouts=layouts,
        visuals=False,
    )
    try:
        return cell.verify(check_cancel)
    finally:
        cell.close()
