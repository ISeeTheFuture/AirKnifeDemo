"""Reuse deterministic preflight work in the running application."""

import hashlib
import struct
import xml.etree.ElementTree as ET
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import numpy as np
import pybullet_data
from simulator import (
    DT,
    FIRST_BOARD_PREFLIGHT_FRAMES,
    MAX_PREFLIGHT_FRAMES,
    BoardFailure,
    Cell,
)

# Store only small verdicts and digests. Each concurrent search owns its Bullet clients.
_verdicts = OrderedDict()
_lock = Lock()
_missing = object()


def _cached(key):
    with _lock:
        result = _verdicts.get(key, _missing)
        if result is not _missing:
            _verdicts.move_to_end(key)
        return result


def _remember(key, result):
    with _lock:
        _verdicts[key] = result
        _verdicts.move_to_end(key)
        if len(_verdicts) > 256:
            _verdicts.popitem(last=False)


def _add_array(digest, values):
    array = np.ascontiguousarray(values)
    digest.update(repr((array.dtype.str, array.shape)).encode("ascii") + b"\0")
    digest.update(memoryview(array).cast("B"))


def _scene_key(scan, base_mm, layouts):
    model = Path(pybullet_data.getDataPath()) / "kuka_iiwa" / "model.urdf"
    urdf = model.read_bytes()
    digest = hashlib.sha256(urdf)
    # The headless robot still uses all collision meshes referenced by its URDF.
    for mesh in ET.fromstring(urdf).findall(".//collision/geometry/mesh"):
        digest.update(
            hashlib.sha256(
                (model.parent / mesh.attrib["filename"]).read_bytes()
            ).digest()
        )
    digest.update(struct.pack("<dI", base_mm, len(layouts)))
    _add_array(digest, scan.vertices)
    for layout in layouts:
        digest.update(
            struct.pack("<q", layout.columns if layout.columns is not None else -1)
        )
        _add_array(digest, layout.positions)
        _add_array(digest, layout.support_heights)
    return digest.digest()


def _add_segment(digest, segment):
    digest.update(struct.pack("<dd?", segment.t0, segment.t1, segment.spraying))
    _add_array(digest, segment.a)
    _add_array(digest, segment.b)


def _add_board_geometry(digest, run, i):
    board = run.boards[i]
    _add_array(digest, run.starts[i : i + 1])
    digest.update(
        struct.pack(
            "<4dI",
            board.origin_x,
            board.speed,
            board.length,
            board.effective,
            len(board.layout.positions),
        )
    )
    _add_array(digest, board.targets)


def _add_board(digest, run, i):
    board = run.boards[i]
    _add_board_geometry(digest, run, i)
    _add_array(digest, run.ends[i : i + 1])
    _add_array(digest, board.ends)
    for segment in board.segments:
        _add_segment(digest, segment)


def _first_spray_key(run):
    """Identify all commands and exposure inputs through the first board's last spray."""
    board = run.boards[0]
    count = len(board.segments)
    while count and not board.segments[count - 1].spraying:
        count -= 1
    if not count:
        return None
    digest = hashlib.sha256()
    _add_board_geometry(digest, run, 0)
    _add_array(digest, board.ends[:count])
    for segment in board.segments[:count]:
        _add_segment(digest, segment)
    return digest.digest(), count == len(board.segments)


def _boundary_keys(scene_key, run):
    """Sign the complete executed history at each work end and following handoff."""
    digest = hashlib.sha256(scene_key)
    keys = []
    for i in range(len(run.boards)):
        _add_board(digest, run, i)
        work = digest.digest()
        idle = run.idle_moves.get(i)
        digest.update(bytes([idle is not None]))
        if idle is not None:
            _add_segment(digest, idle)
        _add_array(digest, run.starts[i + 1 : i + 2])
        keys.append((work, digest.digest()))
    return keys


@contextmanager
def verification_session(scan, base_mm, layouts, check_cancel):
    """Validate candidates for one physical scene; release all native state on exit."""
    check_cancel()
    scene_key = _scene_key(scan, base_mm, layouts)
    cell = None
    active_keys = []

    def validate(run):
        nonlocal cell, active_keys
        check_cancel()
        keys = _boundary_keys(scene_key, run)
        key = keys[-1][0]
        result = _cached(key)
        if result is not _missing:
            return result
        if cell is not None and cell.run.boards[0].length != run.boards[0].length:
            cell.close()
            cell = None
        if cell is None:
            cell = Cell(
                scan,
                run,
                base_mm,
                measure_clearance=False,
                scene_layouts=layouts,
                visuals=False,
                reuse_ik=True,
            )
        elif cell.verified_boundary is not None:
            i, after_idle = cell.verified_boundary
            # Column zero ends at the last work command, column one includes the wait.
            if i + 1 < len(keys) and keys[i][after_idle] == active_keys[i][after_idle]:
                cell.continue_run(run)
            else:
                cell.start_run(run)
        else:
            cell.start_run(run)
        active_keys = keys
        result = cell.verify(check_cancel)
        check_cancel()
        _remember(key, result)
        return result

    try:
        yield validate
    finally:
        if cell is not None:
            cell.close()


class _Checkpoint:
    """Dynamic solver state and the Python counters at a work end or handoff."""

    def __init__(self, cell):
        self.id = cell.world.saveState()
        self.time_step = cell.substep_time
        # At these boundaries later commands can update contact/IK counts for the
        # finished board during idle, but never spray onto its completed dwell array.
        self.state = {
            name: getattr(cell, name)
            for name in (
                "elapsed",
                "frame",
                "board_index",
                "last_command",
                "last_ik",
                "poses",
                "dwells",
                "ik_ok",
                "ik_total",
                "contact_frames",
                "first_contact",
                "min_clearance",
                "had_contact",
                "spraying",
                "phase",
            )
        }
        self.contacts = cell.board_contacts.copy()
        self.misses = cell.board_ik_misses.copy()
        self.pairs = cell.pairs.copy()

    def restore(self, cell, run):
        cell.world.restoreState(stateId=self.id)
        cell.world.setTimeStep(self.time_step)
        cell.substep_time = self.time_step
        for name, value in self.state.items():
            setattr(cell, name, value)
        cell.board_contacts = self.contacts
        cell.board_ik_misses = self.misses
        cell.pairs = self.pairs.copy()
        cell.verified_boundary = None
        cell.extend_run(run, cell.board_index + 1)

    def release(self, cell):
        cell.world.removeState(self.id)


class _RollingVerifier:
    def __init__(self, scan, base, layouts, cancel):
        self.scan, self.base, self.layouts, self.cancel = scan, base, layouts, cancel
        self.history = _scene_key(scan, base, layouts)
        self.cell = self.endpoint = self.completed = None
        self.handoffs = {}
        self.failed_first_sprays = set()

    def _handoff_key(self, run):
        i = len(run.boards) - 2
        digest = hashlib.sha256(self.history)
        if i >= 0:
            idle = run.idle_moves.get(i)
            if idle:
                _add_segment(digest, idle)
            _add_array(digest, run.starts[-1:])
        return digest.digest()

    def _work_key(self, run, handoff):
        # The digest already includes the entire committed prefix.
        digest = hashlib.sha256(handoff)
        _add_board(digest, run, len(run.boards) - 1)
        return digest.digest()

    def _start(self, run, handoff):
        self.completed = None
        if self.endpoint:
            self.handoffs.get(handoff, self.endpoint).restore(self.cell, run)
        elif self.cell is None:
            self.cell = Cell(
                self.scan,
                run,
                self.base,
                measure_clearance=False,
                scene_layouts=self.layouts,
                visuals=False,
                reuse_ik=True,
            )
        else:
            self.cell.start_run(run)

    def validate(self, run):
        self.cancel()
        handoff = self._handoff_key(run)
        key = self._work_key(run, handoff)
        result = _cached(key)
        if result is not _missing:
            return result
        first = len(run.boards) == 1
        spray_key = _first_spray_key(run) if first else None
        if spray_key in self.failed_first_sprays:
            result = BoardFailure(0)
            _remember(key, result)
            return result
        self._start(run, handoff)

        def remember_handoff():
            if handoff not in self.handoffs:
                self.handoffs[handoff] = _Checkpoint(self.cell)

        result = self.cell.verify(
            self.cancel,
            on_handoff=remember_handoff,
            lookahead_frames=FIRST_BOARD_PREFLIGHT_FRAMES
            if first
            else MAX_PREFLIGHT_FRAMES,
        )
        if result is None:
            self.completed = run
        elif spray_key is not None and result.command_time is None:
            # On board one, reaching work end without any earlier IK/contact
            # failure leaves insufficient dwell as the only cause of rejection.
            self.failed_first_sprays.add(spray_key)
            signature, ends_spraying = spray_key
            end = run.duration
            if ends_spraying and abs(end - round(end / DT) * DT) > 1e-10:
                # Stopping mid-tick grants an extra spraying frame. A dry return
                # after the same passes cannot recover that exposure. Keep the
                # two endings separate near a snapped controller tick.
                self.failed_first_sprays.add((signature, False))
        self.cancel()
        _remember(key, result)
        return result

    def commit(self, run):
        """Materialize only the chosen suffix, including an unavoidably failing board."""
        self.failed_first_sprays.clear()
        handoff = self._handoff_key(run)
        if self.completed is not run:
            self._start(run, handoff)
            for frame in self.cell.commands:
                self.cancel()
                self.cell._apply_frame(frame, diagnostics=False)
        self._release_states()
        self.endpoint = _Checkpoint(self.cell)
        self.history = self._work_key(run, handoff)

    def _release_states(self):
        if self.endpoint:
            self.endpoint.release(self.cell)
            self.endpoint = None
        for checkpoint in self.handoffs.values():
            checkpoint.release(self.cell)
        self.handoffs.clear()

    def close(self):
        if self.cell:
            try:
                self._release_states()
            finally:
                self.cell.close()


@contextmanager
def rolling_verification(scan, base, layouts, cancel):
    cancel()
    verifier = _RollingVerifier(scan, base, layouts, cancel)
    try:
        yield verifier
    finally:
        verifier.close()
