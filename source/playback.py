"""Run playback locally and keep next-board verification in a separate process."""

from concurrent.futures import CancelledError, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from types import SimpleNamespace
from uuid import uuid4

from automatic import automatic_boards
from planner import FULL_AUTO, RunPlan
from simulator import Cell

# Only accessed inside the single worker process. A paused demo and the comparison
# window can each retain their own solver state while sharing the verdict cache.
_sessions = {}


def _discard(session):
    entry = _sessions.pop(session, None)
    if entry:
        boards, reader, _ = entry
        try:
            boards.close()
        finally:
            reader.close()


def _next_board(session):
    if session not in _sessions:
        raise RuntimeError("검증 세션이 종료되었습니다. 처음부터 다시 재생하세요.")
    boards, _, count = _sessions[session]
    try:
        run = next(boards)
        i = len(run.boards) - 1
        # Transfer only the chosen board and its incoming handoff. Large target
        # arrays and commands for previous boards already exist in the foreground.
        result = (
            run.starts[i],
            run.boards[i],
            run.decisions[i] if run.decisions else None,
            run.decision(i - 1)[2] if i else "",
            run.idle_moves.get(i - 1),
            i in run.fallback_boards,
        )
        if i + 1 == count:
            _discard(session)
        return result
    except BaseException:
        _discard(session)
        raise


def _start(
    session, reader, factory, geometry, layouts, speed, length, motion, base, mode
):
    def check_cancel():
        # The parent closes the write end on reset, cancellation or shutdown.
        if reader.poll():
            raise CancelledError()

    boards = factory(
        geometry,
        layouts,
        speed,
        length,
        motion,
        base,
        mode=mode,
        check_cancel=check_cancel,
    )
    _sessions[session] = boards, reader, len(layouts)
    return _next_board(session)


class VerificationWorker:
    """One spawned process for the life of the demo, created on first use."""

    def __init__(self):
        self.context = get_context("spawn")
        self.pool = None

    def start(self, session, scan, layouts, speed, length, motion, base, mode):
        geometry = SimpleNamespace(surface=scan.surface, vertices=scan.vertices)
        reader, writer = self.context.Pipe(duplex=False)
        args = (
            session,
            reader,
            automatic_boards,
            geometry,
            layouts,
            speed,
            length,
            motion,
            base,
            mode,
        )
        try:
            if self.pool is not None:
                try:
                    future = self.pool.submit(_start, *args)
                except BrokenProcessPool:
                    self.close()
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=1, mp_context=self.context)
                future = self.pool.submit(_start, *args)
        except BaseException:
            reader.close()
            writer.close()
            raise
        # The executor's queue transfers the handle asynchronously.
        future.add_done_callback(lambda _: reader.close())
        return writer, future

    def next_board(self, session):
        if self.pool is None:
            raise RuntimeError(
                "검증 프로세스가 종료되었습니다. 처음부터 다시 재생하세요."
            )
        return self.pool.submit(_next_board, session)

    def discard(self, session):
        if self.pool is not None:
            try:
                self.pool.submit(_discard, session)
            except BrokenProcessPool:
                self.close()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False)
            self.pool = None


class AutomaticPlayback:
    def __init__(
        self,
        scan,
        layouts,
        speed,
        length,
        motion,
        base,
        *,
        visuals=True,
        worker=None,
        mode=FULL_AUTO,
    ):
        if not layouts:
            raise ValueError("연속 판 수는 1 이상이어야 합니다.")
        self.scan, self.layouts, self.base, self.visuals = scan, layouts, base, visuals
        self.cell = None
        self.prefix = RunPlan.empty(motion.for_layout(layouts[0]))
        self.session = uuid4().hex
        self.owns_worker = worker is None
        self.worker = worker or VerificationWorker()
        self.cancel, self.pending = self.worker.start(
            self.session, scan, layouts, speed, length, motion, base, mode
        )
        self.closed = False

    def _receive(self):
        self.prefix = self.prefix.append_board(*self.pending.result())
        self.pending = None
        return self.prefix

    def prepare(self, *, block=False):
        if self.cell is None:
            if not block and not self.pending.done():
                return False
            self.cell = Cell(
                self.scan,
                self._receive(),
                self.base,
                scene_layouts=self.layouts,
                visuals=self.visuals,
            )
        return True

    @property
    def finished(self):
        return (
            self.cell is not None
            and self.cell.finished
            and len(self.cell.run.boards) == len(self.layouts)
        )

    def step(self, *, block=False):
        if not self.prepare(block=block) or self.finished:
            return False
        cell = self.cell
        planned = len(cell.run.boards)
        if (
            planned < len(self.layouts)
            and cell.run.index_at(cell.elapsed) == planned - 1
        ):
            if self.pending is None:
                self.pending = self.worker.next_board(self.session)
            if cell.finished:
                if not block and not self.pending.done():
                    return False
                cell.extend_run(self._receive())
        cell.step()
        return True

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancel.close()
        if self.pending:
            self.pending.cancel()
        self.worker.discard(self.session)
        if self.owns_worker:
            self.worker.close()
        if self.cell:
            self.cell.close()
