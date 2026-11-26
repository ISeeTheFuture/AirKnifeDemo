"""Select one board at a time from the preceding board's committed work end."""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
from planner import (
    ADVANCE_OPTIONS,
    FULL_AUTO,
    LEADING,
    PITCH,
    ROUTES,
    START_X,
    Plan,
    RunPlan,
    Segment,
    _idle_move,
    layout_targets,
    route_candidates,
)


def _command(segment):
    return (
        segment.t0,
        segment.t1,
        segment.spraying,
        segment.a.tobytes(),
        segment.b.tobytes(),
    )


@dataclass
class Choice:
    board: Plan
    decision: tuple | None
    advance: str
    idle: Segment | None
    work: tuple
    handoff: tuple | None


def _choices(scan, layouts, speed_mm, length_mm, prefix, check_cancel, mode, base_mm):
    i = len(prefix.boards)
    motion = prefix.motion
    layout = layouts[i]
    speed = speed_mm / 1000
    arrival = i * PITCH / speed
    previous_end = prefix.duration if i else 0.0
    begin = max(previous_end, arrival)
    position = (
        prefix.boards[-1].sample(prefix.boards[-1].duration)[0]
        if i
        else np.asarray(motion.start_mm) / 1000
    )
    advances = (
        ADVANCE_OPTIONS
        if i and prefix.decision(-1)[1] == "continue" and previous_end < arrival
        else ("",)
    )
    following = layouts[min(i + 1, len(layouts) - 1)]
    finishes = {
        "home": np.asarray(motion.for_layout(following).start_mm) / 1000,
        "point": np.asarray(motion.point_mm) / 1000,
        "continue": None,
    }
    routes = ROUTES if mode == FULL_AUTO else (mode,)
    if mode != FULL_AUTO:
        finishes = {motion.action: finishes[motion.action]}
        advances = (motion.advance,) if i and previous_end < arrival else ("",)
    geometry = layout_targets(scan, layout)
    choices = {}
    handoffs = set()
    for advance in advances:
        idle = _idle_move(position, previous_end, arrival, layout, advance)
        handoff = _command(idle) if idle else None
        if handoff in handoffs:
            continue
        handoffs.add(handoff)
        for route in routes:
            for action, finish in finishes.items():
                check_cancel()
                candidates = route_candidates(
                    geometry,
                    speed_mm,
                    length_mm,
                    route,
                    layout,
                    start=idle.b if idle else position,
                    finish=finish,
                    origin_x=START_X + speed * (begin - arrival),
                    base_mm=base_mm,
                )
                for board in candidates:
                    check_cancel()
                    work = tuple(map(_command, board.segments))
                    key = handoff, work
                    if key not in choices:
                        choices[key] = Choice(
                            board,
                            (route, action, "") if mode == FULL_AUTO else None,
                            advance,
                            idle,
                            work,
                            handoff,
                        )
    return begin, sorted(choices.values(), key=lambda c: c.board.duration)


def _append(prefix, begin, choice, *, fallback=False):
    return prefix.append_board(
        begin, choice.board, choice.decision, choice.advance, choice.idle, fallback
    )


def _select_board(
    scan, layouts, speed_mm, length_mm, prefix, validate, check_cancel, mode, base_mm
):
    begin, choices = _choices(
        scan, layouts, speed_mm, length_mm, prefix, check_cancel, mode, base_mm
    )
    rejected = set()
    for index, choice in enumerate(choices):
        check_cancel()
        if index in rejected:
            continue
        run = _append(prefix, begin, choice)
        failure = validate(run)
        if failure is None:
            return run
        # The committed physical history is identical. Reject only choices whose
        # incoming idle and commands through the failing instant also match.
        t = failure.command_time
        if failure.during_idle:
            rejected.update(
                j for j, c in enumerate(choices) if c.handoff == choice.handoff
            )
        elif t == 0:
            initial = choice.board.sample(0)[0]
            rejected.update(
                j
                for j, c in enumerate(choices)
                if np.array_equal(c.board.sample(0)[0], initial)
            )
        elif t is not None and t < run.duration - 1e-12:
            n = choice.board.ends.searchsorted(t - begin, side="right") + 1
            rejected.update(
                j
                for j, c in enumerate(choices)
                if c.handoff == choice.handoff
                and begin + c.board.duration > t + 1e-12
                and c.work[:n] == choice.work[:n]
            )
    return _append(prefix, begin, choices[0], fallback=True)


def automatic_boards(
    scan,
    layouts,
    speed_mm,
    length_mm,
    motion,
    base_mm,
    *,
    mode=FULL_AUTO,
    verify=None,
    check_cancel=lambda: None,
):
    """Yield a committed prefix only when its next board is explicitly requested."""
    if not layouts:
        raise ValueError("연속 판 수는 1 이상이어야 합니다.")
    if (
        mode not in (FULL_AUTO, LEADING)
        or speed_mm not in (60, 65, 70)
        or length_mm not in (200, 250, 300)
    ):
        raise ValueError("지원하지 않는 시나리오 값입니다.")
    prefix = RunPlan.empty(motion.for_layout(layouts[0]))
    if verify is None:
        from verification import rolling_verification

        session = rolling_verification(scan, base_mm, layouts, check_cancel)
    else:
        session = nullcontext()
    with session as physical:
        validate = (
            physical.validate
            if physical
            else lambda r: verify(scan, r, base_mm, layouts, check_cancel)
        )
        for i in range(len(layouts)):
            check_cancel()
            if i and physical:
                physical.commit(prefix)
            prefix = _select_board(
                scan,
                layouts,
                speed_mm,
                length_mm,
                prefix,
                validate,
                check_cancel,
                mode,
                base_mm,
            )
            yield prefix


def automatic_run(*args, **kwargs):
    """Collect a complete plan for offline callers that explicitly need one."""
    for run in automatic_boards(*args, **kwargs):
        pass
    return run
