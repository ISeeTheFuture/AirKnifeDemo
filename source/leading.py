"""Candidates for spraying the leading part of each row before the trailing part."""

from functools import lru_cache
from itertools import product

import numpy as np
from planner import (
    LEADING,
    RETRACT,
    SPRAY_SPEED,
    TRANSFER_SPEED,
    Plan,
    _finish_segments,
    _spray_strips,
    _sweep_segments,
)


def _strips(geometry, layout, width):
    rows = [
        sorted(slots, key=lambda i: layout.positions[i][0])
        for _, slots in sorted(layout.slots_by_height.items(), reverse=True)
    ]
    waves = [[], []]
    for slots in rows:
        mid = len(slots) // 2
        for wave, group in zip(waves, (slots[mid:], slots[:mid]), strict=True):
            if group:
                wave.extend(_spray_strips(geometry[group], width, group=True))
    return np.array(waves[0] + waves[1]), len(waves[0])


def _orders(n):
    """Vertical sweeps, including a neighbouring row visited ahead of its turn."""
    if n < 2:
        return [tuple(range(n))]
    orders = set()
    for initial in (list(range(n)), list(reversed(range(n)))):
        orders.add(tuple(initial))
        # Limit dense-rack search while retaining top, bottom and mid-row swaps.
        for i in np.linspace(0, n - 2, min(n - 1, 5), dtype=int):
            other = initial.copy()
            other[i], other[i + 1] = other[i + 1], other[i]
            orders.add(tuple(other))
    return sorted(orders)


@lru_cache(maxsize=8)
def _routes(front, rear):
    n = front + rear
    if n <= 8:
        directions = np.array(list(product((0, 1), repeat=n)), dtype=np.int16)
    else:
        # Keep direction search bounded when a random layout has many occupied rows.
        zigzag = np.arange(n, dtype=np.int16) % 2
        patterns = [zigzag, 1 - zigzag, np.zeros_like(zigzag), np.ones_like(zigzag)]
        for i in (0, front - 1, min(front, n - 1), n - 1):
            other = zigzag.copy()
            other[i] ^= 1
            patterns.append(other)
        directions = np.unique(patterns, axis=0)
    orders = np.array(
        [
            a + tuple(front + i for i in b)
            for a, b in product(_orders(front), _orders(rear))
        ],
        dtype=np.int16,
    )
    return (2 * orders[:, None, :] + directions).reshape(-1, n)


def candidates(
    geometry, speed_mm, length_mm, layout, *, start, finish, origin_x, base_mm=820
):
    by_slot, slot_ids = geometry
    speed, length = speed_mm / 1000, length_mm / 1000
    effective = 0.8 * length
    strips, front = _strips(by_slot, layout, effective - 0.020)
    routes = _routes(front, len(strips) - front)
    entry = strips.reshape(-1, 3)
    exit_pos = strips[:, ::-1].reshape(-1, 3)
    start_local = np.asarray(start) - [origin_x, 0, 0]
    approach = np.linalg.norm(entry + RETRACT - start_local, axis=1) / TRANSFER_SPEED
    retract_s = RETRACT[1] / TRANSFER_SPEED
    transfer = np.linalg.norm(exit_pos[:, None] - entry, axis=2) / TRANSFER_SPEED
    spray = np.linalg.norm(exit_pos - entry, axis=1) / SPRAY_SPEED

    # Estimate shoulder-to-wrist span for the demo iiwa's fixed spray orientation.
    # This ranks candidates only. The coupled-arm physics check decides acceptance.
    def squared_wrist_span(points, time):
        return (
            (points[:, 0] + origin_x + speed * time) ** 2
            + (base_mm / 1000 - points[:, 1] - 0.171) ** 2
            + (points[:, 2] - 0.830) ** 2
        )

    first = routes[:, 0]
    t = approach[first] + retract_s
    worst = np.zeros(len(routes))
    for i, state in enumerate(routes.T):
        if i:
            t += transfer[routes[:, i - 1], state]
        worst = np.maximum(worst, squared_wrist_span(entry[state], t))
        t += spray[state]
        worst = np.maximum(worst, squared_wrist_span(exit_pos[state], t))
    if finish is not None:
        t += retract_s
        end = exit_pos[routes[:, -1]] + RETRACT
        end[:, 0] += origin_x + speed * t
        t += np.maximum(0.30, np.linalg.norm(end - finish, axis=1) / TRANSFER_SPEED)

    distance = np.sqrt(worst)
    by_time = np.argsort(t, kind="stable")
    within_span = by_time[distance[by_time] <= 0.820]
    # Include the shortest fallback and alternatives away from full extension.
    # Only materialize these finalists as plans; the point samples are shared.
    # Rank near-extension alternatives with a 1 s penalty per cm in the final 5 cm.
    balanced = np.argsort(t + 100 * np.maximum(0.0, distance - 0.77), kind="stable")
    selected = dict.fromkeys([*within_span[:48], *balanced[:32], by_time[0]])
    # Avoid spending the next board's lead time on dozens of direction variants
    # of a single row order. Keep the absolute fastest fallback independently.
    order_counts = {}
    for i in selected:
        route = routes[i]
        order = (route // 2).tobytes()
        count = order_counts.get(order, 0)
        if count >= 6 and i != by_time[0]:
            continue
        order_counts[order] = count + 1
        sweeps = list(zip(entry[route], exit_pos[route], strict=True))
        segments = _sweep_segments(start_local, sweeps)
        _finish_segments(segments, finish, origin_x, speed)
        yield Plan(
            segments,
            np.array([s.t1 for s in segments]),
            by_slot.reshape(-1, 3),
            slot_ids,
            speed,
            length,
            effective,
            LEADING,
            layout,
        )
