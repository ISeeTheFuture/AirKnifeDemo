"""Robot-independent planning in rack coordinates (u, unsigned face distance, z)."""

import itertools
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

RACK_WIDTH = 0.700
RACK_HEIGHT = 1.100
RACK_BOTTOM = 0.29
PITCH = 0.990
FACE = 0.026
GAP = 0.042
START_X = -0.34
HOME = np.array([0.0, 0.16, 0.98])
SLOTS = ((-0.16, 1.03), (0.16, 1.03), (-0.16, 0.69), (0.16, 0.69))
SPRAY_SPEED = 0.32
TRANSFER_SPEED = 0.49
BEAM_WIDTH = 0.026
MIN_DWELL = 0.025
RETRACT = np.array([0.0, 0.046, 0.0])
END_ACTIONS = {
    "home": "시작 위치로 복귀",
    "point": "지정 위치로 이동",
    "continue": "마지막 분사 위치에서 계속",
}
ADVANCE_OPTIONS = {
    "": "",
    "top": "상단",
    "bottom": "하단",
    "leading_top": "좌우(상 우선)",
    "leading_bottom": "좌우(하 우선)",
}
FULL_AUTO = "full_auto"
LEADING = "leading"
START_LEVELS = {"top": "상", "middle": "중", "bottom": "하"}


@dataclass(frozen=True)
class Motion:
    """World-frame TCP settings in mm. Side A mirrors side B across Y=0."""

    start_mm: tuple = (0, 160, 980)
    action: str = "home"
    point_mm: tuple = (0, 200, 900)
    advance: str = ""
    start_level: str = ""

    def __post_init__(self):
        if self.start_level and self.start_level not in START_LEVELS:
            raise ValueError("제품 기준 시작 높이는 상·중·하 중에서 선택하세요.")
        if self.action not in END_ACTIONS:
            raise ValueError("판 종료 동작을 선택하세요.")
        if self.advance not in ADVANCE_OPTIONS:
            raise ValueError(
                "미리 이동은 빈칸, 상단, 하단, 좌우(상 우선), 좌우(하 우선) 중에서 선택하세요."
            )
        if self.advance and self.action != "continue":
            raise ValueError(
                "다음 판 여유 시 미리 이동은 ‘마지막 분사 위치에서 계속’에서만 사용할 수 있습니다."
            )
        for xyz in (self.start_mm, self.point_mm):
            if len(xyz) != 3 or not np.isfinite(xyz).all() or xyz[1] < 0:
                raise ValueError("XYZ는 유한한 mm 값이고 B면 Y는 0 이상이어야 합니다.")

    @property
    def end_label(self):
        label = END_ACTIONS[self.action]
        if self.advance:
            label += f" · 미리 이동 {ADVANCE_OPTIONS[self.advance]}"
        return label

    def finish(self):
        if self.action == "continue":
            return None
        xyz = self.start_mm if self.action == "home" else self.point_mm
        return np.asarray(xyz) / 1000

    def for_layout(self, layout):
        if not self.start_level:
            return self
        z = layout.start_height_mm(self.start_level)
        return replace(self, start_mm=(*self.start_mm[:2], z))


DEFAULT_MOTION = Motion()

# Ordered by total time over 150 single-board layouts, scoring each failure as 20 seconds.
ROUTES = {
    "horizontal": (
        "가로 묶음 분사",
        "유효 폭 안의 여러 제품을 가로로 한 번에 분사합니다.",
    ),
    "direct": (
        "제품별 분사 후 직결 이동",
        "제품 하나를 분사한 뒤 공기를 끄고 다음 제품의 시작점까지 직선으로 이동합니다.",
    ),
    "oblique": (
        "사선 묶음 분사",
        "공기를 켠 채 대각선으로 이동하며 여러 제품을 함께 분사합니다. 단독 제품은 보통 수평으로 분사합니다.",
    ),
    "order": (
        "방문 순서·분사 방향 최적화",
        "방문 순서와 좌우 진입 방향을 탐색하고, 각 분사선 끝에서 46 mm 후퇴합니다.",
    ),
    "target": (
        "기존 제품별 분사",
        "줄마다 진행 방향을 바꾸며 제품을 하나씩 수평 분사하고, 각 분사선 끝에서 46 mm 후퇴합니다.",
    ),
    "raster": (
        "전체 랙 왕복",
        "나이프의 유효 폭에 맞춰 위에서 아래로 랙의 전체 너비를 왕복 분사합니다.",
    ),
    LEADING: (
        "선두 제품 우선 분사",
        "각 줄에서 이송 방향 앞쪽 제품을 먼저 묶어 분사합니다. 후미 제품이 도달하면 나머지 묶음을 처리합니다.",
    ),
}


@dataclass(frozen=True)
class RackLayout:
    columns: int | None  # Random rows have no fixed column count.
    positions: tuple[tuple[float, float], ...]

    @property
    def slots_by_height(self):
        """Product indices grouped by row, top to bottom and left to right."""
        rows = {}
        for slot, (_, z) in enumerate(self.positions):
            rows.setdefault(z, []).append(slot)
        return rows

    @property
    def rows(self):
        return len(self.slots_by_height)

    @property
    def row_heights(self):
        return tuple(self.slots_by_height)

    def start_height_mm(self, level):
        """Select an occupied product row and convert its world Z to millimetres."""
        heights = self.row_heights
        low, high = min(heights), max(heights)
        if level == "top":
            return 1000 * high
        if level == "bottom":
            return 1000 * low
        if level != "middle":
            raise ValueError("제품 기준 시작 높이는 상·중·하 중에서 선택하세요.")
        midpoint = (low + high) / 2
        # Resolve physically equal distances in favour of the upper row.
        return 1000 * min(heights, key=lambda z: (round(abs(z - midpoint), 12), -z))

    @property
    def support_heights(self):
        # Random rows use the support centreline as the mounting height.
        offset = 0.0 if self.columns is None else 0.02
        return tuple(z - offset for z in self.row_heights)

    @property
    def label(self):
        count = len(self.positions)
        if self.columns is None:
            return f"랜덤 {self.rows}줄 · {count}개"
        return f"격자 {self.columns}열 × {self.rows}횡 · {count}개"

    def spray_order(self):
        """Return product indices and sweep directions for the current layout."""
        for row, slots in enumerate(self.slots_by_height.values()):
            for slot in reversed(slots) if row % 2 else slots:
                yield slot, row % 2


DEFAULT_LAYOUT = RackLayout(2, SLOTS)
FRAME_INSET = 0.012  # Outside edge to inner edge of each vertical rack bar.


def _placement_area(scan):
    xy = scan.xyz[scan.selected, :2] - scan.center
    left, low_y = xy.min(axis=0)
    right, high_y = xy.max(axis=0)
    size = np.array([right - left, high_y - low_y])
    space = np.array([RACK_WIDTH, RACK_HEIGHT]) - 2 * FRAME_INSET
    # Anchor of an asymmetric product with its bottom left corner at the inner frame.
    origin = np.array([-RACK_WIDTH / 2 - left, RACK_BOTTOM + high_y]) + FRAME_INSET
    return origin, space, size


def grid_layout(scan, columns, rows):
    if columns < 1 or rows < 1:
        raise ValueError("열 개수와 횡 개수는 1 이상의 정수로 입력하세요.")
    origin, space, size = _placement_area(scan)
    max_cols, max_rows = np.floor(space / size).astype(int)
    if columns > max_cols or rows > max_rows:
        raise ValueError(
            f"촬영 세트 {scan.sample}은 한 줄당 {max_cols}개, 총 {max_rows}줄까지 배치할 수 있습니다."
        )
    pitch = space / (columns, rows)
    start = origin + (pitch - size) / 2
    xs = start[0] + np.arange(columns) * pitch[0]
    zs = start[1] + np.arange(rows - 1, -1, -1) * pitch[1]
    return RackLayout(columns, tuple((float(x), float(z)) for z in zs for x in xs))


def _random_axis_positions(origin, span, size, count, rng):
    """Sample ordered anchors, reserving one product width or height per item."""
    slack = max(0.0, span - count * size)
    return origin + np.arange(count) * size + np.sort(rng.uniform(0, slack, count))


def random_layout(scan, count, rng=None):
    """Generate occupied horizontal rows, then place count products along them."""
    origin, space, size = _placement_area(scan)
    max_cols, max_rows = np.floor(space / size).astype(int)
    maximum = int(max_cols * max_rows)
    if not 1 <= count <= maximum:
        raise ValueError(
            f"촬영 세트 {scan.sample}의 랜덤 총 개수는 1~{maximum}개로 입력하세요."
        )
    rng = np.random.default_rng() if rng is None else rng
    min_rows = (count + max_cols - 1) // max_cols
    rows = rng.integers(min_rows, min(count, max_rows) + 1)
    heights = _random_axis_positions(origin[1], space[1], size[1], rows, rng)
    positions = []
    remaining = count
    for row, z in enumerate(reversed(heights), start=1):
        # Leave at least one product and enough capacity for each remaining row.
        rows_left = rows - row
        least = max(1, remaining - rows_left * max_cols)
        most = min(max_cols, remaining - rows_left)
        on_row = rng.integers(least, most + 1)
        xs = _random_axis_positions(origin[0], space[0], size[0], on_row, rng)
        positions.extend((float(x), float(z)) for x in xs)
        remaining -= on_row
    return RackLayout(None, tuple(positions))


def random_boards(scan, products_per_board, board_count, seed=None):
    """Generate independent layouts; a supplied seed reproduces the three-board study."""
    return tuple(
        random_layout(
            scan,
            products_per_board,
            np.random.default_rng(None if seed is None else seed + i * 1_000_000),
        )
        for i in range(board_count)
    )


@dataclass
class Segment:
    t0: float
    t1: float
    a: np.ndarray
    b: np.ndarray
    spraying: bool
    label: str

    def position_at(self, t):
        f = np.clip((t - self.t0) / (self.t1 - self.t0), 0.0, 1.0)
        return (1 - f) * self.a + f * self.b


@dataclass
class Plan:
    segments: list
    ends: np.ndarray
    targets: np.ndarray
    slot_ids: np.ndarray
    speed: float
    length: float
    effective: float
    mode: str
    layout: RackLayout = DEFAULT_LAYOUT
    slope: float | None = None
    origin_x: float = START_X
    traversal: str = ""

    @property
    def duration(self):
        return self.ends[-1]

    def rack_x(self, t):
        return self.origin_x + self.speed * t

    def sample(self, t):
        k = min(np.searchsorted(self.ends, t, side="right"), len(self.segments) - 1)
        s = self.segments[k]
        p = s.position_at(t)
        p[0] += self.rack_x(t)
        return p, s.spraying, s.label


def _spray_strips(by_slot, width, slope=0.0, group=False):
    """Build feasible sweeps from measured points, splitting oversized surfaces."""
    pieces, bounds, split = [], [], []
    for pts in by_slot:
        residual = pts[:, 2] - slope * pts[:, 0]
        low, high = residual.min(), residual.max()
        count = max(1, int(np.ceil((high - low) / width)))
        if count == 1:
            pieces.append(pts)
            bounds.append((low, high))
            split.append(False)
        else:
            bands = np.minimum(
                ((residual - low) / (high - low) * count).astype(int), count - 1
            )
            for band in reversed(np.unique(bands)):
                mask = bands == band
                pieces.append(pts[mask])
                bounds.append((residual[mask].min(), residual[mask].max()))
                split.append(True)

    lo, hi = np.asarray(bounds).T
    groups = []
    if group:
        top, bottom = -np.inf, np.inf
        for i in np.argsort(-(lo + hi), kind="stable"):
            if groups and max(top, hi[i]) - min(bottom, lo[i]) <= width:
                groups[-1].append(i)
                top, bottom = max(top, hi[i]), min(bottom, lo[i])
            else:
                groups.append([i])
                top, bottom = hi[i], lo[i]
    else:
        groups = [[i] for i in range(len(pieces))]
    strips = []
    for ids in groups:
        pts = np.concatenate([pieces[i] for i in ids])
        m = slope if len(ids) > 1 or split[ids[0]] else 0.0
        residual = pts[:, 2] - m * pts[:, 0]
        intercept = (residual.min() + residual.max()) / 2
        xs = np.array([pts[:, 0].min() - 0.010, pts[:, 0].max() + 0.010])
        strips.append(
            np.column_stack((xs, np.full(2, pts[:, 1].max() + GAP), m * xs + intercept))
        )
    return np.asarray(strips)


@lru_cache(maxsize=5)
def _all_routes(n):
    directions = np.asarray(list(itertools.product(range(2), repeat=n)))
    orders = np.asarray(list(itertools.permutations(range(n))))
    return (2 * orders[:, None, :] + directions).reshape(-1, n)


@lru_cache(maxsize=4)
def _reversal_moves(n):
    # Reverse the visit order and the sweep direction of a subsequence.
    # Keep interactive planning bounded on densely populated racks.
    span = n if n <= 30 else 12
    moves, flip = [], []
    for first in range(n):
        for last in range(first + 1, min(n, first + span) + 1):
            idx = np.arange(n)
            idx[first:last] = idx[first:last][::-1]
            moves.append(idx)
            flip.append((idx >= first) & (idx < last))
    return np.asarray(moves), np.asarray(flip)


def _choose_route(strips, speed, direct, endpoints, initial=None):
    n = len(strips)
    entry = strips.reshape(2 * n, 3)
    exit_point = strips[:, ::-1].reshape(2 * n, 3)
    v = TRANSFER_SPEED
    start_local, finish, origin_x = endpoints
    first = np.linalg.norm(entry + RETRACT - start_local, axis=1) / v
    travel = np.linalg.norm(exit_point[:, None] - entry, axis=2) / v
    fixed = np.linalg.norm(strips[:, 1] - strips[:, 0], axis=1).sum() / SPRAY_SPEED
    fixed += (2 if direct else 2 * n) * RETRACT[1] / v
    if finish is None:
        fixed -= RETRACT[1] / v

    def times(routes):
        elapsed = first[routes[:, 0]] + fixed
        elapsed += travel[routes[:, :-1], routes[:, 1:]].sum(axis=1)
        if finish is None:
            return elapsed
        home_delta = exit_point[routes[:, -1]] + RETRACT - finish
        home_delta[:, 0] += origin_x + speed * elapsed
        return elapsed + np.maximum(0.30, np.linalg.norm(home_delta, axis=1) / v)

    if n <= 5:
        candidates = _all_routes(n)
        best = candidates[np.argmin(times(candidates))]
    else:
        seeds = [] if initial is None else [np.asarray(initial)]
        for start in np.argsort(first)[:6]:
            route = [start]
            available = np.ones(2 * n, dtype=bool)
            for _ in range(n - 1):
                available[2 * (route[-1] // 2) : 2 * (route[-1] // 2) + 2] = False
                route.append(np.argmin(np.where(available, travel[route[-1]], np.inf)))
            seeds.append(route)
        seeds = np.asarray(seeds)
        indexes, flips = _reversal_moves(n)
        finals = []
        for route in seeds[np.argsort(times(seeds))[:3]]:
            cost = times(route[None])[0]
            while True:
                neighbors = route[indexes] ^ flips
                costs = times(neighbors)
                k = costs.argmin()
                if costs[k] >= cost - 1e-9:
                    break
                route, cost = neighbors[k], costs[k]
            finals.append(route)
        finals = np.asarray(finals)
        best = finals[np.argmin(times(finals))]
    return best, times(best[None])[0]


def _optimized_sweeps(by_slot, effective, speed, mode, layout, endpoints):
    if mode in ("order", "direct"):
        strips = _spray_strips(by_slot, effective)
        initial = None
        if len(strips) == len(layout.positions):
            initial = [2 * i + rev for i, rev in layout.spray_order()]
        direct = mode == "direct"
        if direct and len(strips) > 5:
            initial, _ = _choose_route(strips, speed, False, endpoints, initial)
        route, _ = _choose_route(strips, speed, direct, endpoints, initial)
        return strips, route, None

    width = effective - 0.020
    slopes = (0.0,)
    if mode == "oblique":
        # Prefer entire products inside a band, as in the ranking experiment.
        slopes = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)
        product = by_slot[0]
        fitting = [
            m for m in slopes if np.ptp(product[:, 2] - m * product[:, 0]) <= width
        ]
        slopes = fitting or slopes
    best_time = np.inf
    for slope in slopes:
        strips = _spray_strips(by_slot, width, slope, group=True)
        route, duration = _choose_route(strips, speed, True, endpoints)
        if duration < best_time:
            best_time = duration
            best = strips, route, slope
    return best


def layout_targets(scan, layout):
    src = scan.surface
    by_slot = np.stack(
        [
            np.column_stack((u + src[:, 0], FACE + src[:, 2], z + src[:, 1]))
            for u, z in layout.positions
        ]
    )
    return by_slot, np.repeat(np.arange(len(layout.positions)), len(src))


def make_plan(
    scan,
    speed_mm=65,
    length_mm=250,
    mode="target",
    layout=DEFAULT_LAYOUT,
    *,
    start=HOME,
    finish=HOME,
    origin_x=START_X,
):
    return plan_from_targets(
        layout_targets(scan, layout),
        speed_mm,
        length_mm,
        mode,
        layout,
        start=start,
        finish=finish,
        origin_x=origin_x,
    )


def plan_from_targets(
    geometry, speed_mm, length_mm, mode, layout, *, start, finish, origin_x
):
    if (
        speed_mm not in (60, 65, 70)
        or length_mm not in (200, 250, 300)
        or mode not in ROUTES
    ):
        raise ValueError("지원하지 않는 시나리오 값입니다.")
    if mode == LEADING:
        from leading import candidates

        return next(
            candidates(
                geometry,
                speed_mm,
                length_mm,
                layout,
                start=start,
                finish=finish,
                origin_x=origin_x,
            )
        )
    speed, length = speed_mm / 1000, length_mm / 1000
    effective = length * 0.8
    by_slot, ids = geometry
    targets = by_slot.reshape(-1, 3)
    prev = np.asarray(start, dtype=float) - [origin_x, 0, 0]
    slope = None
    sweeps = []

    def stripe(a, b, d, z):
        sweeps.append((np.array((a, d, z)), np.array((b, d, z))))

    if mode == "raster":
        n = int(np.ceil(RACK_HEIGHT / effective))
        for i in range(n):
            z = RACK_BOTTOM + RACK_HEIGHT - (i + 0.5) * RACK_HEIGHT / n
            near = targets[np.abs(targets[:, 2] - z) <= effective / 2]
            d = np.max(near[:, 1]) if len(near) else FACE
            a, b = (-0.345, 0.345) if i % 2 == 0 else (0.345, -0.345)
            stripe(a, b, d + GAP, z)
    elif mode == "target":
        for slot, reverse in layout.spray_order():
            pts = by_slot[slot]
            lo, hi = np.min(pts[:, 2]), np.max(pts[:, 2])
            n = max(1, int(np.ceil((hi - lo) / effective)))
            for j in range(n):
                z = hi - (j + 0.5) * (hi - lo) / n
                part = pts[np.abs(pts[:, 2] - z) <= effective / 2 + 1e-9]
                a, b = part[:, 0].min() - 0.010, part[:, 0].max() + 0.010
                if (reverse + j) % 2:
                    a, b = b, a
                stripe(a, b, part[:, 1].max() + GAP, z)
    else:
        strips, route, slope = _optimized_sweeps(
            by_slot, effective, speed, mode, layout, (prev, finish, origin_x)
        )
        sweeps = [
            strips[state // 2][::-1] if state % 2 else strips[state // 2]
            for state in route
        ]
    segments = _sweep_segments(
        prev, sweeps, retract_each=mode in ("order", "target", "raster")
    )
    _finish_segments(segments, finish, origin_x, speed)
    return Plan(
        segments,
        np.array([s.t1 for s in segments]),
        targets,
        ids,
        speed,
        length,
        effective,
        mode,
        layout,
        slope,
        origin_x,
    )


def _sweep_segments(start, sweeps, *, retract_each=False):
    segments = []
    prev, t = start, 0.0
    for i, (a, b) in enumerate(sweeps):
        approach = i == 0 or retract_each
        points = [(a + RETRACT, False, "접근")] if approach else []
        points.extend(
            [
                (a, False, "분사 위치 진입" if approach else "직결 이동"),
                (b, True, "분사"),
            ]
        )
        if retract_each or i + 1 == len(sweeps):
            points.append((b + RETRACT, False, "이격"))
        for end, spraying, label in points:
            distance = np.linalg.norm(end - prev)
            if distance > 1e-6:
                until = t + distance / (SPRAY_SPEED if spraying else TRANSFER_SPEED)
                segments.append(Segment(t, until, prev, end, spraying, label))
                t = until
            prev = end
    return segments


def _finish_segments(segments, finish, origin_x, speed):
    if finish is None:
        # Every strategy ends its last spray with an explicit normal retraction.
        segments.pop()
        return
    t, prev = segments[-1].t1, segments[-1].b
    world_end = prev + [origin_x + speed * t, 0, 0]
    dt = max(0.30, np.linalg.norm(world_end - finish) / TRANSFER_SPEED)
    end_local = np.asarray(finish) - [origin_x + speed * (t + dt), 0, 0]
    segments.append(Segment(t, t + dt, prev, end_local, False, "종료 위치 이동"))


def automatic_variants(board, finish):
    """Also try horizontal passes from the other row or the other side."""
    yield board
    if board.mode != "horizontal":
        return
    original = [(s.a, s.b) for s in board.segments if s.spraying]
    for reverse in (False, True):
        rows = original[::-1] if reverse else original
        for direction in ("기본 방향", "반대 방향", "+X 방향", "−X 방향"):
            if not reverse and direction == "기본 방향":
                continue
            sweeps = []
            for a, b in rows:
                flip = (
                    direction == "반대 방향"
                    or (direction == "+X 방향" and a[0] > b[0])
                    or (direction == "−X 방향" and a[0] < b[0])
                )
                sweeps.append((b, a) if flip else (a, b))
            segments = _sweep_segments(board.segments[0].a, sweeps)
            _finish_segments(segments, finish, board.origin_x, board.speed)
            traversal = f"{'방문 역순' if reverse else '방문 정순'} · {direction}"
            yield replace(
                board,
                segments=segments,
                ends=np.array([s.t1 for s in segments]),
                traversal=traversal,
            )


def route_candidates(
    geometry, speed_mm, length_mm, route, layout, *, start, finish, origin_x, base_mm
):
    if route == LEADING:
        from leading import candidates

        yield from candidates(
            geometry,
            speed_mm,
            length_mm,
            layout,
            start=start,
            finish=finish,
            origin_x=origin_x,
            base_mm=base_mm,
        )
    else:
        board = plan_from_targets(
            geometry,
            speed_mm,
            length_mm,
            route,
            layout,
            start=start,
            finish=finish,
            origin_x=origin_x,
        )
        yield from automatic_variants(board, finish)


@dataclass
class RunPlan:
    boards: tuple
    starts: np.ndarray
    ends: np.ndarray
    motion: Motion
    # World-frame moves keyed by the board that just finished.
    idle_moves: dict[int, Segment] = field(default_factory=dict)
    # One (route, end action, idle advance) decision per board in automatic mode.
    decisions: tuple = ()
    # At least one board used the fastest fallback from its committed entry state.
    fastest_fallback: bool = False
    fallback_boards: tuple = ()

    @classmethod
    def empty(cls, motion):
        return cls((), np.empty(0), np.empty(0), motion)

    def append_board(self, begin, board, decision, advance, idle, fallback=False):
        decisions = self.decisions
        moves = self.idle_moves.copy()
        if self.boards and idle is not None:
            moves[len(self.boards) - 1] = idle
        if decisions:
            decisions = (*decisions[:-1], (*decisions[-1][:2], advance))
        return replace(
            self,
            boards=(*self.boards, board),
            starts=np.append(self.starts, begin),
            ends=np.append(self.ends, begin + board.duration),
            decisions=(*decisions, decision) if decision is not None else (),
            idle_moves=moves,
            fastest_fallback=self.fastest_fallback or fallback,
            fallback_boards=(*self.fallback_boards, len(self.boards))
            if fallback
            else self.fallback_boards,
        )

    def decision(self, index):
        if self.decisions:
            return self.decisions[index]
        return self.boards[index].mode, self.motion.action, self.motion.advance

    @property
    def duration(self):
        return self.ends[-1]

    def index_at(self, t):
        return int(np.searchsorted(self.starts, t, side="right") - 1)

    def sample(self, t, index):
        board = self.boards[index]
        if t > self.ends[index]:
            move = self.idle_moves.get(index)
            if move is not None:
                if t <= move.t1:
                    return move.position_at(t), False, move.label
                return move.b.copy(), False, "다음 판 대기"
            pos, _, _ = board.sample(board.duration)
            return (
                pos,
                False,
                "다음 판 대기" if index + 1 < len(self.boards) else "완료",
            )
        return board.sample(min(board.duration, t - self.starts[index]))

    def next_boundary(self, t, index):
        if t < self.ends[index] or index == len(self.boards) - 1:
            return self.ends[index]
        move = self.idle_moves.get(index)
        if move is not None and t < move.t1:
            return move.t1
        return self.starts[index + 1]


def _idle_move(start, finish_time, arrival, next_layout, option):
    if not option or arrival <= finish_time:
        return None
    target = start.copy()
    if option in ("top", "bottom"):
        extreme = max if option == "top" else min
        target[2] = extreme(next_layout.row_heights)
    else:
        # With a +X conveyor, the largest rack-local X arrives first.
        leading_x = max(x for x, _ in next_layout.positions)
        tied = [z for x, z in next_layout.positions if leading_x - x <= 1e-9]
        target[0] = START_X + leading_x
        target[2] = (max if option == "leading_top" else min)(tied)
    travel_time = np.linalg.norm(target - start) / TRANSFER_SPEED
    if travel_time <= 1e-12:
        return None
    dt = min(travel_time, arrival - finish_time)
    end = target if dt == travel_time else start + (target - start) * (dt / travel_time)
    return Segment(
        finish_time,
        min(arrival, finish_time + dt),
        start,
        end,
        False,
        f"다음 판 {ADVANCE_OPTIONS[option]} 미리 이동",
    )


def make_run(
    scan,
    layouts,
    speed_mm=65,
    length_mm=250,
    mode="target",
    motion=DEFAULT_MOTION,
    base_mm=820,
):
    if not layouts:
        raise ValueError("연속 판 수는 1 이상이어야 합니다.")
    if mode in (LEADING, FULL_AUTO):
        from automatic import automatic_run

        return automatic_run(
            scan, layouts, speed_mm, length_mm, motion, base_mm, mode=mode
        )
    boards, starts, ends = [], [], []
    idle_moves = {}
    motion = motion.for_layout(layouts[0])
    pos = np.asarray(motion.start_mm) / 1000
    finish = motion.finish()
    speed = speed_mm / 1000
    for i, layout in enumerate(layouts):
        arrival = i * PITCH / speed
        begin = max(ends[-1], arrival) if i else 0.0
        if motion.start_level and motion.action == "home":
            # Return before handoff so the next board actually starts at its own product row.
            home_layout = layouts[min(i + 1, len(layouts) - 1)]
            finish = motion.for_layout(home_layout).finish()
        if i and motion.advance and ends[-1] < arrival:
            move = _idle_move(pos, ends[-1], arrival, layout, motion.advance)
            if move is not None:
                idle_moves[i - 1] = move
                pos = move.b
        origin = START_X + speed * (begin - arrival)
        board = make_plan(
            scan,
            speed_mm,
            length_mm,
            mode,
            layout,
            start=pos,
            finish=finish,
            origin_x=origin,
        )
        boards.append(board)
        starts.append(begin)
        ends.append(begin + board.duration)
        pos, _, _ = board.sample(board.duration)
    return RunPlan(
        tuple(boards), np.asarray(starts), np.asarray(ends), motion, idle_moves
    )
