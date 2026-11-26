"""Air Knife Track 2 demo: local desktop playback, comparative runs and Markdown results."""

import argparse
import subprocess
import sys
import tarfile
import time
import tkinter as tk
from datetime import date
from multiprocessing import freeze_support
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pybullet as pb
from PIL import Image, ImageDraw, ImageTk
from planner import (
    ADVANCE_OPTIONS,
    DEFAULT_LAYOUT,
    DEFAULT_MOTION,
    END_ACTIONS,
    FULL_AUTO,
    LEADING,
    MIN_DWELL,
    PITCH,
    RACK_BOTTOM,
    RACK_HEIGHT,
    ROUTES,
    START_LEVELS,
    Motion,
    grid_layout,
    make_run,
    random_boards,
    random_layout,
)
from playback import AutomaticPlayback, VerificationWorker
from route_benchmark import ranked_routes
from scan import CAPTURES, load_scan, locate_tar, open_cloud
from simulator import DT, Cell

TEAL = "#076d70"
INK = "#20323b"
MUTED = "#526b73"
PAPER = "#f4f7f6"
FONT = "Malgun Gothic" if sys.platform == "win32" else "sans-serif"
PLAYBACK_RENDER_SIZE = (560, 370)
STILL_RENDER_SIZE = (850, 560)
MODES = {"전자동": FULL_AUTO, **{label: mode for mode, (label, _) in ROUTES.items()}}
MODE_LABELS = {mode: label for label, mode in MODES.items()}
FALLBACK_LABEL = "통과 경로가 없는 판에 최단 경로 적용"
FIRST_MODE = next(iter(ROUTES))
ROUTE_FIELD = "분사 경로 · 누적시간 단축률 순"


class RoutePicker(ttk.Combobox):
    """Show the benchmark rank and description for the selected or hovered route."""

    def __init__(self, parent, *, motion=DEFAULT_MOTION, **options):
        super().__init__(parent, **options)
        self._tip = self._timer = self._index = None
        self.set_motion(motion)
        self.bind("<Enter>", self._hover_selected)
        for event in ("<Leave>", "<ButtonPress-1>", "<Unmap>", "<<ComboboxSelected>>"):
            self.bind(event, self._hide, add="+")
        # The native ttk dropdown owns a Tcl listbox, separate from the entry.
        popup = self.tk.call("ttk::combobox::PopdownWindow", self._w)
        self._listbox = f"{popup}.f.l"
        self.tk.call(
            "bind", self._listbox, "<Motion>", f"+{self.register(self._hover_row)} %y"
        )
        hide = self.register(self._hide)
        for event in ("<Leave>", "<Unmap>", "<ButtonPress-1>"):
            self.tk.call("bind", self._listbox, event, f"+{hide}")

    def set_motion(self, motion):
        self._hide()
        results = ranked_routes(motion)
        rack_seconds = results["raster"].penalized_total_s
        self.configure(values=tuple(MODE_LABELS[mode] for mode in results))
        self._tips = []
        for rank, (mode, result) in enumerate(results.items()):
            if mode == FULL_AUTO:
                description = "판마다 물리 검증을 통과한 가장 빠른 후보를 선택합니다. 모두 실패하면 가장 빠른 후보를 사용합니다."
                end_label = f"비교 기준 설정 · {motion.end_label}"
            else:
                description = ROUTES[mode][1]
                end_label = motion.end_label
            saved = rack_seconds - result.penalized_total_s
            minutes, cs = divmod(round(saved * 100), 6000)
            self._tips.append(
                f"{rank}순위 · {MODE_LABELS[mode]}\n{description}\n{end_label}\n"
                f"전체 랙 왕복 대비 누적시간 {saved / rack_seconds:.2%} 단축 ({minutes}분 {cs / 100:05.2f}초)\n"
                f"처리속도 {result.boards_per_minute:.2f}판/분 · 판 성공률 {result.passed_boards / 600:.1%}\n"
                "세트 2 · 제품 1~24개 · 연속 5판 × 120묶음 · 실패 판당 20초"
            )

    def _hover_selected(self, _event):
        self._schedule(self.current(), self.winfo_rooty())

    def _hover_row(self, y):
        index = int(self.tk.call(self._listbox, "nearest", y))
        top = int(self.tk.call("winfo", "rooty", self._listbox))
        self._schedule(index, top + int(y))

    def _schedule(self, index, y):
        if index == self._index or index < 0:
            return
        self._hide()
        self._index = index
        self._timer = self.after(300, self._show, index, y)

    def _show(self, index, y):
        self._timer = None
        tip = self._tip = tk.Toplevel(self, takefocus=False)
        tip.withdraw()
        tip.overrideredirect(True)
        tip.attributes("-topmost", True)
        tk.Label(
            tip,
            text=self._tips[index],
            justify="left",
            wraplength=275,
            font=(FONT, 10),
            bg=INK,
            fg="white",
            padx=10,
            pady=6,
        ).pack()
        tip.update_idletasks()
        x = self.winfo_rootx() + self.winfo_width() + 6
        if x + tip.winfo_reqwidth() > self.winfo_screenwidth():
            x = max(0, self.winfo_rootx() - tip.winfo_reqwidth() - 6)
        y = max(0, min(y, self.winfo_screenheight() - tip.winfo_reqheight()))
        tip.geometry(f"+{x}+{y}")
        tip.deiconify()

    def _hide(self, *_):
        if self._timer is not None:
            self.after_cancel(self._timer)
        if self._tip is not None:
            self._tip.destroy()
        self._tip = self._timer = self._index = None

    def destroy(self):
        self._hide()
        super().destroy()


def layout_report(layout):
    lines = [
        layout.label,
        "",
        "| 줄 (위부터) | 높이 Z (mm) | 각 제품의 가로 위치 X (mm, 랙 중심 기준) |",
        "| ---: | ---: | --- |",
    ]
    for row, (z, slots) in enumerate(layout.slots_by_height.items(), start=1):
        xs = ", ".join(f"{layout.positions[i][0] * 1000:.1f}" for i in slots)
        lines.append(f"| {row} | {z * 1000:.1f} | {xs} |")
    return "\n".join(lines)


def decision_label(decision, traversal="", *, show_advance=True):
    route, action, advance = decision
    path = " · ".join(filter(None, (MODE_LABELS[route], traversal)))
    label = f"{path} · {END_ACTIONS[action]}"
    if show_advance:
        label += f" · {ADVANCE_OPTIONS[advance] or '미리 이동 없음'}"
    return label


def run_report(layouts, motion, decisions=(), traversals=()):
    start = ", ".join(f"{x:g}" for x in motion.for_layout(layouts[0]).start_mm)
    point = ", ".join(f"{x:g}" for x in motion.point_mm)
    end = f"전자동 (지정 위치 후보 {point} mm)" if decisions else motion.end_label
    if not decisions and motion.action == "point":
        end += f" ({point}) mm"
    info = [f"연속 {len(layouts)}판 · 시작 TCP ({start}) mm · 종료 동작 {end}"]
    if motion.start_level:
        info[0] += f" · 제품 줄 기준 {START_LEVELS[motion.start_level]}단 시작"
    for i, layout in enumerate(layouts, 1):
        info.append(f"### {i}번 판 배치")
        if decisions:
            info.append(
                decision_label(
                    decisions[i - 1], traversals[i - 1] if traversals else ""
                )
            )
        previous_home = (
            decisions[i - 2][1] == "home"
            if decisions and i > 1
            else motion.action == "home"
        )
        if motion.start_level and (i == 1 or previous_home):
            info.append(
                f"제품 줄 기준 시작 Z {layout.start_height_mm(motion.start_level):.3f} mm"
            )
        info.append(layout_report(layout))
    return "\n\n".join(info)


def run_performance(boards):
    """Format speed and board success rate for a completed simulation."""
    if not boards:
        return "—", "—"
    total = len(boards)
    passed = sum(board["passed"] for board in boards)
    seconds = boards[-1]["end"] - boards[0]["start"]
    speed = f"{60 * total / seconds:.2f} 판/분" if seconds > 0 else "—"
    return speed, f"{passed}/{total} ({passed / total:.1%})"


def details(scan, plan, base, result, run=None):
    r = result
    speed, success = run_performance(r.get("boards", ()))

    def fmt(value):
        return "대기" if value is None else str(value)

    selection = (
        [
            "",
            f"**최단 경로 적용** · {', '.join(str(i + 1) for i in run.fallback_boards)}번 판",
        ]
        if run and run.fastest_fallback
        else []
    )
    lines = [
        "# Air Knife System 트랙 2 시연 결과",
        "",
        f"{date.today().isoformat()} · 촬영 세트 {scan.sample} · PCL {CAPTURES[scan.sample][0]}",
        "",
        f"분사 경로는 **{'전자동' if run and run.decisions else MODE_LABELS[plan.mode]}**이다. 랙 속도 {plan.speed * 1000:.0f} mm/s, 나이프 길이 {plan.length * 1000:.0f} mm, 베이스 중심 간격 {2 * base} mm를 적용했다.",
        *selection,
        "",
        "| 지표 | 계산 결과 |",
        "| --- | ---: |",
        f"| 모의 실행시간 / 계획시간 | {r['elapsed_s']:.3f} / {r['plan_s']:.3f} s |",
        f"| 묶음 처리속도 | {speed} |",
        f"| 이번 실행의 판 성공률 | {success} |",
        f"| 다음 랙 선단 도착 주기 / 계획상 여유 | {r['arrival_interval_s']:.3f} / {r['margin_s']:+.3f} s |",
        f"| 양면 가시 표적의 기하 피복률 A / B | {r['coverage_pct'][0]:.2f} / {r['coverage_pct'][1]:.2f} % |",
        f"| 양팔 목표 중 IK 허용범위 충족 비율 | {fmt(r['ik_ok_pct'])} % |",
        f"| 실제 TCP 추종오차 P95 / 최대 | {fmt(r['tcp_p95_mm'])} / {fmt(r['tcp_max_mm'])} mm |",
        f"| 실제 자세오차 최대 | {fmt(r['angle_max_deg'])} ° |",
        f"| 접촉이 발생한 제어 프레임 수 | {r['contact_frames']} |",
        f"| 로봇-랙 및 양 로봇 간 최소 간격 (표시 상한 150 mm) | {r['rack_or_peer_clearance_mm_capped_at_150']} mm |",
        "",
        f"유효점 {scan.valid.sum():,}개, 선택점 {scan.selected.sum():,}개. 가시 외곽 {scan.size[0] * 1000:.1f} × {scan.size[1] * 1000:.1f} mm, 장축각 {scan.angle:.1f}° (180° 주기).",
        "",
        "## 모델과 해석",
        "",
        run_report(
            [b.layout for b in run.boards],
            run.motion,
            run.decisions,
            [b.traversal for b in run.boards],
        )
        if run
        else layout_report(plan.layout),
        "",
        "- 각 제품 위치에 실측 단일 시점의 선택 표면을 복제했다. 반대면은 가정한 대칭 사본이다. 나사나 홈의 자동 식별은 수행하지 않는다.",
        "- KUKA iiwa 예시 모델 2대, 가상 공구 질량 0.8 kg, 관절 속도 제한 2.2 rad/s를 적용했다. 제조사 로봇·호스·보정 데이터는 미반영이다.",
        "- 피복률은 면별 표적 샘플 중 폭·거리·방향 조건을 만족하며 25 ms 이상 노출된 비율이다. 유효 폭은 나이프 길이의 80%, 이격 범위는 25~85 mm다. 차폐·유동·잔류액 제거 성능은 별도 시험 대상이다.",
        "- IK는 위치 4 mm·자세 3° 이내와 관절 범위로 판단한다. 접촉은 240 Hz, 실제 추종과 피복은 60 Hz, 최소 간격은 10 Hz로 샘플링한다. 제품 충돌체는 가시 표면의 보수적 상자다.",
        "- 총시간에는 각 판의 종료 동작과 다음 판의 도착 대기가 포함된다. 시간 여유는 전체 판 수 × 990 mm ÷ 속도에서 총시간을 뺀 값이다.",
    ]
    if "boards" in r:
        lines += [
            "",
            "## 판별 결과",
            "",
            "| 판 | 시작 (초) | 종료 (초) | 작업 (초) | 최저 면 피복률 | 접촉 프레임 | IK 미달 | 판정 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for i, b in enumerate(r["boards"], 1):
            lines.append(
                f"| {i} | {b['start']:.3f} | {b['end']:.3f} | {b['work_s']:.3f} | {b['worst_pct']:.2f}% | {b['contacts']} | {b['ik_misses']} | {'통과' if b['passed'] else '미달'} |"
            )
    if r["first_contact_s"] is not None:
        lines += [
            "",
            f"첫 접촉 시각은 {r['first_contact_s']:.3f}초다.",
            "",
            *["- " + " ↔ ".join(pair) for pair in r["contact_pairs"]],
        ]
    return "\n".join(lines) + "\n"


def batch(
    scan,
    speed,
    length,
    mode,
    base,
    *,
    run=None,
    layouts=(DEFAULT_LAYOUT,),
    motion=DEFAULT_MOTION,
    return_run=False,
):
    playback = None
    if run is None and mode in (FULL_AUTO, LEADING):
        playback = AutomaticPlayback(
            scan, layouts, speed, length, motion, base, visuals=False, mode=mode
        )
    else:
        run = run or make_run(scan, layouts, speed, length, mode, motion, base)
        sim = Cell(scan, run, base, visuals=False)
    try:
        if playback:
            while not playback.finished:
                playback.step(block=True)
            sim = playback.cell
            run = sim.run
        else:
            while not sim.finished:
                sim.step()
        return (run if return_run else run.boards[0]), {
            **sim.result(),
            "boards": sim.board_results(),
            "decisions": run.decisions,
            "fastest_fallback": run.fastest_fallback,
            "traversals": [b.traversal for b in run.boards],
        }
    finally:
        (playback or sim).close()


def summary_table(results):
    rows = [
        "| 경로 | 처리속도 | 판 성공률 | 계획시간 | 도착 여유 | 피복 A / B | IK 충족 | 접촉 프레임 | TCP P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, r in results:
        if r.get("fastest_fallback"):
            mode = f"{mode} ({FALLBACK_LABEL})"
        speed, success = run_performance(r.get("boards", ()))
        rows.append(
            f"| {mode} | {speed} | {success} | {r['plan_s']:.2f} s | {r['margin_s']:+.2f} s | "
            f"{r['coverage_pct'][0]:.1f} / {r['coverage_pct'][1]:.1f}% | "
            f"{r['ik_ok_pct']:.1f}% | {r['contact_frames']} | {r['tcp_p95_mm']:.1f} mm |"
        )
    return "\n".join(rows)


def sensor_image(scan):
    z = scan.xyz[:, :, 2]
    valid = scan.valid
    low, high = np.quantile(z[valid], [0.015, 0.985])
    depth = np.clip((high - z) / max(high - low, 1e-8), 0, 1)
    rgb = np.uint8(
        np.array([18, 40, 58]) + depth[:, :, None] * np.array([52, 107, 112])
    )
    rgb[~valid] = (15, 27, 37)
    rgb[scan.selected] = (244, 181, 65)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    x0, y0, x1, y1 = scan.roi
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(221, 240, 237), width=1)
    return im


class Demo:
    def __init__(
        self,
        root,
        archive,
        sample=2,
        speed=65,
        length=250,
        mode=FIRST_MODE,
        base=820,
        motion=DEFAULT_MOTION,
        board_count=1,
        random_count=None,
        seed=None,
    ):
        self.root = root
        self.archive = None
        self.initial_random = random_count
        self.initial_seed = seed
        self.scan = None
        self.cell = None
        self.motion = motion
        self.board_count = board_count
        self.layouts = ()
        self.playing = False
        self.after_id = None
        self.mode_view = "작업 셀"
        self.yaw = 49.0
        self.pitch = -17.0
        self.distance = 2.85
        self.photo = None
        self.drag = None
        self.sim_credit = 0.0
        self.last_draw = 0.0
        self.last_stats = 0.0
        self.loaded = None
        self.compare_window = None
        self.compare_after = None
        self.compare_cell = None
        self.playback = self.compare_playback = None
        self.verification_worker = VerificationWorker()
        self.search_after = None
        self.plan_status = ""
        self.children = []
        root.title("Air Knife System | 트랙 2 시연")
        root.geometry("1200x760")
        root.minsize(900, 655)
        root.configure(bg=PAPER)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TCombobox", font=(FONT, 10), padding=3)
        style.configure(
            "Run.Horizontal.TProgressbar", background=TEAL, troughcolor="#dce7e6"
        )
        top = tk.Frame(root, bg="#16383f")
        top.pack(fill="x")
        self.text(top, "AIR KNIFE", 15, "#e5efec", True, "#16383f").pack(
            side="left", padx=(18, 12), pady=12
        )
        self.text(top, "이동 랙 분사 시뮬레이션", 12, "white", False, "#16383f").pack(
            side="left"
        )
        self.text(
            top, "트랙 2  /  기술 검토 데모", 10, "#b0d5d2", False, "#16383f"
        ).pack(side="right", padx=18)
        body = tk.Frame(root, bg=PAPER)
        body.pack(fill="both", expand=True, padx=12, pady=12)
        side = tk.Frame(body, bg="white", width=225, padx=12, pady=6)
        side.pack(side="left", fill="y", padx=(0, 10))
        side.pack_propagate(False)
        self.text(side, "시연 조건", 13, INK, True).pack(anchor="w")
        self.file_label = self.text(side, "TAR를 선택하세요", 9, MUTED)
        self.file_label.pack(anchor="w", pady=(1, 2))
        self.button(side, "원본 TAR 열기", self.pick_file).pack(fill="x")
        self.vars = []
        config = [
            ("촬영 세트", tuple(map(str, CAPTURES)), str(sample)),
            (ROUTE_FIELD, (), MODE_LABELS[mode]),
            ("컨베이어 속도  ·  mm/s", ("60", "65", "70"), str(speed)),
            ("나이프 길이  ·  mm", ("200", "250", "300"), str(length)),
            ("베이스 중심 간격  ·  mm", ("1640", "2020"), str(base * 2)),
        ]
        for name, values, value in config:
            self.text(side, name, 9, MUTED).pack(anchor="w", pady=(3, 0))
            var = tk.StringVar(value=value)
            self.vars.append(var)
            if name == ROUTE_FIELD:
                box = self.route_picker = RoutePicker(
                    side,
                    textvariable=var,
                    motion=self.motion,
                    state="readonly",
                    width=18,
                )
            else:
                box = ttk.Combobox(
                    side, textvariable=var, values=values, state="readonly", width=18
                )
            box.pack(fill="x")
            box.bind("<<ComboboxSelected>>", lambda e: self.change(), add="+")
            if name == "촬영 세트":
                self.add_layout_controls(side)
        self.run_button = self.button(side, "▶  재생", self.toggle, True)
        self.run_button.pack(fill="x", pady=(6, 3))
        action = tk.Frame(side, bg="white")
        action.pack(fill="x")
        self.button(action, "처음으로", self.reset).pack(
            side="left", fill="x", expand=True
        )
        self.button(action, "경로 비교", self.compare).pack(
            side="right", fill="x", expand=True, padx=(3, 0)
        )
        self.save_button = self.button(side, "결과 저장 (.md)", self.save)
        self.save_button.pack(fill="x", pady=(4, 0))
        self.save_button.config(state="disabled")
        self.layout_label = self.text(side, "", 9, MUTED)
        self.layout_label.configure(wraplength=195, justify="left")
        self.layout_label.pack(anchor="w", pady=(2, 0))
        self.add_motion_controls(body)
        main = tk.Frame(body, bg=PAPER)
        main.pack(side="left", expand=True, fill="both")
        row = tk.Frame(main, bg="white")
        row.pack(fill="x")
        self.tab_buttons = {}
        for name in ("작업 셀", "피복 지도", "촬영 데이터"):
            b = self.button(row, name, lambda n=name: self.view(n))
            b.pack(side="left", padx=2, pady=4)
            self.tab_buttons[name] = b
        self.button(row, "원본 3D 점군", self.cloud).pack(side="right", padx=4)
        self.canvas = tk.Canvas(main, bg="#e6eeed", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(4, 0))
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.canvas.bind("<ButtonPress-1>", lambda e: setattr(self, "drag", (e.x, e.y)))
        self.canvas.bind("<B1-Motion>", self.orbit)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", lambda e: self.zoom(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self.zoom(e, -1))
        self.canvas.bind("<Double-Button-1>", lambda e: self.camera_default())
        status = tk.Frame(main, bg="white", padx=10, pady=5)
        status.pack(fill="x")
        self.status = self.text(status, "자료 대기", 10, INK, True)
        self.status.pack(anchor="w")
        self.progress = ttk.Progressbar(status, style="Run.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(3, 4))
        self.stats = self.text(status, "", 10, INK)
        self.stats.pack(anchor="w")
        self.data_label = self.text(status, "", 9, MUTED)
        self.data_label.pack(anchor="w", pady=(3, 0))

        def wrap_status(event):
            width = max(1, event.width - 24)
            for label in (self.status, self.stats, self.data_label):
                label.configure(wraplength=width)

        status.bind("<Configure>", wrap_status)
        self.view("작업 셀")
        root.bind("<space>", self.space)
        if random_count is not None:
            self.random_count.set(str(random_count))
        if archive:
            self.load(Path(archive))
        else:
            self.draw()

    @staticmethod
    def text(parent, value, size, color, bold=False, bg="white"):
        return tk.Label(
            parent,
            text=value,
            fg=color,
            bg=bg,
            anchor="w",
            justify="left",
            font=(FONT, size, "bold" if bold else "normal"),
        )

    @staticmethod
    def button(parent, label, action, primary=False):
        return tk.Button(
            parent,
            text=label,
            command=action,
            cursor="hand2",
            font=(FONT, 10, "bold" if primary else "normal"),
            bg=TEAL if primary else "#edf3f2",
            fg="white" if primary else INK,
            borderwidth=0,
            activebackground="#c9ddda",
            padx=9,
            pady=4,
        )

    def pick_file(self):
        p = filedialog.askopenfilename(
            parent=self.root,
            title="촬영 세트 TAR 선택",
            filetypes=[("ToF 촬영 TAR", "*.tar"), ("모든 파일", "*.*")],
        )
        if p:
            self.load(Path(p))

    def add_layout_controls(self, side):
        fields = tk.Frame(side, bg="white")
        fields.pack(fill="x", pady=(4, 2))
        self.columns = tk.StringVar(value=str(DEFAULT_LAYOUT.columns))
        self.rows = tk.StringVar(value=str(DEFAULT_LAYOUT.rows))
        for i, (label, var) in enumerate(
            (("열 · 줄당 수", self.columns), ("횡 · 줄 수", self.rows))
        ):
            fields.columnconfigure(i, weight=1)
            self.text(fields, label, 9, MUTED).grid(row=0, column=i, sticky="w")
            ttk.Entry(fields, textvariable=var, width=4).grid(
                row=1, column=i, sticky="ew", padx=(0, 3)
            )
        self.grid_button = self.button(fields, "격자 적용", self.apply_grid)
        self.grid_button.grid(row=0, column=2, rowspan=2, sticky="se")

        random_fields = tk.Frame(side, bg="white")
        random_fields.pack(fill="x", pady=(1, 2))
        self.random_count = tk.StringVar(value="4")
        self.text(random_fields, "랜덤 총 개수", 9, MUTED).pack(side="left")
        ttk.Entry(random_fields, textvariable=self.random_count, width=4).pack(
            side="left", fill="x", expand=True, padx=3
        )
        self.random_button = self.button(
            random_fields, "랜덤 생성", self.generate_layout
        )
        self.random_button.pack(side="right")
        for var in (self.columns, self.rows, self.random_count):
            var.trace_add("write", self.update_layout_buttons)
        self.update_layout_buttons()

    @staticmethod
    def positive_count(var):
        try:
            value = int(var.get())
        except ValueError:
            return None
        return value if value > 0 else None

    def update_layout_buttons(self, *_):
        grid_ready = (
            self.scan is not None
            and self.positive_count(self.columns)
            and self.positive_count(self.rows)
        )
        self.grid_button.config(state="normal" if grid_ready else "disabled")
        self.random_button.config(
            state="normal"
            if self.scan is not None and self.positive_count(self.random_count)
            else "disabled"
        )

    def apply_grid(self):
        columns, rows = (
            self.positive_count(self.columns),
            self.positive_count(self.rows),
        )
        if self.archive and columns and rows:
            self.load(self.archive, grid=(columns, rows))

    def generate_layout(self):
        count = self.positive_count(self.random_count)
        if self.archive and count:
            self.load(self.archive, random_count=count)

    def add_motion_controls(self, body):
        pane = tk.Frame(body, bg="white", padx=12, pady=6)
        pane.pack(side="right", fill="y", padx=(10, 0))
        for i in range(3):
            pane.columnconfigure(i, weight=1, uniform="xyz")
        self.text(pane, "연속 작업 설정", 13, INK, True).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        self.text(
            pane,
            "X 이송 방향 · Y 랙 중심에서 B면 방향\nZ 바닥 높이 · A면은 Y 대칭",
            9,
            MUTED,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(3, 5))

        def xyz_fields(row, label, coords):
            self.text(pane, label, 10, INK, True).grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(7, 0)
            )
            variables = []
            entries = []
            for i, (axis, value) in enumerate(zip("XYZ", coords)):
                self.text(pane, axis, 9, MUTED).grid(row=row + 1, column=i, sticky="w")
                var = tk.StringVar(pane, value=f"{value:g}")
                entry = ttk.Entry(pane, width=6, textvariable=var)
                entry.grid(row=row + 2, column=i, padx=(0, 3), sticky="ew")
                variables.append(var)
                entries.append(entry)
            return variables, entries

        start, start_entries = xyz_fields(
            2, "시작 TCP (mm) · 제품 줄 기준", self.motion.start_mm
        )
        self.start_z_field = start[2]
        self.start_level_field = tk.StringVar(pane, value=self.motion.start_level)

        def use_manual_z():
            self.start_level_field.set("")
            return True

        start_entries[2].configure(
            validate="key", validatecommand=pane.register(use_manual_z)
        )
        for i, (level, name) in enumerate(START_LEVELS.items()):
            ttk.Radiobutton(
                pane,
                text=name,
                value=level,
                variable=self.start_level_field,
                style="Toolbutton",
                command=self.update_start_height,
            ).grid(row=5, column=i, sticky="ew", padx=(0, 3), pady=(4, 2))
        self.update_start_height()
        self.text(pane, "매 판 종료 동작", 10, INK, True).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(8, 2)
        )
        action = tk.StringVar(pane, value=END_ACTIONS[self.motion.action])
        choice = ttk.Combobox(
            pane,
            values=tuple(END_ACTIONS.values()),
            textvariable=action,
            state="readonly",
        )
        choice.grid(row=7, column=0, columnspan=3, sticky="ew")
        point, entries = xyz_fields(8, "지정 위치 (mm)", self.motion.point_mm)

        self.text(pane, "다음 판 여유 시 미리 이동", 10, INK, True).grid(
            row=11, column=0, columnspan=3, sticky="w", pady=(10, 2)
        )
        advance = tk.StringVar(pane, value=ADVANCE_OPTIONS[self.motion.advance])
        advance_choice = ttk.Combobox(
            pane,
            values=tuple(ADVANCE_OPTIONS.values()),
            textvariable=advance,
            state="readonly",
        )
        advance_choice.grid(row=12, column=0, columnspan=3, sticky="ew")
        self.auto_action = tk.StringVar(pane, value="판별 선택")
        self.auto_advance = tk.StringVar(pane, value="판별 선택")

        def toggle_action(*_):
            automatic = MODES.get(self.vars[1].get()) == FULL_AUTO
            choice.configure(
                state="disabled" if automatic else "readonly",
                textvariable=self.auto_action if automatic else action,
            )
            for entry in entries:
                entry.configure(
                    state="normal"
                    if automatic or action.get() == END_ACTIONS["point"]
                    else "disabled"
                )
            continuing = action.get() == END_ACTIONS["continue"]
            if not continuing:
                advance.set("")
            advance_choice.configure(
                state="readonly" if continuing and not automatic else "disabled",
                textvariable=self.auto_advance if automatic else advance,
            )

        choice.bind("<<ComboboxSelected>>", toggle_action)
        self.vars[1].trace_add("write", toggle_action)
        toggle_action()
        count = tk.StringVar(pane, value=str(self.board_count))
        self.text(pane, "연속 판 수", 10, INK, True).grid(
            row=13, column=0, columnspan=2, sticky="w", pady=10
        )
        ttk.Entry(pane, textvariable=count, width=6).grid(row=13, column=2, sticky="ew")
        error = self.text(pane, "", 9, "#a2442b")
        error.configure(wraplength=190)
        error.grid(row=15, column=0, columnspan=3, sticky="w", pady=(3, 0))

        def restore():
            for fields, coords in (
                (start, self.motion.start_mm),
                (point, self.motion.point_mm),
            ):
                for var, value in zip(fields, coords):
                    var.set(f"{value:g}")
            self.start_level_field.set(self.motion.start_level)
            self.update_start_height()
            action.set(END_ACTIONS[self.motion.action])
            advance.set(ADVANCE_OPTIONS[self.motion.advance])
            toggle_action()
            count.set(str(self.board_count))
            error.config(text="")

        def apply():
            try:
                num = self.positive_count(count)
                if num is None:
                    raise ValueError("연속 판 수는 1 이상이어야 합니다.")
                key = next(k for k, v in END_ACTIONS.items() if v == action.get())
                level = self.start_level_field.get()
                motion = Motion(
                    (
                        float(start[0].get()),
                        float(start[1].get()),
                        self.motion.start_mm[2] if level else float(start[2].get()),
                    ),
                    key,
                    tuple(float(v.get()) for v in point)
                    if key == "point" or MODES[self.vars[1].get()] == FULL_AUTO
                    else self.motion.point_mm,
                    next(k for k, v in ADVANCE_OPTIONS.items() if v == advance.get()),
                    start_level=level,
                )
            except (TypeError, ValueError) as exc:
                error.config(text=str(exc))
                return
            if self.archive:
                if not self.load(self.archive, motion=motion, board_count=num):
                    return
            else:
                self.motion, self.board_count = motion, num
                self.route_picker.set_motion(motion)
            restore()

        buttons = tk.Frame(pane, bg="white")
        buttons.grid(row=14, column=0, columnspan=3, sticky="ew")
        self.button(buttons, "변경 취소", restore).pack(
            side="left", fill="x", expand=True
        )
        self.button(buttons, "적용", apply, True).pack(
            side="left", padx=(4, 0), fill="x", expand=True
        )

    def update_start_height(self):
        level = self.start_level_field.get()
        if level:
            value = (
                f"{self.layouts[0].start_height_mm(level):g}" if self.layouts else ""
            )
            self.start_z_field.set(value)

    def cancel_search(self):
        if self.playback:
            self.playback.close()
            self.playback = None
            self.cell = None
        if self.search_after is not None:
            self.root.after_cancel(self.search_after)
            self.search_after = None

    def load(
        self,
        path,
        *,
        grid=None,
        random_count=None,
        motion=None,
        board_count=None,
        seed=None,
    ):
        self.pause()
        values = tuple(v.get() for v in self.vars)
        sample, mode, speed, length, base = values
        if self.scan is None and grid is None and random_count is None:
            random_count, seed = self.initial_random, self.initial_seed
        try:
            s = (
                self.scan
                if self.archive == path
                and self.scan is not None
                and self.scan.sample == int(sample)
                else load_scan(path, int(sample))
            )
            num = self.board_count if board_count is None else board_count
            motion = self.motion if motion is None else motion
            old = self.layouts[:num]
            layout = old[0] if old else DEFAULT_LAYOUT
            if grid is not None:
                layouts = (grid_layout(s, *grid),) * num
            elif random_count is not None:
                layouts = random_boards(s, random_count, num, seed)
            elif self.scan is not s and layout is not DEFAULT_LAYOUT:
                if layout.columns is None:
                    layouts = tuple(
                        random_layout(s, len(layout.positions)) for _ in range(num)
                    )
                else:
                    layouts = (grid_layout(s, layout.columns, layout.rows),) * num
            else:
                layouts = old + tuple(
                    random_layout(s, len(layout.positions))
                    if layout.columns is None
                    else layout
                    for _ in range(num - len(old))
                )
            verify_route = MODES[mode] in (FULL_AUTO, LEADING)
            half_base = int(base) // 2
            cell = None
            playback = None
            if verify_route:
                playback = AutomaticPlayback(
                    s,
                    layouts,
                    int(speed),
                    int(length),
                    motion,
                    half_base,
                    worker=self.verification_worker,
                    mode=MODES[mode],
                )
            else:
                run = make_run(
                    s, layouts, int(speed), int(length), MODES[mode], motion, half_base
                )
                cell = Cell(s, run, half_base)
        except (OSError, ValueError, RuntimeError, tarfile.TarError, pb.error) as e:
            if self.loaded:
                for v, old in zip(self.vars, self.loaded):
                    v.set(old)
            messagebox.showerror(
                "자료 또는 시나리오를 열 수 없습니다", str(e), parent=self.root
            )
            return
        self.close_compare()
        self.cancel_search()
        self.playback = playback
        if self.cell:
            self.cell.close()
        self.cell = cell
        self.scan = s
        self.archive = path
        self.loaded = values
        self.layouts, self.board_count, self.motion = (
            layouts,
            num,
            motion.for_layout(layouts[0]),
        )
        self.plan_status = ""
        self.route_picker.set_motion(self.motion)
        self.update_start_height()
        self.update_layout_buttons()
        self.sensor = sensor_image(s)
        self.file_label.config(
            text=f"세트 {s.sample:02}  ·  PCL {CAPTURES[s.sample][0]}"
        )
        self.save_button.config(state="disabled")
        self.run_button.config(state="disabled" if verify_route else "normal")
        if verify_route:
            self.plan_status = "첫 판 준비 중…"

            def poll():
                self.search_after = None
                try:
                    if not self.playback.prepare():
                        self.search_after = self.root.after(40, poll)
                        return
                    self.cell = self.playback.cell
                except (OSError, ValueError, RuntimeError, pb.error) as exc:
                    self.cancel_search()
                    self.plan_status = f"재생 준비 오류 · {exc}"
                else:
                    self.plan_status = ""
                    self.run_button.config(state="normal")
                self.refresh()

            self.search_after = self.root.after(75, poll)
        self.refresh()
        return True

    def change(self):
        if self.archive:
            self.load(self.archive)

    def reset(self):
        if self.archive and (
            not self.cell or len(self.cell.run.boards) < self.board_count
        ):
            self.load(self.archive)
            return
        if self.cell:
            self.pause()
            cell = Cell(self.scan, self.cell.run, int(self.loaded[4]) // 2)
            if self.playback:
                self.cancel_search()
            else:
                self.cell.close()
            self.cell = cell
            self.save_button.config(state="disabled")
            self.refresh()

    def space(self, event):
        if event.widget.winfo_class() not in (
            "TCombobox",
            "TEntry",
            "Button",
            "TButton",
            "TRadiobutton",
        ):
            self.toggle()
            return "break"

    def pause(self):
        self.playing = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.run_button.config(text="▶  재생")

    def toggle(self):
        if not self.cell:
            if self.scan is None:
                self.pick_file()
            return
        if self.playing:
            self.pause()
            self.refresh()
            return
        if self.finished:
            self.reset()
        self.playing = True
        self.last_time = time.monotonic()
        self.sim_credit = 0.0
        self.run_button.config(text="일시정지")
        self.tick()

    @property
    def finished(self):
        return self.playback.finished if self.playback else self.cell.finished

    def tick(self):
        self.after_id = None
        if not self.playing:
            return
        now = time.monotonic()
        self.sim_credit += min(now - self.last_time, 0.20)
        self.last_time = now
        steps = min(int(self.sim_credit / DT), 12)
        before = self.cell.elapsed
        for _ in range(steps):
            if self.finished:
                break
            if self.playback:
                try:
                    advanced = self.playback.step()
                except (OSError, ValueError, RuntimeError, pb.error) as exc:
                    self.pause()
                    self.cancel_search()
                    self.plan_status = f"재생 오류 · {exc}"
                    self.run_button.config(state="disabled")
                    self.refresh()
                    return
                if not advanced:
                    self.sim_credit = 0.0
                    break
            else:
                self.cell.step()
        self.sim_credit = max(0.0, self.sim_credit - (self.cell.elapsed - before))
        if self.finished:
            self.pause()
            self.save_button.config(state="normal")
            self.run_button.config(text="다시 재생")
            self.refresh()
            return
        draw_interval = 1 / 30 if self.mode_view == "작업 셀" else 0.065
        if now - self.last_stats >= 0.20:
            self.refresh()
            self.last_stats = self.last_draw = now
        elif self.mode_view != "촬영 데이터" and now - self.last_draw >= draw_interval:
            self.draw()
            self.last_draw = now
        self.after_id = self.root.after(8, self.tick)

    def view(self, name):
        self.mode_view = name
        for n, b in self.tab_buttons.items():
            b.config(
                bg=TEAL if n == name else "#edf3f2", fg="white" if n == name else INK
            )
        self.draw()

    def orbit(self, e):
        if self.drag and self.mode_view == "작업 셀":
            x, y = self.drag
            self.yaw += (e.x - x) * 0.35
            self.pitch = float(np.clip(self.pitch + (e.y - y) * 0.25, -78, 5))
            self.draw()
        self.drag = (e.x, e.y)

    def zoom(self, e, sign=None):
        if self.mode_view == "작업 셀":
            self.distance = float(
                np.clip(
                    self.distance
                    * (0.92 if (e.delta if sign is None else sign) > 0 else 1.09),
                    1.2,
                    5.0,
                )
            )
            self.draw()

    def camera_default(self):
        self.yaw, self.pitch, self.distance = 49.0, -17.0, 2.85
        self.draw()

    def cloud(self):
        if not self.scan:
            return
        args = [sys.executable]
        if not getattr(sys, "frozen", False):
            args.append(str(Path(__file__).resolve()))
        args.extend(
            [
                "--cloud",
                "--tar",
                str(self.archive),
                "--sample",
                str(self.scan.sample),
            ]
        )
        self.children = [p for p in self.children if p.poll() is None]
        if self.children:
            return
        self.children.append(subprocess.Popen(args))

    def draw(self):
        c = self.canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 20 or h < 20:
            return
        c.delete("all")
        if not self.cell and (self.scan is None or self.mode_view != "촬영 데이터"):
            c.create_text(
                w / 2,
                h / 2,
                text=self.plan_status
                or "원본 TAR를 열면 촬영 데이터와 작업 셀이 표시됩니다.",
                width=w - 40,
                fill=INK,
                font=(FONT, 12),
            )
            return
        if self.mode_view == "촬영 데이터":
            fit = min((w - 20) / 640, (h - 48) / 480)
            size = (max(1, int(640 * fit)), max(1, int(480 * fit)))
            self.photo = ImageTk.PhotoImage(
                self.sensor.resize(size, Image.Resampling.NEAREST)
            )
            c.create_image(w / 2, (h - 26) / 2, image=self.photo)
            c.create_text(
                w / 2,
                h - 16,
                text="깊이 영상  ·  노랑 = 선택 표면  /  흰 테두리 = 샘플별 수동 관심 영역",
                fill=INK,
                font=(FONT, 9),
            )
        elif self.mode_view == "피복 지도":
            self.draw_map(w, h)
        else:
            max_w, max_h = PLAYBACK_RENDER_SIZE if self.playing else STILL_RENDER_SIZE
            scale = min(1.0, max_w / w, max_h / h)
            rw = max(20, round(w * scale))
            rh = max(20, round(h * scale))
            rgb, m = self.cell.render(
                rw, rh, self.yaw, self.pitch, self.distance, shadows=not self.playing
            )
            # Scale only the 3D image. Spray paths and beams stay sharp at canvas resolution.
            im = Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR)

            def pixel(point):
                q = m @ np.append(point, 1.0)
                return ((q[0] / q[3] + 1) * w / 2, (1 - q[1] / q[3]) * h / 2)

            pen = ImageDraw.Draw(im)
            for seg in self.cell.plan.segments:
                if not seg.spraying:
                    continue
                a, b = seg.a.copy(), seg.b.copy()
                for p in (a, b):
                    p[0] += self.cell.active_rack_x
                    p[1] = -p[1]
                pen.line((pixel(a), pixel(b)), fill=(38, 128, 129), width=1)
            if self.cell.spraying:
                from pybullet import getMatrixFromQuaternion

                for tcp, quat in self.cell.poses:
                    rot = np.array(getMatrixFromQuaternion(quat)).reshape(3, 3)
                    for dy in np.linspace(
                        -self.cell.plan.effective / 2, self.cell.plan.effective / 2, 7
                    ):
                        start = tcp + dy * rot[:, 1]
                        pen.line(
                            (pixel(start), pixel(start + 0.055 * rot[:, 2])),
                            fill=(12, 224, 213),
                            width=2,
                        )
            self.photo = ImageTk.PhotoImage(im)
            c.create_image(0, 0, image=self.photo, anchor="nw")
            c.create_rectangle(8, 8, 235, 34, fill="#203e47", outline="")
            c.create_text(
                17,
                21,
                anchor="w",
                text="A 주황  /  B 파랑    ·    청록 분사 경로",
                font=(FONT, 9),
                fill="white",
            )
            hint = c.create_text(
                10,
                h - 10,
                anchor="sw",
                text="드래그 회전  ·  휠 확대  ·  더블클릭 기본 시점",
                fill="#19333c",
                font=(FONT, 9),
            )
            x0, y0, x1, y1 = c.bbox(hint)
            back = c.create_rectangle(
                x0 - 4, y0 - 2, x1 + 4, y1 + 2, fill="#eaf2f0", outline=""
            )
            c.tag_lower(back, hint)

    def draw_map(self, w, h):
        c = self.canvas
        p = self.cell.plan
        pts = p.targets
        c.create_rectangle(0, 0, w, h, fill="#142d38", outline="")
        scale = min((w / 2 - 40) / 0.72, (h - 75) / 1.10)
        if scale <= 0:
            return
        for k, ratio in enumerate((0.25, 0.75)):
            cx = w * ratio
            top = 35
            bottom = top + RACK_HEIGHT * scale
            c.create_rectangle(
                cx - 0.35 * scale, top, cx + 0.35 * scale, bottom, outline="#64858d"
            )
            c.create_text(
                cx,
                17,
                text=("A면 · 실측 형상 복제" if not k else "B면 · 대칭 복제 가정"),
                fill="white",
                font=(FONT, 10),
            )
            # Draw each sample once on an in-memory raster instead of thousands of Tk widgets.
            iw = max(2, int(0.7 * scale))
            ih = max(2, int(RACK_HEIGHT * scale))
            raster = Image.new("RGB", (iw, ih), "#183640")
            draw = ImageDraw.Draw(raster)
            for z in p.layout.support_heights:
                y = (RACK_BOTTOM + RACK_HEIGHT - z) * scale
                draw.line((0, y, iw - 1, y), fill="#375360")
            for seg in p.segments:
                if seg.spraying:
                    draw.line(
                        [
                            (
                                (v[0] + 0.35) * scale,
                                (RACK_BOTTOM + RACK_HEIGHT - v[2]) * scale,
                            )
                            for v in (seg.a, seg.b)
                        ],
                        fill="#44656d",
                        width=1,
                    )
            for (u, _, z), hit in zip(pts, self.cell.dwell[k] >= MIN_DWELL):
                x = (u + 0.35) * scale
                yy = (RACK_HEIGHT + RACK_BOTTOM - z) * scale
                draw.rectangle(
                    (x, yy, x + 1, yy + 1), fill="#38d0b3" if hit else "#eeb360"
                )
            img = ImageTk.PhotoImage(raster)
            if k == 0:
                self.map_images = []
            self.map_images.append(img)
            c.create_image(cx, top + 1, image=img, anchor="n")
        c.create_text(
            w / 2,
            h - 15,
            text="황색  미충족     /     청록  이격·방향·체류 조건 충족",
            fill="#d2e4e4",
            font=(FONT, 10),
        )

    def refresh(self):
        if not self.cell:
            if self.plan_status:
                self.status.config(text=self.plan_status, fg=INK)
                self.stats.config(text="")
                self.progress.config(value=0)
                self.auto_action.set("—")
                self.auto_advance.set("—")
                self.data_label.config(text="")
                self.layout_label.config(
                    text=f"연속 {self.board_count}판 · {self.layouts[0].label}"
                )
                self.draw()
            return
        s = self.cell
        p = s.plan
        r = s.result()
        if s.run.decisions:
            _, action, advance = s.run.decision(s.board_index)
            show_advance = (
                action != "continue"
                or s.board_index + 1 < len(s.run.boards)
                or s.board_index + 1 == self.board_count
            )
            self.auto_action.set(END_ACTIONS[action])
            self.auto_advance.set(
                (ADVANCE_OPTIONS[advance] or "미리 이동 없음") if show_advance else "—"
            )
            policy = decision_label(
                s.run.decision(s.board_index), p.traversal, show_advance=show_advance
            )
        else:
            policy = self.motion.end_label
        if s.board_index in s.run.fallback_boards:
            policy = f"이 판은 최단 경로 적용 · 통과 경로 없음\n{policy}"
        self.layout_label.config(
            text=f"{s.board_index + 1}/{self.board_count}판 · {p.layout.label}\n{policy}"
        )
        phase = (
            "완료"
            if self.finished
            else "판 작업 완료"
            if s.finished and self.playing
            else s.phase
            if self.playing
            else "일시정지"
            if s.frame
            else "대기"
        )
        if s.had_contact:
            phase += " · 접촉"
        elif not all(s.last_ik):
            phase += " · IK 목표 미충족" if s.frame else " · 시작 TCP 도달 미달"
        self.status.config(
            fg="#a2442b" if (s.had_contact or not all(s.last_ik)) else INK
        )
        timing = f"{s.elapsed:05.2f} / {s.run.duration:.2f}초"
        margin = f"여유 {r['margin_s']:+.2f}초"
        progress = s.elapsed / s.run.duration
        if (
            (self.playback or s.run.decisions)
            and not self.finished
            and self.board_count > 1
        ):
            timing = f"경과 {s.elapsed:05.2f}초"
            margin = f"현재 판 작업 {p.duration:.2f}초"
            fraction = np.clip(
                (s.elapsed - s.run.starts[s.board_index]) / p.duration, 0, 1
            )
            progress = (s.board_index + fraction) / self.board_count
        self.status.config(
            text=f"{s.board_index + 1}/{self.board_count}판 · {phase}    {timing}     |     도착 주기 {PITCH / p.speed:.2f}초     |     {margin}"
        )
        self.progress.config(value=100 * progress)
        ik = "—" if r["ik_ok_pct"] is None else f"{r['ik_ok_pct']:.0f}%"
        err = "—" if r["tcp_p95_mm"] is None else f"{r['tcp_p95_mm']:.1f} mm"
        performance = ""
        if self.finished:
            speed, success = run_performance(s.board_results())
            performance = f"\n처리속도 {speed}     ·     판 성공률 {success}"
        self.stats.config(
            text=f"기하 피복  A {r['coverage_pct'][0]:.0f}%  /  B {r['coverage_pct'][1]:.0f}%     ·     IK {ik}     ·     TCP P95 {err}     ·     접촉 {r['contact_frames']}프레임{performance}",
            justify="left",
        )
        sc = self.scan
        axis_note = " · 방향 모호" if sc.axis_ratio < 1.5 else ""
        self.data_label.config(
            text=f"센서 관측    유효 {sc.valid.sum():,}점    /    선택 {sc.selected.sum():,}점    /    외곽 {sc.size[0] * 1000:.1f} × {sc.size[1] * 1000:.1f} mm    /    장축 {sc.angle:.1f}°{axis_note}"
        )
        self.draw()

    def save(self):
        if not self.cell or not self.finished:
            return
        dest = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".md",
            filetypes=[("Markdown", "*.md")],
            initialfile=f"AirKnife_세트{self.scan.sample}_{MODES[self.loaded[1]]}.md",
        )
        if dest:
            try:
                Path(dest).write_text(
                    details(
                        self.scan,
                        self.cell.plan,
                        int(self.loaded[4]) // 2,
                        {**self.cell.result(), "boards": self.cell.board_results()},
                        self.cell.run,
                    ),
                    encoding="utf-8",
                )
            except OSError as e:
                messagebox.showerror("결과 저장", str(e), parent=self.root)

    def compare(self):
        if not self.scan:
            return
        if self.compare_window:
            self.compare_window.lift()
            return
        self.pause()
        self.refresh()
        self.compare_after = None
        win = tk.Toplevel(self.root)
        win.title("동일 조건 경로 비교")
        self.compare_window = win
        win.configure(bg="white")
        win.transient(self.root)
        win.protocol("WM_DELETE_WINDOW", self.close_compare)
        sample = self.scan
        speed = int(self.loaded[2])
        length = int(self.loaded[3])
        base = int(self.loaded[4]) // 2
        layouts, motion = self.layouts, self.motion
        self.text(
            win,
            f"세트 {sample.sample} / 연속 {len(layouts)}판 / 수동 종료 설정 {motion.end_label} / {speed} mm/s / 나이프 {length} mm / 베이스 간격 {base * 2} mm",
            11,
            INK,
            True,
        ).pack(anchor="w", padx=12, pady=9)
        cols = (
            "경로",
            "처리속도",
            "판 성공률",
            "계획시간",
            "도착 여유",
            "피복 A / B",
            "IK 충족",
            "TCP P95",
            "접촉 프레임",
        )
        tree = ttk.Treeview(win, columns=cols, show="headings", height=len(MODES))
        for col in cols:
            tree.heading(col, text=col)
            tree.column(
                col, width=200 if col == "경로" else 100, anchor="center", stretch=True
            )
        tree.pack(padx=12, fill="x")
        note = self.text(
            win,
            f"같은 제품 배치에서 수동 {len(ROUTES)}개 경로와 전자동을 비교합니다.",
            9,
            MUTED,
        )
        note.pack(anchor="w", padx=12, pady=8)
        jobs = [(MODE_LABELS[mode], mode) for mode in ranked_routes(motion)]
        results = []

        def advance():
            self.compare_after = None
            try:
                label, mode = jobs[0]
                if self.compare_cell is None:
                    if (
                        mode in (FULL_AUTO, LEADING)
                        and self.cell
                        and MODES[self.loaded[1]] == mode
                        and len(self.cell.run.boards) == len(layouts)
                    ):
                        self.compare_cell = Cell(
                            sample, self.cell.run, base, visuals=False
                        )
                    elif mode in (FULL_AUTO, LEADING):
                        if self.compare_playback is None:
                            self.compare_playback = AutomaticPlayback(
                                sample,
                                layouts,
                                speed,
                                length,
                                motion,
                                base,
                                visuals=False,
                                worker=self.verification_worker,
                                mode=mode,
                            )
                        if not self.compare_playback.prepare():
                            self.compare_after = win.after(30, advance)
                            return
                        self.compare_cell = self.compare_playback.cell
                    else:
                        run = make_run(
                            sample, layouts, speed, length, mode, motion, base
                        )
                        self.compare_cell = Cell(sample, run, base, visuals=False)
                    note.config(text=f"{label} 재생 결과를 측정하고 있습니다.")
                sim = self.compare_cell
                driver = self.compare_playback or sim
                for _ in range(24):
                    if driver.finished:
                        break
                    if self.compare_playback:
                        if not driver.step():
                            break
                    else:
                        sim.step()
                if driver.finished:
                    r = {
                        **sim.result(),
                        "boards": sim.board_results(),
                        "decisions": sim.run.decisions,
                        "fastest_fallback": sim.run.fastest_fallback,
                        "traversals": [b.traversal for b in sim.run.boards],
                    }
                    speed_text, success = run_performance(r["boards"])
                    results.append((label, r))
                    jobs.pop(0)
                    tree.insert(
                        "",
                        "end",
                        values=(
                            f"{label} (통과 경로 없음 · 최단)"
                            if sim.run.fastest_fallback
                            else label,
                            speed_text,
                            success,
                            f"{r['plan_s']:.2f} s",
                            f"{r['margin_s']:+.2f} s",
                            f"{r['coverage_pct'][0]:.1f} / {r['coverage_pct'][1]:.1f}%",
                            f"{r['ik_ok_pct']:.1f}%",
                            f"{r['tcp_p95_mm']:.1f} mm",
                            r["contact_frames"],
                        ),
                    )
                    driver.close()
                    self.compare_playback = None
                    self.compare_cell = None
                if jobs:
                    self.compare_after = win.after(1, advance)
                else:
                    note.config(
                        text="처리속도는 판 사이 대기를 포함합니다. 판 성공률은 이번 실행에서 도달·접촉·피복 기준을 모두 통과한 판의 비율입니다."
                    )

                    def export():
                        path = filedialog.asksaveasfilename(
                            parent=win,
                            defaultextension=".md",
                            filetypes=[("Markdown", "*.md")],
                            initialfile=f"AirKnife_세트{sample.sample}_경로비교.md",
                        )
                        if path:
                            doc = compare_text(
                                sample,
                                speed,
                                length,
                                base,
                                results,
                                layouts=layouts,
                                motion=motion,
                            )
                            try:
                                Path(path).write_text(doc, encoding="utf-8")
                            except OSError as e:
                                messagebox.showerror("비교 저장", str(e), parent=win)

                    self.button(win, "비교 저장 (.md)", export).pack(
                        anchor="e", padx=12, pady=8
                    )
            except (OSError, ValueError, RuntimeError, pb.error) as e:
                self.close_compare()
                messagebox.showerror("경로 비교", str(e), parent=self.root)

        self.compare_after = win.after(1, advance)

    def close_compare(self):
        if self.compare_playback:
            self.compare_playback.close()
            self.compare_playback = self.compare_cell = None
        if self.compare_after is not None and self.compare_window:
            self.compare_window.after_cancel(self.compare_after)
            self.compare_after = None
        if self.compare_cell:
            self.compare_cell.close()
            self.compare_cell = None
        if self.compare_window:
            self.compare_window.destroy()
            self.compare_window = None

    def close(self):
        self.pause()
        self.close_compare()
        self.cancel_search()
        self.verification_worker.close()
        if self.cell:
            self.cell.close()
            self.cell = None
        for child in self.children:
            if child.poll() is None:
                child.terminate()
        self.root.destroy()


def compare_text(
    scan,
    speed,
    length,
    base,
    results,
    layout=DEFAULT_LAYOUT,
    *,
    layouts=None,
    motion=DEFAULT_MOTION,
):
    header = f"# Air Knife 경로 비교\n\n촬영 세트 {scan.sample} · 컨베이어 {speed} mm/s · 나이프 {length} mm · 베이스 간격 {base * 2} mm"
    return (
        header
        + "\n\n"
        + run_report(layouts or (layout,), motion)
        + "\n\n"
        + summary_table(results)
        + "".join(
            "\n\n### 전자동 판별 선택\n\n"
            + "\n".join(
                f"- {i}판 · {decision_label(d, r['traversals'][i - 1])}"
                for i, d in enumerate(r["decisions"], 1)
            )
            for _, r in results
            if r.get("decisions")
        )
        + "\n\n가상 랙과 예시 로봇에서 기하 피복과 전체 작업시간을 비교했다. 실제 액체 제거 성능은 실물 검증 대상이다.\n"
    )


def main():
    ap = argparse.ArgumentParser(description="Air Knife 트랙 2 시연 프로그램")
    ap.add_argument("--tar", type=Path)
    ap.add_argument("--sample", type=int, choices=CAPTURES, default=2)
    ap.add_argument("--speed", type=int, choices=(60, 65, 70), default=65)
    ap.add_argument("--length", type=int, choices=(200, 250, 300), default=250)
    ap.add_argument(
        "--base-mm",
        type=int,
        choices=(820, 1010),
        default=820,
        help="랙 중심에서 한쪽 로봇까지의 거리",
    )
    ap.add_argument("--mode", choices=tuple(MODE_LABELS))
    ap.add_argument("--boards", type=int, default=1, help="연속 판 수")
    ap.add_argument("--random-count", type=int, help="판 한 면당 랜덤 제품 수")
    ap.add_argument("--seed", type=int, help="첫 판 랜덤 시드")
    ap.add_argument(
        "--start-xyz",
        nargs=3,
        type=float,
        default=DEFAULT_MOTION.start_mm,
        metavar=("X", "Y", "Z"),
        help="B면 시작 TCP (mm)",
    )
    ap.add_argument(
        "--start-level",
        choices=tuple(START_LEVELS),
        default="",
        help="판의 제품 줄에서 시작 Z 선택 (생략 시 입력한 Z 사용)",
    )
    ap.add_argument("--end-action", choices=tuple(END_ACTIONS), default="home")
    ap.add_argument(
        "--advance",
        choices=tuple(k for k in ADVANCE_OPTIONS if k),
        default="",
        help="다음 판의 상·하단 또는 선두 제품에 미리 정렬 (생략 시 제자리 대기)",
    )
    ap.add_argument(
        "--end-xyz",
        nargs=3,
        type=float,
        default=DEFAULT_MOTION.point_mm,
        metavar=("X", "Y", "Z"),
        help="지정 종료 TCP (mm)",
    )
    group = ap.add_mutually_exclusive_group()
    for option in ("headless", "compare", "cloud"):
        group.add_argument("--" + option, action="store_true")
    ap.add_argument("--report", type=Path, help="일괄 실행 결과를 저장할 .md 경로")
    args = ap.parse_args()
    if args.boards < 1:
        ap.error("--boards는 1 이상이어야 합니다.")
    if args.random_count is not None and args.random_count < 1:
        ap.error("--random-count는 1 이상이어야 합니다.")
    if args.seed is not None and (args.seed < 0 or args.random_count is None):
        ap.error("--seed는 --random-count와 함께 0 이상의 정수로 지정하세요.")
    try:
        motion = Motion(
            tuple(args.start_xyz),
            args.end_action,
            tuple(args.end_xyz),
            args.advance,
            start_level=args.start_level,
        )
    except ValueError as exc:
        ap.error(str(exc))
    if args.report and (
        not (args.headless or args.compare) or args.report.suffix.lower() != ".md"
    ):
        ap.error("--report는 --headless 또는 --compare와 .md 경로를 사용하세요.")
    archive = args.tar or locate_tar()
    if args.headless or args.compare or args.cloud:
        if archive is None:
            ap.error("--tar로 제공된 원본을 지정하세요.")
        sc = load_scan(archive, args.sample)
        if args.cloud:
            return open_cloud(sc)
        try:
            layouts = (
                random_boards(sc, args.random_count, args.boards, args.seed)
                if args.random_count is not None
                else (DEFAULT_LAYOUT,) * args.boards
            )
        except ValueError as exc:
            ap.error(str(exc))
        if args.compare:
            res = []
            for mode in ranked_routes(motion):
                _, result = batch(
                    sc,
                    args.speed,
                    args.length,
                    mode,
                    args.base_mm,
                    layouts=layouts,
                    motion=motion,
                )
                res.append((MODE_LABELS[mode], result))
            text = compare_text(
                sc,
                args.speed,
                args.length,
                args.base_mm,
                res,
                layouts=layouts,
                motion=motion,
            )
        else:
            mode = args.mode or "target"
            run, res = batch(
                sc,
                args.speed,
                args.length,
                mode,
                args.base_mm,
                layouts=layouts,
                motion=motion,
                return_run=True,
            )
            text = details(sc, run.boards[0], args.base_mm, res, run)
        print(text)
        if args.report:
            args.report.write_text(text, encoding="utf-8")
    else:
        root = tk.Tk()
        Demo(
            root,
            archive,
            args.sample,
            args.speed,
            args.length,
            args.mode or FIRST_MODE,
            args.base_mm,
            motion,
            args.boards,
            args.random_count,
            args.seed,
        )
        root.mainloop()


if __name__ == "__main__":
    freeze_support()
    main()
