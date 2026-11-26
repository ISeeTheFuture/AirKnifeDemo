"""Validate three-board benchmark checkpoints and write the Korean Markdown results."""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

LABELS = {
    "horizontal": "가로",
    "direct": "직결",
    "oblique": "사선",
    "order": "순서",
    "target": "기존",
    "raster": "전체",
}
FULL_NAMES = {
    "horizontal": "가로 묶음 분사",
    "direct": "제품별 분사 후 직결 이동",
    "oblique": "사선 묶음 분사",
    "order": "방문 순서·분사 방향 최적화",
    "target": "기존 제품별 분사",
    "raster": "전체 랙 왕복",
}
HEIGHTS = {"상": 1150, "중": 840, "하": 530}
ACTIONS = {"home": "시작점 복귀", "point": "지정점 이동", "continue": "끝점 유지"}
INTERVAL = 990 / 65


@dataclass
class Record:
    mode: str
    elapsed: float
    boards: list

    @property
    def passed(self):
        return all(b["passed"] for b in self.boards)

    @property
    def score(self):
        return sum(b["work_s"] if b["passed"] else 20 for b in self.boards)


def read_trials(folder):
    trials = {}
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        n, rep, height, action = (data[k] for k in ("n", "repeat", "height", "action"))
        key = (n, rep, height, action)
        if key in trials or data["seed"] != 20261124 + n * 100 + rep:
            raise ValueError(f"Duplicate or invalid trial: {path}")
        methods = data["methods"]
        if set(methods) != set(LABELS):
            raise ValueError(f"Missing methods: {path}")
        records = {}
        for mode, m in methods.items():
            boards = m["boards"]
            if len(boards) != 3 or not math.isclose(m["elapsed"], boards[-1]["end"]):
                raise ValueError(f"Incomplete run: {path}, {mode}")
            end = 0.0
            for idx, board in enumerate(boards):
                passed = (
                    board["contacts"] == board["ik_misses"] == 0
                    and board["worst_pct"] >= 95
                )
                if (
                    passed != board["passed"]
                    or not 0 <= board["worst_pct"] <= 100
                    or not math.isclose(
                        board["start"], max(idx * INTERVAL, end), abs_tol=1e-8
                    )
                    or not math.isclose(
                        board["end"] - board["start"], board["work_s"], abs_tol=1e-8
                    )
                ):
                    raise ValueError(f"Invalid board: {path}, {mode}, {idx + 1}")
                end = board["end"]
            records[mode] = Record(mode, m["elapsed"], boards)
        trials[key] = records
    expected = {
        (n, r, h, a)
        for n in range(1, 31)
        for r in range(5)
        for h in HEIGHTS
        for a in ACTIONS
    }
    if set(trials) != expected:
        raise ValueError(f"Found {len(trials)} settings; expected all {len(expected)}.")
    return trials


def table(headings, rows):
    def line(cells):
        return "| " + " | ".join(map(str, cells)) + " |"

    return "\n".join(
        [line(headings), line(["---"] * len(headings)), *(line(row) for row in rows)]
    )


def hms(seconds):
    minutes, cs = divmod(round(seconds * 100), 6000)
    h, m = divmod(minutes, 60)
    return f"{h:02}:{m:02}:{cs / 100:05.2f}"


def fastest(values):
    best = min(values.values())
    return [key for key, value in values.items() if value <= best + 0.001]


def names(keys):
    return "·".join(LABELS[key] for key in keys)


def success_rate(passed, total):
    return f"{passed:,}/{total:,} ({passed / total:.1%})"


def board_speed(seconds):
    """Convert the elapsed time for one three-board batch to boards per minute."""
    return f"{3 * 60 / seconds:.2f}"


def metrics(cases):
    scores, times, successes = {}, {}, {}
    wins = Counter()
    valid_times = []
    for mode in LABELS:
        runs = [c[mode] for c in cases]
        scores[mode] = sum(r.score for r in runs)
        times[mode] = median(r.elapsed for r in runs)
        successes[mode] = sum(r.passed for r in runs)
    for case in cases:
        eligible = {m: r.elapsed for m, r in case.items() if r.passed}
        if eligible:
            wins.update(fastest(eligible))
            valid_times.append(min(eligible.values()))
    return scores, times, successes, wins, valid_times


def split_by(trials, fields):
    groups = defaultdict(list)
    for key, case in trials.items():
        groups[tuple(key[i] for i in fields)].append(case)
    return groups


def generate(trials):
    cases = list(trials.values())
    scores, times, successes, wins, valid = metrics(cases)
    rankings = sorted(LABELS, key=scores.get)
    calibrated = []
    for m in rankings:
        runs = [case[m] for case in cases]
        boards = [b for r in runs for b in r.boards]
        successful = [r for r in runs if r.passed]
        calibrated.append(
            [
                FULL_NAMES[m],
                sum(b["passed"] for b in boards),
                len(boards) - sum(b["passed"] for b in boards),
                success_rate(successes[m], len(runs)),
                f"{times[m]:.2f}",
                board_speed(times[m]),
                wins[m],
                hms(scores[m]),
                f"{median(r.elapsed for r in successful):.2f}" if successful else "—",
                sum(r.elapsed <= 3 * INTERVAL for r in successful),
            ]
        )
    lines = [
        "# 에어 나이프 연속 3판 실험",
        "",
        "촬영 세트 2에서 판당 제품 1~30개를 랜덤 생성했다. 개수당 5개씩 총 150개 3판 묶음(450개 판 배치)을 만들고, 시작 높이 3개·판 종료 동작 3개·분사 방식 6개를 교차했다. 총 8,100회 연속 운전과 24,300개 판 작업을 끝까지 시뮬레이션했다.",
        "",
        f"판별 실패에 20초를 적용한 누적시간에서 **{names(fastest(scores))}** 방식이 1위를 기록했다. 배치·높이·종료 동작의 1,350개 조합 중 {len(valid)}개에서 여섯 후보 중 하나 이상이 세 판을 모두 통과했다.",
        "",
        "## 전체 방식 비교",
        "",
        "누적시간은 한 방식당 4,050개 판에서 성공한 판의 작업시간과 실패한 판당 20초를 더한 값이다. 작업시간에는 판 종료 동작이 포함된다. 다음 판이 도착하기까지의 공통 대기는 판별 점수에서 제외하고, 성공 묶음의 전체 경과시간에는 포함했다. ‘통과 1위’는 같은 배치·설정에서 세 판 모두 통과한 후보 중 가장 빨랐던 횟수다. 0.001초 이내는 공동 1위로 집계했다.",
        "",
        "처리속도(판/분)는 180을 해당 방식의 3판 경과시간 중앙값(초)으로 나눈 값이다. 성공과 실패를 모두 포함하며 첫 판 시작부터 마지막 판 종료까지 판 사이 대기와 종료 동작을 센다. 첫 판은 시작 시점에 이미 도착해 있으므로, 짧은 묶음에서 환산한 속도는 컨베이어의 장기 공급 속도(3.94판/분)보다 높을 수 있다. 3판 성공률은 세 판 모두 도달·접촉·피복 기준을 통과한 묶음의 비율이다.",
        "",
        table(
            [
                "분사 방식",
                "성공 판",
                "실패 판",
                "3판 성공률",
                "전체 3판 시간 중앙값 (초)",
                "처리속도 (판/분)",
                "통과 1위",
                "판별 20초 적용 합계 (시:분:초)",
                "성공 묶음 경과시간 중앙값 (초)",
                "다음 묶음 도착 전 성공",
            ],
            calibrated,
        ),
        "",
        "가로는 여러 제품을 수평으로 함께 분사하고, 사선은 여러 제품을 경사선으로 함께 분사한다. 직결은 제품 사이에서 공기를 끄고 직선 이동한다. 순서는 제품별 분사와 46 mm 이격을 유지하며 방문 순서·방향을 탐색한다. 기존은 줄마다 방향을 바꾸며 제품별로 분사하고, 전체는 랙 전체를 왕복한다.",
        "",
        "## 제품 개수별 결과",
        "",
        "각 행은 3판 배치 5개 × 시작 높이·종료 동작 9개인 45개 조건이다. ‘단순 시간 1위’는 물리 판정에 관계없이 방식별 묶음 경과시간 중앙값을 비교한 결과다. ‘통과 최단’은 가능한 조건마다 가장 빠른 성공 경로를 고른 뒤 그 경과시간의 중앙값을 구했다. 일부 판만 통과해도 20초 점수에는 반영된다. 통과 조건이 0이면 해당 개수에서 확인된 3판 성공 방식은 없다.",
        "",
    ]
    grouped = split_by(trials, [0])
    rows = []
    infeasible_counts = []
    for n in range(1, 31):
        cost, t, _, w, v = metrics(grouped[(n,)])
        if not v:
            infeasible_counts.append(str(n))
        rows.append(
            [
                n,
                names(fastest(cost)),
                names(fastest(t)),
                f"{min(t.values()):.2f}",
                board_speed(min(t.values())),
                " · ".join(f"{LABELS[m]} {w[m]}" for m in LABELS if w[m]) or "—",
                success_rate(len(v), len(grouped[(n,)])),
                f"{median(v):.2f}" if v else "—",
                board_speed(median(v)) if v else "—",
            ]
        )
    lines.extend(
        [
            table(
                [
                    "제품 수/판",
                    "20초 점수 1위",
                    "단순 시간 1위",
                    "중앙값 (초)",
                    "단순 시간 1위 속도 (판/분)",
                    "통과 1위 횟수",
                    "한 방식 이상 3판 성공",
                    "통과 최단 (초)",
                    "통과 최단 환산 속도 (판/분)",
                ],
                rows,
            ),
            "",
            (
                f"세 판을 모두 통과한 조합이 관측되지 않은 제품 수는 {', '.join(infeasible_counts)}개다. 각 개수에서 다섯 배치와 54개 설정·방식 조합을 시험했다."
                if infeasible_counts
                else "모든 제품 수에서 한 조합 이상이 세 판을 모두 통과했다."
            ),
            "",
            "## 시작 높이와 종료 동작별 비교",
            "",
            "각 설정에서 동일한 150묶음에 판별 실패 20초를 적용해 누적시간 1위를 골랐다. 아래 속도와 성공률은 해당 1위 방식의 기록이다.",
            "",
        ]
    )
    grouped = split_by(trials, [2, 3])
    rows = []
    winners = []
    combinations = []
    for height, z in HEIGHTS.items():
        for action, label in ACTIONS.items():
            cases = grouped[height, action]
            costs, times, passes, _, valid_times = metrics(cases)
            for mode in fastest(costs):
                winners.append(
                    [
                        f"{height} ({z:,} mm)",
                        label,
                        FULL_NAMES[mode],
                        f"{times[mode]:.2f}",
                        board_speed(times[mode]),
                        success_rate(passes[mode], len(cases)),
                    ]
                )
            rows.append(
                [
                    height,
                    label,
                    *[
                        f"{board_speed(times[m])} · {success_rate(passes[m], len(cases))} · {hms(costs[m])}"
                        for m in LABELS
                    ],
                    names(fastest(costs)),
                    len(valid_times),
                ]
            )
            combinations.extend(
                (costs[m], passes[m], f"{height}단 · {label} · {LABELS[m]}")
                for m in LABELS
            )
    lowest = min(cost for cost, _, _ in combinations)
    highest = max(passed for _, passed, _ in combinations)
    fastest_settings = ", ".join(
        label for cost, _, label in combinations if cost <= lowest + 0.001
    )
    most_successful = ", ".join(
        label for _, passed, label in combinations if passed == highest
    )
    lines.extend(
        [
            table(
                [
                    "시작 높이 Z",
                    "종료 동작",
                    "누적시간 1위 방식",
                    "전체 3판 시간 중앙값 (초)",
                    "처리속도 (판/분)",
                    "3판 성공률",
                ],
                winners,
            ),
            "",
            "<details>",
            "<summary>9개 설정 × 6개 방식 전체 비교</summary>",
            "",
            "방식별 칸은 ‘처리속도(판/분) · 3판 성공률(횟수와 비율) · 판별 20초 적용 누적시간(시:분:초)’ 순서다.",
            "",
            table(
                [
                    "높이",
                    "매 판 종료 동작",
                    *LABELS.values(),
                    "20초 점수 1위",
                    "통과 조건 / 150",
                ],
                rows,
            ),
            "",
            "</details>",
            "",
            f"54개 조합 중 **{fastest_settings}** 조합이 가장 짧은 누적시간을 기록했다. 150묶음에 판별 실패 20초를 적용한 합계는 {hms(lowest)}다. 세 판 모두 성공한 묶음 수는 **{most_successful}** 조합이 150묶음 중 {highest}묶음으로 가장 많았다.",
            "",
            "## 높이·종료 동작·제품 수별 상세 결과",
            "",
            "각 행은 배치 5개다. 방식별 칸은 ‘3판 경과시간 중앙값(초) · 처리속도(판/분) · 3판 성공률(횟수와 비율)’ 순서다. 마지막 두 열은 판별 20초 점수와 통과한 후보의 실제 경과시간을 각각 비교한다.",
            "",
        ]
    )
    groups = split_by(trials, [0, 2, 3])
    for h, z in HEIGHTS.items():
        for a, label in ACTIONS.items():
            rows = []
            for n in range(1, 31):
                cases = groups[n, h, a]
                c, t, s, w, _ = metrics(cases)
                rows.append(
                    [
                        n,
                        *[
                            f"{t[m]:.2f} · {board_speed(t[m])} · {success_rate(s[m], len(cases))}"
                            for m in LABELS
                        ],
                        names(fastest(c)),
                        " · ".join(f"{LABELS[m]} {w[m]}" for m in LABELS if w[m])
                        or "—",
                    ]
                )
            lines.extend(
                [
                    "<details>",
                    f"<summary>{h}단 {z} mm · {label}</summary>",
                    "",
                    table(
                        [
                            "제품 수/판",
                            *LABELS.values(),
                            "20초 점수 1위",
                            "통과 1위 횟수",
                        ],
                        rows,
                    ),
                    "",
                    "</details>",
                    "",
                ]
            )
    lines.extend(
        [
            "## 시험 조건과 해석",
            "",
            "- 입력은 B면 TCP의 월드 XYZ(mm)다. A면은 Y 부호를 반전한다. 시작 X=0, Y=160으로 고정하고 Z=1,150·840·530을 각각 상·중·하로 사용했다. 지정 종료점은 (0, 200, 900) mm로 고정했다. 모두 시연용 가정값이다.",
            "- 시작점 복귀는 매 판 최초 시작 XYZ로 돌아간다. 지정점 이동은 매 판 지정한 XYZ로 간다. 끝점 유지는 마지막 분사 후 공기를 끄고 추가 후퇴·복귀 없이 월드 좌표를 유지한다. 마지막 판에도 선택한 종료 동작을 적용했다.",
            "- 컨베이어 65 mm/s, 랙 피치 990 mm, 판 도착 주기 15.231초, 3판 주기 45.692초다. 한 판이 일찍 끝나면 종료 위치에서 대기한다. 늦게 끝나면 다음 판의 실제 이동 위치에서 계획한다. 같은 양팔·같은 물리 장면을 세 판 동안 계속 사용했다.",
            "- 나이프 250 mm, 유효 폭 200 mm, 베이스 중심 간격 1,640 mm다. 두 면에는 같은 측정 표면을 대칭 복제했다. 판 통과 조건은 접촉 프레임 0개, 양팔의 모든 목표에서 IK 오차 4 mm·3° 이내, 각 제품의 각 면에서 기하 피복률 95% 이상이다. 피복은 누적 노출 25 ms 기준이다. 대기 중 접촉과 도달 실패는 직전 판에 귀속했다.",
            "- 각 시험은 마지막 판의 종료 시점까지 실행했다. 세 판 모두 통과한 경우에만 성공 묶음으로 집계했다. 판당 20초 점수, 묶음 경과시간, 다음 묶음 도착 전 완료 여부는 별도로 기록했다. 표본은 개수당 5개로, 숫자가 같은 다른 배치에서는 결과가 달라질 수 있다. 액체 제거 성능은 실물 시험 대상이다.",
            "- 첫 판 시드는 20261124 + 100 × 제품 수 + 반복 번호(0~4)다. 둘째·셋째 판은 각각 1,000,000·2,000,000을 더한다. 동일한 시드의 세 판 좌표는 모든 방식과 설정이 공유한다.",
            "",
            "## 재현",
            "",
            "프로젝트 루트에서 데모 환경으로 실행한다. 완료한 조건은 파일 단위로 저장되며 같은 출력 폴더로 재실행하면 누락된 조건부터 계산한다. 원본 TAR 위치는 필요하면 `--tar`로 지정한다.",
            "",
            "```shell",
            "python benchmarks/continuous_routes.py --workers 4 --output-dir three-board-results",
            "python benchmarks/continuous_report.py three-board-results --output docs/트랙2_연속3판_경로비교.md",
            "```",
            "",
        ]
    )
    examples = (
        (-n, record.elapsed, repeat, height, action, mode)
        for (n, repeat, height, action), case in trials.items()
        for mode, record in case.items()
        if record.passed
    )
    example = min(examples, default=None)
    if example:
        negative_n, elapsed, repeat, height, action, mode = example
        n = -negative_n
        seed = 20261124 + 100 * n + repeat
        lines.extend(
            [
                "### 화면에서 재현할 수 있는 통과 사례",
                "",
                f"시험에서 3판 모두 통과한 최대 제품 수는 판당 {n}개였다. 그 개수에서 가장 빨리 통과한 사례의 시작 높이는 {height}단, 종료 동작은 {ACTIONS[action]}, 분사 방식은 {FULL_NAMES[mode]}다. 판 사이 대기를 포함해 {elapsed:.3f}초가 걸렸다. 프로젝트 루트의 PowerShell에서 실행한 뒤 **재생**을 누르면 같은 세 판을 볼 수 있다.",
                "",
                "```powershell",
                f".\\run.cmd --sample 2 --boards 3 --random-count {n} --seed {seed} --mode {mode} --start-xyz 0 160 {HEIGHTS[height]} --end-action {action} --end-xyz 0 200 900",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def raw_records(trials):
    lines = [
        "# 연속 3판 실험 개별 기록",
        "",
        "각 행은 동일한 세 판 배치를 여섯 방식으로 실행한 결과다. 방식별 칸의 첫 숫자는 대기를 포함한 3판 총 경과시간(초)이다. 이어지는 세 항목은 판 순서대로 작업시간(초)과 판정이다. P는 통과, I는 IK 허용오차 초과, C는 접촉, F는 제품 면 최저 피복률 95% 미달이다. 여러 사유가 있으면 함께 표시했다. 시간은 읽기 쉽게 소수 셋째 자리까지 표시하고 본문 점수는 반올림 전 시간으로 계산했다.",
        "",
    ]
    for n in range(1, 31):
        rows = []
        for repeat in range(5):
            for height in HEIGHTS:
                for action, label in ACTIONS.items():
                    row = [repeat, height, label]
                    for mode in LABELS:
                        run = trials[n, repeat, height, action][mode]
                        results = []
                        for board in run.boards:
                            reason = "".join(
                                marker
                                for marker, failed in (
                                    ("I", board["ik_misses"]),
                                    ("C", board["contacts"]),
                                    ("F", board["worst_pct"] < 95),
                                )
                                if failed
                            )
                            results.append(f"{board['work_s']:.3f}{reason or 'P'}")
                        row.append(f"{run.elapsed:.3f} ({', '.join(results)})")
                    rows.append(row)
        lines.extend(
            [
                "<details>",
                f"<summary>판당 {n}개 · 45개 조건</summary>",
                "",
                table(["반복 번호", "높이", "종료 동작", *LABELS.values()], rows),
                "",
                "</details>",
                "",
            ]
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--raw-output", type=Path, help="Optional Markdown appendix with every run"
    )
    args = p.parse_args()
    trials = read_trials(args.folder)
    report = generate(trials)
    if args.raw_output:
        args.raw_output.write_text(raw_records(trials), encoding="utf-8")
        link = Path(os.path.relpath(args.raw_output, args.output.parent)).as_posix()
        report += f"\n[8,100회 개별 실행 기록]({link})\n"
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
