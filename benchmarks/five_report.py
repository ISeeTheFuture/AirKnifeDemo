"""Validate 150 paired groups, tabulate every strategy, and save Markdown records."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from five_boards import FULL_AUTO, ROUTES, settings
from planner import ADVANCE_OPTIONS, END_ACTIONS, LEADING

SHORT = {
    LEADING: "선두",
    "horizontal": "가로",
    "direct": "직결",
    "oblique": "사선",
    "order": "순서",
    "target": "기존",
    "raster": "전체",
}
END = {"home": "복귀", "point": "지정", "continue": "계속"}


def name(key):
    route, action, advance = key
    if route == FULL_AUTO:
        return "전자동"
    return " / ".join([SHORT[route], END[action], ADVANCE_OPTIONS[advance] or "—"])


def hms(seconds):
    cs = round(100 * seconds)
    minutes, cs = divmod(cs, 6000)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{cs / 100:05.2f}"


def rate(count, total):
    return f"{count}/{total} ({count / total:.1%})"


def score(boards):
    return math.fsum(b["work_s"] if b["passed"] else 20.0 for b in boards)


def read_runs(folder):
    routes = [
        r for r in ROUTES if r != LEADING or any(folder.glob(f"*-{LEADING}.json"))
    ] + [FULL_AUTO]
    strategies = {key for r in routes for key in settings(r)}
    table = defaultdict(list)
    physics = {}
    decisions = Counter()
    for n in range(1, 31):
        for rep in range(5):
            for route in routes:
                group = json.loads(
                    (folder / f"{n:02}-{rep}-{route}.json").read_text("utf-8")
                )
                assert (group["n"], group["repeat"], group["seed"]) == (
                    n,
                    rep,
                    20261124 + 100 * n + rep,
                )
                assert [tuple(r["setting"]) for r in group["rows"]] == settings(route)
                for row in group["rows"]:
                    key = tuple(row["setting"])
                    if "unavailable" in row:
                        assert route == FULL_AUTO
                        # Five requested boards incur five failure penalties. No elapsed
                        # time or per-board measurements exist for a refused run.
                        run = {
                            "unavailable": row["unavailable"],
                            "elapsed": None,
                            "boards": [{"passed": False} for _ in range(5)],
                        }
                        table[key].append((n, rep, run, ()))
                        continue
                    ref = row["physics"]
                    assert ref.startswith(f"{n:02}-{rep}-") and Path(ref).name == ref
                    if ref not in physics:
                        physics[ref] = json.loads(
                            (folder / "physics" / ref).read_text("utf-8")
                        )
                    run = {
                        **physics[ref],
                        "fastest_fallback": row.get("fastest_fallback", False),
                    }
                    boards = run["boards"]
                    assert len(boards) == len(row["decisions"]) == 5
                    assert (
                        boards[0]["start"] == 0 and run["elapsed"] == boards[-1]["end"]
                    )
                    for i, b in enumerate(boards):
                        assert math.isclose(
                            b["end"] - b["start"], b["work_s"], abs_tol=1e-9
                        )
                        assert math.isclose(
                            b["start"],
                            max(i * 0.990 / 0.065, boards[i - 1]["end"] if i else 0),
                            abs_tol=1e-9,
                        )
                        assert b["passed"] == (
                            b["worst_pct"] >= 95
                            and b["contacts"] == b["ik_misses"] == 0
                        )
                        decision = tuple(row["decisions"][i])
                        assert decision in strategies and decision[0] != FULL_AUTO
                        if route == FULL_AUTO:
                            decisions[decision] += 1
                        else:
                            assert decision == key
                    table[key].append((n, rep, run, row["decisions"]))
    assert set(table) == strategies and all(len(v) == 150 for v in table.values())
    return table, decisions, len(physics)


def totals(records):
    runs = [r for _, _, r, _ in records]
    boards = [b for run in runs for b in run["boards"]]
    return (
        score(boards),
        sum(b["passed"] for b in boards),
        sum(all(b["passed"] for b in r["boards"]) for r in runs),
        statistics.median(times)
        if (times := [r["elapsed"] for r in runs if r["elapsed"] is not None])
        else float("nan"),
    )


def md_table(headers, rows):
    return [
        "| " + " | ".join(map(str, cells)) + " |"
        for cells in (headers, ["---"] * len(headers), *rows)
    ]


def report(folder, docs):
    table, decisions, unique = read_runs(folder)
    stats = {k: totals(v) for k, v in table.items()}
    ordered = sorted(table, key=lambda k: stats[k][0])
    auto = (FULL_AUTO, "", "")
    best_fixed = next(k for k in ordered if k != auto)
    max_groups = max(v[2] for v in stats.values())
    group_winners = [name(k) for k in ordered if stats[k][2] == max_groups]
    fastest_time = min(v[3] for v in stats.values() if math.isfinite(v[3]))
    fastest = [name(k) for k in ordered if abs(stats[k][3] - fastest_time) < 1e-8]

    def rank(k):
        return 1 + sum(v[0] < stats[k][0] - 1e-8 for v in stats.values())

    overview = [
        "# 에어 나이프 연속 5판 전략 비교",
        "",
        f"판당 제품 1~30개, 개수별 독립 배치 5회, 배치당 연속 5판으로 {len(table)}개 전략을 비교했다. 전략마다 같은 150묶음·750판을 사용했다. 총 {150 * len(table):,}개 전략 실행과 {750 * len(table):,}개 판 판정에 대응한다. 같은 다섯 판에 모든 TCP 명령과 타이밍이 일치하면 동일한 물리 결과를 공유했다. 고유한 5판 물리 기록은 {unique:,}개다.",
        "",
        f"실패 판당 20초를 적용한 누적시간에서 **전자동은 {rank(auto)}위({hms(stats[auto][0])})**였다. 판 성공률은 {rate(stats[auto][1], 750)}, 5판 모두 성공한 비율은 {rate(stats[auto][2], 150)}다. 수동 조합 중 가장 짧은 기록은 **{name(best_fixed)}**의 {hms(stats[best_fixed][0])}이다.",
        "",
        f"5판 연속 성공률은 **{' · '.join(group_winners)}**에서 {rate(max_groups, 150)}로 가장 높았다. 성공·실패를 모두 포함한 경과시간 중앙값은 **{' · '.join(fastest)}**에서 {fastest_time:.2f}초로 가장 짧았다.",
        "",
        "## 조건과 점수",
        "",
        "- 촬영 세트 2의 제품을 각 판 한 면에 지정한 수만큼 배치하고, 반대면에는 이를 대칭 복제했다. 줄 수·높이·줄 안의 X를 랜덤 생성했다. 첫 판 시드는 20261124 + 100 × 제품 수 + 반복 번호(0~4)이며 다음 판마다 1,000,000씩 늘렸다.",
        "- 컨베이어 65 mm/s, 판 피치 990 mm, 나이프 250 mm, 베이스 간격 1,640 mm를 적용했다. B면 시작 XYZ는 (0, 160, 980) mm, 지정 종료점은 (0, 200, 900) mm로 고정했다. A면은 Y를 대칭으로 적용했다.",
        "- 성공 판은 실제 계획 작업시간, 실패 판은 20초를 합산했다. 판 작업시간에는 종료 위치 이동을 포함한다. 판 사이 도착 대기와 그 안에서의 미리 이동은 이 점수에서 제외한다.",
        "- 5판 경과시간은 첫 판 시작부터 다섯째 판 종료까지로, 중간 대기와 실패한 판의 실제 운전시간도 포함한다. 처리속도는 300을 5판 경과시간 중앙값(초)으로 나눈 값이다. 판 성공률의 분모는 750, 5판 성공률의 분모는 150이다.",
        "- 두 팔의 모든 IK 명령이 위치 4 mm·자세 3° 이내이고, 접촉 0회이며, 모든 제품 면에서 표적의 95% 이상이 25 ms 이상 노출되어야 판을 통과한다. 판 사이 대기 중의 IK·접촉은 직전 판에 귀속한다. 60 Hz 제어와 240 Hz 접촉 검사를 적용했다. 시뮬레이션의 기하 피복을 판정했으며 실제 잔류액 제거율은 실물 측정 대상이다.",
        "",
        f"## 전체 {len(table)}개 전략",
        "",
        "복귀는 시작 위치, 지정은 별도 종료 위치, 계속은 마지막 분사 위치를 뜻한다. ‘—’는 미리 이동을 하지 않는 설정이다. 좌우 두 전략의 차이는 같은 선두 X에 제품이 여러 개일 때만 발생한다. 누적시간이 같으면 공동 순위다.",
        "",
    ]
    rows = []
    for k in ordered:
        points, passed, groups, elapsed = stats[k]
        route, action, advance = k
        rows.append(
            [
                rank(k),
                "전자동" if k == auto else ROUTES[route][0],
                "판별 선택" if k == auto else END[action],
                "판별 선택" if k == auto else ADVANCE_OPTIONS[advance] or "—",
                hms(points),
                rate(passed, 750),
                rate(groups, 150),
                f"{elapsed:.2f}" if math.isfinite(elapsed) else "—",
                f"{300 / elapsed:.2f}" if math.isfinite(elapsed) else "—",
            ]
        )
    overview += md_table(
        [
            "순위",
            "분사 경로",
            "매판 종료",
            "다음 판 여유 시",
            "누적 시:분:초",
            "판 성공률",
            "5판 성공률",
            "5판 중앙값(초)",
            "속도(판/분)",
        ],
        rows,
    )
    overview += [
        "",
        "## 판당 제품 수별 결과",
        "",
        f"점수 1위는 같은 제품 수에서 5묶음·25판의 점수를 합산해 정했다. 공동은 0.001초 이내이며, 공간상 처음 세 전략과 남은 수를 표시했다. ‘통과 최단’은 그 개수에서 실제로 다섯 판을 모두 통과한 단일 묶음 중 최단 경과시간이다. 성공 배치는 {len(table)}개 중 한 전략 이상이 5판을 모두 통과한 배치 수다.",
        "",
    ]
    counts = []
    for n in range(1, 31):
        subset = {k: [a for a in v if a[0] == n] for k, v in table.items()}
        partial = {k: totals(v) for k, v in subset.items()}
        low = min(v[0] for v in partial.values())
        wins = [name(k) for k, v in partial.items() if v[0] <= low + 0.001]
        passed_runs = [
            (r["elapsed"], name(k), rep)
            for k, v in subset.items()
            for _, rep, r, _ in v
            if all(b["passed"] for b in r["boards"])
        ]
        winner = min(passed_runs) if passed_runs else None
        counts.append(
            [
                n,
                "전략 전체 공동 (통과한 판 0개)"
                if not any(v[1] for v in partial.values())
                else " · ".join(wins[:3])
                + (f" 외 {len(wins) - 3}개" if len(wins) > 3 else ""),
                f"{low:.2f}",
                len({r[2] for r in passed_runs}),
                f"{winner[1]} ({winner[0]:.2f}초)" if winner else "—",
                f"{partial[auto][0]:.2f}",
                rate(partial[auto][1], 25),
            ]
        )
    overview += md_table(
        [
            "제품 수",
            "20초 점수 1위",
            "1위 누적(초)",
            "성공 배치/5",
            "통과 최단 전략·시간",
            "전자동 누적(초)",
            "전자동 판 성공률",
        ],
        counts,
    )
    overview += [
        "",
        "## 전자동이 선택한 동작",
        "",
        "전체 통과 경로가 없어 최단 경로를 선택한 묶음도 실제 실행시간과 판별 통과 여부로 집계한다. 예전 기록의 미실행 묶음은 판당 20초로 채점하고 경과시간 중앙값과 처리속도에서 제외한다.",
        "",
    ]
    overview += md_table(
        ["분사 경로", "종료", "여유 시", "선택 판 수"],
        [
            [ROUTES[r][0], END_ACTIONS[a], ADVANCE_OPTIONS[d] or "—", count]
            for (r, a, d), count in decisions.most_common()
        ],
    )
    failures = Counter()
    for _, _, r, _ in table[auto]:
        if "unavailable" in r:
            failures["미실행"] += len(r["boards"])
            continue
        for b in r["boards"]:
            if not b["passed"]:
                failures["피복"] += int(b["worst_pct"] < 95)
                failures["접촉"] += int(b["contacts"] > 0)
                failures["IK"] += int(b["ik_misses"] > 0)
    overview += [
        "",
        f"전자동의 실패 점수 적용 {750 - stats[auto][1]}판 중 미실행 {failures['미실행']}판, 도달 미달 {failures['IK']}판, 접촉 {failures['접촉']}판, 최저 면 피복 미달 {failures['피복']}판이다. 원인이 겹치면 각각 센다.",
        "",
        "[전략별 150묶음의 개별 기록](트랙2_연속5판_개별기록.md)에서 판 순서별 시간과 통과 여부를 확인할 수 있다.",
        "",
        "## 재현",
        "",
        "프로젝트 루트와 데모 실행 환경에서 다음 명령을 사용한다. 기존 체크포인트를 지정하면 완료된 그룹을 재사용한다.",
        "",
        "```shell",
        "python benchmarks/five_boards.py --output-dir five-runs --workers 6",
        "python benchmarks/five_report.py --input-dir five-runs --docs docs",
        "```",
    ]
    detail = [
        "# 연속 5판 전략별 개별 기록",
        "",
        "각 행은 동일한 제품 수와 반복 번호를 가진 연속 5판이다. 판별 칸은 실제 작업시간(초)과 통과 여부이며, 점수는 실패 판에 20초를 적용한 합계다. 경과시간에는 대기가 포함된다. 전략 이름은 전체 비교표와 같다.",
        "",
    ]
    for key in ordered:
        detail += [
            f"## {name(key)}",
            "",
            "| 제품/판 | 반복(0~4) | 1판 | 2판 | 3판 | 4판 | 5판 | 점수(초) | 경과(초) |"
            + (" 전자동 1→5판 선택 |" if key == auto else ""),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            + (" --- |" if key == auto else ""),
        ]
        for n, rep, r, selected in table[key]:
            boards = r["boards"]
            cells = [
                str(n),
                str(rep),
                *[
                    f"{b['work_s']:.3f} {'통과' if b['passed'] else '미달'}"
                    if "work_s" in b
                    else "미실행"
                    for b in boards
                ],
                f"{score(boards):.3f}",
                f"{r['elapsed']:.3f}" if r["elapsed"] is not None else "—",
            ]
            if key == auto:
                cells.append(
                    r.get("unavailable") or " → ".join(name(tuple(s)) for s in selected)
                )
                if r.get("fastest_fallback"):
                    cells[-1] += " (전체 통과 경로 없음 · 최단)"
            detail.append("| " + " | ".join(cells) + " |")
        detail.append("")
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "트랙2_연속5판_전략비교.md").write_text(
        "\n".join(overview) + "\n", encoding="utf-8"
    )
    (docs / "트랙2_연속5판_개별기록.md").write_text("\n".join(detail), encoding="utf-8")
    print(
        f"Verified {len(table) * 150} groups, {len(table) * 750} boards, {unique} distinct physical records"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--docs", default=Path("docs"), type=Path)
    args = parser.parse_args()
    report(args.input_dir, args.docs)
