"""Frozen statistical audit for the preregistered Hidden Test-v3."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..reproducibility import stable_hash
from .core import path_hash
from .final_evaluation_audit import (
    _arm_summary,
    _holm_adjust,
    _map_outcome,
    _normalize,
    _quantile,
    _sign_test_pvalue,
)
from .final_evaluation_v3_common import EXPECTED_RECORDS, sha256_file, write_json
from .final_evaluation_v3_executor import FINAL_V3_ARM_IDS


PRIMARY_ARM = "evolutionary_afl_uav_v2"
COMPARATOR_ARM = "evolutionary_afl_uav_v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026081802
CONFIDENCE_LEVEL = 0.95
TIE_TOLERANCE = 1.0e-9


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSONL row at line {line_number}")
            values.append(value)
    return values


def _key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row["planner"]),
        str(row["arm_id"]),
        str(row["map_id"]),
        int(row["repetition"]),
        int(row["seed"]),
    )


def _validate_inputs(
    raw_rows: list[dict[str, str]],
    path_rows: list[dict[str, Any]],
    *,
    expected_records: int,
    schedule_path: Path | None,
) -> None:
    if len(raw_rows) != expected_records or len(path_rows) != expected_records:
        raise RuntimeError(
            f"V3 result count is {len(raw_rows)}/{len(path_rows)}, expected {expected_records}"
        )
    keys, path_keys = [_key(row) for row in raw_rows], [_key(row) for row in path_rows]
    if len(set(keys)) != expected_records or len(set(path_keys)) != expected_records:
        raise RuntimeError("V3 results contain duplicate record keys")
    if set(keys) != set(path_keys):
        raise RuntimeError("benchmark and path-log key sets differ")
    observed_arms = {str(row["arm_id"]) for row in raw_rows}
    if observed_arms != set(FINAL_V3_ARM_IDS):
        raise RuntimeError("V3 result arm set differs from preregistration")
    by_path_key = {_key(row): row for row in path_rows}
    for row in raw_rows:
        logged_path = by_path_key[_key(row)].get("path")
        recorded_hash = row.get("path_hash") or None
        if path_hash(logged_path) != recorded_hash:
            raise RuntimeError(f"path hash mismatch: {_key(row)}")
    if schedule_path is not None:
        schedule = _read_csv(schedule_path)
        expected = {_key(row) for row in schedule}
        if len(schedule) != expected_records or len(expected) != expected_records:
            raise RuntimeError("V3 seed schedule is incomplete or duplicated")
        if expected != set(keys):
            raise RuntimeError("V3 result keys differ from frozen seed schedule")


def _paired_comparison(
    by_arm_map: dict[str, dict[str, list[dict[str, Any]]]],
    primary: str,
    comparator: str,
    *,
    map_ids: list[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    outcomes = [
        _map_outcome(
            by_arm_map[primary].get(map_id, []),
            by_arm_map[comparator].get(map_id, []),
            TIE_TOLERANCE,
        )
        for map_id in map_ids
    ]
    labels = [value[0] for value in outcomes]
    scores = np.asarray([value[1] for value in outcomes], dtype=float)
    wins, ties, losses = labels.count("win"), labels.count("tie"), labels.count("loss")
    draws = rng.integers(0, len(scores), size=(BOOTSTRAP_REPLICATES, len(scores)))
    values = scores[draws].mean(axis=1)
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        "primary": primary,
        "comparator": comparator,
        "maps": len(map_ids),
        "both_feasible_maps": sum(
            any(row["trusted_feasible"] for row in by_arm_map[primary].get(item, []))
            and any(row["trusted_feasible"] for row in by_arm_map[comparator].get(item, []))
            for item in map_ids
        ),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "half_tie_win_rate": (wins + 0.5 * ties) / len(map_ids),
        "half_tie_win_rate_bootstrap_95_ci": [
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1.0 - alpha)),
        ],
        "two_sided_exact_sign_p": _sign_test_pvalue(wins, losses),
    }


def _diversity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm_id"]), str(row["map_id"]))].append(row)
    result: dict[str, Any] = {}
    for arm_id in FINAL_V3_ARM_IDS:
        groups = [group for (arm, _), group in grouped.items() if arm == arm_id]
        unique_counts: list[float] = []
        cost_iqrs: list[float] = []
        for group in groups:
            feasible = [row for row in group if row["trusted_feasible"]]
            unique_counts.append(
                float(len({row["path_hash"] for row in feasible if row["path_hash"]}))
            )
            costs = [float(row["total_cost"]) for row in feasible]
            if len(costs) > 1:
                cost_iqrs.append(float(np.quantile(costs, 0.75) - np.quantile(costs, 0.25)))
        result[arm_id] = {
            "maps": len(groups),
            "mean_unique_feasible_path_hashes_per_map": float(np.mean(unique_counts)),
            "median_unique_feasible_path_hashes_per_map": float(np.median(unique_counts)),
            "median_within_map_cost_iqr": _quantile(cost_iqrs, 0.5),
        }
    return result


def _select_paper_method(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    v1, v2 = summaries[COMPARATOR_ARM], summaries[PRIMARY_ARM]
    if v2["trusted_feasible_rate"] > v1["trusted_feasible_rate"]:
        selected, reason = PRIMARY_ARM, "higher trusted feasible rate"
    elif v2["trusted_feasible_rate"] < v1["trusted_feasible_rate"]:
        selected, reason = COMPARATOR_ARM, "higher trusted feasible rate"
    else:
        v1_cost, v2_cost = v1["median_trusted_cost"], v2["median_trusted_cost"]
        if v2_cost is not None and (v1_cost is None or v2_cost < v1_cost):
            selected, reason = PRIMARY_ARM, "equal feasibility and lower median trusted cost"
        else:
            selected, reason = COMPARATOR_ARM, "equal feasibility and lower or tied median trusted cost"
    return {
        "selected_method": selected,
        "rule": "trusted feasible rate first; median trusted feasible cost second",
        "reason": reason,
        "v1_feasible_rate": v1["trusted_feasible_rate"],
        "v2_feasible_rate": v2["trusted_feasible_rate"],
        "v1_median_trusted_cost": v1["median_trusted_cost"],
        "v2_median_trusted_cost": v2["median_trusted_cost"],
    }


def _write_flat_summary(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    rows = []
    for arm_id in FINAL_V3_ARM_IDS:
        row = {"arm_id": arm_id}
        row.update(
            {key: value for key, value in summaries[arm_id].items() if not isinstance(value, dict)}
        )
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_normalized(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: dict[str, Any]) -> str:
    decision = report["paper_method_decision"]
    lines = [
        "# UAV2D Hidden Test-v3 final audit",
        "",
        f"Status: **{report['status']}**  ",
        f"Records: **{report['records']}**  ",
        f"Paper method selected by preregistered rule: **{decision['selected_method']}**  ",
        f"Reason: {decision['reason']}.",
        "",
        "| Rank | Arm | Trusted feasible rate | Median trusted cost |",
        "|---:|---|---:|---:|",
    ]
    for item in report["ranking"]:
        cost = item["median_trusted_cost"]
        lines.append(
            f"| {item['rank']} | {item['arm_id']} | {item['trusted_feasible_rate']:.4f} | "
            + ("—" if cost is None else f"{cost:.4f}")
            + " |"
        )
    lines.extend(["", "## Frozen comparisons", ""])
    for name, item in report["hypothesis_tests"].items():
        lines.append(
            f"- {name}: {item['wins']}W/{item['ties']}T/{item['losses']}L, "
            f"half-tie rate {item['half_tie_win_rate']:.4f}."
        )
    return "\n".join(lines) + "\n"


def audit_results(
    run_dir: str | Path,
    *,
    time_limit_seconds: float,
    schedule_path: str | Path | None = None,
    preflight: bool = False,
    expected_records: int = EXPECTED_RECORDS,
) -> dict[str, Any]:
    if not preflight and expected_records != EXPECTED_RECORDS:
        raise ValueError("final V3 audit requires exactly 6,360 records")
    if not preflight and schedule_path is None:
        raise ValueError("final V3 audit requires the frozen seed schedule")
    run_dir = Path(run_dir)
    runs_path, paths_path = run_dir / "benchmark_runs.csv", run_dir / "benchmark_paths.jsonl"
    raw_rows, path_rows = _read_csv(runs_path), _read_jsonl(paths_path)
    _validate_inputs(
        raw_rows,
        path_rows,
        expected_records=expected_records,
        schedule_path=None if schedule_path is None else Path(schedule_path),
    )
    rows = [_normalize(row, time_limit_seconds=time_limit_seconds) for row in raw_rows]
    root_rng = np.random.default_rng(BOOTSTRAP_SEED)
    by_arm = {
        arm_id: [row for row in rows if row["arm_id"] == arm_id]
        for arm_id in FINAL_V3_ARM_IDS
    }
    summaries = {
        arm_id: _arm_summary(
            by_arm[arm_id],
            replicates=BOOTSTRAP_REPLICATES,
            rng=np.random.default_rng(int(root_rng.integers(0, 2**63 - 1))),
            confidence_level=CONFIDENCE_LEVEL,
        )
        for arm_id in FINAL_V3_ARM_IDS
    }
    rooms = {
        arm_id: _arm_summary(
            [row for row in by_arm[arm_id] if row["difficulty"] == "rooms_maze"],
            replicates=BOOTSTRAP_REPLICATES,
            rng=np.random.default_rng(int(root_rng.integers(0, 2**63 - 1))),
            confidence_level=CONFIDENCE_LEVEL,
        )
        for arm_id in FINAL_V3_ARM_IDS
    }
    order = sorted(
        FINAL_V3_ARM_IDS,
        key=lambda arm: (
            -float(summaries[arm]["trusted_feasible_rate"]),
            float("inf")
            if summaries[arm]["median_trusted_cost"] is None
            else float(summaries[arm]["median_trusted_cost"]),
            arm,
        ),
    )
    ranking = [
        {
            "rank": index + 1,
            "arm_id": arm,
            "trusted_feasible_rate": summaries[arm]["trusted_feasible_rate"],
            "median_trusted_cost": summaries[arm]["median_trusted_cost"],
        }
        for index, arm in enumerate(order)
    ]
    by_arm_map: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    difficulty: dict[str, str] = {}
    for row in rows:
        by_arm_map[str(row["arm_id"])][str(row["map_id"])].append(row)
        difficulty[str(row["map_id"])] = str(row["difficulty"])
    all_maps = sorted(difficulty)
    room_maps = [item for item in all_maps if difficulty[item] == "rooms_maze"]
    definitions = {
        "H1_v2_vs_v1": (PRIMARY_ARM, COMPARATOR_ARM, all_maps),
        "H2_v2_vs_frozen_afl": (PRIMARY_ARM, "frozen_afl_uav", all_maps),
        "H3_v2_vs_astar": (PRIMARY_ARM, "astar", all_maps),
        "H4_v2_vs_theta_star": (PRIMARY_ARM, "theta_star", all_maps),
        "E1_rooms_v2_vs_v1": (PRIMARY_ARM, COMPARATOR_ARM, room_maps),
    }
    tests = {
        name: _paired_comparison(
            by_arm_map,
            primary,
            comparator,
            map_ids=map_ids,
            rng=np.random.default_rng(int(root_rng.integers(0, 2**63 - 1))),
        )
        for name, (primary, comparator, map_ids) in definitions.items()
    }
    adjusted = _holm_adjust(
        [
            (name, float(tests[name]["two_sided_exact_sign_p"]))
            for name in ("H2_v2_vs_frozen_afl", "H3_v2_vs_astar", "H4_v2_vs_theta_star")
        ]
    )
    for name, value in adjusted.items():
        tests[name]["holm_adjusted_p_secondary_family"] = value
    report: dict[str, Any] = {
        "schema_version": "uav2d-final-statistical-audit-v3",
        "status": "passed",
        "mode": "preflight" if preflight else "final",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "records": len(rows),
        "unique_records": len({_key(row) for row in rows}),
        "arms": len(by_arm),
        "timeouts_counted_as_failures": sum(row["timed_out"] for row in rows),
        "ranking_rule": "trusted feasible rate descending, then median trusted feasible cost ascending",
        "ranking": ranking,
        "paper_method_decision": _select_paper_method(summaries),
        "arm_statistics": summaries,
        "rooms_maze_statistics": rooms,
        "path_diversity": _diversity(rows),
        "hypothesis_tests": tests,
        "statistical_contract": {
            "primary": "H1_v2_vs_v1",
            "bootstrap_unit": "map_id",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "paired_aggregation": "median feasible cost over five shared seeds; feasibility first",
            "tie_tolerance": TIE_TOLERANCE,
            "exact_test": "two-sided paired sign test excluding ties",
            "holm_family": ["H2", "H3", "H4"],
            "rooms_maze_comparison": "exploratory only",
            "post_result_changes_allowed": False,
        },
    }
    report["audit_content_id"] = stable_hash(report)
    write_json(run_dir / "audit_report.json", report)
    _write_flat_summary(run_dir / "audit_summary.csv", summaries)
    _write_normalized(run_dir / "normalized_benchmark_runs.csv", rows)
    (run_dir / "audit_report.md").write_text(_markdown(report), encoding="utf-8")
    receipt: dict[str, Any] = {
        "schema_version": "uav2d-final-audit-receipt-v3",
        "status": "passed",
        "mode": report["mode"],
        "records": len(rows),
        "unique_records": len({_key(row) for row in rows}),
        "audit_content_id": report["audit_content_id"],
        "benchmark_runs_sha256": sha256_file(runs_path),
        "benchmark_paths_sha256": sha256_file(paths_path),
        "audit_report_sha256": sha256_file(run_dir / "audit_report.json"),
        "audit_summary_sha256": sha256_file(run_dir / "audit_summary.csv"),
        "normalized_runs_sha256": sha256_file(run_dir / "normalized_benchmark_runs.csv"),
    }
    receipt["audit_receipt_id"] = stable_hash(receipt)
    write_json(run_dir / "audit_receipt.json", receipt)
    return report


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "EXPECTED_RECORDS",
    "audit_results",
]
