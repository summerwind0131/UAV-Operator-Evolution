"""Frozen statistical audit for the preregistered UAV2D final evaluation."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..reproducibility import stable_hash
from .core import path_hash
from .final_evaluation_common import sha256_file, write_json
from .final_evaluation_executor import FINAL_ARM_IDS


PRIMARY_ARM = "evolutionary_afl_uav_v1"
FINAL_EXPECTED_RECORDS = 6960
FINAL_BOOTSTRAP_REPLICATES = 10_000
FINAL_BOOTSTRAP_SEED = 2026081702
FINAL_CONFIDENCE_LEVEL = 0.95
FINAL_TIE_TOLERANCE = 1.0e-9


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
                raise RuntimeError(f"non-object JSONL record at line {line_number}")
            values.append(value)
    return values


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", ""}:
        return False
    raise ValueError(f"invalid boolean field: {value!r}")


def _float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite numeric field: {value!r}")
    return parsed


def _key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row["planner"]),
        str(row["arm_id"]),
        str(row["map_id"]),
        int(row["repetition"]),
        int(row["seed"]),
    )


def _normalize(
    raw: dict[str, Any],
    *,
    time_limit_seconds: float,
) -> dict[str, Any]:
    row = dict(raw)
    elapsed = float(row["elapsed_seconds"])
    original_status = str(row["status"])
    timed_out = original_status == "timeout" or elapsed >= time_limit_seconds
    original_feasible = _bool(row["feasible"])
    trusted_feasible = bool(original_feasible and not timed_out)
    total_cost = _float_or_none(row.get("total_cost"))
    if trusted_feasible and total_cost is None:
        raise RuntimeError(f"feasible row has no cost: {_key(row)}")
    row.update(
        {
            "repetition": int(row["repetition"]),
            "seed": int(row["seed"]),
            "elapsed_seconds": elapsed,
            "objective_evaluations": int(row["objective_evaluations"]),
            "collision_checks": int(row["collision_checks"]),
            "node_expansions": int(row["node_expansions"]),
            "waypoint_count": int(row["waypoint_count"]),
            "original_status": original_status,
            "effective_status": "timeout" if timed_out else original_status,
            "timed_out": timed_out,
            "original_feasible": original_feasible,
            "trusted_feasible": trusted_feasible,
            "total_cost": total_cost if trusted_feasible else None,
        }
    )
    return row


def _validate_inputs(
    raw_rows: list[dict[str, str]],
    path_rows: list[dict[str, Any]],
    *,
    expected_records: int,
    schedule_path: Path | None,
) -> None:
    if len(raw_rows) != expected_records:
        raise RuntimeError(
            f"benchmark has {len(raw_rows)} rows; expected exactly {expected_records}"
        )
    if len(path_rows) != expected_records:
        raise RuntimeError(
            f"path log has {len(path_rows)} rows; expected exactly {expected_records}"
        )
    keys = [_key(row) for row in raw_rows]
    path_keys = [_key(row) for row in path_rows]
    if len(set(keys)) != expected_records:
        raise RuntimeError("benchmark rows contain duplicate record keys")
    if len(set(path_keys)) != expected_records:
        raise RuntimeError("path log contains duplicate record keys")
    if set(keys) != set(path_keys):
        raise RuntimeError("benchmark and path-log key sets differ")
    observed_arms = {row["arm_id"] for row in raw_rows}
    if observed_arms != set(FINAL_ARM_IDS):
        missing = sorted(set(FINAL_ARM_IDS) - observed_arms)
        extra = sorted(observed_arms - set(FINAL_ARM_IDS))
        raise RuntimeError(f"14-arm set mismatch: missing={missing}, extra={extra}")
    by_path_key = {_key(row): row for row in path_rows}
    for row in raw_rows:
        logged_path = by_path_key[_key(row)].get("path")
        recorded_hash = row.get("path_hash") or None
        if path_hash(logged_path) != recorded_hash:
            raise RuntimeError(f"path hash mismatch: {_key(row)}")
    if schedule_path is not None:
        schedule = _read_csv(schedule_path)
        if len(schedule) != expected_records or len({_key(row) for row in schedule}) != expected_records:
            raise RuntimeError("seed schedule is incomplete or contains duplicates")
        expected = {_key(row) for row in schedule}
        observed = set(keys)
        missing = expected - observed
        extra = observed - expected
        if missing or extra:
            raise RuntimeError(
                f"result/schedule mismatch: {len(missing)} missing, {len(extra)} extra"
            )


def _quantile(values: Iterable[float], probability: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return float(np.quantile(array, probability)) if array.size else None


def _iqr(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return float(np.quantile(array, 0.75) - np.quantile(array, 0.25)) if array.size else None


def _bootstrap_arm(
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    rng: np.random.Generator,
    confidence_level: float,
) -> dict[str, list[float] | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["map_id"])].append(row)
    maps = sorted(grouped)
    if not maps:
        return {"feasible_rate": None, "median_cost": None, "map_success_rate": None}
    totals = np.asarray([len(grouped[item]) for item in maps], dtype=float)
    feasible_counts = np.asarray(
        [sum(row["trusted_feasible"] for row in grouped[item]) for item in maps],
        dtype=float,
    )
    success = feasible_counts > 0
    costs = [
        np.asarray(
            [
                float(row["total_cost"])
                for row in grouped[item]
                if row["trusted_feasible"]
            ],
            dtype=float,
        )
        for item in maps
    ]
    draws = rng.integers(0, len(maps), size=(replicates, len(maps)))
    rate_values = feasible_counts[draws].sum(axis=1) / totals[draws].sum(axis=1)
    success_values = success[draws].mean(axis=1)
    median_values: list[float] = []
    for sample in draws:
        selected = [costs[index] for index in sample if costs[index].size]
        if selected:
            median_values.append(float(np.median(np.concatenate(selected))))
    alpha = (1.0 - confidence_level) / 2.0

    def interval(values: np.ndarray | list[float]) -> list[float] | None:
        array = np.asarray(values, dtype=float)
        if not array.size:
            return None
        return [
            float(np.quantile(array, alpha)),
            float(np.quantile(array, 1.0 - alpha)),
        ]

    return {
        "feasible_rate": interval(rate_values),
        "median_cost": interval(median_values),
        "map_success_rate": interval(success_values),
    }


def _arm_summary(
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    rng: np.random.Generator,
    confidence_level: float,
) -> dict[str, Any]:
    feasible = [row for row in rows if row["trusted_feasible"]]
    costs = [float(row["total_cost"]) for row in feasible]
    successful_maps = {str(row["map_id"]) for row in feasible}
    all_maps = {str(row["map_id"]) for row in rows}
    return {
        "runs": len(rows),
        "trusted_feasible_runs": len(feasible),
        "trusted_feasible_rate": len(feasible) / len(rows),
        "successful_maps": len(successful_maps),
        "maps": len(all_maps),
        "map_success_rate": len(successful_maps) / len(all_maps),
        "timeouts_as_failures": sum(row["timed_out"] for row in rows),
        "median_trusted_cost": _quantile(costs, 0.5),
        "iqr_trusted_cost": _iqr(costs),
        "median_elapsed_seconds": _quantile(
            (float(row["elapsed_seconds"]) for row in rows), 0.5
        ),
        "median_objective_evaluations": _quantile(
            (float(row["objective_evaluations"]) for row in rows), 0.5
        ),
        "bootstrap_95_ci": _bootstrap_arm(
            rows,
            replicates=replicates,
            rng=rng,
            confidence_level=confidence_level,
        ),
    }


def _map_outcome(
    primary: list[dict[str, Any]],
    comparator: list[dict[str, Any]],
    tolerance: float,
) -> tuple[str, float]:
    primary_costs = [float(row["total_cost"]) for row in primary if row["trusted_feasible"]]
    comparator_costs = [float(row["total_cost"]) for row in comparator if row["trusted_feasible"]]
    if primary_costs and not comparator_costs:
        return "win", 1.0
    if comparator_costs and not primary_costs:
        return "loss", 0.0
    if not primary_costs and not comparator_costs:
        return "tie", 0.5
    first = float(np.median(primary_costs))
    second = float(np.median(comparator_costs))
    threshold = tolerance * max(1.0, abs(first), abs(second))
    if first < second - threshold:
        return "win", 1.0
    if first > second + threshold:
        return "loss", 0.0
    return "tie", 0.5


def _sign_test_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(trials, value) for value in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def _paired_comparison(
    by_arm_map: dict[str, dict[str, list[dict[str, Any]]]],
    comparator: str,
    *,
    map_ids: list[str],
    tolerance: float,
    replicates: int,
    rng: np.random.Generator,
    confidence_level: float,
) -> dict[str, Any]:
    outcomes = [
        _map_outcome(
            by_arm_map[PRIMARY_ARM].get(map_id, []),
            by_arm_map[comparator].get(map_id, []),
            tolerance,
        )
        for map_id in map_ids
    ]
    labels = [item[0] for item in outcomes]
    scores = np.asarray([item[1] for item in outcomes], dtype=float)
    wins, ties, losses = labels.count("win"), labels.count("tie"), labels.count("loss")
    draws = rng.integers(0, len(scores), size=(replicates, len(scores)))
    values = scores[draws].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    both_feasible = sum(
        bool(
            any(row["trusted_feasible"] for row in by_arm_map[PRIMARY_ARM].get(map_id, []))
            and any(row["trusted_feasible"] for row in by_arm_map[comparator].get(map_id, []))
        )
        for map_id in map_ids
    )
    return {
        "primary": PRIMARY_ARM,
        "comparator": comparator,
        "maps": len(map_ids),
        "both_feasible_maps": both_feasible,
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


def _holm_adjust(items: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def _diversity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm_id"]), str(row["map_id"]))].append(row)
    by_arm: dict[str, dict[str, Any]] = {}
    for arm_id in FINAL_ARM_IDS:
        map_groups = [group for (arm, _), group in grouped.items() if arm == arm_id]
        unique_counts: list[float] = []
        cost_iqrs: list[float] = []
        for group in map_groups:
            feasible = [row for row in group if row["trusted_feasible"]]
            unique_counts.append(
                float(len({row["path_hash"] for row in feasible if row["path_hash"]}))
            )
            costs = [float(row["total_cost"]) for row in feasible]
            if len(costs) >= 2:
                cost_iqrs.append(float(np.quantile(costs, 0.75) - np.quantile(costs, 0.25)))
        by_arm[arm_id] = {
            "maps": len(map_groups),
            "mean_unique_feasible_path_hashes_per_map": float(np.mean(unique_counts)),
            "median_unique_feasible_path_hashes_per_map": float(np.median(unique_counts)),
            "median_within_map_cost_iqr": _quantile(cost_iqrs, 0.5),
        }
    return by_arm


def _write_flat_summary(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for arm_id in FINAL_ARM_IDS:
        row = {"arm_id": arm_id}
        row.update({key: value for key, value in summaries[arm_id].items() if not isinstance(value, dict)})
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_normalized(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# UAV2D final-evaluation audit",
        "",
        f"Status: **{report['status']}**  ",
        f"Records: **{report['records']}**  ",
        "Timeouts are treated as failures. Ranking is feasibility first, then trusted cost.",
        "",
        "| Rank | Arm | Feasible rate | Median feasible cost |",
        "|---:|---|---:|---:|",
    ]
    for item in report["ranking"]:
        cost = item["median_trusted_cost"]
        lines.append(
            f"| {item['rank']} | {item['arm_id']} | {item['trusted_feasible_rate']:.4f} | "
            + ("—" if cost is None else f"{cost:.4f}")
            + " |"
        )
    lines.extend(["", "## Preregistered paired comparisons", ""])
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
    expected_records: int = FINAL_EXPECTED_RECORDS,
) -> dict[str, Any]:
    """Audit a matrix. Final mode is hard-coded to the preregistered statistics."""

    if not preflight and expected_records != FINAL_EXPECTED_RECORDS:
        raise ValueError("final audit always requires exactly 6,960 records")
    if not preflight and schedule_path is None:
        raise ValueError("final audit requires the preregistered seed schedule")
    run_dir = Path(run_dir)
    runs_path = run_dir / "benchmark_runs.csv"
    paths_path = run_dir / "benchmark_paths.jsonl"
    raw_rows = _read_csv(runs_path)
    path_rows = _read_jsonl(paths_path)
    _validate_inputs(
        raw_rows,
        path_rows,
        expected_records=expected_records,
        schedule_path=None if schedule_path is None else Path(schedule_path),
    )
    rows = [
        _normalize(row, time_limit_seconds=time_limit_seconds) for row in raw_rows
    ]
    replicates = FINAL_BOOTSTRAP_REPLICATES
    confidence = FINAL_CONFIDENCE_LEVEL
    root_rng = np.random.default_rng(FINAL_BOOTSTRAP_SEED)
    by_arm = {
        arm_id: [row for row in rows if row["arm_id"] == arm_id]
        for arm_id in FINAL_ARM_IDS
    }
    summaries = {
        arm_id: _arm_summary(
            by_arm[arm_id],
            replicates=replicates,
            rng=np.random.default_rng(int(root_rng.integers(0, 2**63 - 1))),
            confidence_level=confidence,
        )
        for arm_id in FINAL_ARM_IDS
    }
    rooms_summaries = {
        arm_id: _arm_summary(
            [row for row in by_arm[arm_id] if row["difficulty"] == "rooms_maze"],
            replicates=replicates,
            rng=np.random.default_rng(int(root_rng.integers(0, 2**63 - 1))),
            confidence_level=confidence,
        )
        for arm_id in FINAL_ARM_IDS
    }
    ranking_order = sorted(
        FINAL_ARM_IDS,
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
        for index, arm in enumerate(ranking_order)
    ]
    by_arm_map: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    difficulty_by_map: dict[str, str] = {}
    for row in rows:
        by_arm_map[str(row["arm_id"])][str(row["map_id"])].append(row)
        difficulty_by_map[str(row["map_id"])] = str(row["difficulty"])
    all_maps = sorted(difficulty_by_map)
    room_maps = [item for item in all_maps if difficulty_by_map[item] == "rooms_maze"]
    comparison_definitions = {
        "H1_evo_vs_frozen_afl": ("frozen_afl_uav", all_maps),
        "H2a_evo_vs_astar": ("astar", all_maps),
        "H2b_evo_vs_theta_star": ("theta_star", all_maps),
        "H3_rooms_evo_vs_no_rooms_strategy": ("evo_no_rooms_strategy", room_maps),
        "H4_rooms_evo_vs_fixed_length": ("evo_fixed_length", room_maps),
    }
    tests = {
        name: _paired_comparison(
            by_arm_map,
            comparator,
            map_ids=map_ids,
            tolerance=FINAL_TIE_TOLERANCE,
            replicates=replicates,
            rng=np.random.default_rng(int(root_rng.integers(0, 2**63 - 1))),
            confidence_level=confidence,
        )
        for name, (comparator, map_ids) in comparison_definitions.items()
    }
    secondary_adjusted = _holm_adjust(
        [
            (name, float(tests[name]["two_sided_exact_sign_p"]))
            for name in (
                "H3_rooms_evo_vs_no_rooms_strategy",
                "H4_rooms_evo_vs_fixed_length",
            )
        ]
    )
    for name, adjusted in secondary_adjusted.items():
        tests[name]["holm_adjusted_p_secondary_family"] = adjusted
    report: dict[str, Any] = {
        "schema_version": "uav2d-final-statistical-audit-v1",
        "status": "passed",
        "mode": "preflight" if preflight else "final",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "records": len(rows),
        "unique_records": len({_key(row) for row in rows}),
        "arms": len(by_arm),
        "timeouts_counted_as_failures": sum(row["timed_out"] for row in rows),
        "ranking_rule": "trusted feasible rate descending, then trusted feasible cost median ascending",
        "ranking": ranking,
        "arm_statistics": summaries,
        "rooms_maze_statistics": rooms_summaries,
        "path_diversity": _diversity(rows),
        "hypothesis_tests": tests,
        "statistical_contract": {
            "bootstrap_unit": "map_id",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": FINAL_BOOTSTRAP_SEED,
            "confidence_level": confidence,
            "paired_aggregation": "median feasible cost over shared seeds, feasibility lexicographically first",
            "tie_tolerance": FINAL_TIE_TOLERANCE,
            "exact_test": "two-sided paired sign test excluding ties",
            "holm_family": ["H3", "H4"],
            "post_result_changes_allowed": False,
        },
    }
    report["audit_content_id"] = stable_hash(report)
    write_json(run_dir / "audit_report.json", report)
    _write_flat_summary(run_dir / "audit_summary.csv", summaries)
    _write_normalized(run_dir / "normalized_benchmark_runs.csv", rows)
    (run_dir / "audit_report.md").write_text(_markdown(report), encoding="utf-8")
    receipt: dict[str, Any] = {
        "schema_version": "uav2d-final-audit-receipt-v1",
        "status": "passed",
        "mode": report["mode"],
        "records": len(rows),
        "unique_records": len({_key(row) for row in rows}),
        "audit_content_id": report["audit_content_id"],
        "benchmark_runs_sha256": sha256_file(runs_path),
        "benchmark_paths_sha256": sha256_file(paths_path),
        "audit_report_sha256": sha256_file(run_dir / "audit_report.json"),
        "audit_summary_sha256": sha256_file(run_dir / "audit_summary.csv"),
        "normalized_runs_sha256": sha256_file(
            run_dir / "normalized_benchmark_runs.csv"
        ),
    }
    receipt["audit_receipt_id"] = stable_hash(receipt)
    write_json(run_dir / "audit_receipt.json", receipt)
    return report


__all__ = [
    "FINAL_BOOTSTRAP_REPLICATES",
    "FINAL_BOOTSTRAP_SEED",
    "FINAL_EXPECTED_RECORDS",
    "audit_results",
]
