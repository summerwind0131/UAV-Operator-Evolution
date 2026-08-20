"""Audit and compare Evolutionary AFL-UAV on uav2d-v1 Validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from uav_operator_evolution.planning_benchmarks import path_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVOLUTIONARY = (
    ROOT / "artifacts" / "planning_benchmarks" / "evolutionary-afl-uav-validation-v1"
)
DEFAULT_FROZEN = (
    ROOT / "artifacts" / "planning_benchmarks" / "deepseek-v4pro-strict-validation-v2"
)
DEFAULT_TRADITIONAL = (
    ROOT / "artifacts" / "planning_benchmarks" / "offline-traditional-validation-v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "planning_benchmarks"
    / "evolutionary-afl-uav-validation-analysis-v1"
)
TOLERANCE = 1e-9


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _quantile(values: Iterable[float], q: float) -> float | None:
    materialized = list(values)
    return float(np.quantile(materialized, q)) if materialized else None


def _median(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return float(statistics.median(materialized)) if materialized else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cost(row: dict[str, Any]) -> float | None:
    return _float(row.get("total_cost")) if _boolean(row.get("feasible")) else None


def _group_by_map(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["map_id"])].append(row)
    return dict(grouped)


def _bootstrap_map_metrics(
    map_rows: list[dict[str, Any]],
    *,
    seed: int = 20260816,
    replicates: int = 2_000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas = np.asarray([float(row["median_delta"]) for row in map_rows])
    outcomes = np.asarray(
        [1.0 if value < -TOLERANCE else 0.5 if abs(value) <= TOLERANCE else 0.0 for value in deltas]
    )
    median_samples: list[float] = []
    win_samples: list[float] = []
    for _ in range(replicates):
        indices = rng.integers(0, len(map_rows), size=len(map_rows))
        median_samples.append(float(np.median(deltas[indices])))
        win_samples.append(float(np.mean(outcomes[indices])))
    return {
        "cluster_bootstrap_replicates": replicates,
        "median_delta_ci_low": float(np.quantile(median_samples, 0.025)),
        "median_delta_ci_high": float(np.quantile(median_samples, 0.975)),
        "half_tie_win_rate_ci_low": float(np.quantile(win_samples, 0.025)),
        "half_tie_win_rate_ci_high": float(np.quantile(win_samples, 0.975)),
    }


def _pair_evolutionary_and_frozen(
    evolutionary: list[dict[str, Any]],
    frozen: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frozen_index = {(row["map_id"], row["seed"]): row for row in frozen}
    seed_pairs: list[dict[str, Any]] = []
    for row in evolutionary:
        baseline = frozen_index[(row["map_id"], row["seed"])]
        candidate_cost = _cost(row)
        baseline_cost = _cost(baseline)
        delta = (
            None
            if candidate_cost is None or baseline_cost is None
            else candidate_cost - baseline_cost
        )
        seed_pairs.append(
            {
                "map_id": row["map_id"],
                "difficulty": row["difficulty"],
                "seed": int(row["seed"]),
                "evolutionary_feasible": _boolean(row["feasible"]),
                "frozen_feasible": _boolean(baseline["feasible"]),
                "evolutionary_cost": candidate_cost,
                "frozen_cost": baseline_cost,
                "cost_delta": delta,
                "evolutionary_path_hash": row["path_hash"],
                "frozen_path_hash": baseline["path_hash"],
            }
        )

    by_map_evolutionary = _group_by_map(evolutionary)
    by_map_frozen = _group_by_map(frozen)
    map_pairs: list[dict[str, Any]] = []
    for map_id in sorted(by_map_evolutionary):
        candidate_rows = by_map_evolutionary[map_id]
        baseline_rows = by_map_frozen[map_id]
        candidate_costs = [value for row in candidate_rows if (value := _cost(row)) is not None]
        baseline_costs = [value for row in baseline_rows if (value := _cost(row)) is not None]
        candidate_median = _median(candidate_costs)
        baseline_median = _median(baseline_costs)
        delta = (
            None
            if candidate_median is None or baseline_median is None
            else candidate_median - baseline_median
        )
        map_pairs.append(
            {
                "map_id": map_id,
                "difficulty": candidate_rows[0]["difficulty"],
                "evolutionary_feasible_seeds": len(candidate_costs),
                "frozen_feasible_seeds": len(baseline_costs),
                "evolutionary_median_cost": candidate_median,
                "frozen_median_cost": baseline_median,
                "median_delta": delta,
                "evolutionary_unique_paths": len({row["path_hash"] for row in candidate_rows}),
                "frozen_unique_paths": len({row["path_hash"] for row in baseline_rows}),
                "evolutionary_cost_iqr": (
                    float(np.quantile(candidate_costs, 0.75) - np.quantile(candidate_costs, 0.25))
                    if candidate_costs
                    else None
                ),
            }
        )

    comparable = [row for row in map_pairs if row["median_delta"] is not None]
    wins = sum(float(row["median_delta"]) < -TOLERANCE for row in comparable)
    ties = sum(abs(float(row["median_delta"])) <= TOLERANCE for row in comparable)
    losses = len(comparable) - wins - ties
    summary = {
        "paired_seed_records": len(seed_pairs),
        "paired_maps": len(map_pairs),
        "both_feasible_maps": len(comparable),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "strict_win_rate": wins / len(comparable),
        "half_tie_win_rate": (wins + 0.5 * ties) / len(comparable),
        "median_map_cost_delta": _median(float(row["median_delta"]) for row in comparable),
        "median_relative_map_improvement": _median(
            -float(row["median_delta"]) / float(row["frozen_median_cost"])
            for row in comparable
            if float(row["frozen_median_cost"]) > 0
        ),
        **_bootstrap_map_metrics(comparable),
    }
    return seed_pairs, map_pairs, summary


def _difficulty_summary(map_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in map_pairs:
        grouped[str(row["difficulty"])].append(row)
    result: list[dict[str, Any]] = []
    for difficulty, rows in sorted(grouped.items()):
        comparable = [row for row in rows if row["median_delta"] is not None]
        deltas = [float(row["median_delta"]) for row in comparable]
        result.append(
            {
                "difficulty": difficulty,
                "maps": len(rows),
                "wins": sum(value < -TOLERANCE for value in deltas),
                "ties": sum(abs(value) <= TOLERANCE for value in deltas),
                "losses": sum(value > TOLERANCE for value in deltas),
                "evolutionary_median_cost": _median(
                    float(row["evolutionary_median_cost"]) for row in comparable
                ),
                "frozen_median_cost": _median(
                    float(row["frozen_median_cost"]) for row in comparable
                ),
                "median_delta": _median(deltas),
                "median_relative_improvement": _median(
                    -float(row["median_delta"]) / float(row["frozen_median_cost"])
                    for row in comparable
                ),
                "maps_with_multiple_final_paths": sum(
                    int(row["evolutionary_unique_paths"]) > 1 for row in rows
                ),
                "median_unique_final_paths": _median(
                    int(row["evolutionary_unique_paths"]) for row in rows
                ),
            }
        )
    return result


def _component_comparison(
    evolutionary: list[dict[str, Any]],
    frozen: list[dict[str, Any]],
) -> dict[str, Any]:
    frozen_index = {(row["map_id"], row["seed"]): row for row in frozen}
    components = (
        "path_length",
        "risk_penalty",
        "smoothness_penalty",
        "waypoint_penalty",
    )
    result: dict[str, Any] = {}
    for component in components:
        deltas: list[float] = []
        for row in evolutionary:
            baseline = frozen_index[(row["map_id"], row["seed"])]
            candidate_value = _float(row.get(component))
            baseline_value = _float(baseline.get(component))
            if candidate_value is not None and baseline_value is not None:
                deltas.append(candidate_value - baseline_value)
        result[component] = {
            "paired_runs": len(deltas),
            "improved": sum(value < -TOLERANCE for value in deltas),
            "tied": sum(abs(value) <= TOLERANCE for value in deltas),
            "worsened": sum(value > TOLERANCE for value in deltas),
            "median_delta": _median(deltas),
            "q1_delta": _quantile(deltas, 0.25),
            "q3_delta": _quantile(deltas, 0.75),
        }
    return result


def _deterministic_baseline_comparison(
    map_pairs: list[dict[str, Any]],
    traditional: list[dict[str, Any]],
    planner_name: str,
) -> dict[str, Any]:
    baseline = {
        row["map_id"]: row for row in traditional if row["planner"] == planner_name
    }
    comparisons: list[tuple[float, float]] = []
    candidate_only = 0
    baseline_only = 0
    for row in map_pairs:
        candidate_cost = _float(row["evolutionary_median_cost"])
        baseline_cost = _cost(baseline[row["map_id"]])
        if candidate_cost is not None and baseline_cost is not None:
            comparisons.append((candidate_cost, baseline_cost))
        elif candidate_cost is not None:
            candidate_only += 1
        elif baseline_cost is not None:
            baseline_only += 1
    deltas = [candidate - baseline_cost for candidate, baseline_cost in comparisons]
    return {
        "baseline": planner_name,
        "maps": len(map_pairs),
        "both_feasible_maps": len(comparisons),
        "candidate_only_feasible_maps": candidate_only,
        "baseline_only_feasible_maps": baseline_only,
        "wins": sum(value < -TOLERANCE for value in deltas),
        "ties": sum(abs(value) <= TOLERANCE for value in deltas),
        "losses": sum(value > TOLERANCE for value in deltas),
        "strict_win_rate": sum(value < -TOLERANCE for value in deltas) / len(deltas),
        "half_tie_win_rate": (
            sum(value < -TOLERANCE for value in deltas)
            + 0.5 * sum(abs(value) <= TOLERANCE for value in deltas)
        )
        / len(deltas),
        "median_cost_delta": _median(deltas),
    }


def _diagnostic_audit(
    evolutionary_rows: list[dict[str, Any]],
    path_records: list[dict[str, Any]],
) -> dict[str, Any]:
    run_index = {(row["map_id"], int(row["seed"])): row for row in evolutionary_rows}
    errors: list[str] = []
    operator_attempts: Counter[str] = Counter()
    operator_successes: Counter[str] = Counter()
    improvements: list[float] = []
    unique_candidates: list[int] = []
    archive_unique: list[int] = []
    hash_mismatches = 0
    final_cost_mismatches = 0
    nonzero_llm_calls = 0
    path_record_keys: list[tuple[str, int]] = []
    for record in path_records:
        key = (record["map_id"], int(record["seed"]))
        path_record_keys.append(key)
        row = run_index.get(key)
        if row is None:
            errors.append(f"path record has no matching run: {key}")
            continue
        if path_hash(record.get("path")) != row["path_hash"]:
            hash_mismatches += 1
        diagnostics = record.get("diagnostics", {})
        operator_attempts.update(diagnostics.get("operator_attempts", {}))
        operator_successes.update(diagnostics.get("operator_successes", {}))
        improvements.append(float(diagnostics.get("relative_improvement", 0.0)))
        unique_candidates.append(int(diagnostics.get("unique_candidates_evaluated", 0)))
        archive_unique.append(int(diagnostics.get("archive_unique_paths", 0)))
        nonzero_llm_calls += int(diagnostics.get("llm_calls_during_planning", 0)) != 0
        reported = _float(diagnostics.get("best_cost"))
        actual = _cost(row)
        if reported is None or actual is None or abs(reported - actual) > 1e-7:
            final_cost_mismatches += 1
        seed_cost = _float(diagnostics.get("seed_cost"))
        if seed_cost is not None and reported is not None and reported > seed_cost + 1e-7:
            errors.append(f"evolution worsened its frozen seed: {key}")
    if hash_mismatches:
        errors.append(f"{hash_mismatches} path hashes do not match benchmark rows")
    if final_cost_mismatches:
        errors.append(f"{final_cost_mismatches} diagnostic best costs do not match trusted costs")
    if nonzero_llm_calls:
        errors.append(f"{nonzero_llm_calls} planning records report nonzero LLM calls")
    if len(path_record_keys) != 300 or len(set(path_record_keys)) != 300:
        errors.append(
            "expected exactly 300 unique Evolutionary path-record map/seed keys"
        )
    return {
        "status": "passed" if not errors else "failed",
        "path_records": len(path_records),
        "hash_mismatches": hash_mismatches,
        "final_cost_mismatches": final_cost_mismatches,
        "nonzero_llm_call_records": nonzero_llm_calls,
        "operator_attempts": dict(sorted(operator_attempts.items())),
        "operator_successes": dict(sorted(operator_successes.items())),
        "runs_improved_over_seed": sum(value > TOLERANCE for value in improvements),
        "median_relative_improvement": _median(improvements),
        "median_unique_candidates_evaluated": _median(unique_candidates),
        "median_archive_unique_paths": _median(archive_unique),
        "errors": errors,
    }


def _contract_audit(
    evolutionary: list[dict[str, Any]],
    frozen: list[dict[str, Any]],
    metadata: dict[str, Any],
    diagnostic_audit: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    expected = 300
    execution_keys = [
        (row["planner"], row["arm_id"], row["map_id"], row["seed"])
        for row in evolutionary
    ]
    map_counts = Counter(row["difficulty"] for row in evolutionary)
    if len(evolutionary) != expected:
        errors.append(f"Evolutionary run has {len(evolutionary)} rows, expected {expected}")
    if len(set(execution_keys)) != len(execution_keys):
        errors.append("Evolutionary run contains duplicate planner/arm/map/seed keys")
    if set(row["split"] for row in evolutionary) != {"validation"}:
        errors.append("Evolutionary input is not Validation-only")
    if len({row["map_id"] for row in evolutionary}) != 60:
        errors.append("Evolutionary run does not contain exactly 60 maps")
    if any(count != 50 for count in map_counts.values()) or len(map_counts) != 6:
        errors.append(f"Expected 50 rows per difficulty; got {dict(map_counts)}")
    if any(not _boolean(row["feasible"]) for row in evolutionary):
        errors.append("At least one Evolutionary record is infeasible")
    if any(int(row["objective_evaluations"]) > 2_000 for row in evolutionary):
        errors.append("At least one Evolutionary record exceeds 2,000 evaluations")
    if any(float(row["elapsed_seconds"]) > 1.0 + 1e-6 for row in evolutionary):
        errors.append("At least one Evolutionary record exceeds one second")
    if any(row["status"] != "success" for row in evolutionary):
        errors.append("At least one Evolutionary record has non-success status")
    evolutionary_keys = {(row["map_id"], row["seed"]) for row in evolutionary}
    frozen_keys = {(row["map_id"], row["seed"]) for row in frozen}
    if evolutionary_keys != frozen_keys:
        errors.append("Evolutionary and frozen inputs do not share exact map/seed keys")
    if metadata.get("split") != "validation":
        errors.append("Benchmark metadata split is not Validation")
    if metadata.get("actual_records") != expected or metadata.get("expected_records") != expected:
        errors.append("Benchmark metadata record counts do not equal 300")
    if diagnostic_audit["status"] != "passed":
        errors.extend(diagnostic_audit["errors"])
    return {
        "status": "passed" if not errors else "failed",
        "expected_records": expected,
        "actual_records": len(evolutionary),
        "unique_record_keys": len(set(execution_keys)),
        "unique_maps": len({row["map_id"] for row in evolutionary}),
        "rows_per_difficulty": dict(sorted(map_counts.items())),
        "all_validation_only": set(row["split"] for row in evolutionary) == {"validation"},
        "exact_frozen_map_seed_alignment": evolutionary_keys == frozen_keys,
        "max_elapsed_seconds": max(float(row["elapsed_seconds"]) for row in evolutionary),
        "max_objective_evaluations": max(int(row["objective_evaluations"]) for row in evolutionary),
        "all_feasible": all(_boolean(row["feasible"]) for row in evolutionary),
        "errors": errors,
    }


def analyse(
    evolutionary_dir: Path,
    frozen_dir: Path,
    traditional_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evolutionary_path = evolutionary_dir / "benchmark_runs.csv"
    frozen_path = frozen_dir / "benchmark_runs.csv"
    traditional_path = traditional_dir / "benchmark_runs.csv"
    evolutionary = _read_csv(evolutionary_path)
    frozen = _read_csv(frozen_path)
    traditional = _read_csv(traditional_path)
    metadata = _read_json(evolutionary_dir / "benchmark_metadata.json")
    path_records = _read_jsonl(evolutionary_dir / "benchmark_paths.jsonl")

    seed_pairs, map_pairs, frozen_comparison = _pair_evolutionary_and_frozen(
        evolutionary, frozen
    )
    difficulty = _difficulty_summary(map_pairs)
    diagnostics = _diagnostic_audit(evolutionary, path_records)
    audit = _contract_audit(evolutionary, frozen, metadata, diagnostics)
    if audit["status"] != "passed":
        raise RuntimeError("Evolutionary AFL-UAV Validation audit failed: " + "; ".join(audit["errors"]))

    feasible_costs = [value for row in evolutionary if (value := _cost(row)) is not None]
    unique_by_map = [int(row["evolutionary_unique_paths"]) for row in map_pairs]
    overall = {
        "runs": len(evolutionary),
        "maps": len(map_pairs),
        "feasible_runs": len(feasible_costs),
        "feasible_rate": len(feasible_costs) / len(evolutionary),
        "median_cost": _median(feasible_costs),
        "cost_q1": _quantile(feasible_costs, 0.25),
        "cost_q3": _quantile(feasible_costs, 0.75),
        "median_elapsed_seconds": _median(float(row["elapsed_seconds"]) for row in evolutionary),
        "max_elapsed_seconds": max(float(row["elapsed_seconds"]) for row in evolutionary),
        "median_objective_evaluations": _median(
            int(row["objective_evaluations"]) for row in evolutionary
        ),
        "max_objective_evaluations": max(int(row["objective_evaluations"]) for row in evolutionary),
        "maps_with_multiple_final_paths": sum(value > 1 for value in unique_by_map),
        "median_unique_final_paths_per_map": _median(unique_by_map),
        "research_claim_eligible": all(
            _boolean(row["research_claim_eligible"]) for row in evolutionary
        ),
    }
    summary = {
        "analysis_id": "evolutionary-afl-uav-validation-analysis-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "uav2d-v1 Validation only; Test was not read",
        "confidence": "ready_to_share_with_experimental_caveats",
        "contract": audit,
        "evolutionary_afl_uav": overall,
        "vs_frozen_afl_uav": frozen_comparison,
        "objective_component_tradeoffs_vs_frozen": _component_comparison(
            evolutionary, frozen
        ),
        "vs_astar": _deterministic_baseline_comparison(map_pairs, traditional, "astar"),
        "vs_theta_star": _deterministic_baseline_comparison(
            map_pairs, traditional, "theta_star"
        ),
        "operator_and_path_diagnostics": diagnostics,
        "methodology": {
            "ranking": "feasible rate first, then trusted feasible total cost",
            "frozen_comparison": "exact map and shared seed; primary headline uses per-map median",
            "deterministic_comparison": "per-map median Evolutionary cost versus one deterministic result",
            "uncertainty": "cluster bootstrap over map_id, 2000 replicates, seed 20260816",
            "tie_tolerance": TOLERANCE,
        },
        "inputs": {
            str(evolutionary_path.relative_to(ROOT)): _sha256(evolutionary_path),
            str(frozen_path.relative_to(ROOT)): _sha256(frozen_path),
            str(traditional_path.relative_to(ROOT)): _sha256(traditional_path),
        },
    }

    _write_csv(output_dir / "paired_seed_comparison.csv", seed_pairs)
    _write_csv(output_dir / "paired_map_comparison.csv", map_pairs)
    _write_csv(output_dir / "by_difficulty_comparison.csv", difficulty)
    _write_json(output_dir / "validation_audit.json", audit)
    _write_json(output_dir / "analysis_summary.json", summary)
    outputs = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "analysis_receipt.json"
    )
    receipt = {
        "analysis_id": summary["analysis_id"],
        "status": "passed",
        "files": {path.name: _sha256(path) for path in outputs},
    }
    _write_json(output_dir / "analysis_receipt.json", receipt)
    return {
        "status": "passed",
        "output_dir": str(output_dir),
        "records": len(evolutionary),
        "maps": len(map_pairs),
        "median_cost": overall["median_cost"],
        "files": receipt["files"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evolutionary-dir", type=Path, default=DEFAULT_EVOLUTIONARY)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--traditional-dir", type=Path, default=DEFAULT_TRADITIONAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            analyse(
                args.evolutionary_dir.resolve(),
                args.frozen_dir.resolve(),
                args.traditional_dir.resolve(),
                args.output_dir.resolve(),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
