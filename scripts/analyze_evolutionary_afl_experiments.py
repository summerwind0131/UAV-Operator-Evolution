"""Audit and analyse frozen Evolutionary AFL-UAV v1 experiments.

This script is intentionally Validation-only.  It treats a record that crosses
the advertised wall-clock boundary as an effective timeout even when an older
runner wrote ``status=success`` before the trusted boundary check was added.
"""

from __future__ import annotations

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
RUN_ROOT = ROOT / "artifacts" / "planning_benchmarks"
OUTPUT = RUN_ROOT / "evolutionary-afl-uav-experiments-analysis-v1"
TOLERANCE = 1e-9
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_REPLICATES = 2_000

DATASETS = {
    "full_v1": {
        "directory": RUN_ROOT / "evolutionary-afl-uav-validation-v1",
        "expected_rows": 300,
        "expected_arms": {"deepseek_v4pro_evo"},
        "time_limit": 1.0,
        "group": "reference",
    },
    "frozen_afl": {
        "directory": RUN_ROOT / "deepseek-v4pro-strict-validation-v2",
        "expected_rows": 300,
        "expected_arms": {"deepseek_v4pro_strict"},
        "time_limit": 1.0,
        "group": "reference",
    },
    "ablation": {
        "directory": RUN_ROOT / "evolutionary-afl-uav-ablation-validation-v1",
        "expected_rows": 1_500,
        "expected_arms": {
            "evo_no_qd_archive",
            "evo_no_crossover",
            "evo_move_only",
            "evo_no_rooms_strategy",
            "evo_fixed_length",
        },
        "time_limit": 1.0,
        "group": "ablation",
    },
    "sensitivity_algorithm": {
        "directory": RUN_ROOT
        / "evolutionary-afl-uav-sensitivity-algorithm-validation-v1",
        "expected_rows": 1_800,
        "expected_arms": {
            "evo_population_16",
            "evo_population_24",
            "evo_archive_4",
            "evo_archive_12",
            "evo_generations_6",
            "evo_generations_12",
        },
        "time_limit": 1.0,
        "group": "sensitivity",
    },
    "sensitivity_time_025": {
        "directory": RUN_ROOT
        / "evolutionary-afl-uav-sensitivity-time025-validation-v1",
        "expected_rows": 300,
        "expected_arms": {"evo_time_025"},
        "time_limit": 0.25,
        "group": "sensitivity",
    },
    "sensitivity_time_050": {
        "directory": RUN_ROOT
        / "evolutionary-afl-uav-sensitivity-time050-validation-v1",
        "expected_rows": 300,
        "expected_arms": {"evo_time_050"},
        "time_limit": 0.5,
        "group": "sensitivity",
    },
}

ARM_LABELS = {
    "deepseek_v4pro_evo": "完整 Evolutionary AFL-UAV v1",
    "deepseek_v4pro_strict": "原始冻结 AFL-UAV",
    "evo_no_qd_archive": "去掉质量—多样性档案",
    "evo_no_crossover": "去掉交叉",
    "evo_move_only": "仅移动航点",
    "evo_no_rooms_strategy": "去掉 rooms_maze 专用策略",
    "evo_fixed_length": "固定长度种群",
    "evo_population_16": "种群 16",
    "evo_population_24": "种群 24",
    "evo_archive_4": "精英档案 4",
    "evo_archive_12": "精英档案 12",
    "evo_generations_6": "进化代数 6",
    "evo_generations_12": "进化代数 12",
    "evo_time_025": "时间预算 0.25 秒",
    "evo_time_050": "时间预算 0.5 秒",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.median(values)) if values else None


def _quantile(values: Iterable[float], q: float) -> float | None:
    values = list(values)
    return float(np.quantile(values, q)) if values else None


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["map_id"]), int(row["seed"])


def _effective_timeout(row: dict[str, Any], time_limit: float) -> bool:
    return str(row.get("status")) == "timeout" or float(row["elapsed_seconds"]) >= time_limit


def _official_feasible(row: dict[str, Any], time_limit: float) -> bool:
    return _boolean(row.get("feasible")) and not _effective_timeout(row, time_limit)


def _cost(row: dict[str, Any], time_limit: float) -> float | None:
    return _number(row.get("total_cost")) if _official_feasible(row, time_limit) else None


def _bootstrap_map_deltas(deltas: list[float]) -> dict[str, float | int | None]:
    if not deltas:
        return {
            "cluster_bootstrap_replicates": 0,
            "median_delta_ci_low": None,
            "median_delta_ci_high": None,
            "half_tie_win_rate_ci_low": None,
            "half_tie_win_rate_ci_high": None,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray(deltas)
    medians: list[float] = []
    win_rates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = values[rng.integers(0, len(values), size=len(values))]
        medians.append(float(np.median(sample)))
        win_rates.append(
            float(np.mean(np.where(sample < -TOLERANCE, 1.0, np.where(abs(sample) <= TOLERANCE, 0.5, 0.0))))
        )
    return {
        "cluster_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "median_delta_ci_low": float(np.quantile(medians, 0.025)),
        "median_delta_ci_high": float(np.quantile(medians, 0.975)),
        "half_tie_win_rate_ci_low": float(np.quantile(win_rates, 0.025)),
        "half_tie_win_rate_ci_high": float(np.quantile(win_rates, 0.975)),
    }


def _audit_dataset(
    name: str,
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    paths: list[dict[str, Any]],
    reference_keys: set[tuple[str, int]] | None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    time_limit = float(spec["time_limit"])
    arms = {str(row["arm_id"]) for row in rows}
    record_keys = [
        (row["planner"], row["arm_id"], row["map_id"], int(row["seed"]))
        for row in rows
    ]
    per_arm = Counter(str(row["arm_id"]) for row in rows)
    map_seed_by_arm: dict[str, set[tuple[str, int]]] = defaultdict(set)
    difficulty_by_arm: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        map_seed_by_arm[str(row["arm_id"])].add(_key(row))
        difficulty_by_arm[str(row["arm_id"])][str(row["difficulty"])] += 1

    if len(rows) != int(spec["expected_rows"]):
        errors.append(f"rows={len(rows)}, expected={spec['expected_rows']}")
    if len(set(record_keys)) != len(record_keys):
        errors.append("duplicate planner/arm/map/seed keys")
    if arms != set(spec["expected_arms"]):
        errors.append(f"arms={sorted(arms)}, expected={sorted(spec['expected_arms'])}")
    if {row["split"] for row in rows} != {"validation"}:
        errors.append("non-Validation records detected")
    if any(count != 300 for count in per_arm.values()):
        errors.append(f"expected 300 rows per arm, got {dict(per_arm)}")
    for arm, counts in difficulty_by_arm.items():
        if len(counts) != 6 or any(count != 50 for count in counts.values()):
            errors.append(f"{arm} difficulty balance is {dict(counts)}")
    if reference_keys is not None:
        for arm, keys in map_seed_by_arm.items():
            if keys != reference_keys:
                errors.append(f"{arm} does not share the exact reference map/seed keys")
    if any(int(row["objective_evaluations"]) > 2_000 for row in rows):
        errors.append("objective-evaluation budget exceeded")
    if any(not _boolean(row["research_claim_eligible"]) for row in rows):
        errors.append("research_claim_eligible is false")

    raw_timeouts = sum(str(row["status"]) == "timeout" for row in rows)
    effective_timeouts = sum(_effective_timeout(row, time_limit) for row in rows)
    stale_successes = [
        row
        for row in rows
        if str(row["status"]) == "success"
        and float(row["elapsed_seconds"]) >= time_limit
    ]
    if stale_successes:
        warnings.append(
            f"{len(stale_successes)} legacy success labels crossed the wall-clock boundary; "
            "analysis reclassifies them as effective timeouts"
        )

    run_index = {
        (row["arm_id"], row["map_id"], int(row["seed"])): row for row in rows
    }
    path_keys: list[tuple[str, str, int]] = []
    hash_mismatches = 0
    nonzero_llm_calls = 0
    for record in paths:
        path_key = (record["arm_id"], record["map_id"], int(record["seed"]))
        path_keys.append(path_key)
        row = run_index.get(path_key)
        if row is None:
            errors.append(f"orphan path record {path_key}")
            continue
        if path_hash(record.get("path")) != row["path_hash"]:
            hash_mismatches += 1
        diagnostics = record.get("diagnostics", {})
        nonzero_llm_calls += int(diagnostics.get("llm_calls_during_planning", 0)) != 0
    if len(paths) != len(rows) or len(set(path_keys)) != len(paths):
        errors.append(
            f"path record count/uniqueness mismatch: paths={len(paths)}, runs={len(rows)}"
        )
    if hash_mismatches:
        errors.append(f"{hash_mismatches} path hashes mismatch")
    if nonzero_llm_calls:
        errors.append(f"{nonzero_llm_calls} records report online LLM calls")

    return {
        "dataset": name,
        "status": "passed_with_warnings" if warnings and not errors else "passed" if not errors else "failed",
        "expected_rows": int(spec["expected_rows"]),
        "actual_rows": len(rows),
        "unique_record_keys": len(set(record_keys)),
        "arms": sorted(arms),
        "rows_per_arm": dict(sorted(per_arm.items())),
        "validation_only": {row["split"] for row in rows} == {"validation"},
        "time_limit_seconds": time_limit,
        "raw_timeout_records": raw_timeouts,
        "effective_timeout_records": effective_timeouts,
        "legacy_success_beyond_boundary": len(stale_successes),
        "legacy_success_keys": [
            {"arm_id": row["arm_id"], "map_id": row["map_id"], "seed": int(row["seed"])}
            for row in stale_successes
        ],
        "max_elapsed_seconds": max(float(row["elapsed_seconds"]) for row in rows),
        "max_objective_evaluations": max(int(row["objective_evaluations"]) for row in rows),
        "path_records": len(paths),
        "path_hash_mismatches": hash_mismatches,
        "nonzero_llm_call_records": nonzero_llm_calls,
        "errors": errors,
        "warnings": warnings,
    }


def _arm_statistics(
    dataset: str,
    group: str,
    arm: str,
    rows: list[dict[str, Any]],
    time_limit: float,
) -> dict[str, Any]:
    official = [row for row in rows if _official_feasible(row, time_limit)]
    costs = [float(row["total_cost"]) for row in official]
    by_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in official:
        by_map[str(row["map_id"])].append(row)
    unique_paths = [len({row["path_hash"] for row in map_rows}) for map_rows in by_map.values()]
    map_iqrs = [
        float(
            np.quantile([float(row["total_cost"]) for row in map_rows], 0.75)
            - np.quantile([float(row["total_cost"]) for row in map_rows], 0.25)
        )
        for map_rows in by_map.values()
    ]
    return {
        "dataset": dataset,
        "experiment_group": group,
        "arm_id": arm,
        "label": ARM_LABELS[arm],
        "runs": len(rows),
        "trusted_feasible_runs": len(official),
        "trusted_feasible_rate": len(official) / len(rows),
        "maps_with_trusted_feasible_path": len(by_map),
        "effective_timeouts": sum(_effective_timeout(row, time_limit) for row in rows),
        "median_cost": _median(costs),
        "cost_q1": _quantile(costs, 0.25),
        "cost_q3": _quantile(costs, 0.75),
        "median_elapsed_seconds": _median(float(row["elapsed_seconds"]) for row in rows),
        "max_elapsed_seconds": max(float(row["elapsed_seconds"]) for row in rows),
        "median_objective_evaluations": _median(int(row["objective_evaluations"]) for row in rows),
        "max_objective_evaluations": max(int(row["objective_evaluations"]) for row in rows),
        "maps_with_multiple_paths": sum(value > 1 for value in unique_paths),
        "median_unique_paths_per_map": _median(unique_paths),
        "median_within_map_cost_iqr": _median(map_iqrs),
    }


def _compare_to_full(
    arm: str,
    candidate: list[dict[str, Any]],
    candidate_limit: float,
    full: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_index = {_key(row): row for row in full}
    candidate_index = {_key(row): row for row in candidate}
    seed_deltas: list[float] = []
    for key in sorted(full_index):
        candidate_cost = _cost(candidate_index[key], candidate_limit)
        full_cost = _cost(full_index[key], 1.0)
        if candidate_cost is not None and full_cost is not None:
            seed_deltas.append(candidate_cost - full_cost)

    map_rows: list[dict[str, Any]] = []
    map_ids = sorted({key[0] for key in full_index})
    for map_id in map_ids:
        baseline_rows = [row for key, row in full_index.items() if key[0] == map_id]
        candidate_rows = [row for key, row in candidate_index.items() if key[0] == map_id]
        baseline_costs = [
            value for row in baseline_rows if (value := _cost(row, 1.0)) is not None
        ]
        candidate_costs = [
            value
            for row in candidate_rows
            if (value := _cost(row, candidate_limit)) is not None
        ]
        baseline_median = _median(baseline_costs)
        candidate_median = _median(candidate_costs)
        delta = (
            candidate_median - baseline_median
            if candidate_median is not None and baseline_median is not None
            else None
        )
        map_rows.append(
            {
                "arm_id": arm,
                "label": ARM_LABELS[arm],
                "map_id": map_id,
                "difficulty": baseline_rows[0]["difficulty"],
                "candidate_feasible_seeds": len(candidate_costs),
                "full_v1_feasible_seeds": len(baseline_costs),
                "candidate_median_cost": candidate_median,
                "full_v1_median_cost": baseline_median,
                "candidate_minus_full_cost": delta,
            }
        )

    comparable = [row for row in map_rows if row["candidate_minus_full_cost"] is not None]
    deltas = [float(row["candidate_minus_full_cost"]) for row in comparable]
    wins = sum(value < -TOLERANCE for value in deltas)
    ties = sum(abs(value) <= TOLERANCE for value in deltas)
    losses = sum(value > TOLERANCE for value in deltas)
    summary = {
        "arm_id": arm,
        "label": ARM_LABELS[arm],
        "paired_seed_runs": len(seed_deltas),
        "candidate_seed_wins": sum(value < -TOLERANCE for value in seed_deltas),
        "candidate_seed_ties": sum(abs(value) <= TOLERANCE for value in seed_deltas),
        "candidate_seed_losses": sum(value > TOLERANCE for value in seed_deltas),
        "paired_maps": len(comparable),
        "candidate_map_wins": wins,
        "candidate_map_ties": ties,
        "candidate_map_losses": losses,
        "half_tie_map_win_rate": (wins + 0.5 * ties) / len(comparable),
        "median_candidate_minus_full_cost": _median(deltas),
        "median_relative_change_vs_full": _median(
            float(row["candidate_minus_full_cost"]) / float(row["full_v1_median_cost"])
            for row in comparable
        ),
        **_bootstrap_map_deltas(deltas),
    }

    rooms = [row for row in comparable if row["difficulty"] == "rooms_maze"]
    room_deltas = [float(row["candidate_minus_full_cost"]) for row in rooms]
    rooms_summary = {
        "arm_id": arm,
        "label": ARM_LABELS[arm],
        "rooms_maze_maps": len(rooms),
        "candidate_wins": sum(value < -TOLERANCE for value in room_deltas),
        "ties": sum(abs(value) <= TOLERANCE for value in room_deltas),
        "candidate_losses": sum(value > TOLERANCE for value in room_deltas),
        "median_candidate_minus_full_cost": _median(room_deltas),
    }
    return summary, rooms_summary


def _operator_rows(
    dataset: str,
    group: str,
    arm: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    attempts: Counter[str] = Counter()
    changed: Counter[str] = Counter()
    accepted_offspring = 0
    unique_candidates = 0
    archive_unique: list[int] = []
    for record in records:
        if record["arm_id"] != arm:
            continue
        diagnostics = record.get("diagnostics", {})
        attempts.update(diagnostics.get("operator_attempts", {}))
        changed.update(diagnostics.get("operator_successes", {}))
        accepted_offspring += int(diagnostics.get("accepted_offspring", 0))
        unique_candidates += int(diagnostics.get("unique_candidates_evaluated", 0))
        archive_unique.append(int(diagnostics.get("archive_unique_paths", 0)))
    result: list[dict[str, Any]] = []
    for operator in sorted(set(attempts) | set(changed)):
        result.append(
            {
                "dataset": dataset,
                "experiment_group": group,
                "arm_id": arm,
                "label": ARM_LABELS[arm],
                "operator": operator,
                "attempts": attempts[operator],
                "structure_changes": changed[operator],
                "structure_change_rate": changed[operator] / attempts[operator] if attempts[operator] else None,
                "accepted_offspring_all_operators": accepted_offspring,
                "unique_candidates_all_operators": unique_candidates,
                "median_archive_unique_paths": _median(archive_unique),
                "interpretation": "structure_change_rate_is_not_cost_improvement_rate",
            }
        )
    return result


def analyse() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    for name, spec in DATASETS.items():
        directory = Path(spec["directory"])
        runs_path = directory / "benchmark_runs.csv"
        paths_path = directory / "benchmark_paths.jsonl"
        metadata_path = directory / "benchmark_metadata.json"
        loaded[name] = {
            "rows": _read_csv(runs_path),
            "paths": _read_jsonl(paths_path),
            "metadata": _read_json(metadata_path),
        }
        for path in (runs_path, paths_path, metadata_path):
            input_hashes[str(path.relative_to(ROOT))] = _sha256(path)

    full_rows = loaded["full_v1"]["rows"]
    reference_keys = {_key(row) for row in full_rows}
    audits = [
        _audit_dataset(
            name,
            spec,
            loaded[name]["rows"],
            loaded[name]["paths"],
            None if name == "full_v1" else reference_keys,
        )
        for name, spec in DATASETS.items()
    ]
    failures = [audit for audit in audits if audit["errors"]]
    if failures:
        raise RuntimeError(
            "experiment audit failed: "
            + "; ".join(
                f"{audit['dataset']}: {', '.join(audit['errors'])}" for audit in failures
            )
        )

    statistics_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    rooms_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    paired_map_rows: list[dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        rows = loaded[dataset]["rows"]
        paths = loaded[dataset]["paths"]
        for arm in sorted(spec["expected_arms"]):
            arm_rows = [row for row in rows if row["arm_id"] == arm]
            statistics_rows.append(
                _arm_statistics(
                    dataset,
                    str(spec["group"]),
                    arm,
                    arm_rows,
                    float(spec["time_limit"]),
                )
            )
            operator_rows.extend(
                _operator_rows(dataset, str(spec["group"]), arm, paths)
            )
            if dataset != "full_v1":
                paired, rooms = _compare_to_full(
                    arm, arm_rows, float(spec["time_limit"]), full_rows
                )
                paired["dataset"] = dataset
                paired["experiment_group"] = spec["group"]
                rooms["dataset"] = dataset
                rooms["experiment_group"] = spec["group"]
                paired_rows.append(paired)
                rooms_rows.append(rooms)

                full_index_by_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
                candidate_index_by_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in full_rows:
                    full_index_by_map[row["map_id"]].append(row)
                for row in arm_rows:
                    candidate_index_by_map[row["map_id"]].append(row)
                for map_id in sorted(full_index_by_map):
                    base_costs = [
                        value
                        for row in full_index_by_map[map_id]
                        if (value := _cost(row, 1.0)) is not None
                    ]
                    cand_costs = [
                        value
                        for row in candidate_index_by_map[map_id]
                        if (value := _cost(row, float(spec["time_limit"]))) is not None
                    ]
                    base_median = _median(base_costs)
                    cand_median = _median(cand_costs)
                    paired_map_rows.append(
                        {
                            "dataset": dataset,
                            "experiment_group": spec["group"],
                            "arm_id": arm,
                            "label": ARM_LABELS[arm],
                            "map_id": map_id,
                            "difficulty": full_index_by_map[map_id][0]["difficulty"],
                            "candidate_feasible_seeds": len(cand_costs),
                            "full_v1_feasible_seeds": len(base_costs),
                            "candidate_median_cost": cand_median,
                            "full_v1_median_cost": base_median,
                            "candidate_minus_full_cost": (
                                cand_median - base_median
                                if cand_median is not None and base_median is not None
                                else None
                            ),
                        }
                    )

    ablation_pairs = [row for row in paired_rows if row["experiment_group"] == "ablation"]
    sensitivity_pairs = [
        row for row in paired_rows if row["experiment_group"] == "sensitivity"
    ]
    audit_status = (
        "passed_with_caveats"
        if any(audit["warnings"] for audit in audits)
        else "passed"
    )
    summary = {
        "analysis_id": "evolutionary-afl-uav-experiments-analysis-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "uav2d-v1 Validation only; Test was not read; zero API calls",
        "overall_assessment": "ready_to_share_with_caveats",
        "audit_status": audit_status,
        "data_quality": audits,
        "headline": {
            "full_v1": next(row for row in statistics_rows if row["arm_id"] == "deepseek_v4pro_evo"),
            "original_frozen_afl": next(row for row in statistics_rows if row["arm_id"] == "deepseek_v4pro_strict"),
            "ablation_vs_full": ablation_pairs,
            "sensitivity_vs_full": sensitivity_pairs,
            "rooms_maze": rooms_rows,
        },
        "methodology": {
            "grain": "planner arm × Validation map × shared seed",
            "primary_comparison": "per-map median over five shared seeds",
            "delta_definition": "candidate cost minus frozen full-v1 cost; positive means full v1 is better",
            "ranking": "trusted feasible rate first, then trusted feasible total cost",
            "effective_timeout": "raw timeout OR elapsed_seconds >= advertised time limit",
            "uncertainty": f"cluster bootstrap over 60 map_id values, {BOOTSTRAP_REPLICATES} replicates",
            "operator_success_definition": "operator_successes counts valid structural changes, not cost improvements",
            "operator_quality_contribution": "inferred from controlled ablation comparisons, not per-operation counters",
            "sensitivity_design": "one factor at a time; baseline pop32/archive8/gens20/time1 reused without tuning",
            "tie_tolerance": TOLERANCE,
        },
        "required_caveats": [
            "Validation has already been observed and is used for explanation, not further v1 tuning.",
            "Three ablation records crossed the one-second boundary; they are preserved and treated as effective timeouts.",
            "Ablation supports component attribution within this implementation, but does not establish universal causality on unseen distributions.",
            "No hidden Test result is included; final generalization remains unmeasured.",
        ],
        "inputs": input_hashes,
    }

    _write_csv(OUTPUT / "arm_statistics.csv", statistics_rows)
    _write_csv(OUTPUT / "paired_vs_full.csv", paired_rows)
    _write_csv(OUTPUT / "paired_map_details.csv", paired_map_rows)
    _write_csv(OUTPUT / "rooms_maze_comparison.csv", rooms_rows)
    _write_csv(OUTPUT / "operator_diagnostics.csv", operator_rows)
    _write_csv(OUTPUT / "sensitivity_summary.csv", sensitivity_pairs)
    _write_json(OUTPUT / "validation_audit.json", {"status": audit_status, "datasets": audits})
    _write_json(OUTPUT / "analysis_summary.json", summary)
    outputs = sorted(
        path
        for path in OUTPUT.iterdir()
        if path.is_file() and path.name != "analysis_receipt.json"
    )
    receipt = {
        "analysis_id": summary["analysis_id"],
        "status": audit_status,
        "generated_at_utc": summary["generated_at_utc"],
        "source_input_hashes": input_hashes,
        "output_hashes": {path.name: _sha256(path) for path in outputs},
    }
    _write_json(OUTPUT / "analysis_receipt.json", receipt)
    return {
        "status": audit_status,
        "output_directory": str(OUTPUT),
        "datasets": {name: len(loaded[name]["rows"]) for name in DATASETS},
        "arms": len(statistics_rows),
        "api_calls": 0,
    }


if __name__ == "__main__":
    print(json.dumps(analyse(), ensure_ascii=False, indent=2))
