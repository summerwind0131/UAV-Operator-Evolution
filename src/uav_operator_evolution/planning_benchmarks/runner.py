"""Experiment orchestration and durable outputs for planner benchmarks."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import logging
import math
import platform
import socket
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import ExperimentConfig
from ..environment.environment import Environment2D
from ..environment.generator import DatasetManifest, load_dataset_split
from ..path.evaluator import PathEvaluator
from ..reproducibility import derive_seed, stable_hash
from .core import PlanningBudget, path_hash, run_with_trusted_validation
from .planners import build_planners

LOGGER = logging.getLogger(__name__)


class _FinalEvaluationAuthorization:
    """Internal capability issued only by the hash-audited final-test entry."""

    __slots__ = ("benchmark_id", "opening_id", "preregistration_id")

    def __init__(
        self,
        *,
        benchmark_id: str,
        opening_id: str,
        preregistration_id: str,
    ) -> None:
        self.benchmark_id = benchmark_id
        self.opening_id = opening_id
        self.preregistration_id = preregistration_id


def _issue_final_evaluation_authorization(
    *,
    benchmark_id: str,
    opening_id: str,
    preregistration_id: str,
) -> _FinalEvaluationAuthorization:
    """Issue the private runner capability after the entrypoint validates receipts."""

    if not benchmark_id or not opening_id or not preregistration_id:
        raise ValueError("final-evaluation authorization fields must be non-empty")
    return _FinalEvaluationAuthorization(
        benchmark_id=benchmark_id,
        opening_id=opening_id,
        preregistration_id=preregistration_id,
    )


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite diagnostics with strict-JSON nulls."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=str,
        )
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


def _write_paths_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _json_safe(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def _selected_maps(
    maps: Iterable[Environment2D],
    maps_per_class: int | None,
) -> list[Environment2D]:
    grouped: dict[str, list[Environment2D]] = defaultdict(list)
    for environment in maps:
        grouped[environment.difficulty].append(environment)
    selected: list[Environment2D] = []
    for difficulty in sorted(grouped):
        ordered = sorted(grouped[difficulty], key=lambda item: item.map_id)
        selected.extend(
            ordered if maps_per_class is None else ordered[:maps_per_class]
        )
    return sorted(selected, key=lambda item: item.map_id)


def _quantile(values: list[float], quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values else None


def _summary_row(
    planner: str,
    arm_id: str,
    map_class: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    feasible_rows = [row for row in rows if row["feasible"]]
    costs = [float(row["total_cost"]) for row in feasible_rows]
    runtimes = [float(row["elapsed_seconds"]) for row in rows]
    q25 = _quantile(costs, 0.25)
    q75 = _quantile(costs, 0.75)
    return {
        "planner": planner,
        "arm_id": arm_id,
        "map_class": map_class,
        "runs": len(rows),
        "feasible_runs": len(feasible_rows),
        "feasible_rate": len(feasible_rows) / len(rows) if rows else 0.0,
        "timeout_rate": (
            sum(row["status"] == "timeout" for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "budget_exhausted_rate": (
            sum(row["status"] == "budget_exhausted" for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "median_total_cost_feasible": _quantile(costs, 0.5),
        "iqr_total_cost_feasible": (
            None if q25 is None or q75 is None else q75 - q25
        ),
        "median_runtime_seconds": _quantile(runtimes, 0.5),
    }


def summarize_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate overall and per-class metrics without penalized fake successes."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        arm_id = str(row.get("arm_id") or row["planner"])
        grouped[(str(row["planner"]), arm_id, "all")].append(row)
        grouped[
            (str(row["planner"]), arm_id, str(row["difficulty"]))
        ].append(row)
    summary_rows = [
        _summary_row(planner, arm_id, map_class, group)
        for (planner, arm_id, map_class), group in sorted(grouped.items())
    ]
    overall = [row for row in summary_rows if row["map_class"] == "all"]
    ranking = sorted(
        overall,
        key=lambda row: (
            -float(row["feasible_rate"]),
            float("inf")
            if row["median_total_cost_feasible"] is None
            else float(row["median_total_cost_feasible"]),
            row["planner"],
            row["arm_id"],
        ),
    )
    paired_comparisons: list[dict[str, Any]] = []
    by_arm_map: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["feasible"] and row["total_cost"] is not None:
            arm_id = str(row.get("arm_id") or row["planner"])
            by_arm_map[(str(row["planner"]), arm_id)][str(row["map_id"])].append(
                float(row["total_cost"])
            )
    afl_arms = sorted(
        {
            (str(row["planner"]), str(row.get("arm_id") or row["planner"]))
            for row in rows
            if row["planner"] == "afl_uav"
        }
    )
    for afl_key in afl_arms:
        for baseline in ("astar", "theta_star"):
            baseline_key = (baseline, baseline)
            common_maps = sorted(
                set(by_arm_map[afl_key]) & set(by_arm_map.get(baseline_key, {}))
            )
            wins = ties = losses = 0
            for map_id in common_maps:
                afl_cost = float(np.median(by_arm_map[afl_key][map_id]))
                baseline_cost = float(np.median(by_arm_map[baseline_key][map_id]))
                tolerance = 1e-9 * max(1.0, abs(afl_cost), abs(baseline_cost))
                if afl_cost < baseline_cost - tolerance:
                    wins += 1
                elif afl_cost > baseline_cost + tolerance:
                    losses += 1
                else:
                    ties += 1
            paired_comparisons.append(
                {
                    "planner": "afl_uav",
                    "arm_id": afl_key[1],
                    "baseline": baseline,
                    "both_feasible_maps": len(common_maps),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate_on_both_feasible": (
                        wins / len(common_maps) if common_maps else None
                    ),
                }
            )
    return {
        "ranking_rule": (
            "descending feasible_rate, then ascending median unified total cost "
            "among feasible paths"
        ),
        "ranking": [
            {
                "rank": index + 1,
                "planner": row["planner"],
                "arm_id": row["arm_id"],
                "feasible_rate": row["feasible_rate"],
                "median_total_cost_feasible": row[
                    "median_total_cost_feasible"
                ],
            }
            for index, row in enumerate(ranking)
        ],
        "statistics": summary_rows,
        "paired_comparisons": paired_comparisons,
    }


def run_planner_benchmark(
    config: ExperimentConfig,
    *,
    split: str = "test",
    planners: list[str] | None = None,
    maps_per_class: int | None = None,
    time_limit_seconds: float | None = None,
    max_objective_evaluations: int | None = None,
    repetitions: int | None = None,
    afl_artifact_path: str | Path | None = None,
    afl_artifacts: dict[str, str | Path] | None = None,
    evolutionary_afl_artifacts: dict[str, str | Path] | None = None,
    planner_overrides: dict[str, object] | None = None,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    _final_evaluation_authorization: _FinalEvaluationAuthorization | None = None,
) -> dict[str, Any]:
    """Run all selected planner/map/seed combinations and write four artifacts."""

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    seal_marker = Path(config.output.data_dir) / "SEALED.json"
    if seal_marker.exists():
        raise PermissionError(
            "benchmark dataset is sealed; an explicit, hash-audited opening "
            f"receipt is required before any planner may run: {seal_marker}"
        )
    contains_restricted_test_arm = bool(
        afl_artifact_path is not None
        or afl_artifacts
        or evolutionary_afl_artifacts
        or any(
            getattr(planner, "name", "") in {"afl_uav", "evolutionary_afl_uav"}
            for planner in (planner_overrides or {}).values()
        )
    )
    if split == "test" and contains_restricted_test_arm:
        authorization = _final_evaluation_authorization
        authorized = bool(
            isinstance(authorization, _FinalEvaluationAuthorization)
            and authorization.benchmark_id == config.planning_benchmark.benchmark_id
            and authorization.opening_id
            and authorization.preregistration_id
        )
        if not authorized:
            raise ValueError(
                "AFL-UAV and Evolutionary AFL-UAV are restricted to Train/Validation "
                "unless the dedicated hash-audited final-evaluation entry issues an "
                "authorization capability"
            )
    benchmark_config = config.planning_benchmark
    planner_registry = build_planners(
        grid_resolution=config.maps.grid_resolution,
        population_size=benchmark_config.population_size,
        waypoint_count=benchmark_config.waypoint_count,
        afl_artifact_path=(
            None if afl_artifact_path is None else str(afl_artifact_path)
        ),
        afl_artifacts={
            arm_id: str(path) for arm_id, path in (afl_artifacts or {}).items()
        },
        evolutionary_afl_artifacts={
            arm_id: str(path)
            for arm_id, path in (evolutionary_afl_artifacts or {}).items()
        },
    )
    duplicate_override_keys = sorted(
        set(planner_registry).intersection(planner_overrides or {})
    )
    if duplicate_override_keys:
        raise ValueError(
            "planner overrides conflict with built-in registry keys: "
            + ", ".join(duplicate_override_keys)
        )
    planner_registry.update(planner_overrides or {})
    frozen_artifact_planner_names = [
        name for name in planner_registry if name == "afl_uav" or name.startswith("afl_uav:")
    ]
    evolutionary_artifact_planner_names = [
        name
        for name in planner_registry
        if name == "evolutionary_afl_uav"
        or name.startswith("evolutionary_afl_uav:")
    ]
    artifact_planner_names = [
        *frozen_artifact_planner_names,
        *evolutionary_artifact_planner_names,
    ]
    if planners is None:
        selected_planner_names = list(benchmark_config.planners)
        selected_planner_names.extend(
            name for name in artifact_planner_names if name not in selected_planner_names
        )
    else:
        selected_planner_names = []
        for name in planners:
            if name == "afl_uav" and frozen_artifact_planner_names:
                selected_planner_names.extend(frozen_artifact_planner_names)
            elif (
                name == "evolutionary_afl_uav"
                and evolutionary_artifact_planner_names
            ):
                selected_planner_names.extend(evolutionary_artifact_planner_names)
            else:
                selected_planner_names.append(name)
    unknown = sorted(set(selected_planner_names) - set(planner_registry))
    if unknown:
        raise ValueError(f"unknown planners: {', '.join(unknown)}")
    if len(selected_planner_names) != len(set(selected_planner_names)):
        raise ValueError("planner selection contains duplicates")
    dataset_root = config.output.data_dir
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        from ..environment.generator import generate_dataset

        generate_dataset(config)
    manifest = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    environments = _selected_maps(
        load_dataset_split(dataset_root, split),
        maps_per_class,
    )
    budget = PlanningBudget(
        time_limit_seconds=(
            time_limit_seconds
            if time_limit_seconds is not None
            else benchmark_config.time_limit_seconds
        ),
        max_objective_evaluations=(
            max_objective_evaluations
            if max_objective_evaluations is not None
            else benchmark_config.max_objective_evaluations
        ),
    )
    stochastic_repetitions = (
        repetitions
        if repetitions is not None
        else benchmark_config.stochastic_repetitions
    )
    if stochastic_repetitions < 1:
        raise ValueError("repetitions must be positive")
    identifier = run_id or (
        f"{benchmark_config.benchmark_id}-{split}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    destination = (
        Path(run_dir)
        if run_dir is not None
        else config.output.results_dir / identifier
    )
    destination.mkdir(parents=True, exist_ok=True)
    evaluator = PathEvaluator(config.objective)

    records: list[dict[str, Any]] = []
    path_records: list[dict[str, Any]] = []
    for environment in environments:
        for planner_name in selected_planner_names:
            planner = planner_registry[planner_name]
            arm_id = str(getattr(planner, "arm_id", planner.name))
            planner_repetitions = (
                stochastic_repetitions if planner.stochastic else 1
            )
            for repetition in range(planner_repetitions):
                seed = derive_seed(
                    config.seed,
                    "planner-benchmark",
                    split,
                    environment.map_id,
                    repetition,
                )
                result = run_with_trusted_validation(
                    planner,
                    environment,
                    evaluator,
                    budget,
                    np.random.default_rng(seed),
                )
                trusted = result.trusted_evaluation
                feasible = bool(trusted is not None and trusted.feasible)
                row = {
                    "benchmark_id": benchmark_config.benchmark_id,
                    "split": split,
                    "map_id": environment.map_id,
                    "difficulty": environment.difficulty,
                    "layout_subtype": environment.layout_subtype or "",
                    "planner": planner.name,
                    "arm_id": arm_id,
                    "execution_arm": planner_name,
                    "repetition": repetition,
                    "seed": seed,
                    "status": result.status,
                    "feasible": feasible,
                    "research_claim_eligible": bool(
                        planner.research_claim_eligible
                    ),
                    "total_cost": trusted.total_cost if feasible else None,
                    "path_length": trusted.path_length if feasible else None,
                    "collision_penalty": (
                        trusted.collision_penalty if trusted is not None else None
                    ),
                    "smoothness_penalty": (
                        trusted.smoothness_penalty if feasible else None
                    ),
                    "risk_penalty": trusted.risk_penalty if feasible else None,
                    "waypoint_penalty": (
                        trusted.waypoint_penalty if feasible else None
                    ),
                    "minimum_clearance": (
                        trusted.minimum_clearance if trusted is not None else None
                    ),
                    "elapsed_seconds": result.elapsed_seconds,
                    "objective_evaluations": result.objective_evaluations,
                    "collision_checks": result.collision_checks,
                    "node_expansions": result.node_expansions,
                    "waypoint_count": len(result.path) if result.path else 0,
                    "path_hash": path_hash(result.path),
                }
                records.append(row)
                path_records.append(
                    {
                        "benchmark_id": benchmark_config.benchmark_id,
                        "split": split,
                        "map_id": environment.map_id,
                        "planner": planner.name,
                        "arm_id": arm_id,
                        "execution_arm": planner_name,
                        "repetition": repetition,
                        "seed": seed,
                        "status": result.status,
                        "feasible": feasible,
                        "path": result.path,
                        "message": result.message,
                        "diagnostics": result.diagnostics,
                    }
                )
                if len(records) % 100 == 0:
                    _write_csv(destination / "benchmark_runs.csv", records)
                    _write_paths_jsonl(
                        destination / "benchmark_paths.jsonl",
                        path_records,
                    )
                    LOGGER.info(
                        "planner benchmark progress: %d/%d map-planner runs",
                        len(records),
                        sum(
                            len(environments)
                            * (
                                stochastic_repetitions
                                if planner_registry[name].stochastic
                                else 1
                            )
                            for name in selected_planner_names
                        ),
                    )

    unique_keys = {
        (row["planner"], row["arm_id"], row["map_id"], row["seed"])
        for row in records
    }
    if len(unique_keys) != len(records):
        raise RuntimeError(
            "benchmark produced duplicate planner/arm/map/seed rows"
        )
    expected_records = sum(
        len(environments)
        * (stochastic_repetitions if planner_registry[name].stochastic else 1)
        for name in selected_planner_names
    )
    if len(records) != expected_records:
        raise RuntimeError(
            f"benchmark produced {len(records)} rows; expected {expected_records}"
        )

    summary = summarize_records(records)
    _write_csv(destination / "benchmark_runs.csv", records)
    _write_paths_jsonl(destination / "benchmark_paths.jsonl", path_records)
    _json_dump(destination / "benchmark_summary.json", summary)
    _write_csv(destination / "benchmark_summary.csv", summary["statistics"])

    overrides = {
        "planners": planners,
        "maps_per_class": maps_per_class,
        "time_limit_seconds": time_limit_seconds,
        "max_objective_evaluations": max_objective_evaluations,
        "repetitions": repetitions,
        "afl_artifact_path": (
            None if afl_artifact_path is None else str(Path(afl_artifact_path).resolve())
        ),
        "afl_artifacts": (
            None
            if not afl_artifacts
            else {
                arm_id: str(Path(path).resolve())
                for arm_id, path in sorted(afl_artifacts.items())
            }
        ),
        "evolutionary_afl_artifacts": (
            None
            if not evolutionary_afl_artifacts
            else {
                arm_id: str(Path(path).resolve())
                for arm_id, path in sorted(evolutionary_afl_artifacts.items())
            }
        ),
        "planner_overrides": sorted((planner_overrides or {}).keys()) or None,
    }
    afl_artifact_metadata: list[dict[str, Any]] = []
    for planner_key in artifact_planner_names:
        afl_planner = planner_registry[planner_key]
        artifact = afl_planner.artifact
        usage = getattr(artifact, "llm_usage", None)
        qualifications = list(getattr(artifact, "qualification_results", []))
        afl_artifact_metadata.append(
            {
                "arm_id": afl_planner.arm_id,
                "planner": afl_planner.name,
                "execution_arm": planner_key,
                "artifact_path": str(Path(afl_planner.artifact_path).resolve()),
                "artifact_id": artifact.artifact_id,
                "artifact_schema_version": artifact.schema_version,
                "candidate_source_hash": getattr(
                    artifact, "candidate_source_hash", artifact.solver_hash
                ),
                "approved_source_hash": getattr(
                    artifact, "approved_source_hash", None
                ),
                "solver_hash": artifact.solver_hash,
                "provider": artifact.provider,
                "model": artifact.model,
                "provider_sdk_versions": getattr(
                    artifact, "provider_sdk_versions", {}
                ),
                "research_claim_eligible": artifact.research_claim_eligible,
                "generated_from_split": artifact.generated_from_split,
                "generated_from_map_id": artifact.generated_from_map_id,
                "solver_cli_contract": artifact.solver_cli_contract,
                "generation_success": bool(
                    artifact.provider_calls
                    and all(
                        call.get("status") == "success"
                        for call in artifact.provider_calls
                    )
                ),
                "provider_call_count": len(artifact.provider_calls),
                "provider_retry_count": sum(
                    max(0, int(call.get("retry_count", 0)))
                    for call in artifact.provider_calls
                ),
                "provider_latency_ms": sum(
                    max(0.0, float(call.get("latency_ms", 0.0)))
                    for call in artifact.provider_calls
                ),
                "description_revisions": getattr(
                    artifact, "description_revisions", 0
                ),
                "code_stage_revisions": getattr(
                    artifact, "code_stage_revisions", 0
                ),
                "llm_usage": (
                    None if usage is None else usage.model_dump(mode="json")
                ),
                "qualification_passed": bool(
                    qualifications
                    and all(item.passed for item in qualifications)
                ),
                "human_review": (
                    None
                    if getattr(artifact, "human_review", None) is None
                    else artifact.human_review.model_dump(mode="json")
                ),
                "evolutionary_parameters": getattr(
                    afl_planner, "algorithm_parameters", None
                ),
            }
        )
    unique_generation_artifacts = {
        item["artifact_id"]: item for item in afl_artifact_metadata
    }
    metadata = {
        "benchmark_id": benchmark_config.benchmark_id,
        "run_id": identifier,
        "split": split,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_hash": manifest.manifest_hash,
        "config_hash": config.config_hash,
        "budget": budget.model_dump(mode="json"),
        "selected_planners": selected_planner_names,
        "selected_maps": len(environments),
        "stochastic_repetitions": stochastic_repetitions,
        "expected_records": expected_records,
        "actual_records": len(records),
        "overrides": {
            key: value for key, value in overrides.items() if value is not None
        },
        "algorithm_parameters": {
            "grid_resolution": config.maps.grid_resolution,
            "rrt_step_size": 4.0,
            "rrt_goal_bias": 0.1,
            "rrt_max_samples": 1_000,
            "prm_neighbor_count": 10,
            "prm_sample_count": 180,
            "population_size": benchmark_config.population_size,
            "waypoint_count": benchmark_config.waypoint_count,
            "population_max_generations": 20,
            "aco_label": "aco_acor",
            "afl_uav_iteration_limit": 256,
            "evolutionary_afl_uav": {
                item["execution_arm"]: item["evolutionary_parameters"]
                for item in afl_artifact_metadata
                if item["evolutionary_parameters"] is not None
            },
        },
        "afl_uav_artifacts": afl_artifact_metadata,
        "afl_uav_artifact": (
            afl_artifact_metadata[0]
            if len(afl_artifact_metadata) == 1
            else None
        ),
        "afl_generation_summary": {
            "artifacts": len(unique_generation_artifacts),
            "successful_artifacts": sum(
                bool(item["generation_success"])
                for item in unique_generation_artifacts.values()
            ),
            "generation_success_rate": (
                sum(
                    bool(item["generation_success"])
                    for item in unique_generation_artifacts.values()
                )
                / len(unique_generation_artifacts)
                if unique_generation_artifacts
                else None
            ),
            "logical_calls": sum(
                int((item["llm_usage"] or {}).get("logical_calls", 0))
                for item in unique_generation_artifacts.values()
            ),
            "total_tokens": sum(
                int((item["llm_usage"] or {}).get("total_tokens", 0))
                for item in unique_generation_artifacts.values()
            ),
            "provider_latency_ms": sum(
                float(item["provider_latency_ms"])
                for item in unique_generation_artifacts.values()
            ),
        },
        "objective": config.objective.model_dump(mode="json"),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
        },
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pydantic", "PyYAML")
        },
        "record_key_hash": stable_hash(
            [
                [
                    row["planner"],
                    row["arm_id"],
                    row["map_id"],
                    row["seed"],
                ]
                for row in records
            ]
        ),
        "final_evaluation_authorization": (
            None
            if _final_evaluation_authorization is None
            else {
                "benchmark_id": _final_evaluation_authorization.benchmark_id,
                "opening_id": _final_evaluation_authorization.opening_id,
                "preregistration_id": (
                    _final_evaluation_authorization.preregistration_id
                ),
            }
        ),
    }
    _json_dump(destination / "benchmark_metadata.json", metadata)
    return {
        "run_id": identifier,
        "run_dir": str(destination.resolve()),
        "records": len(records),
        "expected_records": expected_records,
        "selected_maps": len(environments),
        "ranking": summary["ranking"],
    }


__all__ = ["run_planner_benchmark", "summarize_records"]
