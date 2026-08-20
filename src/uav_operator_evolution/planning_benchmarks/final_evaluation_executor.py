"""Resumable, arm-isolated executor for the preregistered UAV2D final matrix."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, load_config
from ..reproducibility import stable_hash
from .evolutionary_afl_experiments import EvolutionaryAFLExperimentPlanner
from .final_evaluation_common import (
    ROOT,
    project_relative,
    read_json,
    read_protocol,
    require_hash,
    resolve_project_path,
    sha256_file,
    write_json,
)
from .runner import (
    _FinalEvaluationAuthorization,
    run_planner_benchmark,
)


FINAL_ARM_IDS = (
    "dijkstra",
    "astar",
    "theta_star",
    "rrt",
    "rrt_star",
    "prm",
    "ga",
    "pso",
    "de",
    "aco_acor",
    "frozen_afl_uav",
    "evolutionary_afl_uav_v1",
    "evo_no_rooms_strategy",
    "evo_fixed_length",
)


@dataclass(frozen=True)
class MatrixRunSettings:
    """Execution settings fixed before a matrix starts."""

    split: str
    maps_per_class: int | None
    time_limit_seconds: float
    max_objective_evaluations: int
    stochastic_repetitions: int
    expected_maps: int
    expected_records: int
    mode: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty benchmark table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"invalid JSONL object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def _artifact_context(
    protocol: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    frozen_arm = next(
        arm for arm in protocol["arms"] if arm["arm_id"] == "frozen_afl_uav"
    )
    method_arm = next(
        arm
        for arm in protocol["arms"]
        if arm["arm_id"] == "evolutionary_afl_uav_v1"
    )
    frozen_artifact = resolve_project_path(frozen_arm["artifact"])
    method_artifact = resolve_project_path(method_arm["artifact"])
    require_hash(
        frozen_artifact,
        frozen_arm["expected_artifact_sha256"],
        "frozen AFL-UAV artifact",
    )
    require_hash(
        method_artifact,
        method_arm["expected_artifact_sha256"],
        "Evolutionary AFL-UAV v1 method artifact",
    )
    frozen_payload = read_json(frozen_artifact)
    method_payload = read_json(method_artifact)
    if frozen_payload.get("artifact_id") != frozen_arm["expected_artifact_id"]:
        raise RuntimeError("frozen AFL-UAV artifact ID mismatch")
    if method_payload.get("artifact_id") != method_arm["expected_artifact_id"]:
        raise RuntimeError("Evolutionary AFL-UAV method artifact ID mismatch")
    if method_payload.get("method_id") != "evolutionary-afl-uav-v1":
        raise RuntimeError("unexpected evolutionary method ID")
    seed_reference = method_payload["seed_artifact"]
    if seed_reference["artifact_id"] != frozen_payload["artifact_id"]:
        raise RuntimeError("evolutionary method does not use the frozen AFL artifact")
    seed_path = resolve_project_path(seed_reference["path"])
    if seed_path != frozen_artifact:
        raise RuntimeError("evolutionary seed artifact path differs from frozen arm")
    current_core = resolve_project_path(method_payload["source"]["project_path"])
    frozen_core = method_artifact.parent / method_payload["source"]["frozen_filename"]
    expected_core = method_payload["source"]["sha256"]
    require_hash(current_core, expected_core, "Evolutionary AFL-UAV v1 core")
    require_hash(frozen_core, expected_core, "frozen Evolutionary AFL-UAV v1 copy")
    frozen_solver = frozen_artifact.parent / "frozen_solver.py"
    solver_source_hash = stable_hash(
        {"source": frozen_solver.read_text(encoding="utf-8")}
    )
    if solver_source_hash != frozen_payload["solver_hash"]:
        raise RuntimeError("frozen AFL solver semantic source hash mismatch")
    return frozen_artifact.parent, method_payload


def validate_matrix_definition(protocol: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    arms = protocol.get("arms", [])
    ids = tuple(str(arm.get("arm_id")) for arm in arms)
    if ids != FINAL_ARM_IDS:
        raise RuntimeError("the final matrix is not the preregistered 14-arm order")
    if len(ids) != len(set(ids)):
        raise RuntimeError("final matrix contains duplicate arm IDs")
    total = 120 * sum(int(arm["repetitions"]) for arm in arms)
    if total != 6960:
        raise RuntimeError(f"14-arm matrix implies {total} rows instead of 6,960")
    return _artifact_context(protocol)


def _arm_invocation(
    arm: dict[str, Any],
    *,
    seed_artifact_dir: Path,
) -> tuple[str, dict[str, Any]]:
    arm_id = str(arm["arm_id"])
    planner = str(arm["planner"])
    if arm_id in FINAL_ARM_IDS[:10]:
        return planner, {}
    if arm_id == "frozen_afl_uav":
        return f"afl_uav:{arm_id}", {
            "afl_artifacts": {arm_id: seed_artifact_dir}
        }
    if arm_id == "evolutionary_afl_uav_v1":
        return f"evolutionary_afl_uav:{arm_id}", {
            "evolutionary_afl_artifacts": {arm_id: seed_artifact_dir}
        }
    experiment = EvolutionaryAFLExperimentPlanner(
        seed_artifact_dir,
        arm_id=arm_id,
        variant=arm["experiment_variant"],
        population_size=int(arm["population_size"]),
        archive_size=int(arm["archive_size"]),
        max_generations=int(arm["max_generations"]),
        max_waypoints=64,
        base_iteration_limit=64,
        crossover_probability=0.40,
        extra_mutation_probability=0.30,
    )
    key = f"evolutionary_afl_uav:{arm_id}"
    return key, {"planner_overrides": {key: experiment}}


def _expected_arm_rows(arm: dict[str, Any], settings: MatrixRunSettings) -> int:
    repetitions = 1 if arm["arm_id"] in FINAL_ARM_IDS[:3] else (
        settings.stochastic_repetitions
    )
    return settings.expected_maps * repetitions


def _record_key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row["planner"]),
        str(row["arm_id"]),
        str(row["map_id"]),
        int(row["repetition"]),
        int(row["seed"]),
    )


def _verify_arm_checkpoint(
    arm_dir: Path,
    arm: dict[str, Any],
    expected_rows: int,
    plan_id: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    checkpoint_path = arm_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise RuntimeError(f"partial arm directory has no checkpoint: {arm_dir}")
    checkpoint = read_json(checkpoint_path)
    if checkpoint.get("plan_id") != plan_id:
        raise RuntimeError(f"checkpoint belongs to a different plan: {arm_dir}")
    if checkpoint.get("arm_id") != arm["arm_id"]:
        raise RuntimeError(f"checkpoint arm mismatch: {arm_dir}")
    runs_path = arm_dir / "benchmark_runs.csv"
    paths_path = arm_dir / "benchmark_paths.jsonl"
    require_hash(runs_path, checkpoint["benchmark_runs_sha256"], "arm runs")
    require_hash(paths_path, checkpoint["benchmark_paths_sha256"], "arm paths")
    rows = _read_csv(runs_path)
    paths = _read_jsonl(paths_path)
    if len(rows) != expected_rows or len(paths) != expected_rows:
        raise RuntimeError(
            f"arm {arm['arm_id']} has {len(rows)} runs/{len(paths)} paths; "
            f"expected {expected_rows}"
        )
    if {row["arm_id"] for row in rows} != {arm["arm_id"]}:
        raise RuntimeError(f"arm output label mismatch: {arm['arm_id']}")
    keys = [_record_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"arm output contains duplicate keys: {arm['arm_id']}")
    return rows, paths


def _write_checkpoint(
    arm_dir: Path,
    arm: dict[str, Any],
    *,
    plan_id: str,
    expected_rows: int,
) -> None:
    rows = _read_csv(arm_dir / "benchmark_runs.csv")
    paths = _read_jsonl(arm_dir / "benchmark_paths.jsonl")
    if len(rows) != expected_rows or len(paths) != expected_rows:
        raise RuntimeError(f"completed arm has wrong row count: {arm['arm_id']}")
    write_json(
        arm_dir / "checkpoint.json",
        {
            "schema_version": "uav2d-final-arm-checkpoint-v1",
            "status": "complete",
            "plan_id": plan_id,
            "arm_id": arm["arm_id"],
            "rows": expected_rows,
            "benchmark_runs_sha256": sha256_file(
                arm_dir / "benchmark_runs.csv"
            ),
            "benchmark_paths_sha256": sha256_file(
                arm_dir / "benchmark_paths.jsonl"
            ),
            "benchmark_metadata_sha256": sha256_file(
                arm_dir / "benchmark_metadata.json"
            ),
        },
    )


def _validate_final_schedule(
    rows: list[dict[str, str]],
    schedule_path: Path,
) -> None:
    expected = {_record_key(row) for row in _read_csv(schedule_path)}
    observed_keys = [_record_key(row) for row in rows]
    observed = set(observed_keys)
    if len(observed_keys) != 6960 or len(observed) != 6960:
        raise RuntimeError("final output must contain exactly 6,960 unique keys")
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise RuntimeError(
            f"final output differs from seed schedule: {len(missing)} missing, "
            f"{len(extra)} extra"
        )


def execute_matrix(
    config: ExperimentConfig | str | Path,
    *,
    protocol_path: str | Path,
    destination: str | Path,
    settings: MatrixRunSettings,
    authorization: _FinalEvaluationAuthorization | None = None,
    stop_after_arms: int | None = None,
    protocol_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute or resume the 14 arms, checkpointing only complete arm outputs."""

    if settings.mode not in {"preflight", "final"}:
        raise ValueError("matrix mode must be preflight or final")
    if settings.split == "test" and authorization is None:
        raise PermissionError("test execution requires a validated authorization")
    if settings.mode == "preflight" and settings.split == "test":
        raise PermissionError("preflight must not use the hidden test split")
    config = load_config(config) if isinstance(config, (str, Path)) else config
    protocol = read_protocol(protocol_path)
    seed_artifact_dir, method_payload = validate_matrix_definition(protocol)
    destination = Path(destination)
    plan_payload: dict[str, Any] = {
        "schema_version": "uav2d-final-execution-plan-v1",
        "mode": settings.mode,
        "split": settings.split,
        "maps_per_class": settings.maps_per_class,
        "time_limit_seconds": settings.time_limit_seconds,
        "max_objective_evaluations": settings.max_objective_evaluations,
        "stochastic_repetitions": settings.stochastic_repetitions,
        "expected_maps": settings.expected_maps,
        "expected_records": settings.expected_records,
        "arm_ids": list(FINAL_ARM_IDS),
        "config_hash": config.config_hash,
        "protocol_sha256": sha256_file(protocol_path),
        "method_artifact_id": method_payload["artifact_id"],
        "seed_artifact_id": method_payload["seed_artifact"]["artifact_id"],
        "protocol_identity": protocol_identity or {},
    }
    plan_id = stable_hash(plan_payload)
    plan_payload["plan_id"] = plan_id
    plan_path = destination / "execution_plan.json"
    if destination.exists():
        if not plan_path.is_file():
            raise RuntimeError("existing execution directory has no immutable plan")
        if read_json(plan_path) != plan_payload:
            raise RuntimeError("existing execution plan differs from requested plan")
    else:
        destination.mkdir(parents=True)
        write_json(plan_path, plan_payload)
    completed_receipt_path = destination / "execution_receipt.json"
    if completed_receipt_path.is_file():
        completed = read_json(completed_receipt_path)
        if completed.get("status") != "complete" or completed.get("plan_id") != plan_id:
            raise RuntimeError("existing execution receipt is invalid")
        if int(completed.get("records", -1)) != settings.expected_records:
            raise RuntimeError("existing execution receipt has the wrong row count")
        require_hash(
            destination / "benchmark_runs.csv",
            completed["benchmark_runs_sha256"],
            "completed matrix runs",
        )
        require_hash(
            destination / "benchmark_paths.jsonl",
            completed["benchmark_paths_sha256"],
            "completed matrix paths",
        )
        expected_receipt_id = dict(completed)
        recorded_id = expected_receipt_id.pop("execution_receipt_id", None)
        if stable_hash(expected_receipt_id) != recorded_id:
            raise RuntimeError("existing execution receipt self-hash is invalid")
        return completed
    arms_root = destination / "arms"
    arms_root.mkdir(exist_ok=True)

    resumed_arms: list[str] = []
    newly_completed: list[str] = []
    for arm_index, arm in enumerate(protocol["arms"]):
        arm_id = str(arm["arm_id"])
        arm_dir = arms_root / arm_id
        expected_rows = _expected_arm_rows(arm, settings)
        if arm_dir.exists():
            _verify_arm_checkpoint(
                arm_dir, arm, expected_rows=expected_rows, plan_id=plan_id
            )
            resumed_arms.append(arm_id)
        else:
            planner_key, invocation = _arm_invocation(
                arm, seed_artifact_dir=seed_artifact_dir
            )
            run_planner_benchmark(
                config,
                split=settings.split,
                planners=[planner_key],
                maps_per_class=settings.maps_per_class,
                time_limit_seconds=settings.time_limit_seconds,
                max_objective_evaluations=settings.max_objective_evaluations,
                repetitions=settings.stochastic_repetitions,
                run_id=f"{settings.mode}-{arm_index:02d}-{arm_id}",
                run_dir=arm_dir,
                _final_evaluation_authorization=authorization,
                **invocation,
            )
            _write_checkpoint(
                arm_dir,
                arm,
                plan_id=plan_id,
                expected_rows=expected_rows,
            )
            newly_completed.append(arm_id)
        completed_total = len(resumed_arms) + len(newly_completed)
        if stop_after_arms is not None and completed_total >= stop_after_arms:
            return {
                "status": "checkpointed",
                "plan_id": plan_id,
                "completed_arms": completed_total,
                "resumed_arms": resumed_arms,
                "newly_completed_arms": newly_completed,
                "hidden_test_maps_read": False if settings.mode == "preflight" else True,
            }

    merged_rows: list[dict[str, str]] = []
    merged_paths: list[dict[str, Any]] = []
    for arm in protocol["arms"]:
        arm_rows, arm_paths = _verify_arm_checkpoint(
            arms_root / arm["arm_id"],
            arm,
            expected_rows=_expected_arm_rows(arm, settings),
            plan_id=plan_id,
        )
        merged_rows.extend(arm_rows)
        merged_paths.extend(arm_paths)
    keys = [_record_key(row) for row in merged_rows]
    if len(keys) != settings.expected_records or len(set(keys)) != len(keys):
        raise RuntimeError(
            f"merged matrix has {len(keys)} rows/{len(set(keys))} unique; "
            f"expected {settings.expected_records}"
        )
    if settings.mode == "final":
        schedule_path = resolve_project_path(
            protocol["sealing"]["seed_schedule"]
        )
        _validate_final_schedule(merged_rows, schedule_path)
    _write_csv(destination / "benchmark_runs.csv", merged_rows)
    _write_jsonl(destination / "benchmark_paths.jsonl", merged_paths)
    receipt: dict[str, Any] = {
        "schema_version": "uav2d-final-execution-receipt-v1",
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "mode": settings.mode,
        "plan_id": plan_id,
        "records": len(merged_rows),
        "unique_records": len(set(keys)),
        "arm_ids": list(FINAL_ARM_IDS),
        "resumed_arms": resumed_arms,
        "newly_completed_arms": newly_completed,
        "benchmark_runs_sha256": sha256_file(destination / "benchmark_runs.csv"),
        "benchmark_paths_sha256": sha256_file(
            destination / "benchmark_paths.jsonl"
        ),
        "hidden_test_maps_read": settings.mode == "final",
        "api_calls": 0,
    }
    receipt["execution_receipt_id"] = stable_hash(receipt)
    write_json(destination / "execution_receipt.json", receipt)
    return receipt


__all__ = [
    "FINAL_ARM_IDS",
    "MatrixRunSettings",
    "execute_matrix",
    "validate_matrix_definition",
]
