"""Resumable executor for the preregistered 13-arm Hidden Test-v3 matrix."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, load_config
from ..reproducibility import stable_hash
from .evolutionary_afl_v2 import EvolutionaryAFLUAVV2Planner
from .final_evaluation_v3_common import (
    EXPECTED_RECORDS,
    read_json,
    read_protocol,
    require_hash,
    resolve_project_path,
    sha256_file,
    write_json,
)
from .runner import _FinalEvaluationAuthorization, run_planner_benchmark


FINAL_V3_ARM_IDS = (
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
    "evolutionary_afl_uav_v2",
)


@dataclass(frozen=True)
class MatrixRunSettings:
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
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
            values.append(value)
    return values


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def _record_key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row["planner"]),
        str(row["arm_id"]),
        str(row["map_id"]),
        int(row["repetition"]),
        int(row["seed"]),
    )


def _validate_method_artifact(
    arm: dict[str, Any],
    *,
    expected_method_id: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    artifact_path = resolve_project_path(arm["artifact"])
    require_hash(artifact_path, arm["expected_artifact_sha256"], label)
    payload = read_json(artifact_path)
    if payload.get("artifact_id") != arm["expected_artifact_id"]:
        raise RuntimeError(f"{label} artifact ID mismatch")
    if payload.get("method_id") != expected_method_id:
        raise RuntimeError(f"{label} method ID mismatch")
    current = resolve_project_path(payload["source"]["project_path"])
    frozen = artifact_path.parent / payload["source"]["frozen_filename"]
    expected_source = payload["source"]["sha256"]
    require_hash(current, expected_source, f"{label} current source")
    require_hash(frozen, expected_source, f"{label} frozen source")
    return artifact_path, payload


def validate_matrix_definition(protocol: dict[str, Any]) -> dict[str, Any]:
    arms = protocol.get("arms", [])
    ids = tuple(str(arm.get("arm_id")) for arm in arms)
    if ids != FINAL_V3_ARM_IDS or len(ids) != len(set(ids)):
        raise RuntimeError("final matrix is not the preregistered 13-arm V3 order")
    total = 120 * sum(int(arm["repetitions"]) for arm in arms)
    if total != EXPECTED_RECORDS:
        raise RuntimeError(f"13-arm matrix implies {total}, expected 6,360 rows")

    frozen_arm = next(arm for arm in arms if arm["arm_id"] == "frozen_afl_uav")
    seed_path = resolve_project_path(frozen_arm["artifact"])
    require_hash(
        seed_path,
        frozen_arm["expected_artifact_sha256"],
        "frozen AFL-UAV artifact",
    )
    seed = read_json(seed_path)
    if seed.get("artifact_id") != frozen_arm["expected_artifact_id"]:
        raise RuntimeError("frozen AFL-UAV artifact ID mismatch")

    v1_arm = next(arm for arm in arms if arm["arm_id"] == "evolutionary_afl_uav_v1")
    v2_arm = next(arm for arm in arms if arm["arm_id"] == "evolutionary_afl_uav_v2")
    _, v1 = _validate_method_artifact(
        v1_arm,
        expected_method_id="evolutionary-afl-uav-v1",
        label="Evolutionary AFL-UAV v1",
    )
    _, v2 = _validate_method_artifact(
        v2_arm,
        expected_method_id="evolutionary-afl-uav-v2",
        label="Evolutionary AFL-UAV v2",
    )
    for method, label in ((v1, "v1"), (v2, "v2")):
        reference = method["seed_artifact"]
        if reference["artifact_id"] != seed["artifact_id"]:
            raise RuntimeError(f"{label} does not use the preregistered AFL seed")
        if resolve_project_path(reference["path"]) != seed_path:
            raise RuntimeError(f"{label} AFL seed artifact path mismatch")
    if v2.get("research_claim_eligible") is not True:
        raise RuntimeError("frozen V2 artifact is not research-claim eligible")
    return {
        "seed_artifact_dir": seed_path.parent,
        "seed": seed,
        "v1": v1,
        "v2": v2,
    }


def _arm_invocation(
    arm: dict[str, Any],
    *,
    context: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    arm_id = str(arm["arm_id"])
    planner = str(arm["planner"])
    seed_dir = context["seed_artifact_dir"]
    if arm_id in FINAL_V3_ARM_IDS[:10]:
        return planner, {}
    if arm_id == "frozen_afl_uav":
        return f"afl_uav:{arm_id}", {"afl_artifacts": {arm_id: seed_dir}}
    if arm_id == "evolutionary_afl_uav_v1":
        return f"evolutionary_afl_uav:{arm_id}", {
            "evolutionary_afl_artifacts": {arm_id: seed_dir}
        }
    parameters = dict(context["v2"]["parameters"])
    source_quotas = parameters.pop("source_quotas")
    v2 = EvolutionaryAFLUAVV2Planner(
        seed_dir,
        arm_id=arm_id,
        source_quotas=source_quotas,
        **parameters,
    )
    # Eligibility is provenance metadata granted only by the verified frozen
    # artifact. It does not alter planning behavior.
    v2.research_claim_eligible = True
    key = f"evolutionary_afl_uav:{arm_id}"
    return key, {"planner_overrides": {key: v2}}


def _expected_arm_rows(arm: dict[str, Any], settings: MatrixRunSettings) -> int:
    repetitions = 1 if arm["arm_id"] in FINAL_V3_ARM_IDS[:3] else (
        settings.stochastic_repetitions
    )
    return settings.expected_maps * repetitions


def _verify_arm_checkpoint(
    arm_dir: Path,
    arm: dict[str, Any],
    *,
    expected_rows: int,
    plan_id: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    checkpoint_path = arm_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise RuntimeError(f"partial arm directory lacks checkpoint: {arm_dir}")
    checkpoint = read_json(checkpoint_path)
    if checkpoint.get("plan_id") != plan_id or checkpoint.get("arm_id") != arm["arm_id"]:
        raise RuntimeError(f"checkpoint identity mismatch: {arm_dir}")
    runs_path = arm_dir / "benchmark_runs.csv"
    paths_path = arm_dir / "benchmark_paths.jsonl"
    require_hash(runs_path, checkpoint["benchmark_runs_sha256"], "arm runs")
    require_hash(paths_path, checkpoint["benchmark_paths_sha256"], "arm paths")
    rows, paths = _read_csv(runs_path), _read_jsonl(paths_path)
    if len(rows) != expected_rows or len(paths) != expected_rows:
        raise RuntimeError(f"arm {arm['arm_id']} has an incomplete output")
    if {row["arm_id"] for row in rows} != {arm["arm_id"]}:
        raise RuntimeError(f"arm label mismatch: {arm['arm_id']}")
    keys = [_record_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate arm keys: {arm['arm_id']}")
    return rows, paths


def _write_checkpoint(
    arm_dir: Path,
    arm: dict[str, Any],
    *,
    expected_rows: int,
    plan_id: str,
) -> None:
    rows = _read_csv(arm_dir / "benchmark_runs.csv")
    paths = _read_jsonl(arm_dir / "benchmark_paths.jsonl")
    if len(rows) != expected_rows or len(paths) != expected_rows:
        raise RuntimeError(f"completed arm has wrong row count: {arm['arm_id']}")
    write_json(
        arm_dir / "checkpoint.json",
        {
            "schema_version": "uav2d-final-arm-checkpoint-v3",
            "status": "complete",
            "plan_id": plan_id,
            "arm_id": arm["arm_id"],
            "rows": expected_rows,
            "benchmark_runs_sha256": sha256_file(arm_dir / "benchmark_runs.csv"),
            "benchmark_paths_sha256": sha256_file(arm_dir / "benchmark_paths.jsonl"),
            "benchmark_metadata_sha256": sha256_file(
                arm_dir / "benchmark_metadata.json"
            ),
        },
    )


def _validate_final_schedule(rows: list[dict[str, str]], schedule_path: Path) -> None:
    expected = {_record_key(row) for row in _read_csv(schedule_path)}
    keys = [_record_key(row) for row in rows]
    if len(keys) != EXPECTED_RECORDS or len(set(keys)) != EXPECTED_RECORDS:
        raise RuntimeError("final output must contain exactly 6,360 unique keys")
    missing, extra = expected - set(keys), set(keys) - expected
    if missing or extra:
        raise RuntimeError(
            f"final output differs from schedule: {len(missing)} missing, {len(extra)} extra"
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
    if settings.mode not in {"preflight", "final"}:
        raise ValueError("matrix mode must be preflight or final")
    if settings.split == "test" and authorization is None:
        raise PermissionError("test execution requires validated authorization")
    if settings.mode == "preflight" and settings.split == "test":
        raise PermissionError("preflight cannot use hidden test maps")
    config = load_config(config) if isinstance(config, (str, Path)) else config
    protocol = read_protocol(protocol_path)
    context = validate_matrix_definition(protocol)
    destination = Path(destination)
    plan: dict[str, Any] = {
        "schema_version": "uav2d-final-execution-plan-v3",
        "mode": settings.mode,
        "split": settings.split,
        "maps_per_class": settings.maps_per_class,
        "time_limit_seconds": settings.time_limit_seconds,
        "max_objective_evaluations": settings.max_objective_evaluations,
        "stochastic_repetitions": settings.stochastic_repetitions,
        "expected_maps": settings.expected_maps,
        "expected_records": settings.expected_records,
        "arm_ids": list(FINAL_V3_ARM_IDS),
        "config_hash": config.config_hash,
        "protocol_sha256": sha256_file(protocol_path),
        "v1_artifact_id": context["v1"]["artifact_id"],
        "v2_artifact_id": context["v2"]["artifact_id"],
        "seed_artifact_id": context["seed"]["artifact_id"],
        "protocol_identity": protocol_identity or {},
    }
    plan["plan_id"] = stable_hash(plan)
    plan_path = destination / "execution_plan.json"
    if destination.exists():
        if not plan_path.is_file() or read_json(plan_path) != plan:
            raise RuntimeError("existing execution directory has another immutable plan")
    else:
        destination.mkdir(parents=True)
        write_json(plan_path, plan)
    receipt_path = destination / "execution_receipt.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("plan_id") != plan["plan_id"]:
            raise RuntimeError("existing execution receipt plan mismatch")
        require_hash(
            destination / "benchmark_runs.csv",
            receipt["benchmark_runs_sha256"],
            "completed matrix runs",
        )
        return receipt

    arms_root = destination / "arms"
    arms_root.mkdir(exist_ok=True)
    resumed: list[str] = []
    completed: list[str] = []
    for index, arm in enumerate(protocol["arms"]):
        arm_id = str(arm["arm_id"])
        arm_dir = arms_root / arm_id
        expected_rows = _expected_arm_rows(arm, settings)
        if arm_dir.exists():
            _verify_arm_checkpoint(
                arm_dir,
                arm,
                expected_rows=expected_rows,
                plan_id=plan["plan_id"],
            )
            resumed.append(arm_id)
        else:
            planner_key, invocation = _arm_invocation(arm, context=context)
            run_planner_benchmark(
                config,
                split=settings.split,
                planners=[planner_key],
                maps_per_class=settings.maps_per_class,
                time_limit_seconds=settings.time_limit_seconds,
                max_objective_evaluations=settings.max_objective_evaluations,
                repetitions=settings.stochastic_repetitions,
                run_id=f"{settings.mode}-{index:02d}-{arm_id}",
                run_dir=arm_dir,
                _final_evaluation_authorization=authorization,
                **invocation,
            )
            _write_checkpoint(
                arm_dir,
                arm,
                expected_rows=expected_rows,
                plan_id=plan["plan_id"],
            )
            completed.append(arm_id)
        if stop_after_arms is not None and len(resumed) + len(completed) >= stop_after_arms:
            return {
                "status": "checkpointed",
                "plan_id": plan["plan_id"],
                "completed_arms": len(resumed) + len(completed),
                "resumed_arms": resumed,
                "newly_completed_arms": completed,
                "hidden_test_maps_read": settings.mode == "final",
            }

    merged_rows: list[dict[str, str]] = []
    merged_paths: list[dict[str, Any]] = []
    for arm in protocol["arms"]:
        rows, paths = _verify_arm_checkpoint(
            arms_root / arm["arm_id"],
            arm,
            expected_rows=_expected_arm_rows(arm, settings),
            plan_id=plan["plan_id"],
        )
        merged_rows.extend(rows)
        merged_paths.extend(paths)
    keys = [_record_key(row) for row in merged_rows]
    if len(keys) != settings.expected_records or len(set(keys)) != len(keys):
        raise RuntimeError("merged V3 matrix row count or uniqueness failed")
    if settings.mode == "final":
        _validate_final_schedule(
            merged_rows,
            resolve_project_path(protocol["sealing"]["seed_schedule"]),
        )
    _write_csv(destination / "benchmark_runs.csv", merged_rows)
    _write_jsonl(destination / "benchmark_paths.jsonl", merged_paths)
    receipt: dict[str, Any] = {
        "schema_version": "uav2d-final-execution-receipt-v3",
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "mode": settings.mode,
        "plan_id": plan["plan_id"],
        "records": len(merged_rows),
        "unique_records": len(set(keys)),
        "arm_ids": list(FINAL_V3_ARM_IDS),
        "resumed_arms": resumed,
        "newly_completed_arms": completed,
        "benchmark_runs_sha256": sha256_file(destination / "benchmark_runs.csv"),
        "benchmark_paths_sha256": sha256_file(
            destination / "benchmark_paths.jsonl"
        ),
        "hidden_test_maps_read": settings.mode == "final",
        "api_calls": 0,
    }
    receipt["execution_receipt_id"] = stable_hash(receipt)
    write_json(receipt_path, receipt)
    return receipt


__all__ = [
    "FINAL_V3_ARM_IDS",
    "MatrixRunSettings",
    "execute_matrix",
    "validate_matrix_definition",
]
