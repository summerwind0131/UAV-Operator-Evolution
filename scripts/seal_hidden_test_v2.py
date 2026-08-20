"""Generate, audit, preregister, and seal the hidden uav2d Test-v2 dataset.

This command never imports or invokes a planner benchmark.  A sealed marker is
written only after dataset, cross-dataset deduplication, seed-schedule, and
preregistration checks pass.  Re-running it verifies the existing seal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import socket
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from uav_operator_evolution.config import MapSplitConfig
from uav_operator_evolution.environment import Environment2D
from uav_operator_evolution.environment.generator import (
    DatasetManifest,
    GENERATOR_VERSION,
    MapGenerator,
    MapManifestEntry,
    load_dataset_split,
)
from uav_operator_evolution.reproducibility import derive_seed, stable_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "uav_hidden_test_v2.yaml"
SEMANTIC_HASH_FIELDS = (
    "content_hash",
    "terminal_hash",
    "obstacle_layout_hash",
    "geometry_hash",
    "seed",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HiddenDatasetSpec(StrictModel):
    output_dir: Path
    split: str
    count: int = Field(ge=1)
    difficulties: list[str]
    width: float = Field(gt=10)
    height: float = Field(gt=10)
    safety_distance: float = Field(ge=0)
    grid_resolution: float = Field(gt=0)
    generation_attempts: int = Field(ge=1)
    rooms_maze_balance: dict[str, int]
    minimum_terminal_distance_diagonal_ratio: float = Field(gt=0, le=1)
    deduplicate_against: list[Path]
    required_cross_dataset_unique_fields: list[str]


class HiddenBudgetSpec(StrictModel):
    time_limit_seconds: float = Field(gt=0)
    max_objective_evaluations: int = Field(ge=1)
    stochastic_repetitions: int = Field(ge=1)
    deterministic_repetitions: int = Field(ge=1)
    memory_limit: int | None = None


class ArmSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    arm_id: str
    planner: str
    family: str
    repetitions: int = Field(ge=1)


class SealingSpec(StrictModel):
    status: str
    seal_marker: Path
    preregistration_receipt: Path
    seal_receipt: Path
    seed_schedule: Path
    forbidden_results_directory: Path
    api_calls_required: int
    test_maps_may_be_used_for_tuning: bool
    opening_policy: str


class HiddenProtocol(StrictModel):
    schema_version: str
    benchmark_id: str
    parent_benchmark_id: str
    master_seed: int = Field(ge=0)
    generator_version: str
    dataset: HiddenDatasetSpec
    budget: HiddenBudgetSpec
    shared_contract: dict[str, Any]
    arms: list[ArmSpec]
    expected_execution_matrix: dict[str, Any]
    hypotheses: dict[str, Any]
    analysis: dict[str, Any]
    frozen_implementation_hashes: dict[str, dict[str, str]]
    sealing: SealingSpec


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _resolve(relative: str | Path) -> Path:
    path = (ROOT / Path(relative)).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise ValueError(f"hidden-test path escapes project root: {relative}")
    return path


def _load_protocol(config_path: Path) -> tuple[HiddenProtocol, dict[str, Any]]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    protocol = HiddenProtocol.model_validate(raw)
    if protocol.schema_version != "uav2d-hidden-test-protocol-v1":
        raise ValueError("unsupported hidden-test protocol schema")
    if protocol.generator_version != GENERATOR_VERSION:
        raise ValueError(
            f"generator version mismatch: protocol={protocol.generator_version}, "
            f"runtime={GENERATOR_VERSION}"
        )
    if protocol.dataset.split != "test":
        raise ValueError("hidden dataset must use split=test")
    if protocol.dataset.count != 120:
        raise ValueError("uav2d-hidden-test-v2 must contain exactly 120 maps")
    if protocol.dataset.difficulties != [
        "sparse",
        "dense",
        "corridor",
        "clustered",
        "rooms_maze",
        "mixed",
    ]:
        raise ValueError("hidden-test difficulty order is not the preregistered six-class order")
    if set(protocol.dataset.required_cross_dataset_unique_fields) != set(
        SEMANTIC_HASH_FIELDS
    ):
        raise ValueError("cross-dataset uniqueness fields do not match the sealed contract")
    if protocol.sealing.status != "sealed_unrun":
        raise ValueError("hidden-test protocol must start sealed_unrun")
    if protocol.sealing.api_calls_required != 0:
        raise ValueError("hidden-test sealing must require zero API calls")
    if protocol.sealing.test_maps_may_be_used_for_tuning:
        raise ValueError("hidden test maps may not be used for tuning")
    return protocol, raw


def _verify_frozen_inputs(protocol: HiddenProtocol) -> dict[str, str]:
    verified: dict[str, str] = {}
    for name, item in protocol.frozen_implementation_hashes.items():
        path = _resolve(item["path"])
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(
                f"frozen implementation changed for {name}: expected "
                f"{item['sha256']}, got {actual}"
            )
        verified[str(path.relative_to(ROOT)).replace("\\", "/")] = actual

    for arm in protocol.arms:
        extra = arm.model_extra or {}
        artifact_value = extra.get("artifact")
        expected_sha = extra.get("expected_artifact_sha256")
        expected_id = extra.get("expected_artifact_id")
        if not artifact_value:
            continue
        artifact_path = _resolve(str(artifact_value))
        actual_sha = _sha256(artifact_path)
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(f"frozen artifact changed for {arm.arm_id}")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if expected_id and artifact.get("artifact_id") != expected_id:
            raise RuntimeError(f"frozen artifact id changed for {arm.arm_id}")
        verified[str(artifact_path.relative_to(ROOT)).replace("\\", "/")] = actual_sha
    return dict(sorted(verified.items()))


def _comparison_hashes(protocol: HiddenProtocol) -> tuple[list[DatasetManifest], dict[str, str]]:
    manifests: list[DatasetManifest] = []
    file_hashes: dict[str, str] = {}
    for relative in protocol.dataset.deduplicate_against:
        path = _resolve(relative)
        manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        file_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = _sha256(path)
    return manifests, dict(sorted(file_hashes.items()))


def _audit_maps(
    protocol: HiddenProtocol,
    environments: list[Environment2D],
    comparison_manifests: list[DatasetManifest],
) -> dict[str, Any]:
    errors: list[str] = []
    expected_per_class = protocol.dataset.count // len(protocol.dataset.difficulties)
    class_balance = Counter(environment.difficulty for environment in environments)
    expected_balance = {
        difficulty: expected_per_class for difficulty in protocol.dataset.difficulties
    }
    if dict(class_balance) != expected_balance:
        errors.append(f"difficulty balance mismatch: {dict(class_balance)}")
    rooms_balance = Counter(
        environment.layout_subtype
        for environment in environments
        if environment.difficulty == "rooms_maze"
    )
    if dict(rooms_balance) != protocol.dataset.rooms_maze_balance:
        errors.append(f"rooms/maze balance mismatch: {dict(rooms_balance)}")

    for environment in environments:
        if environment.width != protocol.dataset.width or environment.height != protocol.dataset.height:
            errors.append(f"map dimensions mismatch: {environment.map_id}")
        if environment.safety_distance != protocol.dataset.safety_distance:
            errors.append(f"safety distance mismatch: {environment.map_id}")
        if math.dist(environment.start, environment.goal) < (
            protocol.dataset.minimum_terminal_distance_diagonal_ratio
            * environment.diagonal
        ):
            errors.append(f"terminal distance too short: {environment.map_id}")
        if not environment.point_is_collision_free(environment.start):
            errors.append(f"unsafe start: {environment.map_id}")
        if not environment.point_is_collision_free(environment.goal):
            errors.append(f"unsafe goal: {environment.map_id}")

    new_values: dict[str, list[Any]] = {
        field: [getattr(environment, field) for environment in environments]
        for field in SEMANTIC_HASH_FIELDS
    }
    within_duplicates: dict[str, int] = {}
    cross_duplicates: dict[str, int] = {}
    for field, values in new_values.items():
        duplicate_count = len(values) - len(set(values))
        within_duplicates[field] = duplicate_count
        if duplicate_count:
            errors.append(f"within-hidden-test duplicate {field}: {duplicate_count}")
        comparison_values = {
            getattr(entry, field)
            for manifest in comparison_manifests
            for entry in manifest.maps
            if getattr(entry, field) is not None
        }
        overlap = set(values).intersection(comparison_values)
        cross_duplicates[field] = len(overlap)
        if overlap:
            errors.append(f"cross-dataset duplicate {field}: {len(overlap)}")

    if errors:
        raise RuntimeError("hidden-test dataset audit failed: " + "; ".join(errors))
    return {
        "status": "passed",
        "maps": len(environments),
        "difficulty_balance": dict(sorted(class_balance.items())),
        "rooms_maze_balance": dict(sorted(rooms_balance.items())),
        "connected_maps": len(environments),
        "safe_terminal_maps": len(environments),
        "minimum_terminal_distance_ratio": protocol.dataset.minimum_terminal_distance_diagonal_ratio,
        "within_dataset_duplicate_counts": within_duplicates,
        "cross_dataset_duplicate_counts": cross_duplicates,
    }


def _write_seed_schedule(
    path: Path,
    protocol: HiddenProtocol,
    environments: list[Environment2D],
) -> tuple[int, str]:
    rows: list[dict[str, Any]] = []
    record_keys: list[tuple[str, str, str, int]] = []
    for environment in environments:
        for arm in protocol.arms:
            for repetition in range(arm.repetitions):
                seed = derive_seed(
                    protocol.master_seed,
                    "planner-benchmark",
                    "test",
                    environment.map_id,
                    repetition,
                )
                rows.append(
                    {
                        "planner": arm.planner,
                        "arm_id": arm.arm_id,
                        "map_id": environment.map_id,
                        "repetition": repetition,
                        "seed": seed,
                    }
                )
                record_keys.append((arm.planner, arm.arm_id, environment.map_id, seed))
    if len(record_keys) != len(set(record_keys)):
        raise RuntimeError("preregistered execution record keys are not unique")
    expected = int(protocol.expected_execution_matrix["total_records"])
    if len(rows) != expected:
        raise RuntimeError(f"seed schedule has {len(rows)} rows, expected {expected}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("planner", "arm_id", "map_id", "repetition", "seed"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), _canonical_hash(record_keys)


def _map_file_hashes(root: Path, manifest: DatasetManifest) -> dict[str, str]:
    return {
        entry.relative_path: _sha256(root / entry.relative_path)
        for entry in sorted(manifest.maps, key=lambda item: item.relative_path)
    }


def _build_preregistration(
    protocol: HiddenProtocol,
    raw_protocol: dict[str, Any],
    config_path: Path,
    manifest: DatasetManifest,
    manifest_path: Path,
    manifest_sha256: str,
    seed_schedule_path: Path,
    seed_schedule_sha256: str,
    schedule_rows: int,
    record_key_hash: str,
    verified_inputs: dict[str, str],
    comparison_hashes: dict[str, str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "uav2d-final-evaluation-preregistration-v1",
        "status": "preregistered_sealed_unrun",
        "preregistered_at_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": protocol.benchmark_id,
        "parent_benchmark_id": protocol.parent_benchmark_id,
        "protocol_config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "protocol_config_sha256": _sha256(config_path),
        "dataset": {
            "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": manifest_sha256,
            "manifest_content_hash": manifest.manifest_hash,
            "generator_version": manifest.generator_version,
            "master_seed": manifest.master_seed,
            "maps": len(manifest.maps),
            "split": "test",
        },
        "arms": raw_protocol["arms"],
        "budget": raw_protocol["budget"],
        "shared_contract": raw_protocol["shared_contract"],
        "expected_execution_matrix": raw_protocol["expected_execution_matrix"],
        "hypotheses": raw_protocol["hypotheses"],
        "analysis": raw_protocol["analysis"],
        "seed_schedule": {
            "path": str(seed_schedule_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": seed_schedule_sha256,
            "rows": schedule_rows,
            "record_key_hash": record_key_hash,
        },
        "frozen_input_hashes": verified_inputs,
        "comparison_manifest_hashes": comparison_hashes,
        "commitments": {
            "planner_execution_before_opening": False,
            "api_calls_for_dataset_or_execution": 0,
            "test_results_seen": False,
            "post_result_parameter_changes_allowed": False,
            "post_result_metric_changes_allowed": False,
            "test_maps_allowed_for_tuning": False,
            "missing_or_timeout_records_may_be_silently_removed": False,
            "opening_requires_explicit_user_authorization": True,
        },
    }
    payload["preregistration_id"] = _canonical_hash(payload)
    return payload


def _verify_existing(protocol: HiddenProtocol, config_path: Path) -> dict[str, Any]:
    root = _resolve(protocol.dataset.output_dir)
    marker_path = _resolve(protocol.sealing.seal_marker)
    receipt_path = _resolve(protocol.sealing.seal_receipt)
    preregistration_path = _resolve(protocol.sealing.preregistration_receipt)
    schedule_path = _resolve(protocol.sealing.seed_schedule)
    if not marker_path.is_file() or not receipt_path.is_file():
        raise RuntimeError("hidden-test directory exists without a complete seal")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if marker.get("status") != "sealed_unrun" or receipt.get("status") != "sealed_unrun":
        raise RuntimeError("hidden-test seal status changed")
    if marker.get("seal_receipt_sha256") != _sha256(receipt_path):
        raise RuntimeError("seal receipt hash does not match SEALED marker")
    if marker.get("preregistration_sha256") != _sha256(preregistration_path):
        raise RuntimeError("preregistration hash does not match SEALED marker")
    if receipt["protocol_config_sha256"] != _sha256(config_path):
        raise RuntimeError("hidden-test protocol config changed after sealing")
    if receipt["seed_schedule_sha256"] != _sha256(schedule_path):
        raise RuntimeError("hidden-test seed schedule changed after sealing")
    for relative, expected in receipt["comparison_manifest_hashes"].items():
        if _sha256(_resolve(relative)) != expected:
            raise RuntimeError(f"comparison manifest changed after sealing: {relative}")
    if _verify_frozen_inputs(protocol) != receipt["frozen_input_hashes"]:
        raise RuntimeError("a preregistered frozen implementation input changed")

    manifest_path = root / "manifest.json"
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if receipt["manifest_sha256"] != _sha256(manifest_path):
        raise RuntimeError("hidden-test manifest changed after sealing")
    if receipt["manifest_content_hash"] != manifest.manifest_hash:
        raise RuntimeError("hidden-test manifest content hash changed")
    environments = load_dataset_split(manifest_path, "test")
    actual_file_hashes = _map_file_hashes(root, manifest)
    if actual_file_hashes != receipt["map_file_sha256"]:
        raise RuntimeError("one or more hidden map files changed after sealing")
    if _canonical_hash(actual_file_hashes) != receipt["map_file_hash_root"]:
        raise RuntimeError("hidden map file hash root changed")
    if _resolve(protocol.sealing.forbidden_results_directory).exists():
        raise RuntimeError("forbidden hidden-test results directory exists")
    return {
        "status": "verified_sealed_unrun",
        "benchmark_id": protocol.benchmark_id,
        "maps": len(environments),
        "manifest_content_hash": manifest.manifest_hash,
        "preregistration_id": receipt["preregistration_id"],
        "seal_id": receipt["seal_id"],
        "planner_executions": 0,
        "api_calls": 0,
    }


def seal(config_path: Path) -> dict[str, Any]:
    protocol, raw_protocol = _load_protocol(config_path)
    output_root = _resolve(protocol.dataset.output_dir)
    if output_root.exists():
        return _verify_existing(protocol, config_path)
    forbidden_results = _resolve(protocol.sealing.forbidden_results_directory)
    if forbidden_results.exists():
        raise RuntimeError(
            "refusing to seal because the preregistered final-results directory already exists"
        )
    verified_inputs = _verify_frozen_inputs(protocol)
    comparison_manifests, comparison_hashes = _comparison_hashes(protocol)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="uav2d-hidden-test-v2-", dir=output_root.parent
    ) as temporary:
        staging = Path(temporary)
        generator = MapGenerator(
            protocol.master_seed,
            grid_resolution=protocol.dataset.grid_resolution,
            generation_attempts=protocol.dataset.generation_attempts,
        )
        split_config = MapSplitConfig(
            count=protocol.dataset.count,
            difficulties=protocol.dataset.difficulties,
            width=protocol.dataset.width,
            height=protocol.dataset.height,
            safety_distance=protocol.dataset.safety_distance,
        )
        environments = generator.generate_split("test", split_config)
        data_audit = _audit_maps(protocol, environments, comparison_manifests)

        entries: list[MapManifestEntry] = []
        for environment in environments:
            relative = Path("test") / f"{environment.map_id}.json"
            environment.save_json(staging / relative)
            entries.append(
                MapManifestEntry(
                    map_id=environment.map_id,
                    split="test",
                    difficulty=environment.difficulty,
                    seed=environment.seed,
                    relative_path=relative.as_posix(),
                    content_hash=environment.content_hash,
                    layout_subtype=environment.layout_subtype,
                    terminal_hash=environment.terminal_hash,
                    obstacle_layout_hash=environment.obstacle_layout_hash,
                    geometry_hash=environment.geometry_hash,
                )
            )
        manifest = DatasetManifest(
            master_seed=protocol.master_seed,
            config_hash=_sha256(config_path),
            benchmark_id=protocol.benchmark_id,
            maps=entries,
        )
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest.model_dump(mode="json"))
        seed_schedule_path = staging / "seed_schedule.csv"
        schedule_rows, record_key_hash = _write_seed_schedule(
            seed_schedule_path, protocol, environments
        )

        # Build receipts using their final project-relative paths, even though
        # files are still in the atomic staging directory.
        final_manifest_path = output_root / "manifest.json"
        final_seed_schedule_path = output_root / "seed_schedule.csv"
        preregistration = _build_preregistration(
            protocol,
            raw_protocol,
            config_path,
            manifest,
            final_manifest_path,
            _sha256(manifest_path),
            final_seed_schedule_path,
            _sha256(seed_schedule_path),
            schedule_rows,
            record_key_hash,
            verified_inputs,
            comparison_hashes,
        )
        preregistration_without_id = dict(preregistration)
        preregistration_without_id.pop("preregistration_id", None)
        preregistration["preregistration_id"] = _canonical_hash(preregistration_without_id)
        preregistration_path = staging / "preregistration.json"
        _write_json(preregistration_path, preregistration)

        map_hashes = _map_file_hashes(staging, manifest)
        receipt: dict[str, Any] = {
            "schema_version": "uav2d-hidden-test-seal-v1",
            "status": "sealed_unrun",
            "sealed_at_utc": datetime.now(UTC).isoformat(),
            "benchmark_id": protocol.benchmark_id,
            "protocol_config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "protocol_config_sha256": _sha256(config_path),
            "manifest_sha256": _sha256(manifest_path),
            "manifest_content_hash": manifest.manifest_hash,
            "map_file_sha256": map_hashes,
            "map_file_hash_root": _canonical_hash(map_hashes),
            "seed_schedule_sha256": _sha256(seed_schedule_path),
            "seed_schedule_rows": schedule_rows,
            "expected_final_records": int(
                protocol.expected_execution_matrix["total_records"]
            ),
            "record_key_hash": record_key_hash,
            "preregistration_sha256": _sha256(preregistration_path),
            "preregistration_id": preregistration["preregistration_id"],
            "data_audit": data_audit,
            "comparison_manifest_hashes": comparison_hashes,
            "frozen_input_hashes": verified_inputs,
            "sealer_source_sha256": _sha256(Path(__file__)),
            "dependencies": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "pydantic", "PyYAML")
            },
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
            },
            "planner_executions": 0,
            "api_calls": 0,
            "test_results_seen": False,
            "forbidden_results_directory_absent": True,
            "opening_policy": protocol.sealing.opening_policy,
        }
        receipt["seal_id"] = _canonical_hash(receipt)
        receipt_path = staging / "seal_receipt.json"
        _write_json(receipt_path, receipt)
        marker = {
            "schema_version": "uav2d-hidden-test-lock-v1",
            "status": "sealed_unrun",
            "benchmark_id": protocol.benchmark_id,
            "manifest_content_hash": manifest.manifest_hash,
            "preregistration_id": preregistration["preregistration_id"],
            "preregistration_sha256": _sha256(preregistration_path),
            "seal_id": receipt["seal_id"],
            "seal_receipt_sha256": _sha256(receipt_path),
            "opening_policy": protocol.sealing.opening_policy,
            "planner_execution_guard": "run_planner_benchmark rejects while this marker exists",
        }
        _write_json(staging / "SEALED.json", marker)

        # A same-volume rename publishes the fully validated bundle at once.
        staging.replace(output_root)

    return _verify_existing(protocol, config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(seal(args.config.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
