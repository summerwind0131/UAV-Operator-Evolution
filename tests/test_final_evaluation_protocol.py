from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from uav_operator_evolution.planning_benchmarks.final_evaluation_audit import (
    audit_results,
)
from uav_operator_evolution.planning_benchmarks.final_evaluation_common import (
    DEFAULT_PREREGISTRATION,
    DEFAULT_PROTOCOL,
    canonical_id,
    validate_preregistration,
)
from uav_operator_evolution.planning_benchmarks.final_evaluation_executor import (
    FINAL_ARM_IDS,
    validate_matrix_definition,
)


PREREGISTRATION_ID = (
    "8d8dcbb568f25a5fd9da74afbe865c5576ce585e25fc37da36662ad1787c3c99"
)
ROOT = Path(__file__).resolve().parents[1]


def _load_authorized_entry():
    path = ROOT / "scripts/run_authorized_hidden_test_v2.py"
    specification = importlib.util.spec_from_file_location(
        "run_authorized_hidden_test_v2", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _planner(arm_id: str) -> str:
    if arm_id == "frozen_afl_uav":
        return "afl_uav"
    if arm_id.startswith("evo_") or arm_id == "evolutionary_afl_uav_v1":
        return "evolutionary_afl_uav"
    return arm_id


def _write_synthetic_matrix(root: Path, *, duplicate: bool = False) -> None:
    rows = []
    paths = []
    for index, arm_id in enumerate(FINAL_ARM_IDS):
        map_id = f"validation-{index:03d}-rooms_maze-synthetic"
        row = {
            "benchmark_id": "surrogate",
            "split": "validation",
            "map_id": map_id,
            "difficulty": "rooms_maze",
            "layout_subtype": "rooms",
            "planner": _planner(arm_id),
            "arm_id": arm_id,
            "execution_arm": arm_id,
            "repetition": 0,
            "seed": 1000 + index,
            "status": "success",
            "feasible": True,
            "research_claim_eligible": True,
            "total_cost": 100.0 + index,
            "path_length": 90.0,
            "collision_penalty": 0.0,
            "smoothness_penalty": 1.0,
            "risk_penalty": 1.0,
            "waypoint_penalty": 1.0,
            "minimum_clearance": 3.0,
            "elapsed_seconds": 0.01,
            "objective_evaluations": 10,
            "collision_checks": 20,
            "node_expansions": 5,
            "waypoint_count": 0,
            "path_hash": None,
        }
        rows.append(row)
        paths.append(
            {
                "planner": row["planner"],
                "arm_id": arm_id,
                "map_id": map_id,
                "repetition": 0,
                "seed": 1000 + index,
                "path": None,
            }
        )
    if duplicate:
        rows[-1] = dict(rows[0])
        paths[-1] = dict(paths[0])
    root.mkdir()
    with (root / "benchmark_runs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "benchmark_paths.jsonl").open("w", encoding="utf-8") as handle:
        for row in paths:
            handle.write(json.dumps(row) + "\n")


def test_base_preregistration_self_hash_and_matrix_are_valid() -> None:
    prereg = validate_preregistration(
        preregistration_path=DEFAULT_PREREGISTRATION,
        protocol_path=DEFAULT_PROTOCOL,
        requested_id=PREREGISTRATION_ID,
    )
    assert canonical_id(prereg, "preregistration_id") == PREREGISTRATION_ID
    seed_dir, method = validate_matrix_definition(
        __import__("yaml").safe_load(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    )
    assert (seed_dir / "artifact.json").is_file()
    assert method["method_id"] == "evolutionary-afl-uav-v1"


def test_authorized_entry_respects_hidden_test_lifecycle() -> None:
    dataset_root = ROOT / "data/benchmarks/uav2d-hidden-test-v2"
    seal_marker = dataset_root / "SEALED.json"
    if seal_marker.exists():
        run_authorized_final = _load_authorized_entry().run_authorized_final
        with pytest.raises(PermissionError, match="SEALED.json exists"):
            run_authorized_final(
                preregistration_id=PREREGISTRATION_ID,
                opening_receipt_path=Path("does-not-matter.json"),
            )
        return

    # Once the one-time evaluation has been authorized and completed, never
    # call the entry point again from a test. Verify the immutable lifecycle
    # evidence instead.
    assert (dataset_root / "SEALED.preopening.json").is_file()
    assert (dataset_root / "opening_receipt.json").is_file()
    final_results = (
        ROOT / "artifacts/planning_benchmarks/uav2d-hidden-test-v2-final"
    )
    assert (final_results / "execution_receipt.json").is_file()
    assert (final_results / "audit_receipt.json").is_file()


def test_preflight_auditor_exercises_all_14_arms(tmp_path: Path) -> None:
    run_dir = tmp_path / "audit"
    _write_synthetic_matrix(run_dir)
    report = audit_results(
        run_dir,
        time_limit_seconds=0.1,
        preflight=True,
        expected_records=14,
    )
    assert report["status"] == "passed"
    assert report["arms"] == 14
    assert len(report["ranking"]) == 14
    assert report["statistical_contract"]["bootstrap_replicates"] == 10_000


def test_auditor_rejects_duplicate_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "duplicate"
    _write_synthetic_matrix(run_dir, duplicate=True)
    with pytest.raises(RuntimeError, match="duplicate"):
        audit_results(
            run_dir,
            time_limit_seconds=0.1,
            preflight=True,
            expected_records=14,
        )


def test_timeout_boundary_is_counted_as_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "timeout"
    _write_synthetic_matrix(run_dir)
    runs_path = run_dir / "benchmark_runs.csv"
    with runs_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["elapsed_seconds"] = "0.1"
    with runs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = audit_results(
        run_dir,
        time_limit_seconds=0.1,
        preflight=True,
        expected_records=14,
    )
    assert report["timeouts_counted_as_failures"] == 1
    assert report["arm_statistics"][FINAL_ARM_IDS[0]]["trusted_feasible_runs"] == 0
