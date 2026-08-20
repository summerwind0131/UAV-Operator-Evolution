"""Contract tests for the sealed or one-time-consumed Hidden Test-v2 bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from uav_operator_evolution.environment.generator import DatasetManifest


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "benchmarks" / "uav2d-hidden-test-v2"
EXPECTED_MANIFEST_CONTENT_HASH = (
    "ebfb307652363aae4537c0efae8891cbf08fd433b89018b9b9585529408237ac"
)
EXPECTED_PREREGISTRATION_ID = (
    "8d8dcbb568f25a5fd9da74afbe865c5576ce585e25fc37da36662ad1787c3c99"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hidden_test_v2_is_balanced_unique_and_lifecycle_audited() -> None:
    manifest_path = DATASET_ROOT / "manifest.json"
    receipt_path = DATASET_ROOT / "seal_receipt.json"
    preregistration_path = DATASET_ROOT / "preregistration.json"
    live_marker_path = DATASET_ROOT / "SEALED.json"
    archived_marker_path = DATASET_ROOT / "SEALED.preopening.json"
    schedule_path = DATASET_ROOT / "seed_schedule.csv"

    manifest = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    preregistration = json.loads(
        preregistration_path.read_text(encoding="utf-8")
    )
    marker_path = (
        live_marker_path if live_marker_path.exists() else archived_marker_path
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))

    assert manifest.benchmark_id == "uav2d-hidden-test-v2"
    assert manifest.manifest_hash == EXPECTED_MANIFEST_CONTENT_HASH
    assert len(manifest.maps) == 120
    assert {entry.split for entry in manifest.maps} == {"test"}
    assert Counter(entry.difficulty for entry in manifest.maps) == {
        "sparse": 20,
        "dense": 20,
        "corridor": 20,
        "clustered": 20,
        "rooms_maze": 20,
        "mixed": 20,
    }
    assert Counter(
        entry.layout_subtype
        for entry in manifest.maps
        if entry.difficulty == "rooms_maze"
    ) == {"rooms": 10, "maze": 10}
    for field in (
        "content_hash",
        "terminal_hash",
        "obstacle_layout_hash",
        "geometry_hash",
        "seed",
    ):
        values = [getattr(entry, field) for entry in manifest.maps]
        assert all(value is not None for value in values)
        assert len(values) == len(set(values))

    assert receipt["status"] == marker["status"] == "sealed_unrun"
    assert receipt["planner_executions"] == receipt["api_calls"] == 0
    assert receipt["test_results_seen"] is False
    assert receipt["data_audit"]["cross_dataset_duplicate_counts"] == {
        "content_hash": 0,
        "geometry_hash": 0,
        "obstacle_layout_hash": 0,
        "seed": 0,
        "terminal_hash": 0,
    }
    assert receipt["preregistration_id"] == EXPECTED_PREREGISTRATION_ID
    assert preregistration["preregistration_id"] == EXPECTED_PREREGISTRATION_ID
    assert marker["manifest_content_hash"] == EXPECTED_MANIFEST_CONTENT_HASH
    assert marker["seal_receipt_sha256"] == _sha256(receipt_path)
    assert marker["preregistration_sha256"] == _sha256(preregistration_path)
    assert receipt["seed_schedule_sha256"] == _sha256(schedule_path)

    with schedule_path.open(encoding="utf-8", newline="") as handle:
        schedule = list(csv.DictReader(handle))
    assert len(schedule) == receipt["expected_final_records"] == 6960
    assert len(
        {
            (row["planner"], row["arm_id"], row["map_id"], row["seed"])
            for row in schedule
        }
    ) == 6960
    final_results = (
        ROOT / "artifacts" / "planning_benchmarks" / "uav2d-hidden-test-v2-final"
    )
    if live_marker_path.exists():
        assert not final_results.exists()
    else:
        opening = json.loads(
            (DATASET_ROOT / "opening_receipt.json").read_text(encoding="utf-8")
        )
        assert opening["preregistration_id"] == EXPECTED_PREREGISTRATION_ID
        assert (final_results / "execution_receipt.json").is_file()
        assert (final_results / "audit_receipt.json").is_file()
        with (final_results / "benchmark_runs.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == 6960
    assert not list(DATASET_ROOT.rglob("benchmark_*"))


def test_hidden_test_v2_has_no_semantic_overlap_with_uav2d_v1() -> None:
    hidden = DatasetManifest.model_validate_json(
        (DATASET_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    existing = DatasetManifest.model_validate_json(
        (ROOT / "data" / "benchmarks" / "uav2d-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for field in (
        "content_hash",
        "terminal_hash",
        "obstacle_layout_hash",
        "geometry_hash",
        "seed",
    ):
        hidden_values = {getattr(entry, field) for entry in hidden.maps}
        existing_values = {getattr(entry, field) for entry in existing.maps}
        assert hidden_values.isdisjoint(existing_values)
