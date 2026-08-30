from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "artifacts"
    / "releases"
    / "cross-domain-core-qualification-v1.formal.json"
)


def test_formal_jssp_qualification_receipt_is_canonical_and_frozen() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected_hash = payload.pop("receipt_payload_sha256")
    canonical = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert expected_hash == (
        "ad44cc84d67b8c7343fd25667811c2ac3860dd4a4a8b168b57215a460c9b2f7d"
    )
    assert payload["schema_version"] == (
        "cross-domain-core-qualification-formal-receipt-v1"
    )
    assert payload["configuration"] == {
        "candidates_per_generation": 3,
        "generations": 3,
        "master_seed": 20260823,
        "population_slots": 8,
        "runtime_repetitions": 4,
        "test_calls": 400,
        "test_instances": 41,
        "train_calls": 400,
        "train_instances": 60,
        "validation_calls": 240,
        "validation_instances": 41,
    }
    report = payload["report"]
    assert report["training"]["total_search_calls"] == 24_000
    assert report["training"]["trace_count"] == 24_000
    assert report["training"]["profile_count"] == 8
    assert len(report["evolution"]["candidate_records"]) == 9
    assert not any(
        record["retained"]
        for record in report["evolution"]["candidate_records"]
    )
    assert report["initial_population_ids"] == report["final_population_ids"]
    frozen = report["frozen_test"]
    assert frozen["test_instances"] == 41
    assert frozen["p0"]["feasibility_rate"] == 1.0
    assert frozen["pn"]["feasibility_rate"] == 1.0
    assert frozen["p0"]["mean_best_makespan"] == frozen["pn"]["mean_best_makespan"]
    assert frozen["mean_relative_gain"] == 0.0
    assert frozen["win_rate"] == 0.0
    assert frozen["tie_rate"] == 1.0
    assert all(
        outcome["p0_best_makespan"] == outcome["pn_best_makespan"]
        for outcome in frozen["outcomes"]
    )
    assert payload["test_access"] == {
        "freeze_receipt_id": report["freeze_receipt_id"],
        "opened": True,
        "opened_after_population_freeze": True,
        "used_for_retention": False,
    }
    assert all(
        len(value) == 64
        for key, value in payload["artifacts"].items()
        if key.endswith("_sha256")
    )
