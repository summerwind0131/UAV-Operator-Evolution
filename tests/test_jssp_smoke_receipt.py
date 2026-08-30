from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "artifacts"
    / "releases"
    / "cross-domain-core-qualification-v1.smoke.json"
)


def test_registered_jssp_smoke_receipt_is_canonical_and_test_remained_sealed() -> None:
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
    assert payload["schema_version"] == (
        "cross-domain-core-qualification-smoke-receipt-v1"
    )
    assert payload["configuration"] == {
        "candidates_per_generation": 2,
        "fitness_policy": "deterministic-v2",
        "generations": 2,
        "master_seed": 20260823,
        "runtime_repetitions": 2,
        "search_calls": 64,
        "validation_instances": 2,
    }
    assert payload["dataset"]["train_instances"] == 60
    assert payload["dataset"]["validation_instances"] == 41
    assert payload["dataset"]["sealed_test_instances"] == 41
    assert payload["report"]["trace_count"] == 1024
    assert len(payload["report"]["candidate_records"]) == 4
    assert sum(
        record["retained"] for record in payload["report"]["candidate_records"]
    ) == 1
    assert payload["diagnosis"]["attempts"] == 1024
    assert payload["test_accessed"] is False
