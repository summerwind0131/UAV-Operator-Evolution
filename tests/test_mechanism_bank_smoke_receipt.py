from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "artifacts" / "releases" / "mechanism-transfer-v1.bank-smoke.json"


def test_mechanism_bank_smoke_receipt_is_canonical_sealed_and_ir_free() -> None:
    document = json.loads(RECEIPT.read_text(encoding="utf-8"))
    payload = document["payload"]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert hashlib.sha256(canonical).hexdigest() == document["payload_sha256"]
    assert payload["mode"] == "smoke"
    assert payload["test_instances_opened"] is False
    assert payload["remote_provider_calls"] == 0
    assert len(payload["uav_bank"]["records"]) == 8
    assert len(payload["jssp_bank"]["records"]) == 8
    serialized_records = json.dumps(
        [
            *payload["uav_bank"]["records"],
            *payload["jssp_bank"]["records"],
        ],
        sort_keys=True,
    )
    for forbidden in (
        "selection_strategy",
        "transformations",
        "selector",
        "primitive",
        "uav-v1",
        "jssp-v1",
    ):
        assert forbidden not in serialized_records
