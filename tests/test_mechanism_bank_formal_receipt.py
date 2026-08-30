from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import yaml

from operator_evolution_core.evolution import MechanismTransferPreregistrationV1
from operator_evolution_core.memory import MechanismBankV1


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "artifacts" / "releases" / "mechanism-transfer-v1.bank-formal.json"


def test_formal_mechanism_banks_cover_registered_seeds_and_remain_sealed() -> None:
    document = json.loads(RECEIPT.read_text(encoding="utf-8"))
    payload = document["payload"]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    registration = MechanismTransferPreregistrationV1.model_validate(
        yaml.safe_load(
            (ROOT / "configs" / "mechanism_transfer_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    uav_bank = MechanismBankV1.model_validate(payload["uav_bank"])
    jssp_bank = MechanismBankV1.model_validate(payload["jssp_bank"])

    assert hashlib.sha256(canonical).hexdigest() == document["payload_sha256"]
    assert document["payload_sha256"] == (
        "58b02c41b4d2fc99c2c6a73b9661b38d66f79f64bdc1e15a9365763f7cff3b81"
    )
    assert payload["mode"] == "formal"
    assert payload["preregistration_hash"] == registration.content_hash
    assert payload["test_instances_opened"] is False
    assert payload["remote_provider_calls"] == 0
    assert uav_bank.bank_master_seeds == registration.uav_bank_seeds
    assert jssp_bank.bank_master_seeds == registration.jssp_bank_seeds
    assert len(uav_bank.records) == len(jssp_bank.records) == 32
    assert Counter(record.bank_master_seed for record in uav_bank.records) == {
        seed: 8 for seed in registration.uav_bank_seeds
    }
    assert Counter(record.bank_master_seed for record in jssp_bank.records) == {
        seed: 8 for seed in registration.jssp_bank_seeds
    }
    assert {
        split
        for bank in (uav_bank, jssp_bank)
        for record in bank.records
        for split in record.evidence_splits
    } == {"train", "validation"}

    serialized = json.dumps(
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
        assert forbidden not in serialized
