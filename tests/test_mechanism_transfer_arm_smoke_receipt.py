from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from operator_evolution_core.evolution import (
    MechanismTransferPreregistrationV1,
    TransferArmLifecycleV1,
)
from uav_operator_evolution.domain.adapters import UAV_DOMAIN_ID


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT / "artifacts" / "releases" / "mechanism-transfer-v1.arms-smoke.json"
)


def test_bidirectional_three_arm_smoke_receipt_is_complete_and_sealed() -> None:
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
    outcomes = [
        TransferArmLifecycleV1.model_validate(item) for item in payload["outcomes"]
    ]

    assert hashlib.sha256(canonical).hexdigest() == document["payload_sha256"]
    assert document["payload_sha256"] == (
        "f023f6bad255dd175525f8ec62ab1919f7d3c512bac87fabf7e5deab4ff0f92b"
    )
    assert payload["source_commit"] == (
        "12fd8489c66ac414f23ceff86d708dc714c1b1a4"
    )
    assert payload["formal_bank_receipt_sha256"] == (
        "58b02c41b4d2fc99c2c6a73b9661b38d66f79f64bdc1e15a9365763f7cff3b81"
    )
    assert payload["mode"] == "smoke"
    assert payload["preregistration_hash"] == registration.content_hash
    assert payload["master_seeds"] == [registration.master_seeds[0]]
    assert payload["arms"] == list(registration.arms)
    assert payload["retrieval_limit"] == 4
    assert payload["designer"] == "deterministic-heuristic"
    assert payload["test_instances_opened"] is False
    assert payload["remote_provider_calls"] == 0
    assert len(outcomes) == 6
    assert {
        (outcome.target_domain_id, outcome.arm) for outcome in outcomes
    } == {
        (domain, arm)
        for domain in (UAV_DOMAIN_ID, "jssp")
        for arm in registration.arms
    }
    assert all(outcome.search_calls == 64 for outcome in outcomes)
    assert all(outcome.generations == 1 for outcome in outcomes)
    assert all(outcome.candidates_per_generation == 1 for outcome in outcomes)
    assert all(outcome.validation_instances == 1 for outcome in outcomes)
    assert all(outcome.runtime_repetitions == 2 for outcome in outcomes)
    assert all(not outcome.test_instances_opened for outcome in outcomes)
    assert all(outcome.remote_provider_calls == 0 for outcome in outcomes)
    assert all(len(outcome.candidates) == 1 for outcome in outcomes)
    assert all(outcome.candidates[0].smoke_passed for outcome in outcomes)
    assert all(outcome.candidates[0].validation_outcomes == 1 for outcome in outcomes)
    for outcome in outcomes:
        evidence = outcome.candidates[0].evidence
        assert len(evidence.mechanism_ids) == (
            0 if outcome.arm == "scratch" else 4
        )
        if outcome.arm == "scratch":
            assert evidence.source_domain_id is None
        elif outcome.arm == "same-domain":
            assert evidence.source_domain_id == outcome.target_domain_id
        else:
            assert evidence.source_domain_id != outcome.target_domain_id
