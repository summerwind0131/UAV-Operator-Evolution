from __future__ import annotations

from pathlib import Path

import pytest

from jssp_operator_evolution.data import build_jssp_splits
from jssp_operator_evolution.transfer import (
    JSSPMechanismBankConfig,
    build_jssp_mechanism_bank,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "jssp" / "orlib" / "jobshop1.txt"


def test_jssp_bank_uses_train_validation_only_and_emits_no_domain_ir() -> None:
    splits = build_jssp_splits(RAW)
    bank = build_jssp_mechanism_bank(
        splits,
        bank_master_seeds=(2026090301,),
        source_code_commit="4bb54dc067a592a766e649b51751e91cd8c6d888",
        config=JSSPMechanismBankConfig(
            train_calls=8,
            validation_calls=8,
            train_instances=1,
            validation_instances=1,
        ),
    )

    assert bank.source_domain_id == "jssp"
    assert len(bank.records) == 8
    assert {record.evidence_splits for record in bank.records} == {
        ("train", "validation")
    }
    serialized = bank.model_dump_json()
    for forbidden in ("selector", "transform", "primitive", "jssp-v1"):
        assert forbidden not in serialized
    with pytest.raises(PermissionError):
        splits.open_test()
