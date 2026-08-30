from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
import yaml

from operator_evolution_core.evolution import MechanismTransferPreregistrationV1
from operator_evolution_core.memory import create_mechanism_bank_v1


ROOT = Path(__file__).resolve().parents[1]


def test_mechanism_transfer_v1_preregistration_is_fixed_and_content_addressed() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "mechanism_transfer_v1.yaml").read_text(encoding="utf-8")
    )
    registration = MechanismTransferPreregistrationV1.model_validate(payload)

    assert len(registration.master_seeds) == 10
    assert registration.retrieval_limit == 4
    assert registration.remote_provider_allowed is False
    assert registration.budget.model_dump() == {
        "train_search_calls": 400,
        "validation_search_calls": 240,
        "test_search_calls": 400,
        "generations": 3,
        "candidates_per_generation": 3,
        "population_slots": 8,
    }
    assert len(registration.content_hash) == 64

    with pytest.raises(ValidationError, match="disjoint"):
        MechanismTransferPreregistrationV1.model_validate(
            {**payload, "uav_bank_seeds": [registration.master_seeds[0]]}
        )


def test_mechanism_bank_rejects_empty_or_cross_domain_content() -> None:
    with pytest.raises(ValidationError):
        create_mechanism_bank_v1(
            source_domain_id="uav",
            bank_master_seeds=(1,),
            source_code_commit="4de3d6d",
            records=(),
        )
