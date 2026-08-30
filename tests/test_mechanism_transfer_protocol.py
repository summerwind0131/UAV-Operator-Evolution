from __future__ import annotations

import pytest
from pydantic import ValidationError

from operator_evolution_core.memory import (
    AbstractMechanismContextV1,
    ExpectedMechanismEffectV1,
    MechanismRecordV1,
    create_mechanism_record_v1,
    retrieve_top4_mechanisms_v1,
)


COMMIT = "4de3d6d5a39105d52f365865a952168c41b7284c"
POPULATION = "a" * 64


def _record(
    operator_id: str,
    *,
    context: AbstractMechanismContextV1,
    strength: float,
) -> MechanismRecordV1:
    return create_mechanism_record_v1(
        source_domain_id="source-domain",
        mechanism_tags=("repair", "rollback"),
        trigger_context=context,
        expected_effect=ExpectedMechanismEffectV1(
            feasibility="improve",
            cost="preserve",
            diversity="preserve",
            locality="local",
        ),
        failure_modes=("over-correction",),
        evidence_refs=(f"profile:{operator_id}",),
        evidence_strength=strength,
        evidence_sample_count=32,
        evidence_splits=("train", "validation"),
        bank_run_id="bank-seed-17",
        bank_master_seed=17,
        source_operator_id=operator_id,
        source_code_commit=COMMIT,
        source_population_fingerprint=POPULATION,
    )


def test_mechanism_record_v1_is_stable_tamper_evident_and_has_no_ir_slot() -> None:
    context = AbstractMechanismContextV1(
        constraint_pressure="high",
        stagnation="medium",
        feasibility="mixed",
    )
    record = _record("operator-a", context=context, strength=0.8)

    assert MechanismRecordV1.model_validate_json(record.model_dump_json()) == record
    assert record.mechanism_id == f"mechanism-v1-{record.provenance_hash[:16]}"
    with pytest.raises(ValidationError, match="provenance_hash"):
        MechanismRecordV1.model_validate(
            {**record.model_dump(), "evidence_strength": 0.9}
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        MechanismRecordV1.model_validate(
            {**record.model_dump(), "selector": {"primitive": "foreign-ir"}}
        )
    with pytest.raises(ValidationError):
        create_mechanism_record_v1(
            **{
                **record.model_dump(exclude={"mechanism_id", "provenance_hash"}),
                "evidence_splits": ("train", "test"),
            }
        )


def test_top4_retrieval_uses_context_then_evidence_then_stable_id() -> None:
    target = AbstractMechanismContextV1(
        constraint_pressure="high",
        stagnation="high",
        feasibility="mixed",
    )
    exact = AbstractMechanismContextV1(
        constraint_pressure="high",
        stagnation="high",
        feasibility="mixed",
    )
    partial = AbstractMechanismContextV1(
        constraint_pressure="high",
        stagnation="low",
        feasibility="mixed",
    )
    records = [
        _record("exact-low", context=exact, strength=0.4),
        _record("partial-high", context=partial, strength=1.0),
        _record("exact-high-a", context=exact, strength=0.9),
        _record("exact-high-b", context=exact, strength=0.9),
        _record("partial-low", context=partial, strength=0.2),
    ]

    selected = retrieve_top4_mechanisms_v1(records, target)

    assert len(selected) == 4
    assert selected[0].evidence_strength == 0.9
    assert selected[1].evidence_strength == 0.9
    assert selected[0].mechanism_id < selected[1].mechanism_id
    assert selected[2].source_operator_id == "exact-low"
    assert selected[3].source_operator_id == "partial-high"
