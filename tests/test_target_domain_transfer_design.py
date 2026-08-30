from __future__ import annotations

from operator_evolution_core.memory import (
    AbstractMechanismContextV1,
    ExpectedMechanismEffectV1,
    create_mechanism_record_v1,
)
from jssp_operator_evolution.operators import JSSPOperatorCompiler
from jssp_operator_evolution.transfer_design import (
    design_jssp_operator_from_mechanisms,
)
from uav_operator_evolution.operators.compiler import OperatorCompiler
from uav_operator_evolution.transfer_design import design_uav_operator_from_mechanisms


def _foreign_record(source_domain: str, operator_id: str):
    return create_mechanism_record_v1(
        source_domain_id=source_domain,
        mechanism_tags=("repair", "rollback"),
        trigger_context=AbstractMechanismContextV1(
            constraint_pressure="high",
            stagnation="medium",
            feasibility="mixed",
        ),
        expected_effect=ExpectedMechanismEffectV1(
            feasibility="improve",
            cost="preserve",
            diversity="preserve",
            locality="local",
        ),
        failure_modes=("low-yield",),
        evidence_refs=("profile:foreign",),
        evidence_strength=0.9,
        evidence_sample_count=64,
        evidence_splits=("train", "validation"),
        bank_run_id="foreign-bank",
        bank_master_seed=99,
        source_operator_id=operator_id,
        source_code_commit="4bb54dc067a592a766e649b51751e91cd8c6d888",
        source_population_fingerprint="b" * 64,
    )


def test_uav_record_is_redesigned_and_compiled_only_as_jssp_v1() -> None:
    record = _foreign_record("uav-path-planning-2d", "do-not-copy-uav-selector")
    spec = design_jssp_operator_from_mechanisms(
        [record], master_seed=7, candidate_index=0
    )

    compiled = JSSPOperatorCompiler().compile(spec)
    assert compiled.operator_id == spec.operator_id
    serialized = spec.model_dump_json()
    assert "do-not-copy-uav-selector" not in serialized
    assert record.mechanism_id in spec.parent_ids


def test_jssp_record_is_redesigned_and_compiled_only_as_uav_v1() -> None:
    record = _foreign_record("jssp", "do-not-copy-jssp-selector")
    spec = design_uav_operator_from_mechanisms(
        [record], master_seed=8, candidate_index=0
    )

    compiled = OperatorCompiler().compile(spec)
    assert compiled.name == spec.name
    serialized = spec.model_dump_json()
    assert "do-not-copy-jssp-selector" not in serialized
    assert record.mechanism_id in spec.parent_operators


def test_scratch_design_uses_the_same_target_domain_gates() -> None:
    jssp = design_jssp_operator_from_mechanisms([], master_seed=1, candidate_index=0)
    uav = design_uav_operator_from_mechanisms([], master_seed=1, candidate_index=0)

    assert JSSPOperatorCompiler().compile(jssp).operator_id == jssp.operator_id
    assert OperatorCompiler().compile(uav).name == uav.name
