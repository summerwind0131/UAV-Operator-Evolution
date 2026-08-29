from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from operator_evolution_core.proposal import (
    CandidateProposalEnvelope,
    DomainCompatibilityError,
    ProposalBudgetDeclaration,
    ensure_domain_compatibility,
    proposal_hash,
)
from uav_operator_evolution.agents.designer_base import OperatorProposal
from uav_operator_evolution.agents.evidence import DesignBudget, OperatorEvidenceBundle
from uav_operator_evolution.agents.proposal_validation import ProposalValidator
from uav_operator_evolution.agents.tools import AgentToolContext, SmokeTestFixture
from uav_operator_evolution.domain import UAV_DOMAIN_ID, UAVDomainKit, UAV_IR_VERSION
from uav_operator_evolution.environment import Environment2D
from uav_operator_evolution.operators.catalog import manual_operator_specs
from uav_operator_evolution.operators.specs import OperatorSpec, primitive_catalog
from uav_operator_evolution.reproducibility import stable_hash


LEGACY_BUNDLE_HASH = (
    "b0b45eb54fa90a1f7585e665b9fbdbe0feb3fcea4597cc4a06bab43d6defed0d"
)
LEGACY_PROPOSAL_HASH = (
    "1b1af3028fddf40da8694157a1f3f1152babde4286291f10883fb97ead6bc533"
)


def _legacy_bundle() -> OperatorEvidenceBundle:
    parent = manual_operator_specs()["segment_shift"]
    return OperatorEvidenceBundle(
        problem_summary="Step 6 legacy hash fixture.",
        parent_specs=[parent],
        parent_profiles=[{"operator_id": "segment_shift", "attempts": 4}],
        existing_operator_names=["segment_shift"],
        allowed_primitives={
            key: list(values) for key, values in primitive_catalog().items()
        },
        design_budget=DesignBudget(),
        limitations=["fixed fixture"],
    )


def _candidate_spec() -> OperatorSpec:
    payload = manual_operator_specs()["segment_shift"].model_dump(mode="json")
    payload.update(
        {
            "name": "Step6Candidate",
            "description": "A fixed candidate for proposal envelope tests.",
            "parent_operators": ["segment_shift"],
            "fallback_strategy": {"kind": "rollback_on_failure"},
        }
    )
    return OperatorSpec.model_validate(payload)


def test_legacy_uav_json_and_hash_projection_is_unchanged() -> None:
    bundle = _legacy_bundle()
    proposal = OperatorProposal(
        operator_spec=manual_operator_specs()["segment_shift"],
        design_rationale="Fixed legacy proposal hash fixture.",
    )
    bundle_payload = bundle.model_dump(mode="json")
    proposal_payload = proposal.model_dump(mode="json", by_alias=True)

    assert bundle.bundle_hash == LEGACY_BUNDLE_HASH
    assert stable_hash(proposal_payload) == LEGACY_PROPOSAL_HASH
    assert bundle.domain_id == proposal.domain_id == UAV_DOMAIN_ID
    assert bundle.ir_version == proposal.ir_version == UAV_IR_VERSION
    assert "domain_id" not in bundle_payload and "ir_version" not in bundle_payload
    assert "domain_id" not in proposal_payload and "ir_version" not in proposal_payload


def test_typed_proposal_envelope_is_content_addressed_and_tamper_evident() -> None:
    proposal = OperatorProposal(
        operator_spec=_candidate_spec(),
        design_rationale="Use a bounded shift and preserve rollback safety.",
        used_evidence_ids=["fail_0123456789abcdef01234567"],
    )
    envelope = proposal.to_envelope(
        "candidate_step6_01",
        {"search_evaluations": 240, "validation_instances": 4},
    )
    payload = envelope.model_dump(mode="json")

    assert envelope.schema_version == "proposal-envelope-v1"
    assert envelope.domain_id == UAV_DOMAIN_ID
    assert envelope.ir_version == UAV_IR_VERSION
    assert isinstance(envelope.payload, OperatorSpec)
    assert envelope.envelope_hash == proposal_hash(
        envelope.model_dump(mode="json", exclude={"envelope_hash"})
    )
    restored = CandidateProposalEnvelope[OperatorSpec].model_validate_json(
        envelope.model_dump_json()
    )
    assert restored == envelope

    tampered = deepcopy(payload)
    tampered["design_rationale"] = "Changed after hashing."
    with pytest.raises(ValidationError, match="envelope_hash"):
        CandidateProposalEnvelope[OperatorSpec].model_validate(tampered)


def test_domain_and_ir_mismatches_fail_closed_before_capability_access() -> None:
    kit = UAVDomainKit()
    envelope = OperatorProposal(
        operator_spec=_candidate_spec(),
        design_rationale="A compatible envelope.",
    ).to_envelope("candidate_step6_02", {"smoke_seeds": 3})

    ensure_domain_compatibility(kit, envelope)
    with pytest.raises(DomainCompatibilityError, match="domain mismatch"):
        ensure_domain_compatibility(
            kit,
            envelope.model_copy(update={"domain_id": "job-shop-scheduling"}),
        )
    with pytest.raises(DomainCompatibilityError, match="IR version mismatch"):
        ensure_domain_compatibility(
            kit,
            envelope.model_copy(update={"ir_version": "uav-v2"}),
        )
    with pytest.raises(DomainCompatibilityError, match="missing"):
        ensure_domain_compatibility(kit, {})
    assert ensure_domain_compatibility(
        kit, {}, allow_legacy_unversioned=True
    ) == (UAV_DOMAIN_ID, UAV_IR_VERSION)

    mismatched_bundle = {
        **_legacy_bundle().model_dump(mode="json"),
        "domain_id": "job-shop-scheduling",
        "ir_version": "jssp-v1",
    }
    with pytest.raises(DomainCompatibilityError, match="domain mismatch"):
        ProposalValidator(kit).validate(
            OperatorProposal(
                operator_spec=_candidate_spec(),
                design_rationale="Mismatch must be rejected.",
            ),
            mismatched_bundle,
        )
    with pytest.raises(DomainCompatibilityError, match="domain mismatch"):
        AgentToolContext(bundle=mismatched_bundle, domain_kit=kit)


def test_uav_domain_kit_owns_ir_catalog_compile_smoke_and_fingerprints() -> None:
    kit = UAVDomainKit()
    spec = _candidate_spec()
    parsed = kit.parse_ir(spec.model_dump(mode="json"))
    compiled = kit.compile(parsed)
    environment = Environment2D(
        map_id="kit-smoke",
        width=20,
        height=20,
        start=(1, 1),
        goal=(19, 19),
        obstacles=[],
    )
    report = kit.smoke(
        parsed,
        SmokeTestFixture(environment, [(1, 1), (8, 9), (19, 19)]),
    )

    assert compiled.name == spec.name
    assert set(kit.capability_usage(spec)).issubset(
        {
            name
            for names in kit.capability_catalog().values()
            for name in names
        }
    )
    assert len(kit.topology_fingerprint(spec)) == 64
    assert len(kit.behavior_fingerprint(spec)) == 64
    assert kit.static_safety_score(spec) == 1.0
    assert report.smoke_passed and report.seeds_tested == 3

    invalid = spec.model_dump(mode="json")
    invalid["transformations"] = [{"kind": "execute_python", "source": "pass"}]
    with pytest.raises(ValidationError):
        kit.parse_ir(invalid)


def test_budget_declaration_rejects_nonfinite_negative_or_boolean_limits() -> None:
    with pytest.raises(ValidationError):
        ProposalBudgetDeclaration(limits={"calls": -1})
    with pytest.raises(ValidationError):
        ProposalBudgetDeclaration(limits={"calls": float("inf")})
    with pytest.raises(ValidationError):
        ProposalBudgetDeclaration(limits={"calls": True})
