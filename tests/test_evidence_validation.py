from __future__ import annotations

import pytest

from uav_operator_evolution.agents.design_models import (
    DesignHypothesis,
    DiagnosisClaim,
    DiagnosisReport,
)
from uav_operator_evolution.agents.designer_base import OperatorProposal
from uav_operator_evolution.agents.evidence import (
    DesignBudget,
    FailureEvidence,
    OperatorEvidenceBundle,
)
from uav_operator_evolution.agents.proposal_validation import (
    ProposalValidationError,
    ProposalValidator,
)
from uav_operator_evolution.operators.catalog import manual_operator_specs
from uav_operator_evolution.operators.specs import OperatorSpec, primitive_catalog


EVIDENCE_ID = "fail_0123456789abcdef01234567"


def _bundle() -> OperatorEvidenceBundle:
    parent = manual_operator_specs()["segment_shift"]
    failure = FailureEvidence(
        evidence_id=EVIDENCE_ID,
        source_refs=["failure:1"],
        sample_count=12,
        effect_size=-2.0,
        confidence=1.0,
        low_confidence=False,
        operator_id="segment_shift",
        failure_mode="large_cost_increase",
        severity=2.0,
    )
    return OperatorEvidenceBundle(
        problem_summary="Improve segment shifting.",
        parent_specs=[parent],
        parent_profiles=[{"operator_id": "segment_shift", "attempts": 12}],
        failure_modes=[failure],
        existing_operator_names=["segment_shift"],
        allowed_primitives={key: list(values) for key, values in primitive_catalog().items()},
        design_budget=DesignBudget(),
    )


def _proposal(spec: OperatorSpec, *, evidence_id: str = EVIDENCE_ID) -> OperatorProposal:
    diagnosis = DiagnosisReport(
        parent_operator="segment_shift",
        failure_modes=[
            DiagnosisClaim(
                claim="large_cost_increase",
                evidence_ids=[evidence_id],
                confidence=0.9,
                alternative_explanation="The sampled maps may be unusually constrained.",
            )
        ],
    )
    hypothesis = DesignHypothesis(
        hypothesis="Follow a segment shift with local smoothing.",
        target_failure_mode="large_cost_increase",
        expected_mechanism="Remove curvature introduced by a coherent shift.",
        expected_effective_context="Dense maps with high local curvature.",
        possible_side_effects=["Additional runtime"],
        evidence_ids=[evidence_id],
    )
    return OperatorProposal(
        operator_spec=spec,
        design_rationale="The measured failure motivates a bounded smoothing follow-up.",
        evidence_used=["legacy prose remains supported"],
        changes_from_parents=["append smooth_segment"],
        expected_advantages=["lower accepted objective on dense maps"],
        expected_risks=["extra runtime"],
        diagnosis=diagnosis,
        hypothesis=hypothesis,
        used_evidence_ids=[evidence_id],
    )


def _structural_candidate() -> OperatorSpec:
    parent = manual_operator_specs()["segment_shift"]
    payload = parent.model_dump(mode="json")
    payload.update(
        {
            "name": "SegmentShiftThenSmooth",
            "description": "Shift a bounded segment and smooth the result.",
            "parent_operators": ["segment_shift"],
            "transformations": [
                *payload["transformations"],
                {"kind": "smooth_segment", "strength": 0.5, "repeat": 1, "when": None},
            ],
            "fallback_strategy": {"kind": "rollback_on_failure"},
            "expected_mechanism": "shift then smooth",
            "target_failure_modes": ["large_cost_increase"],
        }
    )
    return OperatorSpec.model_validate(payload)


def test_aliases_and_structural_review_are_compatible() -> None:
    proposal = _proposal(_structural_candidate())
    review = ProposalValidator().validate(proposal, _bundle())
    assert proposal.spec is proposal.specification
    assert proposal.operator_spec is proposal.specification
    assert proposal.expected_risks == ["extra runtime"]
    serialized = proposal.model_dump(mode="json", by_alias=True)
    assert "operator_spec" in serialized and "specification" not in serialized
    assert "expected_risks" in serialized and "potential_risks" not in serialized
    assert review.decision == "approve"
    assert review.lineage_relation == "structural_variant"
    assert review.novelty_score == 0.85


def test_unknown_evidence_and_target_mismatch_fail_closed() -> None:
    proposal = _proposal(_structural_candidate(), evidence_id="fail_aaaaaaaaaaaaaaaaaaaaaaaa")
    with pytest.raises(ProposalValidationError, match="unknown evidence IDs"):
        ProposalValidator().validate(proposal, _bundle())

    valid = _proposal(_structural_candidate())
    valid.hypothesis.target_failure_mode = "different_failure"
    with pytest.raises(ProposalValidationError, match="exactly match"):
        ProposalValidator().validate(valid, _bundle())


def test_rename_only_is_rejected_but_parameter_variant_is_low_novelty() -> None:
    parent = manual_operator_specs()["segment_shift"]
    renamed_payload = parent.model_dump(mode="json")
    renamed_payload.update(
        {
            "name": "RenamedSegmentShift",
            "description": "Only renamed.",
            "parent_operators": ["segment_shift"],
        }
    )
    with pytest.raises(ProposalValidationError, match="rename-only"):
        ProposalValidator().validate(
            _proposal(OperatorSpec.model_validate(renamed_payload)), _bundle()
        )

    tuned_payload = dict(renamed_payload)
    tuned_payload["name"] = "TunedSegmentShift"
    tuned_payload["description"] = "Parameter-tuned segment shift."
    tuned_payload["transformations"] = [
        {"kind": "shift_segment", "scale": 7.0, "max_segment_points": 4, "repeat": 1, "when": None}
    ]
    review = ProposalValidator().validate(
        _proposal(OperatorSpec.model_validate(tuned_payload)), _bundle()
    )
    assert review.decision == "approve"
    assert review.lineage_relation == "parameter_variant"
    assert review.novelty_score == 0.35
