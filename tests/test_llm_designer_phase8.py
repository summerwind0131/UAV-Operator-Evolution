from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from uav_operator_evolution.agents.design_models import (
    DesignHypothesis,
    DiagnosisClaim,
    DiagnosisReport,
    OperatorReview,
)
from uav_operator_evolution.agents.designer_base import OperatorProposal
from uav_operator_evolution.agents.evidence import ContextEvidence, OperatorEvidenceBundle
from uav_operator_evolution.agents.llm_designer import (
    LLMDesignError,
    LLMDesignerAdapter,
    LLMDesignerUnavailableError,
)
from uav_operator_evolution.agents.prompts import DESIGNER_V1, DIAGNOSER_V1
from uav_operator_evolution.agents.proposal_validation import ProposalValidationError
from uav_operator_evolution.agents.providers import LLMCallConfig, MockLLMProvider
from uav_operator_evolution.operators.specs import OperatorSpec, primitive_catalog


EVIDENCE_ID = "ctx_" + "a" * 24


def _parent_spec() -> OperatorSpec:
    return OperatorSpec.model_validate(
        {
            "name": "ParentOperator",
            "description": "parent",
            "selection_strategy": {"kind": "select_random_waypoint"},
            "transformations": [{"kind": "perturb_waypoint", "scale": 4.0}],
            "fallback_strategy": {"kind": "rollback_on_failure"},
            "expected_mechanism": "local exploration",
        }
    )


def _bundle() -> OperatorEvidenceBundle:
    evidence = ContextEvidence(
        evidence_id=EVIDENCE_ID,
        source_refs=["profile:1"],
        sample_count=20,
        effect_size=-1.0,
        confidence=0.9,
        low_confidence=False,
        operator_id="ParentOperator",
        classification="failure",
        context={"map_type": "dense"},
        mean_reward=-1.0,
    )
    return OperatorEvidenceBundle(
        problem_summary="repair dense-map failures",
        parent_specs=[_parent_spec()],
        failure_contexts=[evidence],
        existing_operator_names=["ParentOperator"],
        allowed_primitives={key: list(value) for key, value in primitive_catalog().items()},
    )


def _diagnosis(*, evidence_id: str = EVIDENCE_ID, claim: str = "dense failure") -> DiagnosisReport:
    failure = DiagnosisClaim(
        claim=claim,
        evidence_ids=[evidence_id],
        confidence=0.9,
        alternative_explanation="limited sample coverage",
    )
    return DiagnosisReport(parent_operator="ParentOperator", failure_modes=[failure])


def _proposal(diagnosis: DiagnosisReport | None = None) -> OperatorProposal:
    report = diagnosis or _diagnosis()
    hypothesis = DesignHypothesis(
        hypothesis="smoothing after perturbation should reduce dense failures",
        target_failure_mode=report.failure_modes[0].claim,
        expected_mechanism="couple exploration with local smoothing",
        expected_effective_context="dense maps",
        possible_side_effects=["may suppress large exploratory moves"],
        evidence_ids=[EVIDENCE_ID],
    )
    spec = OperatorSpec.model_validate(
        {
            "name": "ParentOperatorSmoothed",
            "description": "bounded structural composite",
            "parent_operators": ["ParentOperator"],
            "selection_strategy": {"kind": "select_random_waypoint"},
            "transformations": [
                {"kind": "perturb_waypoint", "scale": 4.0},
                {"kind": "smooth_segment", "strength": 0.5},
            ],
            "fallback_strategy": {"kind": "rollback_on_failure"},
            "expected_mechanism": "explore then smooth",
            "target_failure_modes": [report.failure_modes[0].claim],
        }
    )
    return OperatorProposal(
        operator_spec=spec,
        design_rationale="directly addresses the diagnosed failure",
        evidence_used=[EVIDENCE_ID],
        target_failure_modes=[report.failure_modes[0].claim],
        changes_from_parents=["add smoothing step"],
        expected_contexts=["dense maps"],
        expected_risks=["reduced exploration"],
        evidence_level="computed",
        diagnosis=report,
        hypothesis=hypothesis,
        expected_advantages=["lower objective in dense maps"],
        used_evidence_ids=[EVIDENCE_ID],
    )


class QueueProvider:
    def __init__(self, outputs: Sequence[Any]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def generate_structured(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return kwargs["output_model"].model_validate(value)


def test_single_call_validates_and_keeps_prompt_audit_metadata() -> None:
    provider = QueueProvider([_proposal()])
    adapter = LLMDesignerAdapter(provider=provider)
    result = adapter.propose_from_evidence(_bundle(), "single_call", LLMCallConfig())
    assert result.spec.name == "ParentOperatorSmoothed"
    assert adapter.last_review is not None
    assert adapter.last_review.decision == "approve"
    assert provider.calls[0]["prompt_version"] == DESIGNER_V1.version
    assert provider.calls[0]["prompt_hash"] == DESIGNER_V1.prompt_hash
    assert provider.calls[0]["output_model"] is OperatorProposal


def test_staged_mode_reuses_the_exact_validated_diagnosis() -> None:
    diagnosis = _diagnosis()
    provider = QueueProvider([diagnosis, _proposal(diagnosis)])
    adapter = LLMDesignerAdapter(provider=provider)
    result = adapter.propose_from_evidence(_bundle(), "staged", LLMCallConfig())
    assert result.diagnosis == diagnosis
    assert [call["prompt_version"] for call in provider.calls] == [
        DIAGNOSER_V1.version,
        DESIGNER_V1.version,
    ]
    assert provider.calls[1]["user_payload"]["diagnosis"] == diagnosis.model_dump(mode="json")


def test_bare_mock_provider_runs_both_adapter_stages_offline() -> None:
    provider = MockLLMProvider()
    adapter = LLMDesignerAdapter(provider=provider)
    proposal = adapter.propose_from_evidence(_bundle(), "staged", LLMCallConfig())
    assert proposal.diagnosis is not None
    assert proposal.used_evidence_ids == [EVIDENCE_ID]
    assert adapter.last_review is not None
    assert adapter.last_review.decision == "approve"
    assert [record.prompt_version for record in provider.call_records] == [
        DIAGNOSER_V1.version,
        DESIGNER_V1.version,
    ]


def test_staged_mode_rejects_a_changed_diagnosis_hash() -> None:
    first = _diagnosis()
    changed = _diagnosis(claim="different failure wording")
    provider = QueueProvider([first, _proposal(changed)])
    adapter = LLMDesignerAdapter(provider=provider)
    with pytest.raises(LLMDesignError, match="diagnosis.*hash"):
        adapter.propose_from_evidence(_bundle(), "staged", LLMCallConfig())


def test_diagnosis_unknown_evidence_stops_before_design_call() -> None:
    invalid = _diagnosis(evidence_id="ctx_" + "b" * 24)
    provider = QueueProvider([invalid, _proposal()])
    adapter = LLMDesignerAdapter(provider=provider)
    with pytest.raises(ProposalValidationError, match="unknown evidence"):
        adapter.propose_from_evidence(_bundle(), "staged", LLMCallConfig())
    assert len(provider.calls) == 1


def test_phase8_path_never_silently_falls_back() -> None:
    with pytest.raises(LLMDesignerUnavailableError, match="explicit structured LLM provider"):
        LLMDesignerAdapter().propose_from_evidence(_bundle())

    provider = QueueProvider([RuntimeError("provider timeout")])
    with pytest.raises(RuntimeError, match="provider timeout"):
        LLMDesignerAdapter(provider=provider).propose_from_evidence(_bundle())


def test_legacy_six_argument_path_still_uses_offline_fallback() -> None:
    adapter = LLMDesignerAdapter()
    proposal = adapter.propose(
        "repair dense maps",
        [_parent_spec()],
        [],
        [],
        [],
        [],
    )
    assert proposal.spec.parent_operators == ["ParentOperator"]
    assert adapter.last_error is not None
    assert "deterministic heuristic designer" in adapter.last_error


def test_llm_review_cannot_override_static_lineage_or_hard_thresholds() -> None:
    bundle = _bundle()
    proposal = _proposal()

    def factory(*, output_model, user_payload):
        assert output_model is OperatorReview
        payload = dict(user_payload["static_review"])
        payload.update(
            {
                "decision": "approve",
                "evidence_alignment_score": 0.2,
                "novelty_score": 1.0,
                "lineage_relation": "parameter_variant",
                "topology_fingerprint": "untrusted",
            }
        )
        return payload

    adapter = LLMDesignerAdapter(provider=MockLLMProvider(factory=factory))
    review = adapter.review_from_evidence(bundle, proposal)
    assert review.decision == "revise"
    assert review.novelty_score == 0.85
    assert review.lineage_relation == "structural_variant"
    assert review.topology_fingerprint != "untrusted"
