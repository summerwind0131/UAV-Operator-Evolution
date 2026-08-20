from __future__ import annotations

from typing import Any

from uav_operator_evolution.agents.design_models import (
    CandidateStatus,
    DesignHypothesis,
    DiagnosisClaim,
    DiagnosisReport,
)
from uav_operator_evolution.agents.designer_base import OperatorProposal
from uav_operator_evolution.agents.evidence import FailureEvidence, OperatorEvidenceBundle
from uav_operator_evolution.agents.providers import MockLLMProvider
from uav_operator_evolution.agents.research_agent import DeterministicMockResearchAgent
from uav_operator_evolution.agents.tools import AgentToolContext, SmokeTestFixture
from uav_operator_evolution.environment import Environment2D
from uav_operator_evolution.operators.compiler import OperatorCompiler
from uav_operator_evolution.operators.specs import OperatorSpec, primitive_catalog


EVIDENCE_ID = "fail_" + "c" * 24


def _parent() -> OperatorSpec:
    return OperatorSpec.model_validate(
        {
            "name": "ParentOperator",
            "description": "bounded parent",
            "selection_strategy": {"kind": "select_random_waypoint"},
            "transformations": [{"kind": "perturb_waypoint", "scale": 2.0}],
            "fallback_strategy": {"kind": "rollback_on_failure"},
            "expected_mechanism": "local exploration",
        }
    )


def _bundle() -> OperatorEvidenceBundle:
    parent = _parent()
    return OperatorEvidenceBundle(
        problem_summary="repair dense-map stagnation",
        parent_specs=[parent],
        parent_profiles=[{"operator_id": parent.name, "sample_count": 24}],
        failure_modes=[
            FailureEvidence(
                evidence_id=EVIDENCE_ID,
                source_refs=["profile:parent:dense"],
                sample_count=24,
                effect_size=-1.25,
                confidence=0.9,
                low_confidence=False,
                operator_id=parent.name,
                failure_mode="dense stagnation",
                severity=1.25,
                context={"map_type": "dense"},
            )
        ],
        existing_operator_names=[parent.name],
        allowed_primitives={key: list(values) for key, values in primitive_catalog().items()},
    )


def _context() -> AgentToolContext:
    environment = Environment2D(
        map_id="agent-smoke",
        width=20,
        height=20,
        start=(1, 1),
        goal=(19, 19),
        obstacles=[],
    )
    return AgentToolContext(
        bundle=_bundle(),
        compiler=OperatorCompiler(),
        smoke_fixture=SmokeTestFixture(
            environment=environment,
            path=[(1, 1), (7, 9), (13, 11), (19, 19)],
        ),
    )


def _diagnosis() -> DiagnosisReport:
    return DiagnosisReport(
        parent_operator="ParentOperator",
        failure_modes=[
            DiagnosisClaim(
                claim="dense stagnation",
                evidence_ids=[EVIDENCE_ID],
                confidence=0.9,
                alternative_explanation="map mix may contribute",
            )
        ],
    )


def _proposal(*, structural: bool) -> OperatorProposal:
    diagnosis = _diagnosis()
    payload = _parent().model_dump(mode="json")
    payload.update(
        {
            "name": "RevisedOperator" if structural else "RenameOnlyOperator",
            "description": "evidence-grounded candidate",
            "parent_operators": ["ParentOperator"],
            "expected_mechanism": "explore and smooth" if structural else "local exploration",
            "target_failure_modes": ["dense stagnation"],
        }
    )
    if structural:
        payload["transformations"] = [
            *payload["transformations"],
            {"kind": "smooth_segment", "strength": 0.4},
        ]
    spec = OperatorSpec.model_validate(payload)
    return OperatorProposal(
        operator_spec=spec,
        design_rationale="address the cited failure",
        evidence_used=[EVIDENCE_ID],
        target_failure_modes=["dense stagnation"],
        changes_from_parents=["add smoothing"] if structural else ["rename"],
        expected_contexts=["dense maps"],
        expected_risks=["may reduce clearance"],
        evidence_level="computed",
        diagnosis=diagnosis,
        hypothesis=DesignHypothesis(
            hypothesis="bounded smoothing should reduce stagnation",
            target_failure_mode="dense stagnation",
            expected_mechanism="smooth noisy local geometry",
            expected_effective_context="dense maps",
            possible_side_effects=["clearance tradeoff"],
            evidence_ids=[EVIDENCE_ID],
        ),
        expected_advantages=["lower objective on dense maps"],
        used_evidence_ids=[EVIDENCE_ID],
    )


def test_mock_research_agent_dispatches_real_tools_and_stops_before_validation() -> None:
    provider = MockLLMProvider()
    result = DeterministicMockResearchAgent(provider).run(_context())

    assert result.status == CandidateStatus.SMOKE_PASSED
    assert result.proposal is not None
    assert result.selected_candidate_id == result.candidates[0].candidate_id
    assert result.candidates[0].status_history == [
        CandidateStatus.PROPOSED,
        CandidateStatus.SCHEMA_VALID,
        CandidateStatus.REVIEWED,
        CandidateStatus.COMPILED,
        CandidateStatus.SMOKE_PASSED,
    ]
    tool_names = [call.tool_name for call in result.tool_calls]
    assert "get_failure_modes" in tool_names
    assert tool_names[-2:] == ["compile_operator_spec", "run_operator_smoke_test"]
    assert all("validat" not in name for name in tool_names)
    assert result.usage.tool_calls == 10
    assert result.usage.smoke_tests == 1
    assert result.usage.turns == 3
    assert result.usage.total_tokens > 0
    assert len(result.provider_call_ids) == 2


def test_mock_research_agent_allows_exactly_one_traced_revision() -> None:
    proposal_calls = 0

    def factory(*, output_model: type[Any], user_payload: Any) -> Any:
        nonlocal proposal_calls
        if output_model is DiagnosisReport:
            return _diagnosis()
        if output_model is OperatorProposal:
            proposal_calls += 1
            return _proposal(structural=proposal_calls == 2)
        raise AssertionError(output_model)

    provider = MockLLMProvider(factory=factory)
    result = DeterministicMockResearchAgent(provider).run(_context())

    assert result.status == CandidateStatus.SMOKE_PASSED
    assert len(result.candidates) == 2
    rejected, revised = result.candidates
    assert rejected.final_status == CandidateStatus.REJECTED
    assert rejected.status_history == [
        CandidateStatus.PROPOSED,
        CandidateStatus.SCHEMA_VALID,
        CandidateStatus.REJECTED,
    ]
    assert "rename-only" in (rejected.rejection_reason or "")
    assert revised.supersedes_candidate_id == rejected.candidate_id
    assert revised.candidate_id != rejected.candidate_id
    assert revised.final_status == CandidateStatus.SMOKE_PASSED
    assert result.selected_candidate_id == revised.candidate_id
    assert result.usage.candidate_specs == 2
    assert result.usage.revisions == 1
    assert result.usage.turns == 5
    assert len(provider.call_records) == 4
