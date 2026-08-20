from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from uav_operator_evolution.agents.design_models import CandidateStatus
from uav_operator_evolution.agents.multi_agent import (
    CandidateCritique,
    CandidatePortfolio,
    DeterministicCritic,
    DeterministicMockMultiAgent,
    ExploitationDesigner,
    ExplorationDesigner,
    MULTI_AGENT_BUDGET,
    MultiAgentRoleTrace,
)
from uav_operator_evolution.agents.providers import LLMCallConfig, MockLLMProvider
from uav_operator_evolution.agents.research_agent import ResearchAgentResult
from uav_operator_evolution.agents.tools import AgentBudget, AgentBudgetExceeded

from test_mock_research_agent import _context, _diagnosis, _proposal


class _FailingCompiler:
    def compile(self, specification: Any) -> Any:
        raise RuntimeError("forced compile failure")


def test_multi_agent_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CandidateCritique.model_validate(
            {
                "candidate_id": "candidate",
                "decision": "approve",
                "evidence_alignment_score": 1.0,
                "safety_score": 1.0,
                "testability_score": 1.0,
                "mechanism_fit_score": 1.0,
                "causal_overclaim": False,
                "evidence_ids": [],
                "strengths": [],
                "concerns": [],
                "required_revisions": [],
                "unexpected": True,
            }
        )


def test_coordinator_exposes_two_designers_and_one_non_selecting_critic() -> None:
    backend = DeterministicMockMultiAgent()

    assert tuple(type(designer) for designer in backend.designers) == (
        ExploitationDesigner,
        ExplorationDesigner,
    )
    assert isinstance(backend.critic, DeterministicCritic)
    assert not hasattr(backend.critic, "select")


def test_offline_multi_agent_returns_compatible_result_with_two_siblings() -> None:
    provider = MockLLMProvider()
    result = DeterministicMockMultiAgent(provider).run(
        _context(), agent_run_id="offline-team"
    )

    assert isinstance(result, ResearchAgentResult)
    assert isinstance(result.portfolio, CandidatePortfolio)
    portfolio = result.portfolio
    assert portfolio.portfolio_id.startswith("portfolio_")
    assert len(portfolio.portfolio_hash) == 64
    assert len(portfolio.diagnosis_hash) == 64
    assert result.status == CandidateStatus.SMOKE_PASSED
    assert len(result.candidates) == 2
    selected, not_selected = result.candidates
    assert selected.candidate_id != not_selected.candidate_id
    assert selected.supersedes_candidate_id is None
    assert not_selected.supersedes_candidate_id is None
    assert selected.final_status == CandidateStatus.SMOKE_PASSED
    assert not_selected.final_status == CandidateStatus.REJECTED
    assert not_selected.status_history[-2:] == [
        CandidateStatus.SMOKE_PASSED,
        CandidateStatus.REJECTED,
    ]
    assert not_selected.rejection_reason == "portfolio_not_selected"
    assert selected.proposal is not None
    assert not_selected.proposal is not None
    assert selected.proposal.spec.name.startswith("MockExploit_")
    assert not_selected.proposal.spec.name.startswith("MockExplore_")
    assert selected.proposal.diagnosis == portfolio.diagnosis
    assert not_selected.proposal.diagnosis == portfolio.diagnosis
    assert result.selected_candidate_id == selected.candidate_id
    assert result.proposal == selected.proposal


def test_python_coordinator_owns_fixed_score_and_selection() -> None:
    result = DeterministicMockMultiAgent().run(_context(), agent_run_id="scores")
    portfolio = result.portfolio
    assert isinstance(portfolio, CandidatePortfolio)

    assert all(item.critique.decision == "approve" for item in portfolio.candidates)
    assert not hasattr(portfolio.candidates[0].critique, "selected_candidate_id")
    for item in portfolio.candidates:
        components = item.score_components
        assert item.portfolio_score == pytest.approx(
            0.30 * components.evidence_alignment
            + 0.20 * components.safety
            + 0.20 * components.topology_diversity
            + 0.15 * components.priority_failure_coverage
            + 0.15 * components.testability
        )
        assert item.topology_fingerprint == item.static_review.topology_fingerprint
        assert item.eligible
    expected = sorted(
        portfolio.candidates,
        key=lambda item: (-item.portfolio_score, item.candidate_id),
    )[0]
    assert portfolio.selected_candidate_id == expected.candidate_id
    assert result.selected_candidate_id == expected.candidate_id


def test_offline_multi_agent_shares_hard_budget_and_dispatcher() -> None:
    audited: list[tuple[int, str]] = []

    def audit(result: Any, arguments: Any) -> None:
        audited.append((result.sequence, result.tool_name))

    result = DeterministicMockMultiAgent().run(
        _context(),
        budget=AgentBudget(
            max_turns=32,
            max_tool_calls=128,
            max_candidate_specs=8,
            max_revisions=4,
            max_smoke_tests=16,
        ),
        audit_sink=audit,
    )

    assert MULTI_AGENT_BUDGET.max_turns == 4
    assert result.usage.turns == 4
    assert result.usage.tool_calls == 12
    assert result.usage.candidate_specs == 2
    assert result.usage.revisions == 0
    assert result.usage.smoke_tests == 2
    assert result.usage.total_tokens > 0
    assert [item[0] for item in audited] == list(range(1, 13))
    assert [call.tool_name for call in result.tool_calls[-4:]] == [
        "compile_operator_spec",
        "run_operator_smoke_test",
        "compile_operator_spec",
        "run_operator_smoke_test",
    ]
    assert all("validat" not in call.tool_name for call in result.tool_calls)
    assert all(
        CandidateStatus.VALIDATED not in attempt.status_history
        and CandidateStatus.ACCEPTED not in attempt.status_history
        for attempt in result.candidates
    )


def test_offline_multi_agent_has_four_complete_role_traces() -> None:
    provider = MockLLMProvider()
    result = DeterministicMockMultiAgent(provider).run(_context())
    assert all(isinstance(trace, MultiAgentRoleTrace) for trace in result.role_traces)
    assert [trace.role for trace in result.role_traces] == [
        "diagnoser",
        "exploitation_designer",
        "exploration_designer",
        "critic",
    ]
    assert [trace.action for trace in result.role_traces] == [
        "diagnose",
        "design",
        "design",
        "review",
    ]
    assert [trace.turn for trace in result.role_traces] == [1, 2, 3, 4]
    assert [trace.prompt_version for trace in result.role_traces] == [
        "diagnoser_v1",
        "designer_exploitation_v1",
        "designer_exploration_v1",
        "critic_v1",
    ]
    assert result.provider_call_ids == [record.call_id for record in provider.call_records]
    assert all(trace.input_hash and trace.output_hash for trace in result.role_traces)
    assert all(trace.latency_ms >= 0 and trace.status == "succeeded" for trace in result.role_traces)
    assert all(trace.error is None for trace in result.role_traces)


def test_critic_gate_runs_compile_and_smoke_only_for_approved_candidate() -> None:
    evidence_id = _context().bundle.evidence_ids()[0]
    provider = MockLLMProvider(
        fixtures={
            "PortfolioCritique": {
                "assessments": [
                    {
                            "candidate_id": "candidate_critic-gate_01",
                            "decision": "approve",
                            "evidence_alignment_score": 1.0,
                            "safety_score": 1.0,
                            "testability_score": 1.0,
                            "mechanism_fit_score": 0.9,
                            "causal_overclaim": False,
                        "evidence_ids": [evidence_id],
                        "strengths": ["bounded refinement"],
                        "concerns": [],
                        "required_revisions": [],
                    },
                    {
                            "candidate_id": "candidate_critic-gate_02",
                            "decision": "reject",
                            "evidence_alignment_score": 0.4,
                            "safety_score": 1.0,
                            "testability_score": 0.8,
                            "mechanism_fit_score": 0.4,
                            "causal_overclaim": False,
                        "evidence_ids": [evidence_id],
                        "strengths": [],
                        "concerns": ["mechanism is weakly supported"],
                        "required_revisions": [],
                    },
                ],
                "comparative_rationale": "Assessments only; Python makes the selection.",
                "used_evidence_ids": [evidence_id],
            }
        }
    )
    result = DeterministicMockMultiAgent(provider).run(
        _context(), agent_run_id="critic-gate"
    )

    assert result.selected_candidate_id == "candidate_critic-gate_01"
    assert result.usage.tool_calls == 10  # eight evidence calls + one compile/smoke pair
    assert [call.tool_name for call in result.tool_calls[-2:]] == [
        "compile_operator_spec",
        "run_operator_smoke_test",
    ]
    rejected = result.candidates[1]
    assert CandidateStatus.COMPILED not in rejected.status_history
    assert CandidateStatus.SMOKE_PASSED not in rejected.status_history
    assert rejected.status_history[-1] == CandidateStatus.REJECTED
    assert (rejected.rejection_reason or "").startswith("critic_reject")


def test_causal_overclaim_is_rejected_before_compile_and_smoke() -> None:
    evidence_id = _context().bundle.evidence_ids()[0]
    provider = MockLLMProvider(
        fixtures={
            "PortfolioCritique": {
                "assessments": [
                    {
                        "candidate_id": "candidate_causal-gate_01",
                        "decision": "approve",
                        "evidence_alignment_score": 1.0,
                        "safety_score": 1.0,
                        "testability_score": 1.0,
                        "mechanism_fit_score": 0.9,
                        "causal_overclaim": True,
                        "evidence_ids": [evidence_id],
                        "strengths": [],
                        "concerns": ["claim exceeds the supplied evidence"],
                        "required_revisions": [],
                    },
                    {
                        "candidate_id": "candidate_causal-gate_02",
                        "decision": "approve",
                        "evidence_alignment_score": 1.0,
                        "safety_score": 1.0,
                        "testability_score": 1.0,
                        "mechanism_fit_score": 0.8,
                        "causal_overclaim": False,
                        "evidence_ids": [evidence_id],
                        "strengths": ["bounded and testable"],
                        "concerns": [],
                        "required_revisions": [],
                    },
                ],
                "comparative_rationale": "Causal claims are a hard pre-compile gate.",
                "used_evidence_ids": [evidence_id],
            }
        }
    )

    result = DeterministicMockMultiAgent(provider).run(
        _context(), agent_run_id="causal-gate"
    )

    assert result.selected_candidate_id == "candidate_causal-gate_02"
    assert result.usage.tool_calls == 10
    rejected = result.candidates[0]
    assert rejected.final_status == CandidateStatus.REJECTED
    assert rejected.rejection_reason == "critic_reject: causal overclaim"
    assert CandidateStatus.COMPILED not in rejected.status_history


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        (AgentBudget(max_turns=3), "max_turns>=4"),
        (AgentBudget(max_tool_calls=11), "max_tool_calls>=12"),
        (AgentBudget(max_candidate_specs=1), "max_candidate_specs>=2"),
        (AgentBudget(max_smoke_tests=1), "max_smoke_tests>=2"),
    ],
)
def test_offline_multi_agent_rejects_budget_that_cannot_run_fixed_topology(
    budget: AgentBudget,
    message: str,
) -> None:
    provider = MockLLMProvider()
    with pytest.raises(AgentBudgetExceeded, match=message):
        DeterministicMockMultiAgent(provider).run(_context(), budget=budget)
    assert provider.call_records == []


def test_portfolio_content_is_deterministic_across_fresh_runs() -> None:
    first = DeterministicMockMultiAgent().run(_context(), agent_run_id="repeatable")
    second = DeterministicMockMultiAgent().run(_context(), agent_run_id="repeatable")

    assert isinstance(first.portfolio, CandidatePortfolio)
    assert isinstance(second.portfolio, CandidatePortfolio)
    assert first.portfolio.portfolio_hash == second.portfolio.portfolio_hash
    assert first.portfolio.portfolio_id == second.portfolio.portfolio_id
    assert first.portfolio.canonical_payload() == second.portfolio.canonical_payload()
    assert first.selected_candidate_id == second.selected_candidate_id


@pytest.mark.parametrize(
    ("failure_sequence", "expected_roles"),
    [
        (["schema_error"], ["diagnoser"]),
        (
            ["success", "success", "timeout", "timeout", "timeout"],
            ["diagnoser", "exploitation_designer", "exploration_designer"],
        ),
        (
            ["success", "success", "success", "refusal"],
            [
                "diagnoser",
                "exploitation_designer",
                "exploration_designer",
                "critic",
            ],
        ),
    ],
)
def test_role_provider_failures_return_two_auditable_terminal_siblings(
    failure_sequence: list[str],
    expected_roles: list[str],
) -> None:
    provider = MockLLMProvider(failure_sequence=failure_sequence)  # type: ignore[arg-type]

    result = DeterministicMockMultiAgent(provider).run(
        _context(), agent_run_id="provider-failure"
    )

    assert result.status == CandidateStatus.REJECTED
    assert result.selected_candidate_id is None
    assert result.portfolio is None
    assert len(result.candidates) == 2
    assert all(
        attempt.final_status == CandidateStatus.REJECTED
        and attempt.status_history[-1] == CandidateStatus.REJECTED
        and attempt.supersedes_candidate_id is None
        for attempt in result.candidates
    )
    assert [trace.role for trace in result.role_traces] == expected_roles
    assert result.role_traces[-1].status == "failed"
    assert result.role_traces[-1].provider_call_id is not None
    assert result.provider_call_ids[-1] == result.role_traces[-1].provider_call_id


def test_token_budget_failure_is_fail_closed_with_partial_role_audit() -> None:
    result = DeterministicMockMultiAgent().run(
        _context(),
        agent_run_id="token-failure",
        call_config=LLMCallConfig(max_total_tokens=1),
    )

    assert result.status == CandidateStatus.REJECTED
    assert all(item.final_status == CandidateStatus.REJECTED for item in result.candidates)
    assert len(result.role_traces) == 1
    assert result.role_traces[0].role == "diagnoser"
    assert result.role_traces[0].status == "failed"
    assert "token budget" in (result.role_traces[0].error or "").lower()


def test_missing_persisted_failure_evidence_safely_rejects_before_tools() -> None:
    context = _context()
    payload = context.bundle.model_dump(mode="json")
    payload["failure_modes"] = []
    payload["bundle_hash"] = ""
    context.bundle = type(context.bundle).model_validate(payload)
    provider = MockLLMProvider()

    result = DeterministicMockMultiAgent(provider).run(
        context, agent_run_id="missing-failure"
    )

    assert result.status == CandidateStatus.REJECTED
    assert result.tool_calls == []
    assert result.role_traces == []
    assert provider.call_records == []
    assert len(result.candidates) == 2
    assert all(
        item.final_status == CandidateStatus.REJECTED
        and "no persisted failure-mode evidence" in (item.rejection_reason or "")
        for item in result.candidates
    )


@pytest.mark.parametrize("failure_stage", ["compile", "smoke"])
def test_both_candidates_reach_terminal_rejection_on_local_gate_failure(
    failure_stage: str,
) -> None:
    context = _context()
    if failure_stage == "compile":
        context.compiler = _FailingCompiler()  # type: ignore[assignment]
    else:
        context.smoke_fixture = None

    result = DeterministicMockMultiAgent().run(
        context, agent_run_id=f"{failure_stage}-failure"
    )

    assert result.status == CandidateStatus.REJECTED
    assert result.selected_candidate_id is None
    assert isinstance(result.portfolio, CandidatePortfolio)
    assert all(item.final_status == CandidateStatus.REJECTED for item in result.candidates)
    assert all(
        failure_stage in (item.rejection_reason or "") for item in result.candidates
    )
    assert all(item.supersedes_candidate_id is None for item in result.candidates)


def test_duplicate_topology_keeps_only_stable_role_priority_winner() -> None:
    evidence_id = _context().bundle.evidence_ids()[0]

    def factory(*, output_model: type[Any], user_payload: Any) -> Any:
        if output_model.__name__ == "DiagnosisReport":
            return _diagnosis()
        if output_model.__name__ == "OperatorProposal":
            role = user_payload["design_role"]
            proposal = _proposal(structural=True)
            spec_payload = proposal.spec.model_dump(mode="json")
            spec_payload["name"] = (
                "TwinExploit" if role == "exploitation_designer" else "TwinExplore"
            )
            if role == "exploration_designer":
                spec_payload["transformations"][-1]["strength"] = 0.55
            return proposal.model_copy(
                update={
                    "specification": type(proposal.spec).model_validate(spec_payload)
                }
            )
        if output_model.__name__ == "PortfolioCritique":
            return {
                "assessments": [
                    {
                        "candidate_id": item["candidate_id"],
                        "decision": "approve",
                        "evidence_alignment_score": 1.0,
                        "safety_score": 1.0,
                        "testability_score": 1.0,
                        "mechanism_fit_score": 1.0,
                        "causal_overclaim": False,
                        "evidence_ids": [evidence_id],
                        "strengths": ["bounded"],
                        "concerns": [],
                        "required_revisions": [],
                    }
                    for item in user_payload["candidates"]
                ],
                "comparative_rationale": "Python resolves duplicate topologies.",
                "used_evidence_ids": [evidence_id],
            }
        raise AssertionError(output_model)

    result = DeterministicMockMultiAgent(MockLLMProvider(factory=factory)).run(
        _context(), agent_run_id="duplicate"
    )
    portfolio = result.portfolio

    assert isinstance(portfolio, CandidatePortfolio)
    assert len({item.topology_fingerprint for item in portfolio.candidates}) == 1
    assert [item.score_components.topology_diversity for item in portfolio.candidates] == [
        0.0,
        0.0,
    ]
    assert portfolio.selected_candidate_id == "candidate_duplicate_01"
    assert portfolio.candidates[0].eligible
    assert not portfolio.candidates[1].eligible
    assert result.candidates[1].rejection_reason == "portfolio_not_selected"


def test_portfolio_rejects_an_ineligible_or_unknown_selected_candidate() -> None:
    result = DeterministicMockMultiAgent().run(_context(), agent_run_id="invalid-winner")
    assert isinstance(result.portfolio, CandidatePortfolio)
    payload = result.portfolio.model_dump(mode="json")
    payload.update(
        {
            "portfolio_id": "",
            "portfolio_hash": "",
            "selected_candidate_id": "candidate_not_in_portfolio",
        }
    )

    with pytest.raises(ValidationError, match="selected candidate must be eligible"):
        CandidatePortfolio.model_validate(payload)


def test_critic_unknown_evidence_id_fails_closed_after_shared_review() -> None:
    provider = MockLLMProvider(
        fixtures={
            "PortfolioCritique": {
                "assessments": [
                    {
                        "candidate_id": f"candidate_unknown-evidence_{index:02d}",
                        "decision": "approve",
                        "evidence_alignment_score": 1.0,
                        "safety_score": 1.0,
                        "testability_score": 1.0,
                        "mechanism_fit_score": 1.0,
                        "causal_overclaim": False,
                        "evidence_ids": ["fail_" + "f" * 24],
                        "strengths": [],
                        "concerns": [],
                        "required_revisions": [],
                    }
                    for index in (1, 2)
                ],
                "comparative_rationale": "invalid evidence fixture",
                "used_evidence_ids": ["fail_" + "f" * 24],
            }
        }
    )

    result = DeterministicMockMultiAgent(provider).run(
        _context(), agent_run_id="unknown-evidence"
    )

    assert result.status == CandidateStatus.REJECTED
    assert result.portfolio is None
    assert len(result.role_traces) == 4
    assert all(item.final_status == CandidateStatus.REJECTED for item in result.candidates)
    assert all(
        "critic cited evidence outside the shared bundle" in (item.rejection_reason or "")
        for item in result.candidates
    )
