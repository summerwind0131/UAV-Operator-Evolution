from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from uav_operator_evolution.agents.audit import AgentAuditStore
from uav_operator_evolution.agents.design_models import (
    CandidateStatus,
    DesignHypothesis,
    DiagnosisClaim,
    DiagnosisReport,
)
from uav_operator_evolution.agents.designer_base import OperatorProposal
from uav_operator_evolution.agents.evidence import EvidenceBundleBuilder, OperatorEvidenceBundle
from uav_operator_evolution.agents.llm_designer import LLMDesignerAdapter
from uav_operator_evolution.agents.multi_agent import DeterministicMockMultiAgent
from uav_operator_evolution.agents.orchestrator import (
    OperatorDesignOrchestrator,
    OperatorDesignRequest,
)
from uav_operator_evolution.agents.proposal_validation import ProposalValidator
from uav_operator_evolution.agents.providers import MockLLMProvider
from uav_operator_evolution.agents.research_agent import DeterministicMockResearchAgent
from uav_operator_evolution.environment import Environment2D
from uav_operator_evolution.evolution.validation import PairedOutcome, ValidationReport
from uav_operator_evolution.memory import MechanismMemory
from uav_operator_evolution.operators.compiler import OperatorCompilationError, OperatorCompiler
from uav_operator_evolution.operators.registry import (
    OperatorRegistry,
    build_manual_operator_registry,
)
from uav_operator_evolution.reproducibility import stable_hash


PARENT = "waypoint_perturb"
CANDIDATE = "EvolvedWaypointPerturb"


def _environments() -> tuple[Environment2D, list[Environment2D]]:
    smoke = Environment2D(
        map_id="design-smoke",
        width=20,
        height=20,
        start=(1, 1),
        goal=(19, 19),
        obstacles=[],
        seed=1,
    )
    validation = [
        Environment2D(
            map_id="validation-a",
            width=21,
            height=21,
            start=(1, 1),
            goal=(20, 20),
            obstacles=[],
            seed=2,
        ),
        Environment2D(
            map_id="validation-b",
            width=22,
            height=22,
            start=(1, 1),
            goal=(21, 21),
            obstacles=[],
            seed=3,
        ),
    ]
    return smoke, validation


def _proposal(bundle: OperatorEvidenceBundle) -> OperatorProposal:
    evidence = bundle.failure_modes[0]
    claim = DiagnosisClaim(
        claim=evidence.failure_mode,
        evidence_ids=[evidence.evidence_id],
        confidence=evidence.confidence,
        alternative_explanation="the map mix may also contribute",
    )
    diagnosis = DiagnosisReport(
        parent_operator=PARENT,
        failure_modes=[claim],
    )
    payload = bundle.parent_specs[0].model_dump(mode="json")
    payload.update(
        name=CANDIDATE,
        description="Evidence-grounded perturbation followed by bounded smoothing.",
        parent_operators=[PARENT],
        transformations=[
            *payload["transformations"],
            {"kind": "smooth_segment", "strength": 0.35},
        ],
        expected_mechanism="preserve exploration while smoothing noisy local geometry",
        target_failure_modes=[evidence.failure_mode],
    )
    return OperatorProposal.model_validate(
        {
            "operator_spec": payload,
            "design_rationale": "Address the highest-severity cited parent failure.",
            "evidence_used": [evidence.evidence_id],
            "target_failure_modes": [evidence.failure_mode],
            "changes_from_parents": ["add bounded smoothing"],
            "expected_contexts": ["locally irregular paths"],
            "expected_risks": ["may trade clearance for smoothness"],
            "evidence_level": "computed",
            "diagnosis": diagnosis.model_dump(mode="json"),
            "hypothesis": DesignHypothesis(
                hypothesis="A bounded smoothing pass will reduce noisy perturbation outcomes.",
                target_failure_mode=evidence.failure_mode,
                expected_mechanism="smooth the perturbed local segment",
                expected_effective_context="paths with excessive local curvature",
                possible_side_effects=["reduced obstacle clearance"],
                evidence_ids=[evidence.evidence_id],
            ).model_dump(mode="json"),
            "expected_advantages": ["lower path smoothness penalty"],
            "used_evidence_ids": [evidence.evidence_id],
        }
    )


class _StaticDesigner:
    provider = None

    def __init__(self, payload: Any | None = None) -> None:
        self.payload = payload
        self.modes: list[str] = []

    def propose_from_evidence(
        self,
        bundle: OperatorEvidenceBundle,
        *,
        mode: str,
        call_config: Any,
    ) -> Any:
        del call_config
        self.modes.append(mode)
        return _proposal(bundle) if self.payload is None else self.payload


class _CountingMockProvider(MockLLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.reset_calls = 0

    def reset_usage(self) -> None:
        self.reset_calls += 1
        super().reset_usage()


@dataclass
class _RecordingCandidateValidator:
    retained: bool = True
    smoke_failures: list[str] = field(default_factory=list)
    validation_map_ids: list[str] = field(default_factory=list)
    smoke_map_ids: list[str] = field(default_factory=list)
    validate_calls: int = 0
    config: Any = field(
        default_factory=lambda: SimpleNamespace(maps=SimpleNamespace(grid_resolution=4.0))
    )

    def contract_failures(
        self,
        candidate: Any,
        environment: Environment2D,
        generation: int,
        candidate_index: int,
    ) -> list[str]:
        del candidate, generation, candidate_index
        self.smoke_map_ids.append(environment.map_id)
        return list(self.smoke_failures)

    def validate(
        self,
        population: list[Any],
        parent_name: str,
        candidate: Any,
        validation_environments: list[Environment2D],
        *,
        generation: int,
        candidate_index: int,
        recorder: Any,
        root_run_id: str,
    ) -> ValidationReport:
        del population, generation, candidate_index, recorder, root_run_id
        self.validate_calls += 1
        self.validation_map_ids = [item.map_id for item in validation_environments]
        outcomes = [
            PairedOutcome(
                map_id=item.map_id,
                difficulty=item.difficulty,
                parent_best_cost=10.0,
                candidate_best_cost=9.0,
                parent_feasible=True,
                candidate_feasible=True,
                parent_runtime_ms=2.0,
                candidate_runtime_ms=1.8,
            )
            for item in validation_environments
        ]
        return ValidationReport(
            parent_operator=parent_name,
            candidate_operator=str(candidate.name),
            safety_passed=True,
            outcomes=outcomes,
            mean_gain=0.1,
            win_rate=1.0,
            parent_feasibility_rate=1.0,
            candidate_feasibility_rate=1.0,
            median_runtime_reduction=0.1,
            retained=self.retained,
            retention_reasons=[
                "global paired gain" if self.retained else "no pre-registered effect threshold met"
            ],
        )


class _RejectingCompiler(OperatorCompiler):
    def compile(self, spec: Any) -> Any:
        name = spec.name if hasattr(spec, "name") else spec.get("name")
        if name == CANDIDATE:
            raise OperatorCompilationError("synthetic compiler rejection")
        return super().compile(spec)


@dataclass
class _Harness:
    orchestrator: OperatorDesignOrchestrator
    memory: MechanismMemory
    audit: AgentAuditStore
    registry: OperatorRegistry
    validator: _RecordingCandidateValidator
    smoke: Environment2D
    validation: list[Environment2D]

    def close(self) -> None:
        self.audit.close()
        self.memory.close()


def _harness(
    tmp_path: Path,
    *,
    retained: bool = True,
    smoke_failures: list[str] | None = None,
    compiler: OperatorCompiler | None = None,
    research_backend: Any | None = None,
    llm_designer: Any | None = None,
) -> _Harness:
    database = tmp_path / "orchestration.sqlite"
    memory = MechanismMemory(database)
    memory.add_failure_mode(
        "poor_smoothness",
        operator_id=PARENT,
        count=12,
        severity=1.5,
        context={"difficulty": "dense"},
        evidence=[{"source": "paired operator history"}],
    )
    registry = build_manual_operator_registry()
    audit = AgentAuditStore(database)
    validator = _RecordingCandidateValidator(
        retained=retained,
        smoke_failures=list(smoke_failures or []),
    )
    resolved_compiler = compiler or OperatorCompiler()
    smoke, validation = _environments()
    return _Harness(
        orchestrator=OperatorDesignOrchestrator(
            evidence_builder=EvidenceBundleBuilder(memory, registry),
            proposal_validator=ProposalValidator(),
            compiler=resolved_compiler,
            candidate_validator=validator,  # type: ignore[arg-type]
            memory=memory,
            registry=registry,
            llm_designer=llm_designer or _StaticDesigner(),  # type: ignore[arg-type]
            research_agent_backend=research_backend,
            audit_store=audit,
        ),
        memory=memory,
        audit=audit,
        registry=registry,
        validator=validator,
        smoke=smoke,
        validation=validation,
    )


def _request(
    harness: _Harness,
    *,
    request_id: str,
    mode: str = "llm_single",
    review_mode: str = "rule_based",
) -> Any:
    return OperatorDesignRequest(
        request_id=request_id,
        experiment_id="orchestrator-test",
        root_run_id=f"root-{request_id}",
        problem_summary="Reduce poor smoothing outcomes without weakening bounded safety.",
        parent_operator_ids=[PARENT],
        smoke_environment=harness.smoke,
        validation_environments=harness.validation,
        design_mode=mode,
        review_mode=review_mode,
    )


def test_acceptance_runs_full_state_machine_and_persists_memory_lineage(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        result = harness.orchestrator.run(_request(harness, request_id="accepted"))

        assert result.outcome == "accepted"
        assert result.final_status == CandidateStatus.ACCEPTED
        assert result.operator_name == CANDIDATE
        assert result.retained is True
        assert CANDIDATE in harness.registry
        assert harness.validator.smoke_map_ids == ["design-smoke"]
        assert harness.validator.validation_map_ids == ["validation-a", "validation-b"]
        assert "design-smoke" not in harness.validator.validation_map_ids

        mechanism = harness.memory.get_mechanism(CANDIDATE)
        assert mechanism is not None
        assert harness.memory.get_insights(CANDIDATE, insight_type="improvement_hypothesis")
        lineage = harness.memory.get_lineage(CANDIDATE, direction="ancestors")
        assert [(edge.parent_id, edge.child_id, edge.relation) for edge in lineage] == [
            (PARENT, CANDIDATE, "structural_variant")
        ]

        events = harness.audit.list_candidate_events(result.candidate_id)
        assert [event.status for event in events] == [
            CandidateStatus.PROPOSED,
            CandidateStatus.SCHEMA_VALID,
            CandidateStatus.REVIEWED,
            CandidateStatus.COMPILED,
            CandidateStatus.SMOKE_PASSED,
            CandidateStatus.VALIDATED,
            CandidateStatus.ACCEPTED,
        ]
        reconstructed = OperatorProposal.model_validate(events[1].details["proposal"])
        assert reconstructed.spec.name == CANDIDATE
        assert events[0].details["proposal"]["operator_spec"]["name"] == CANDIDATE
        assert events[2].details["review"]["decision"] == "approve"
        assert events[3].details["compile"]["operator_name"] == CANDIDATE
        assert events[4].details["smoke"]["map_id"] == "design-smoke"
        assert events[5].details["validation_report"]["retained"] is True

        run = harness.audit.get_agent_run(result.agent_run_id)
        assert run is not None
        assert run.metadata["formal_validation_exposed_as_tool"] is False
        stored_bundle = harness.audit.get_evidence_bundle(result.bundle_id)
        assert stored_bundle is not None
        assert stored_bundle.bundle_hash == result.bundle_hash
    finally:
        harness.close()


def test_retention_rejection_is_audited_and_saved_as_failure_evidence(tmp_path: Path) -> None:
    harness = _harness(tmp_path, retained=False)
    try:
        result = harness.orchestrator.run(_request(harness, request_id="not-retained"))

        assert result.outcome == "rejected"
        assert result.final_status == CandidateStatus.REJECTED
        assert result.rejection_stage == "retention"
        assert result.compiled is True
        assert result.smoke_passed is True
        assert result.retained is False
        assert CANDIDATE not in harness.registry
        assert harness.memory.get_mechanism(CANDIDATE) is None
        failures = harness.memory.get_failure_modes(CANDIDATE)
        assert [item.mode for item in failures] == ["candidate_rejected"]
        assert failures[0].context["stage"] == "retention"
        assert result.rejection_evidence_ids == [failures[0].failure_id]
        assert [event.status for event in harness.audit.list_candidate_events(result.candidate_id)] == [
            CandidateStatus.PROPOSED,
            CandidateStatus.SCHEMA_VALID,
            CandidateStatus.REVIEWED,
            CandidateStatus.COMPILED,
            CandidateStatus.SMOKE_PASSED,
            CandidateStatus.VALIDATED,
            CandidateStatus.REJECTED,
        ]
    finally:
        harness.close()


@pytest.mark.parametrize(
    ("stage", "compiler", "smoke_failures", "expected"),
    [
        (
            "compile",
            _RejectingCompiler(),
            [],
            [
                CandidateStatus.PROPOSED,
                CandidateStatus.SCHEMA_VALID,
                CandidateStatus.REVIEWED,
                CandidateStatus.REJECTED,
            ],
        ),
        (
            "smoke",
            OperatorCompiler(),
            ["synthetic endpoint violation"],
            [
                CandidateStatus.PROPOSED,
                CandidateStatus.SCHEMA_VALID,
                CandidateStatus.REVIEWED,
                CandidateStatus.COMPILED,
                CandidateStatus.REJECTED,
            ],
        ),
    ],
)
def test_compile_and_smoke_rejections_stop_before_formal_validation(
    tmp_path: Path,
    stage: str,
    compiler: OperatorCompiler,
    smoke_failures: list[str],
    expected: list[CandidateStatus],
) -> None:
    harness = _harness(tmp_path, compiler=compiler, smoke_failures=smoke_failures)
    try:
        result = harness.orchestrator.run(_request(harness, request_id=f"reject-{stage}"))

        assert result.rejection_stage == stage
        assert harness.validator.validate_calls == 0
        events = harness.audit.list_candidate_events(result.candidate_id)
        assert [event.status for event in events] == expected
        assert events[-1].details["stage"] == stage
        assert harness.memory.get_failure_modes(CANDIDATE)[0].mode == "candidate_rejected"
    finally:
        harness.close()


def test_request_has_no_test_split_and_rejects_smoke_validation_overlap() -> None:
    smoke, validation = _environments()
    payload = {
        "request_id": "split-contract",
        "experiment_id": "orchestrator-test",
        "root_run_id": "root-split-contract",
        "problem_summary": "Keep the held-out test split outside candidate design.",
        "parent_operator_ids": [PARENT],
        "smoke_environment": smoke,
        "validation_environments": validation,
        "test_environments": [],
    }
    with pytest.raises(ValidationError, match="test_environments"):
        OperatorDesignRequest.model_validate(payload)

    payload.pop("test_environments")
    payload["validation_environments"] = [smoke]
    with pytest.raises(ValidationError, match="smoke environment"):
        OperatorDesignRequest.model_validate(payload)


def test_single_agent_tools_stop_before_formal_validation_and_audit_provider_ids(
    tmp_path: Path,
) -> None:
    provider = MockLLMProvider()
    backend = DeterministicMockResearchAgent(provider)
    harness = _harness(tmp_path, research_backend=backend)
    try:
        result = harness.orchestrator.run(
            _request(harness, request_id="single-agent", mode="single_agent")
        )

        assert result.final_status == CandidateStatus.ACCEPTED
        assert harness.validator.validate_calls == 1
        tool_calls = harness.audit.list_tool_calls(result.agent_run_id)
        assert len(tool_calls) == 10
        assert all("validat" not in call.tool_name.lower() for call in tool_calls)
        assert tool_calls[-2].tool_name == "compile_operator_spec"
        assert tool_calls[-1].tool_name == "run_operator_smoke_test"
        events = harness.audit.list_candidate_events(result.candidate_id)
        assert [event.status for event in events] == [
            CandidateStatus.PROPOSED,
            CandidateStatus.SCHEMA_VALID,
            CandidateStatus.REVIEWED,
            CandidateStatus.COMPILED,
            CandidateStatus.SMOKE_PASSED,
            CandidateStatus.VALIDATED,
            CandidateStatus.ACCEPTED,
        ]
        assert events[0].details["proposal"]["operator_spec"]["name"].startswith(
            "MockEvolved_"
        )
        assert events[0].details["supersedes_candidate_id"] is None
        assert events[2].details["review"]["decision"] == "approve"
        assert events[3].details["compile"]["tool_name"] == "compile_operator_spec"
        assert events[4].details["smoke"]["tool_name"] == "run_operator_smoke_test"
        llm_calls = harness.audit.list_llm_calls(result.agent_run_id)
        assert len(llm_calls) == 2
        assert [call.response_id for call in llm_calls] == [
            "mock_response_000001",
            "mock_response_000002",
        ]
    finally:
        harness.close()


def test_multi_agent_portfolio_audit_and_only_selected_formal_validation(
    tmp_path: Path,
) -> None:
    provider = MockLLMProvider()
    backend = DeterministicMockMultiAgent(provider)
    harness = _harness(tmp_path, research_backend=backend)
    try:
        result = harness.orchestrator.run(
            _request(harness, request_id="multi-agent", mode="multi_agent")
        )

        assert result.final_status == CandidateStatus.ACCEPTED
        assert result.candidate_portfolio is not None
        assert result.candidate_portfolio["selected_candidate_id"] == result.candidate_id
        assert len(result.candidate_portfolio["candidates"]) == 2
        assert harness.validator.validate_calls == 1

        tool_calls = harness.audit.list_tool_calls(result.agent_run_id)
        assert len(tool_calls) == 12
        assert all("validat" not in call.tool_name.lower() for call in tool_calls)
        assert [call.tool_name for call in tool_calls[-4:]] == [
            "compile_operator_spec",
            "run_operator_smoke_test",
            "compile_operator_spec",
            "run_operator_smoke_test",
        ]
        assert len(harness.audit.list_llm_calls(result.agent_run_id)) == 4

        multi_runs = harness.audit.list_multi_agent_runs(result.agent_run_id)
        assert len(multi_runs) == 1
        multi_run = multi_runs[0]
        assert multi_run.selected_candidate_id == result.candidate_id
        portfolio = harness.audit.get_candidate_portfolio(multi_run.portfolio_id or "")
        assert portfolio is not None
        assert portfolio.portfolio_hash == result.candidate_portfolio["portfolio_hash"]
        role_events = harness.audit.list_multi_agent_role_events(
            multi_run.multi_agent_run_id
        )
        assert [event.action for event in role_events] == [
            "diagnose",
            "design",
            "design",
            "review",
            "select",
        ]
        assert len(result.multi_agent_role_traces) == 4
        llm_by_id = {
            call.call_id: call
            for call in harness.audit.list_llm_calls(result.agent_run_id)
        }
        for event, trace in zip(role_events[:-1], result.multi_agent_role_traces):
            assert event.input_hash == trace["input_hash"]
            assert event.output_hash == trace["output_hash"]
            assert event.prompt_hash == trace["prompt_hash"]
            assert event.summary_input_hash == stable_hash(event.input_summary)
            assert event.summary_output_hash == stable_hash(event.output_summary)
            assert event.provider_call_id in llm_by_id
            assert (
                llm_by_id[event.provider_call_id].prompt["provider_prompt_hash"]
                == trace["prompt_hash"]
            )

        sibling_ids = {
            item["candidate_id"] for item in result.candidate_portfolio["candidates"]
        }
        loser_id = (sibling_ids - {result.candidate_id}).pop()
        loser_events = harness.audit.list_candidate_events(loser_id)
        assert loser_events[-1].status == CandidateStatus.REJECTED
        assert "portfolio_not_selected" in loser_events[-1].reason
    finally:
        harness.close()


def test_multi_agent_schema_failure_persists_partial_failed_run_without_portfolio(
    tmp_path: Path,
) -> None:
    provider = MockLLMProvider(failure_sequence=["schema_error"])
    backend = DeterministicMockMultiAgent(provider)
    harness = _harness(tmp_path, research_backend=backend)
    try:
        result = harness.orchestrator.run(
            _request(harness, request_id="multi-agent-schema-failure", mode="multi_agent")
        )

        assert result.final_status == CandidateStatus.REJECTED
        assert result.candidate_portfolio is None
        assert harness.validator.validate_calls == 0
        agent_run = harness.audit.get_agent_run(result.agent_run_id)
        assert agent_run is not None
        candidate_ids = agent_run.metadata["candidate_ids"]
        assert len(candidate_ids) == 2
        assert all(
            harness.audit.list_candidate_events(candidate_id)[-1].status
            == CandidateStatus.REJECTED
            for candidate_id in candidate_ids
        )

        multi_runs = harness.audit.list_multi_agent_runs(result.agent_run_id)
        assert len(multi_runs) == 1
        multi_run = multi_runs[0]
        assert multi_run.status == "failed"
        assert multi_run.portfolio_id is None
        assert multi_run.portfolio is None
        assert harness.audit.list_candidate_portfolios(multi_run.multi_agent_run_id) == []

        events = harness.audit.list_multi_agent_role_events(
            multi_run.multi_agent_run_id
        )
        assert [event.action for event in events] == ["diagnose", "select"]
        failed_role, coordinator = events
        assert failed_role.status == "failed"
        assert failed_role.input_hash == result.multi_agent_role_traces[0]["input_hash"]
        assert failed_role.output_hash is None
        assert failed_role.prompt_hash == result.multi_agent_role_traces[0]["prompt_hash"]
        assert failed_role.provider_call_id is not None
        assert coordinator.status == "failed"
    finally:
        harness.close()


def test_llm_review_runs_after_hard_validation_and_shares_one_provider_budget(
    tmp_path: Path,
) -> None:
    provider = _CountingMockProvider()
    designer = LLMDesignerAdapter(provider=provider, proposal_validator=ProposalValidator())
    harness = _harness(tmp_path, llm_designer=designer)
    try:
        result = harness.orchestrator.run(
            _request(
                harness,
                request_id="llm-review",
                mode="llm_single",
                review_mode="llm",
            )
        )

        assert result.final_status == CandidateStatus.ACCEPTED
        assert result.review is not None
        assert result.review.decision == "approve"
        assert result.review.lineage_relation == "structural_variant"
        assert provider.reset_calls == 1
        assert len(provider.call_records) == 2
        calls = harness.audit.list_llm_calls(result.agent_run_id)
        assert [call.prompt_version for call in calls] == ["designer_v1", "reviewer_v1"]
        assert [call.response_id for call in calls] == [
            "mock_response_000001",
            "mock_response_000002",
        ]
        events = harness.audit.list_candidate_events(result.candidate_id)
        assert events[2].details["review"]["decision"] == "approve"
        assert events[2].details["review"]["lineage_relation"] == "structural_variant"
    finally:
        harness.close()
