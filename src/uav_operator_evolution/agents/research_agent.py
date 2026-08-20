"""Bounded single-agent backends for evidence-grounded operator research.

The deterministic backend is the default offline implementation.  It uses the
same structured designer and the same whitelisted tool dispatcher as an online
agent, making tests and ``agent-demo`` exercise the real permission and budget
boundaries.  Formal paired validation is intentionally absent from this module.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..reproducibility import canonical_json, stable_hash
from .designer_base import OperatorProposal
from .design_models import CandidateStatus, DiagnosisReport, OperatorReview
from .evidence import OperatorEvidenceBundle
from .llm_designer import LLMDesignError, LLMDesignerAdapter
from .prompts import RESEARCH_AGENT_V1
from .proposal_validation import ProposalValidationError, ProposalValidator
from .providers import (
    LLMCallConfig,
    LLMConfigurationError,
    LLMProvider,
    LLMTokenBudgetError,
    MockLLMProvider,
)
from .tools import (
    AgentBudget,
    AgentBudgetController,
    AgentBudgetExceeded,
    AgentToolContext,
    AgentToolDispatcher,
    AgentUsage,
    CaseQueryInput,
    EmptyToolInput,
    LineageQueryInput,
    OperatorIdInput,
    OperatorSpecInput,
    ToolAuditSink,
    ToolExecutionResult,
)


class ResearchAgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateAttempt(ResearchAgentModel):
    """One immutable-in-practice candidate attempt and its local state trace."""

    candidate_id: str = Field(min_length=1, max_length=200)
    proposal: OperatorProposal | None = None
    review: OperatorReview | None = None
    status_history: list[CandidateStatus] = Field(min_length=2, max_length=8)
    final_status: CandidateStatus
    rejection_reason: str | None = None
    supersedes_candidate_id: str | None = None
    compile_result: ToolExecutionResult | None = None
    smoke_result: ToolExecutionResult | None = None

    @model_validator(mode="after")
    def valid_state_history(self) -> "CandidateAttempt":
        if self.status_history[0] != CandidateStatus.PROPOSED:
            raise ValueError("candidate history must start at PROPOSED")
        if self.status_history[-1] != self.final_status:
            raise ValueError("final_status must match the last state transition")
        ordered = [
            CandidateStatus.PROPOSED,
            CandidateStatus.SCHEMA_VALID,
            CandidateStatus.REVIEWED,
            CandidateStatus.COMPILED,
            CandidateStatus.SMOKE_PASSED,
            CandidateStatus.VALIDATED,
        ]
        prefix = (
            self.status_history[:-1]
            if self.final_status in {CandidateStatus.REJECTED, CandidateStatus.ACCEPTED}
            else self.status_history
        )
        if prefix != ordered[: len(prefix)]:
            raise ValueError("candidate states must follow the fixed transition order")
        if self.final_status == CandidateStatus.ACCEPTED and prefix != ordered:
            raise ValueError("ACCEPTED is allowed only after VALIDATED")
        if self.final_status == CandidateStatus.REJECTED and not self.rejection_reason:
            raise ValueError("rejected candidates require a reason")
        if self.final_status != CandidateStatus.REJECTED and self.rejection_reason:
            raise ValueError("non-rejected candidates cannot carry a rejection reason")
        return self


class ResearchAgentResult(ResearchAgentModel):
    """Agent output before any formal fixed-budget candidate validation."""

    agent_run_id: str = Field(min_length=1, max_length=200)
    backend: str = Field(min_length=1, max_length=100)
    bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: CandidateStatus
    proposal: OperatorProposal | None = None
    review: OperatorReview | None = None
    selected_candidate_id: str | None = None
    candidates: list[CandidateAttempt] = Field(min_length=1, max_length=2)
    tool_calls: list[ToolExecutionResult] = Field(default_factory=list)
    usage: AgentUsage
    provider_call_ids: list[str] = Field(default_factory=list)
    sdk_trace_id: str | None = None
    # Optional multi-role metadata.  ``Any`` avoids a circular import while
    # preserving the historical result type and exact single-agent defaults.
    portfolio: Any | None = None
    role_traces: list[Any] = Field(default_factory=list)


@runtime_checkable
class DiagnoserAgent(Protocol):
    def diagnose(
        self,
        bundle: OperatorEvidenceBundle,
        *,
        call_config: LLMCallConfig,
    ) -> DiagnosisReport: ...


@runtime_checkable
class DesignerAgent(Protocol):
    def design(
        self,
        bundle: OperatorEvidenceBundle,
        diagnosis: DiagnosisReport,
        *,
        call_config: LLMCallConfig,
    ) -> OperatorProposal: ...


@runtime_checkable
class ReviewerAgent(Protocol):
    def review(
        self,
        proposal: OperatorProposal,
        bundle: OperatorEvidenceBundle,
    ) -> OperatorReview: ...


@runtime_checkable
class ResearchAgentBackend(Protocol):
    def run(
        self,
        context: AgentToolContext,
        *,
        budget: AgentBudget | None = None,
        call_config: LLMCallConfig | None = None,
        audit_sink: ToolAuditSink | None = None,
        agent_run_id: str | None = None,
    ) -> ResearchAgentResult: ...


def _candidate_id(run_id: str, index: int) -> str:
    return f"candidate_{run_id}_{index:02d}"[:200]


def _revision_bundle(
    bundle: OperatorEvidenceBundle,
    candidate_id: str,
    reason: str,
) -> OperatorEvidenceBundle:
    """Carry bounded failure feedback into the second structured design call."""

    payload = bundle.model_dump(mode="json")
    payload["bundle_hash"] = ""
    feedback = f"revision_feedback[{candidate_id}]: {reason}"[:1_000]
    payload["limitations"] = [*payload.get("limitations", []), feedback]
    return OperatorEvidenceBundle.model_validate(payload)


def _record_rejection(
    *,
    candidate_id: str,
    proposal: OperatorProposal | None,
    review: OperatorReview | None,
    history: list[CandidateStatus],
    reason: str,
    supersedes: str | None,
    compile_result: ToolExecutionResult | None = None,
    smoke_result: ToolExecutionResult | None = None,
) -> CandidateAttempt:
    if not history or history[-1] != CandidateStatus.REJECTED:
        history.append(CandidateStatus.REJECTED)
    return CandidateAttempt(
        candidate_id=candidate_id,
        proposal=proposal,
        review=review,
        status_history=history,
        final_status=CandidateStatus.REJECTED,
        rejection_reason=reason[:4_000],
        supersedes_candidate_id=supersedes,
        compile_result=compile_result,
        smoke_result=smoke_result,
    )


def _check_tool_budget(result: ToolExecutionResult) -> None:
    if result.status == "budget_exceeded":
        raise AgentBudgetExceeded(result.error or "agent tool budget exceeded")


def _evaluate_proposal(
    *,
    candidate_id: str,
    proposal: OperatorProposal,
    bundle: OperatorEvidenceBundle,
    dispatcher: AgentToolDispatcher,
    validator: ProposalValidator,
    supersedes: str | None,
) -> CandidateAttempt:
    history = [CandidateStatus.PROPOSED, CandidateStatus.SCHEMA_VALID]
    try:
        review = validator.validate_and_review(proposal, bundle, review_mode="rule_based")
    except Exception as exc:
        return _record_rejection(
            candidate_id=candidate_id,
            proposal=proposal,
            review=None,
            history=history,
            reason=f"static validation failed: {type(exc).__name__}: {exc}",
            supersedes=supersedes,
        )

    history.append(CandidateStatus.REVIEWED)
    if review.decision != "approve":
        return _record_rejection(
            candidate_id=candidate_id,
            proposal=proposal,
            review=review,
            history=history,
            reason="rule-based review requested revision: " + "; ".join(review.concerns),
            supersedes=supersedes,
        )

    arguments = {"operator_spec": proposal.spec.model_dump(mode="json")}
    compiled = dispatcher.execute("compile_operator_spec", arguments)
    _check_tool_budget(compiled)
    if compiled.status != "ok" or not compiled.payload.get("compiled"):
        return _record_rejection(
            candidate_id=candidate_id,
            proposal=proposal,
            review=review,
            history=history,
            reason=f"compile failed: {compiled.error or compiled.payload}",
            supersedes=supersedes,
            compile_result=compiled,
        )
    history.append(CandidateStatus.COMPILED)

    smoke = dispatcher.execute("run_operator_smoke_test", arguments)
    _check_tool_budget(smoke)
    if smoke.status != "ok" or not smoke.payload.get("smoke_passed", False):
        return _record_rejection(
            candidate_id=candidate_id,
            proposal=proposal,
            review=review,
            history=history,
            reason=f"smoke failed: {smoke.error or smoke.payload.get('failures', smoke.payload)}",
            supersedes=supersedes,
            compile_result=compiled,
            smoke_result=smoke,
        )

    history.append(CandidateStatus.SMOKE_PASSED)
    return CandidateAttempt(
        candidate_id=candidate_id,
        proposal=proposal,
        review=review,
        status_history=history,
        final_status=CandidateStatus.SMOKE_PASSED,
        supersedes_candidate_id=supersedes,
        compile_result=compiled,
        smoke_result=smoke,
    )


def evaluate_candidate_attempt(
    *,
    candidate_id: str,
    proposal: OperatorProposal,
    bundle: OperatorEvidenceBundle,
    dispatcher: AgentToolDispatcher,
    validator: ProposalValidator,
    supersedes_candidate_id: str | None = None,
) -> CandidateAttempt:
    """Publicly reuse the single-agent local gate without adding validation.

    Multi-role research backends use this compatibility seam so every candidate
    goes through the exact same schema/review/compiler/smoke sequence as the
    historical single-agent implementation.  It deliberately exposes neither a
    validation split nor a retention callback.
    """

    return _evaluate_proposal(
        candidate_id=candidate_id,
        proposal=proposal,
        bundle=bundle,
        dispatcher=dispatcher,
        validator=validator,
        supersedes=supersedes_candidate_id,
    )


def _evidence_queries(parent_name: str) -> tuple[tuple[str, dict[str, Any]], ...]:
    return (
        ("get_parent_operator_spec", {"operator_id": parent_name}),
        ("get_operator_profile", {"operator_id": parent_name}),
        ("get_failure_modes", {"operator_id": parent_name}),
        ("get_synergies", {"operator_id": parent_name}),
        ("get_relevant_cases", {"operator_id": parent_name, "limit": 3}),
        ("get_lineage", {"operator_id": parent_name, "direction": "both", "max_depth": 4}),
        ("get_counterfactual_results", {"operator_id": parent_name}),
        ("get_allowed_primitives", {}),
    )


class DeterministicMockResearchAgent:
    """Offline single-agent loop using structured mock calls and real tools."""

    backend_name = "deterministic_mock_research_agent"

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        validator: ProposalValidator | None = None,
    ) -> None:
        self.provider = provider or MockLLMProvider()
        self.validator = validator or ProposalValidator()
        self.designer = LLMDesignerAdapter(
            provider=self.provider,
            proposal_validator=self.validator,
        )

    def run(
        self,
        context: AgentToolContext,
        *,
        budget: AgentBudget | None = None,
        call_config: LLMCallConfig | None = None,
        audit_sink: ToolAuditSink | None = None,
        agent_run_id: str | None = None,
    ) -> ResearchAgentResult:
        bundle = context.bundle
        if not bundle.parent_specs:
            raise ValueError("research agent requires at least one parent operator")
        budget_controller = AgentBudgetController(budget or AgentBudget())
        tool_calls: list[ToolExecutionResult] = []

        def capture(result: ToolExecutionResult, arguments: Mapping[str, Any]) -> None:
            tool_calls.append(result)
            if audit_sink is not None:
                audit_sink(result, arguments)

        dispatcher = AgentToolDispatcher(context, budget_controller, audit_sink=capture)
        run_id = agent_run_id or f"mock_{bundle.bundle_hash[:16]}"
        config = call_config or LLMCallConfig()
        provider_record_start = len(self.provider.call_records)
        reset_usage = getattr(self.provider, "reset_usage", None)
        if callable(reset_usage):
            reset_usage()

        budget_controller.start_turn()
        parent_name = bundle.parent_specs[0].name
        for name, arguments in _evidence_queries(parent_name):
            evidence_result = dispatcher.execute(name, arguments)
            _check_tool_budget(evidence_result)

        attempts: list[CandidateAttempt] = []
        working_bundle = bundle
        supersedes: str | None = None
        maximum_attempts = min(2, budget_controller.budget.max_candidate_specs)

        for index in range(1, maximum_attempts + 1):
            revision = index > 1
            budget_controller.register_candidate(revision=revision)
            candidate_id = _candidate_id(run_id, index)
            before_calls = len(self.provider.call_records)
            proposal: OperatorProposal | None = None
            try:
                # Staged mode has one diagnosis and one design model turn.
                budget_controller.start_turn()
                budget_controller.start_turn()
                proposal = self.designer.propose_from_evidence(
                    working_bundle,
                    mode="staged",
                    call_config=config,
                )
                attempt = _evaluate_proposal(
                    candidate_id=candidate_id,
                    proposal=proposal,
                    bundle=working_bundle,
                    dispatcher=dispatcher,
                    validator=self.validator,
                    supersedes=supersedes,
                )
            except AgentBudgetExceeded:
                raise
            except Exception as exc:
                rejected_history = [CandidateStatus.PROPOSED]
                if isinstance(exc, (ProposalValidationError, LLMDesignError)):
                    rejected_history.append(CandidateStatus.SCHEMA_VALID)
                attempt = _record_rejection(
                    candidate_id=candidate_id,
                    proposal=proposal,
                    review=None,
                    history=rejected_history,
                    reason=f"design failed: {type(exc).__name__}: {exc}",
                    supersedes=supersedes,
                )
            finally:
                for record in self.provider.call_records[before_calls:]:
                    budget_controller.add_tokens(
                        input_tokens=record.usage.input_tokens,
                        output_tokens=record.usage.output_tokens,
                    )

            attempts.append(attempt)
            if attempt.final_status == CandidateStatus.SMOKE_PASSED:
                break
            if index >= maximum_attempts or budget_controller.budget.max_revisions <= 0:
                break
            supersedes = candidate_id
            working_bundle = _revision_bundle(
                bundle,
                candidate_id,
                attempt.rejection_reason or "candidate rejected",
            )

        selected = next(
            (item for item in reversed(attempts) if item.final_status == CandidateStatus.SMOKE_PASSED),
            None,
        )
        last = selected or attempts[-1]
        return ResearchAgentResult(
            agent_run_id=run_id,
            backend=self.backend_name,
            bundle_hash=bundle.bundle_hash,
            status=last.final_status,
            proposal=None if selected is None else selected.proposal,
            review=None if selected is None else selected.review,
            selected_candidate_id=None if selected is None else selected.candidate_id,
            candidates=attempts,
            tool_calls=tool_calls,
            usage=budget_controller.usage,
            provider_call_ids=[
                record.call_id for record in self.provider.call_records[provider_record_start:]
            ],
        )


class OpenAIAgentsSDKUnavailableError(RuntimeError):
    """Raised when the optional Agents SDK backend was selected but unavailable."""


class OpenAIAgentsResearchAgent:
    """Optional OpenAI Agents SDK backend with only ten local bounded tools."""

    backend_name = "openai_agents_research_agent"

    def __init__(
        self,
        *,
        validator: ProposalValidator | None = None,
        remote_tracing: bool = False,
        trace_include_sensitive_data: bool = False,
    ) -> None:
        self.validator = validator or ProposalValidator()
        self.remote_tracing = bool(remote_tracing)
        self.trace_include_sensitive_data = bool(trace_include_sensitive_data)

    @staticmethod
    def _load_sdk() -> tuple[Any, Any, Any, Any]:
        try:
            from agents import Agent, RunConfig, Runner, function_tool  # type: ignore[import-not-found]
        except (ImportError, ModuleNotFoundError) as exc:
            raise OpenAIAgentsSDKUnavailableError(
                "OpenAI Agents SDK is not installed; install the project's 'agent' optional dependency"
            ) from exc
        return Agent, RunConfig, Runner, function_tool

    @classmethod
    def available(cls) -> bool:
        try:
            cls._load_sdk()
        except OpenAIAgentsSDKUnavailableError:
            return False
        return True

    def _typed_tools(self, dispatcher: AgentToolDispatcher, function_tool: Any) -> list[Any]:
        def payload(name: str, arguments: BaseModel) -> str:
            result = dispatcher.execute(name, arguments.model_dump(mode="json"))
            _check_tool_budget(result)
            return result.payload_json if result.status == "ok" else canonical_json(
                {"status": result.status, "error": result.error}
            )

        @function_tool
        def get_operator_profile(query: OperatorIdInput) -> str:
            """Return the compact computed profile for one parent operator."""
            return payload("get_operator_profile", query)

        @function_tool
        def get_failure_modes(query: OperatorIdInput) -> str:
            """Return compact observed failure evidence for one parent."""
            return payload("get_failure_modes", query)

        @function_tool
        def get_synergies(query: OperatorIdInput) -> str:
            """Return compact associative synergy evidence for one parent."""
            return payload("get_synergies", query)

        @function_tool
        def get_relevant_cases(query: CaseQueryInput) -> str:
            """Return bounded scalar-only representative case summaries."""
            return payload("get_relevant_cases", query)

        @function_tool
        def get_lineage(query: LineageQueryInput) -> str:
            """Return bounded mechanism lineage from local memory."""
            return payload("get_lineage", query)

        @function_tool
        def get_counterfactual_results(query: OperatorIdInput) -> str:
            """Return compact counterfactual evidence already in this bundle."""
            return payload("get_counterfactual_results", query)

        @function_tool
        def get_allowed_primitives(query: EmptyToolInput) -> str:
            """Return the authoritative read-only DSL primitive catalog."""
            return payload("get_allowed_primitives", query)

        @function_tool
        def get_parent_operator_spec(query: OperatorIdInput) -> str:
            """Return one parent OperatorSpec from the current bundle."""
            return payload("get_parent_operator_spec", query)

        @function_tool
        def compile_operator_spec(query: OperatorSpecInput) -> str:
            """Compile a bounded OperatorSpec with the trusted interpreter."""
            return payload("compile_operator_spec", query)

        @function_tool
        def run_operator_smoke_test(query: OperatorSpecInput) -> str:
            """Run bounded contract smoke checks; this is not formal validation."""
            return payload("run_operator_smoke_test", query)

        return [
            get_operator_profile,
            get_failure_modes,
            get_synergies,
            get_relevant_cases,
            get_lineage,
            get_counterfactual_results,
            get_allowed_primitives,
            get_parent_operator_spec,
            compile_operator_spec,
            run_operator_smoke_test,
        ]

    def run(
        self,
        context: AgentToolContext,
        *,
        budget: AgentBudget | None = None,
        call_config: LLMCallConfig | None = None,
        audit_sink: ToolAuditSink | None = None,
        agent_run_id: str | None = None,
    ) -> ResearchAgentResult:
        Agent, RunConfig, Runner, function_tool = self._load_sdk()
        config = call_config or LLMCallConfig()
        model = config.model or os.getenv("UOE_LLM_MODEL")
        if not model:
            raise LLMConfigurationError("OpenAI Agents backend requires UOE_LLM_MODEL")
        openai_key = os.getenv("OPENAI_API_KEY")
        uoe_key = os.getenv("UOE_LLM_API_KEY")
        if not (openai_key or uoe_key):
            raise LLMConfigurationError(
                "OpenAI Agents backend requires OPENAI_API_KEY or UOE_LLM_API_KEY"
            )
        if not openai_key and uoe_key:
            # The SDK natively reads OPENAI_API_KEY.  Support the project-level
            # alias explicitly without copying it into audit or trace metadata.
            from agents import set_default_openai_key  # type: ignore[import-not-found]

            set_default_openai_key(
                uoe_key,
                use_for_tracing=self.remote_tracing,
            )

        bundle = context.bundle
        controller = AgentBudgetController(budget or AgentBudget())
        tool_calls: list[ToolExecutionResult] = []

        def capture(result: ToolExecutionResult, arguments: Mapping[str, Any]) -> None:
            tool_calls.append(result)
            if audit_sink is not None:
                audit_sink(result, arguments)

        dispatcher = AgentToolDispatcher(context, controller, audit_sink=capture)
        tools = self._typed_tools(dispatcher, function_tool)
        from agents import ModelSettings  # type: ignore[import-not-found]

        agent = Agent(
            name="Operator Research Agent",
            instructions=RESEARCH_AGENT_V1.system_text,
            model=model,
            model_settings=ModelSettings(
                max_tokens=config.max_output_tokens,
                parallel_tool_calls=False,
            ),
            output_type=OperatorProposal,
            tools=tools,
        )
        run_id = agent_run_id or f"openai_agent_{bundle.bundle_hash[:16]}"
        sdk_trace_id = f"trace_{stable_hash({'agent_run_id': run_id})[:32]}"
        try:
            run_config = RunConfig(
                tracing_disabled=not self.remote_tracing,
                trace_include_sensitive_data=self.trace_include_sensitive_data,
                workflow_name="UAV Operator Research Agent",
                trace_id=sdk_trace_id,
            )
        except TypeError:
            # Older compatible SDK builds may not expose the sensitive-data
            # switch; tracing remains disabled by default in that case.
            run_config = RunConfig(tracing_disabled=not self.remote_tracing)

        attempts: list[CandidateAttempt] = []
        working_bundle = bundle
        supersedes: str | None = None
        trace_id: str | None = sdk_trace_id
        maximum_attempts = min(2, controller.budget.max_candidate_specs)

        for index in range(1, maximum_attempts + 1):
            revision = index > 1
            controller.register_candidate(revision=revision)
            candidate_id = _candidate_id(run_id, index)
            remaining_turns = controller.budget.max_turns - controller.usage.turns
            # Reserve one turn for the only permitted revision. Runner itself
            # applies this per-attempt limit; the local controller tracks the
            # actual response count as a second aggregate limit.
            reserve_revision = int(
                index == 1
                and maximum_attempts > 1
                and controller.budget.max_revisions > 0
            )
            sdk_turn_limit = max(1, remaining_turns - reserve_revision)
            controller.start_turn()
            prompt = canonical_json(
                {
                    "task": "Return one complete evidence-grounded OperatorProposal.",
                    "bundle": working_bundle.model_dump(mode="json"),
                    "formal_validation_available": False,
                    "supersedes_candidate_id": supersedes,
                }
            )
            proposal: OperatorProposal | None = None
            try:
                result = Runner.run_sync(
                    agent,
                    input=prompt,
                    max_turns=sdk_turn_limit,
                    run_config=run_config,
                )
                trace_id = str(getattr(result, "trace_id", "")) or trace_id
                raw_responses = getattr(result, "raw_responses", None)
                observed_turns = len(raw_responses) if isinstance(raw_responses, list) else 1
                for _ in range(max(0, min(sdk_turn_limit, observed_turns) - 1)):
                    controller.start_turn()
                usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
                controller.add_tokens(
                    input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                )
                if controller.usage.total_tokens > config.max_total_tokens:
                    raise LLMTokenBudgetError(
                        "OpenAI Agents run exceeded the local cumulative token budget"
                    )
                proposal = OperatorProposal.model_validate(getattr(result, "final_output", None))
                candidate = _evaluate_proposal(
                    candidate_id=candidate_id,
                    proposal=proposal,
                    bundle=working_bundle,
                    dispatcher=dispatcher,
                    validator=self.validator,
                    supersedes=supersedes,
                )
            except AgentBudgetExceeded:
                raise
            except Exception as exc:
                rejected_history = [CandidateStatus.PROPOSED]
                if isinstance(exc, (ProposalValidationError, LLMDesignError)):
                    rejected_history.append(CandidateStatus.SCHEMA_VALID)
                candidate = _record_rejection(
                    candidate_id=candidate_id,
                    proposal=proposal,
                    review=None,
                    history=rejected_history,
                    reason=f"agent design failed: {type(exc).__name__}: {exc}",
                    supersedes=supersedes,
                )
            attempts.append(candidate)
            if candidate.final_status == CandidateStatus.SMOKE_PASSED:
                break
            if index >= maximum_attempts or controller.budget.max_revisions <= 0:
                break
            supersedes = candidate_id
            working_bundle = _revision_bundle(
                bundle,
                candidate_id,
                candidate.rejection_reason or "candidate rejected",
            )

        selected = next(
            (item for item in reversed(attempts) if item.final_status == CandidateStatus.SMOKE_PASSED),
            None,
        )
        last = selected or attempts[-1]
        return ResearchAgentResult(
            agent_run_id=run_id,
            backend=self.backend_name,
            bundle_hash=bundle.bundle_hash,
            status=last.final_status,
            proposal=None if selected is None else selected.proposal,
            review=None if selected is None else selected.review,
            selected_candidate_id=None if selected is None else selected.candidate_id,
            candidates=attempts,
            tool_calls=tool_calls,
            usage=controller.usage,
            sdk_trace_id=trace_id,
        )


__all__ = [
    "CandidateAttempt",
    "DesignerAgent",
    "DeterministicMockResearchAgent",
    "DiagnoserAgent",
    "OpenAIAgentsResearchAgent",
    "OpenAIAgentsSDKUnavailableError",
    "ResearchAgentBackend",
    "ResearchAgentResult",
    "ReviewerAgent",
    "evaluate_candidate_attempt",
]
