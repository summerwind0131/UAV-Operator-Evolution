"""Bounded offline multi-role research coordinated by deterministic Python.

One diagnoser, one exploitation designer, one exploration designer, and one
critic share an evidence bundle and local tool/budget boundary.  The critic
assesses candidates but cannot select them.  Python applies a fixed portfolio
score, stable tie-break, compiler, and smoke gate; formal validation remains an
orchestrator-only responsibility.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..reproducibility import stable_hash
from .design_models import CandidateStatus, DiagnosisReport, OperatorReview
from .designer_base import OperatorProposal
from .prompts import (
    DIAGNOSER_V1,
    EXPLOITATION_DESIGNER_V1,
    EXPLORATION_DESIGNER_V1,
    PORTFOLIO_CRITIC_V1,
    PromptTemplate,
)
from .proposal_validation import ProposalValidator
from .providers import LLMCallConfig, LLMProvider, LLMUsage, MockLLMProvider
from .research_agent import (
    CandidateAttempt,
    ResearchAgentResult,
    evaluate_candidate_attempt,
)
from .tools import (
    AgentBudget,
    AgentBudgetController,
    AgentBudgetExceeded,
    AgentToolContext,
    AgentToolDispatcher,
    AgentUsage,
    ToolAuditSink,
    ToolExecutionResult,
)


MultiAgentRole = Literal[
    "diagnoser",
    "exploitation_designer",
    "exploration_designer",
    "critic",
]
DesignerRole = Literal["exploitation_designer", "exploration_designer"]
CriticDecision = Literal["approve", "revise", "reject"]


def _exclude_runtime_fields(value: Any) -> Any:
    """Remove wall-clock measurements from content-addressed portfolio data."""

    if isinstance(value, dict):
        return {
            key: _exclude_runtime_fields(item)
            for key, item in value.items()
            if key not in {"latency_ms", "runtime_ms", "created_at", "completed_at"}
        }
    if isinstance(value, list):
        return [_exclude_runtime_fields(item) for item in value]
    return value


class MultiAgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CandidateCritique(MultiAgentModel):
    """Critic assessment for one candidate; it contains no selection field."""

    candidate_id: str = Field(min_length=1, max_length=200)
    decision: CriticDecision
    evidence_alignment_score: float = Field(ge=0.0, le=1.0)
    safety_score: float = Field(ge=0.0, le=1.0)
    testability_score: float = Field(ge=0.0, le=1.0)
    mechanism_fit_score: float = Field(ge=0.0, le=1.0)
    causal_overclaim: bool = False
    evidence_ids: list[str] = Field(default_factory=list, max_length=64)
    strengths: list[str] = Field(default_factory=list, max_length=16)
    concerns: list[str] = Field(default_factory=list, max_length=16)
    required_revisions: list[str] = Field(default_factory=list, max_length=16)


class PortfolioCritique(MultiAgentModel):
    """Exactly two per-candidate assessments; deterministic Python selects later."""

    assessments: list[CandidateCritique] = Field(min_length=2, max_length=2)
    comparative_rationale: str = Field(min_length=1, max_length=4_000)
    used_evidence_ids: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def two_unique_assessments(self) -> "PortfolioCritique":
        identifiers = [item.candidate_id for item in self.assessments]
        if len(set(identifiers)) != 2:
            raise ValueError("critic must assess two unique candidates")
        object.__setattr__(
            self,
            "assessments",
            sorted(self.assessments, key=lambda item: item.candidate_id),
        )
        object.__setattr__(self, "used_evidence_ids", sorted(set(self.used_evidence_ids)))
        return self


class PortfolioScoreComponents(MultiAgentModel):
    evidence_alignment: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)
    topology_diversity: float = Field(ge=0.0, le=1.0)
    priority_failure_coverage: float = Field(ge=0.0, le=1.0)
    testability: float = Field(ge=0.0, le=1.0)

    def weighted_score(self) -> float:
        return (
            0.30 * self.evidence_alignment
            + 0.20 * self.safety
            + 0.20 * self.topology_diversity
            + 0.15 * self.priority_failure_coverage
            + 0.15 * self.testability
        )


class PortfolioCandidate(MultiAgentModel):
    candidate_id: str = Field(min_length=1, max_length=200)
    role: DesignerRole
    proposal: OperatorProposal
    static_review: OperatorReview
    critique: CandidateCritique
    attempt: CandidateAttempt
    topology_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    eligible: bool
    score_components: PortfolioScoreComponents
    portfolio_score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def score_matches_components(self) -> "PortfolioCandidate":
        expected = self.score_components.weighted_score()
        if abs(self.portfolio_score - expected) > 1e-12:
            raise ValueError("portfolio_score does not match the fixed weighted formula")
        if self.critique.candidate_id != self.candidate_id:
            raise ValueError("candidate critique ID mismatch")
        if self.attempt.candidate_id != self.candidate_id:
            raise ValueError("candidate attempt ID mismatch")
        if self.attempt.proposal != self.proposal or self.attempt.review != self.static_review:
            raise ValueError("candidate attempt must carry the reviewed proposal")
        if self.static_review.topology_fingerprint != self.topology_fingerprint:
            raise ValueError("topology fingerprint must come from deterministic static review")
        return self


class CandidatePortfolio(MultiAgentModel):
    portfolio_id: str = ""
    portfolio_hash: str = ""
    bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnosis: DiagnosisReport
    diagnosis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    critic_report: PortfolioCritique
    candidates: list[PortfolioCandidate] = Field(min_length=2, max_length=2)
    selected_candidate_id: str | None = Field(default=None, max_length=200)
    selection_reason: str = Field(min_length=1, max_length=4_000)
    selection_policy: str = (
        "weighted_evidence_0.30_safety_0.20_topology_0.20_"
        "priority_failure_0.15_testability_0.15_then_role_candidate_id"
    )

    def canonical_payload(self, *, include_id: bool = False) -> dict[str, Any]:
        excluded = {"portfolio_hash"} if include_id else {"portfolio_id", "portfolio_hash"}
        return _exclude_runtime_fields(self.model_dump(mode="json", exclude=excluded))

    @model_validator(mode="after")
    def canonicalize_and_hash(self) -> "CandidatePortfolio":
        if self.diagnosis_hash != stable_hash(self.diagnosis.model_dump(mode="json")):
            raise ValueError("diagnosis_hash does not match the shared diagnosis")
        object.__setattr__(
            self,
            "candidates",
            sorted(self.candidates, key=lambda item: item.candidate_id),
        )
        identifiers = [item.candidate_id for item in self.candidates]
        if len(set(identifiers)) != 2:
            raise ValueError("portfolio requires two unique sibling candidates")
        if {item.candidate_id for item in self.critic_report.assessments} != set(identifiers):
            raise ValueError("critic report must cover exactly the portfolio candidates")
        if {item.role for item in self.candidates} != {
            "exploitation_designer",
            "exploration_designer",
        }:
            raise ValueError("portfolio requires exploitation and exploration candidates")
        eligible = {item.candidate_id for item in self.candidates if item.eligible}
        if self.selected_candidate_id is not None and self.selected_candidate_id not in eligible:
            raise ValueError("selected candidate must be eligible")
        base_payload = self.canonical_payload()
        expected_id = f"portfolio_{stable_hash(base_payload)[:24]}"
        hash_payload = self.canonical_payload(include_id=True)
        hash_payload["portfolio_id"] = expected_id
        expected_hash = stable_hash(hash_payload)
        if self.portfolio_hash and self.portfolio_hash != expected_hash:
            raise ValueError("portfolio_hash does not match canonical portfolio content")
        if self.portfolio_id and self.portfolio_id != expected_id:
            raise ValueError("portfolio_id does not match canonical portfolio content")
        object.__setattr__(self, "portfolio_hash", expected_hash)
        object.__setattr__(self, "portfolio_id", expected_id)
        return self


class MultiAgentRoleTrace(MultiAgentModel):
    role: MultiAgentRole
    action: Literal["diagnose", "design", "review"]
    turn: int = Field(ge=1, le=4)
    prompt_version: str = Field(min_length=1, max_length=100)
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_model: str = Field(min_length=1, max_length=200)
    provider_call_id: str | None = Field(default=None, max_length=200)
    response_id: str | None = Field(default=None, max_length=300)
    candidate_id: str | None = Field(default=None, max_length=200)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    latency_ms: float = Field(ge=0.0)
    status: Literal["succeeded", "failed"]
    error: str | None = Field(default=None, max_length=2_000)


@dataclass(frozen=True, slots=True)
class ExploitationDesigner:
    """Deterministic role policy for conservative parent refinement."""

    role: DesignerRole = "exploitation_designer"
    action: Literal["design"] = "design"
    template: PromptTemplate = EXPLOITATION_DESIGNER_V1

    def payload(
        self,
        context: AgentToolContext,
        diagnosis: DiagnosisReport,
        diagnosis_hash: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        return {
            "task": "return one sibling candidate; do not revise another candidate",
            "design_role": self.role,
            "bundle": context.bundle.model_dump(mode="json"),
            "diagnosis": diagnosis.model_dump(mode="json"),
            "diagnosis_hash": diagnosis_hash,
            "candidate_id": candidate_id,
        }


@dataclass(frozen=True, slots=True)
class ExplorationDesigner:
    """Deterministic role policy for topology-diverse bounded exploration."""

    role: DesignerRole = "exploration_designer"
    action: Literal["design"] = "design"
    template: PromptTemplate = EXPLORATION_DESIGNER_V1

    def payload(
        self,
        context: AgentToolContext,
        diagnosis: DiagnosisReport,
        diagnosis_hash: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        return {
            "task": "return one sibling candidate; do not revise another candidate",
            "design_role": self.role,
            "bundle": context.bundle.model_dump(mode="json"),
            "diagnosis": diagnosis.model_dump(mode="json"),
            "diagnosis_hash": diagnosis_hash,
            "candidate_id": candidate_id,
        }


@dataclass(frozen=True, slots=True)
class DeterministicCritic:
    """Adversarial structured reviewer; it deliberately has no select method."""

    role: MultiAgentRole = "critic"
    action: Literal["review"] = "review"
    template: PromptTemplate = PORTFOLIO_CRITIC_V1

    def payload(
        self,
        *,
        bundle_hash: str,
        diagnosis_hash: str,
        candidates: list[dict[str, Any]],
        used_evidence_ids: list[str],
    ) -> dict[str, Any]:
        return {
            "task": "assess each candidate without selecting either candidate",
            "bundle_hash": bundle_hash,
            "diagnosis_hash": diagnosis_hash,
            "candidates": candidates,
            "used_evidence_ids": used_evidence_ids,
            "formal_validation_available": False,
        }


@dataclass(slots=True)
class _CoordinatorState:
    run_id: str
    controller: AgentBudgetController
    tool_calls: list[ToolExecutionResult] = field(default_factory=list)
    traces: list[MultiAgentRoleTrace] = field(default_factory=list)
    proposals: dict[str, OperatorProposal] = field(default_factory=dict)
    reviews: dict[str, OperatorReview] = field(default_factory=dict)
    attempts: dict[str, CandidateAttempt] = field(default_factory=dict)


MULTI_AGENT_BUDGET = AgentBudget(
    max_turns=4,
    max_tool_calls=12,
    max_candidate_specs=2,
    max_revisions=0,
    max_smoke_tests=2,
)


def _effective_budget(requested: AgentBudget | None) -> AgentBudget:
    if requested is None:
        return MULTI_AGENT_BUDGET
    requirements = {
        "max_turns": 4,
        "max_tool_calls": 12,
        "max_candidate_specs": 2,
        "max_smoke_tests": 2,
    }
    for field, minimum in requirements.items():
        if int(getattr(requested, field)) < minimum:
            raise AgentBudgetExceeded(
                f"offline multi-agent topology requires {field}>={minimum}"
            )
    return MULTI_AGENT_BUDGET


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


def _validate_diagnosis(diagnosis: DiagnosisReport, context: AgentToolContext) -> None:
    if diagnosis.parent_operator not in {spec.name for spec in context.bundle.parent_specs}:
        raise ValueError("diagnosis references a parent outside the evidence bundle")
    known = set(context.bundle.evidence_ids())
    cited = {
        evidence_id
        for claims in (
            diagnosis.effective_mechanisms,
            diagnosis.failure_modes,
            diagnosis.useful_synergies,
        )
        for claim in claims
        for evidence_id in claim.evidence_ids
    }
    unknown = sorted(cited - known)
    if unknown:
        raise ValueError(f"diagnosis references unknown evidence IDs: {unknown}")


def _critic_rejected_attempt(
    candidate_id: str,
    proposal: OperatorProposal,
    review: OperatorReview,
    critique: CandidateCritique,
) -> CandidateAttempt:
    critic_reason = (
        "critic_reject: causal overclaim"
        if critique.causal_overclaim
        else (
            f"critic_{critique.decision}: "
            + ("; ".join(critique.concerns) or "candidate was not approved")
        )
    )
    return CandidateAttempt(
        candidate_id=candidate_id,
        proposal=proposal,
        review=review,
        status_history=[
            CandidateStatus.PROPOSED,
            CandidateStatus.SCHEMA_VALID,
            CandidateStatus.REVIEWED,
            CandidateStatus.REJECTED,
        ],
        final_status=CandidateStatus.REJECTED,
        rejection_reason=critic_reason,
    )


def _portfolio_not_selected(attempt: CandidateAttempt) -> CandidateAttempt:
    history = list(attempt.status_history)
    if history[-1] != CandidateStatus.REJECTED:
        history.append(CandidateStatus.REJECTED)
    prior = attempt.rejection_reason
    reason = "portfolio_not_selected"
    if prior:
        reason += f": {prior}"
    return attempt.model_copy(
        update={
            "status_history": history,
            "final_status": CandidateStatus.REJECTED,
            "rejection_reason": reason,
            "supersedes_candidate_id": None,
        }
    )


class MultiAgentCoordinator:
    """Coordinate four structured roles while retaining all decisions in Python."""

    backend_name = "deterministic_mock_multi_agent"

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        validator: ProposalValidator | None = None,
        designers: tuple[ExploitationDesigner | ExplorationDesigner, ...] | None = None,
        critic: DeterministicCritic | None = None,
    ) -> None:
        self.provider = provider or MockLLMProvider()
        self.validator = validator or ProposalValidator()
        self.designers = designers or (ExploitationDesigner(), ExplorationDesigner())
        if tuple(designer.role for designer in self.designers) != (
            "exploitation_designer",
            "exploration_designer",
        ):
            raise ValueError("multi-agent requires exploitation then exploration designers")
        self.critic = critic or DeterministicCritic()

    def run(
        self,
        context: AgentToolContext,
        *,
        budget: AgentBudget | None = None,
        call_config: LLMCallConfig | None = None,
        audit_sink: ToolAuditSink | None = None,
        agent_run_id: str | None = None,
    ) -> ResearchAgentResult:
        if not context.bundle.parent_specs:
            raise ValueError("multi-agent research requires at least one parent operator")
        state = _CoordinatorState(
            run_id=agent_run_id or f"multi_{context.bundle.bundle_hash[:16]}",
            controller=AgentBudgetController(_effective_budget(budget)),
        )
        try:
            return self._run_bounded(
                context,
                state=state,
                call_config=call_config,
                audit_sink=audit_sink,
            )
        except Exception as exc:
            return self._failed_result(context, state, exc)

    def _run_bounded(
        self,
        context: AgentToolContext,
        *,
        state: _CoordinatorState,
        call_config: LLMCallConfig | None,
        audit_sink: ToolAuditSink | None,
    ) -> ResearchAgentResult:
        if not context.bundle.failure_modes:
            raise ValueError(
                "multi-agent safely rejected: no persisted failure-mode evidence"
            )
        controller = state.controller
        tool_calls = state.tool_calls

        def capture(result: ToolExecutionResult, arguments: Mapping[str, Any]) -> None:
            tool_calls.append(result)
            if audit_sink is not None:
                audit_sink(result, arguments)

        dispatcher = AgentToolDispatcher(context, controller, audit_sink=capture)
        parent_name = context.bundle.parent_specs[0].name
        for tool_name, arguments in _evidence_queries(parent_name):
            result = dispatcher.execute(tool_name, arguments)
            if result.status == "budget_exceeded":
                raise AgentBudgetExceeded(result.error or "shared tool budget exceeded")
            if result.status != "ok":
                raise RuntimeError(
                    f"shared evidence tool {tool_name} failed: {result.error or result.status}"
                )

        reset_usage = getattr(self.provider, "reset_usage", None)
        if callable(reset_usage):
            reset_usage()
        config = call_config or LLMCallConfig()
        traces = state.traces

        diagnosis = DiagnosisReport.model_validate(
            self._call_role(
                controller,
                traces,
                role="diagnoser",
                action="diagnose",
                template=DIAGNOSER_V1,
                payload={
                    "task": "produce one shared evidence-grounded diagnosis",
                    "bundle": context.bundle.model_dump(mode="json"),
                },
                output_model=DiagnosisReport,
                config=config,
            )
        )
        _validate_diagnosis(diagnosis, context)
        diagnosis_hash = stable_hash(diagnosis.model_dump(mode="json"))

        run_id = state.run_id
        proposals = state.proposals
        static_reviews = state.reviews
        candidate_roles: dict[str, DesignerRole] = {}
        for index, designer in enumerate(self.designers, start=1):
            role = designer.role
            controller.register_candidate(revision=False)
            candidate_id = f"candidate_{run_id}_{index:02d}"[:200]
            proposal = OperatorProposal.model_validate(
                self._call_role(
                    controller,
                    traces,
                    role=role,
                    action=designer.action,
                    template=designer.template,
                    payload=designer.payload(
                        context, diagnosis, diagnosis_hash, candidate_id
                    ),
                    output_model=OperatorProposal,
                    config=config,
                    candidate_id=candidate_id,
                )
            )
            if proposal.diagnosis is None or stable_hash(
                proposal.diagnosis.model_dump(mode="json")
            ) != diagnosis_hash:
                raise ValueError(f"{role} did not preserve the shared diagnosis")
            static_review = self.validator.validate_and_review(
                proposal,
                context.bundle,
                review_mode="rule_based",
            )
            proposals[candidate_id] = proposal
            static_reviews[candidate_id] = static_review
            candidate_roles[candidate_id] = role

        proposal_values = list(proposals.values())
        if stable_hash(proposal_values[0].spec.model_dump(mode="json")) == stable_hash(
            proposal_values[1].spec.model_dump(mode="json")
        ):
            raise ValueError("exploitation and exploration proposals must be distinct")

        used_evidence = sorted(
            {
                evidence_id
                for proposal in proposal_values
                for evidence_id in proposal.used_evidence_ids
            }
        )
        critique = PortfolioCritique.model_validate(
            self._call_role(
                controller,
                traces,
                role=self.critic.role,
                action=self.critic.action,
                template=self.critic.template,
                payload=self.critic.payload(
                    bundle_hash=context.bundle.bundle_hash,
                    diagnosis_hash=diagnosis_hash,
                    candidates=[
                        {
                            "candidate_id": candidate_id,
                            "role": candidate_roles[candidate_id],
                            "operator_spec": proposal.spec.model_dump(mode="json"),
                            "static_review": static_reviews[candidate_id].model_dump(mode="json"),
                        }
                        for candidate_id, proposal in proposals.items()
                    ],
                    used_evidence_ids=used_evidence,
                ),
                output_model=PortfolioCritique,
                config=config,
            )
        )
        critiques = {item.candidate_id: item for item in critique.assessments}
        if set(critiques) != set(proposals):
            raise ValueError("critic must assess exactly the two sibling candidates")
        critic_evidence = {
            evidence_id
            for item in critique.assessments
            for evidence_id in item.evidence_ids
        } | set(critique.used_evidence_ids)
        if not critic_evidence.issubset(set(context.bundle.evidence_ids())):
            raise ValueError("critic cited evidence outside the shared bundle")

        # Critic gate comes before compile/smoke.  It can reject or request a
        # revision, but this topology never performs a revision.
        attempts = state.attempts
        for candidate_id, proposal in proposals.items():
            review = static_reviews[candidate_id]
            critic = critiques[candidate_id]
            if (
                critic.decision != "approve"
                or critic.causal_overclaim
                or review.decision != "approve"
            ):
                attempts[candidate_id] = _critic_rejected_attempt(
                    candidate_id, proposal, review, critic
                )
                continue
            attempts[candidate_id] = evaluate_candidate_attempt(
                candidate_id=candidate_id,
                proposal=proposal,
                bundle=context.bundle,
                dispatcher=dispatcher,
                validator=self.validator,
            )

        fingerprints: dict[str, str] = {}
        for candidate_id, review in static_reviews.items():
            if review.topology_fingerprint is None:
                raise RuntimeError("deterministic static review omitted topology fingerprint")
            fingerprints[candidate_id] = review.topology_fingerprint
        fingerprint_counts = {
            fingerprint: list(fingerprints.values()).count(fingerprint)
            for fingerprint in set(fingerprints.values())
        }
        failure_claims = [claim.claim for claim in diagnosis.failure_modes]
        component_map: dict[str, PortfolioScoreComponents] = {}
        preliminary_eligible: dict[str, bool] = {}
        for candidate_id, proposal in proposals.items():
            review = static_reviews[candidate_id]
            critic = critiques[candidate_id]
            target = (
                None
                if proposal.hypothesis is None
                else proposal.hypothesis.target_failure_mode
            )
            failure_coverage = (
                1.0
                if failure_claims and target == failure_claims[0]
                else 0.5
                if target in failure_claims[1:]
                else 0.0
            )
            topology_diversity = (
                0.0
                if fingerprint_counts[fingerprints[candidate_id]] > 1
                else 1.0
                if review.lineage_relation == "structural_variant"
                else 0.35
            )
            component_map[candidate_id] = PortfolioScoreComponents(
                evidence_alignment=min(
                    review.evidence_alignment_score,
                    critic.evidence_alignment_score,
                ),
                safety=min(review.safety_score, critic.safety_score),
                topology_diversity=topology_diversity,
                priority_failure_coverage=failure_coverage,
                testability=min(review.testability_score, critic.testability_score),
            )
            preliminary_eligible[candidate_id] = (
                critic.decision == "approve"
                and not critic.causal_overclaim
                and review.decision == "approve"
                and attempts[candidate_id].final_status == CandidateStatus.SMOKE_PASSED
            )

        role_order = {"exploitation_designer": 0, "exploration_designer": 1}
        duplicate_allowed = set(proposals)
        for fingerprint, count in fingerprint_counts.items():
            if count <= 1:
                continue
            group = [
                candidate_id
                for candidate_id, value in fingerprints.items()
                if value == fingerprint and preliminary_eligible[candidate_id]
            ]
            if len(group) > 1:
                keeper = sorted(
                    group,
                    key=lambda candidate_id: (
                        -component_map[candidate_id].weighted_score(),
                        role_order[candidate_roles[candidate_id]],
                        candidate_id,
                    ),
                )[0]
                duplicate_allowed.difference_update(set(group) - {keeper})

        eligibility = {
            candidate_id: preliminary_eligible[candidate_id]
            and candidate_id in duplicate_allowed
            for candidate_id in proposals
        }
        eligible_ids = [candidate_id for candidate_id in proposals if eligibility[candidate_id]]
        selected_id = (
            sorted(
                eligible_ids,
                key=lambda candidate_id: (
                    -component_map[candidate_id].weighted_score(),
                    role_order[candidate_roles[candidate_id]],
                    candidate_id,
                ),
            )[0]
            if eligible_ids
            else None
        )

        ordered_attempts: list[CandidateAttempt] = []
        final_attempts: dict[str, CandidateAttempt] = {}
        for candidate_id in proposals:
            attempt = attempts[candidate_id]
            if (
                candidate_id != selected_id
                and attempt.final_status == CandidateStatus.SMOKE_PASSED
            ):
                attempt = _portfolio_not_selected(attempt)
            ordered_attempts.append(attempt)
            final_attempts[candidate_id] = attempt

        portfolio_candidates = [
            PortfolioCandidate(
                candidate_id=candidate_id,
                role=candidate_roles[candidate_id],
                proposal=proposals[candidate_id],
                static_review=static_reviews[candidate_id],
                critique=critiques[candidate_id],
                attempt=final_attempts[candidate_id],
                topology_fingerprint=fingerprints[candidate_id],
                eligible=eligibility[candidate_id],
                score_components=component_map[candidate_id],
                portfolio_score=component_map[candidate_id].weighted_score(),
            )
            for candidate_id in proposals
        ]
        selection_reason = (
            "no candidate passed static review, critic, compile, smoke, and duplicate gates"
            if selected_id is None
            else (
                f"selected {selected_id} with deterministic portfolio score "
                f"{component_map[selected_id].weighted_score():.6f}"
            )
        )
        portfolio = CandidatePortfolio(
            bundle_hash=context.bundle.bundle_hash,
            diagnosis=diagnosis,
            diagnosis_hash=diagnosis_hash,
            critic_report=critique,
            candidates=portfolio_candidates,
            selected_candidate_id=selected_id,
            selection_reason=selection_reason,
        )

        selected_attempt = next(
            (item for item in ordered_attempts if item.candidate_id == selected_id),
            None,
        )
        selected_proposal = None if selected_attempt is None else selected_attempt.proposal
        selected_review = None if selected_attempt is None else selected_attempt.review
        return ResearchAgentResult(
            agent_run_id=run_id,
            backend=self.backend_name,
            bundle_hash=context.bundle.bundle_hash,
            status=(
                CandidateStatus.REJECTED
                if selected_attempt is None
                else CandidateStatus.SMOKE_PASSED
            ),
            proposal=selected_proposal,
            review=selected_review,
            selected_candidate_id=selected_id,
            candidates=ordered_attempts,
            tool_calls=tool_calls,
            usage=controller.usage,
            provider_call_ids=[
                trace.provider_call_id
                for trace in traces
                if trace.provider_call_id is not None
            ],
            portfolio=portfolio,
            role_traces=traces,
        )

    def _failed_result(
        self,
        context: AgentToolContext,
        state: _CoordinatorState,
        error: Exception,
    ) -> ResearchAgentResult:
        """Convert every bounded execution failure into audited terminal siblings."""

        reason = f"multi_agent_failed: {type(error).__name__}: {error}"[:4_000]
        attempts: list[CandidateAttempt] = []
        for index, _designer in enumerate(self.designers, start=1):
            candidate_id = f"candidate_{state.run_id}_{index:02d}"[:200]
            existing = state.attempts.get(candidate_id)
            if existing is not None:
                if existing.final_status == CandidateStatus.REJECTED:
                    attempts.append(existing)
                    continue
                history = list(existing.status_history)
                history.append(CandidateStatus.REJECTED)
                attempts.append(
                    existing.model_copy(
                        update={
                            "status_history": history,
                            "final_status": CandidateStatus.REJECTED,
                            "rejection_reason": reason,
                            "supersedes_candidate_id": None,
                        }
                    )
                )
                continue

            proposal = state.proposals.get(candidate_id)
            review = state.reviews.get(candidate_id)
            history = [CandidateStatus.PROPOSED]
            if proposal is not None:
                history.append(CandidateStatus.SCHEMA_VALID)
            if review is not None:
                history.append(CandidateStatus.REVIEWED)
            history.append(CandidateStatus.REJECTED)
            attempts.append(
                CandidateAttempt(
                    candidate_id=candidate_id,
                    proposal=proposal,
                    review=review,
                    status_history=history,
                    final_status=CandidateStatus.REJECTED,
                    rejection_reason=reason,
                    supersedes_candidate_id=None,
                )
            )

        return ResearchAgentResult(
            agent_run_id=state.run_id,
            backend=self.backend_name,
            bundle_hash=context.bundle.bundle_hash,
            status=CandidateStatus.REJECTED,
            proposal=None,
            review=None,
            selected_candidate_id=None,
            candidates=attempts,
            tool_calls=state.tool_calls,
            usage=state.controller.usage,
            provider_call_ids=[
                trace.provider_call_id
                for trace in state.traces
                if trace.provider_call_id is not None
            ],
            portfolio=None,
            role_traces=state.traces,
        )

    def _call_role(
        self,
        controller: AgentBudgetController,
        traces: list[MultiAgentRoleTrace],
        *,
        role: MultiAgentRole,
        action: Literal["diagnose", "design", "review"],
        template: PromptTemplate,
        payload: dict[str, Any],
        output_model: type[BaseModel],
        config: LLMCallConfig,
        candidate_id: str | None = None,
    ) -> BaseModel:
        controller.start_turn()
        before = len(self.provider.call_records)
        started = time.perf_counter()
        output: BaseModel | None = None
        failure: Exception | None = None
        try:
            output = self.provider.generate_structured(
                system_prompt=template.system_text,
                user_payload=payload,
                output_model=output_model,
                config=config,
                prompt_version=template.version,
                prompt_hash=template.prompt_hash,
            )
            return output
        except Exception as exc:
            failure = exc
            raise
        finally:
            records = self.provider.call_records[before:]
            for record in records:
                controller.add_tokens(
                    input_tokens=record.usage.input_tokens,
                    output_tokens=record.usage.output_tokens,
                )
            record = records[-1] if records else None
            traces.append(
                MultiAgentRoleTrace(
                    role=role,
                    action=action,
                    turn=len(traces) + 1,
                    prompt_version=template.version,
                    prompt_hash=template.prompt_hash,
                    input_hash=stable_hash(payload),
                    output_hash=(
                        None
                        if output is None
                        else stable_hash(output.model_dump(mode="json", by_alias=True))
                    ),
                    output_model=output_model.__name__,
                    provider_call_id=None if record is None else record.call_id,
                    response_id=None if record is None else record.response_id,
                    candidate_id=candidate_id,
                    usage=LLMUsage() if record is None else record.usage,
                    latency_ms=(time.perf_counter() - started) * 1_000.0,
                    status="failed" if failure is not None else "succeeded",
                    error=None if failure is None else f"{type(failure).__name__}: {failure}",
                )
            )


class DeterministicMockMultiAgent(MultiAgentCoordinator):
    """Named offline backend; defaults to the deterministic mock provider."""

    backend_name = "deterministic_mock_multi_agent"

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        validator: ProposalValidator | None = None,
        designers: tuple[ExploitationDesigner | ExplorationDesigner, ...] | None = None,
        critic: DeterministicCritic | None = None,
    ) -> None:
        super().__init__(
            provider or MockLLMProvider(),
            validator=validator,
            designers=designers,
            critic=critic,
        )


# Compatibility spelling retained for the initial Phase-8 research naming.
DeterministicMockMultiAgentResearch = DeterministicMockMultiAgent
RoleTrace = MultiAgentRoleTrace


__all__ = [
    "CandidateCritique",
    "CandidatePortfolio",
    "DeterministicCritic",
    "DeterministicMockMultiAgent",
    "DeterministicMockMultiAgentResearch",
    "MULTI_AGENT_BUDGET",
    "ExploitationDesigner",
    "ExplorationDesigner",
    "MultiAgentCoordinator",
    "MultiAgentRole",
    "MultiAgentRoleTrace",
    "PortfolioCandidate",
    "PortfolioCritique",
    "PortfolioScoreComponents",
    "RoleTrace",
]
