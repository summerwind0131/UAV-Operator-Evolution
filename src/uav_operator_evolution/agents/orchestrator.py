"""Deterministic, evidence-grounded operator design orchestration.

Formal paired validation is deliberately called only by this module.  Agent
backends receive a compact evidence bundle plus compile/contract-smoke tools,
never a validation dataset or retention decision tool.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from operator_evolution_core.proposal import DomainKit, ensure_domain_compatibility

from ..domain.uav_kit import UAVDomainKit
from ..environment import Environment2D
from ..evolution.candidate_validator import FixedBudgetCandidateValidator
from ..evolution.validation import ValidationReport
from ..memory import MechanismMemory
from ..operators.base import PathOperator
from ..operators.compiler import OperatorCompiler
from ..operators.registry import OperatorRegistry
from ..path.initializer import initialize_path
from ..reproducibility import stable_hash
from ..trajectory import TrajectoryRecorder
from .audit import (
    AgentAuditStore,
    AgentBudget as AuditAgentBudget,
    AgentRunRecord,
    AgentToolCallRecord,
    AgentUsage as AuditAgentUsage,
    AuthorizationDecision,
    CandidatePortfolioRecord,
    CandidateEventRecord,
    EvidenceBundleRecord,
    LLMCallRecord as AuditLLMCallRecord,
    ModelUsage,
    MultiAgentRoleEventRecord,
    MultiAgentRunRecord,
)
from .designer_base import OperatorProposal
from .design_models import CandidateStatus, OperatorReview
from .evidence import DesignBudget, EvidenceBundleBuilder, OperatorEvidenceBundle
from .heuristic_designer import HeuristicDesigner
from .llm_designer import LLMDesignerAdapter
from .prompts import DESIGNER_V1, DIAGNOSER_V1, REVIEWER_V1, get_prompt_template
from .proposal_validation import ProposalValidator
from .providers import LLMCallConfig
from .research_agent import ResearchAgentBackend, ResearchAgentResult
from .tools import (
    AgentBudget as ResearchAgentBudget,
    AgentToolContext,
    SmokeTestFixture,
    ToolExecutionResult,
)

DesignMode = Literal[
    "llm_single", "llm_staged", "single_agent", "multi_agent", "heuristic"
]
ReviewMode = Literal["none", "rule_based", "llm"]


class OrchestratorModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class OperatorDesignRequest(OrchestratorModel):
    """All inputs needed for one candidate decision, excluding held-out test data."""

    request_id: str = Field(min_length=1, max_length=120)
    experiment_id: str = Field(min_length=1, max_length=200)
    root_run_id: str = Field(min_length=1, max_length=200)
    problem_summary: str = Field(min_length=1, max_length=4_000)
    parent_operator_ids: list[str] = Field(min_length=1, max_length=4)
    smoke_environment: Environment2D
    validation_environments: list[Environment2D] = Field(min_length=1)
    design_mode: DesignMode = "llm_single"
    review_mode: ReviewMode = "rule_based"
    generation: int = Field(default=1, ge=1)
    candidate_index: int = Field(default=0, ge=0)
    population_operator_names: list[str] | None = None
    parent_profiles: list[dict[str, Any]] = Field(default_factory=list)
    counterfactual_results: list[Any] = Field(default_factory=list)
    counterfactual_seed: int | None = Field(default=None, ge=0)
    design_budget: DesignBudget = Field(default_factory=DesignBudget)
    llm_call_config: LLMCallConfig = Field(default_factory=LLMCallConfig)
    research_agent_budget: ResearchAgentBudget = Field(default_factory=ResearchAgentBudget)

    @model_validator(mode="after")
    def split_boundaries_are_explicit(self) -> "OperatorDesignRequest":
        names = [name.strip() for name in self.parent_operator_ids]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("parent_operator_ids must be unique non-empty names")
        validation_ids = {environment.map_id for environment in self.validation_environments}
        validation_hashes = {environment.content_hash for environment in self.validation_environments}
        if len(validation_ids) != len(self.validation_environments):
            raise ValueError("validation environment map_ids must be unique")
        if self.smoke_environment.map_id in validation_ids:
            raise ValueError("smoke environment must not be a validation environment")
        if self.smoke_environment.content_hash in validation_hashes:
            raise ValueError("smoke and validation environments must have different contents")
        return self


class OperatorDesignOrchestrationResult(OrchestratorModel):
    request_id: str
    experiment_id: str
    agent_run_id: str
    candidate_id: str
    operator_name: str | None = None
    final_status: CandidateStatus
    outcome: Literal["accepted", "rejected"]
    rejection_stage: str | None = None
    reason: str
    bundle_id: str
    bundle_hash: str
    proposal: OperatorProposal | None = None
    review: OperatorReview | None = None
    validation_report: ValidationReport | None = None
    lineage_relation: Literal["structural_variant", "parameter_variant"] | None = None
    compiled: bool = False
    smoke_passed: bool = False
    retained: bool = False
    mechanism_id: str | None = None
    insight_id: int | None = None
    lineage_ids: list[int] = Field(default_factory=list)
    rejection_evidence_ids: list[int] = Field(default_factory=list)
    audit_event_ids: list[str] = Field(default_factory=list)
    candidate_portfolio: dict[str, Any] | None = None
    multi_agent_role_traces: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class _CandidateDraft:
    candidate_id: str
    proposal: OperatorProposal | None = None
    statuses: list[CandidateStatus] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    def append(
        self,
        status: CandidateStatus,
        reason: str,
        **details: Any,
    ) -> None:
        if self.statuses and self.statuses[-1] == status:
            return
        self.statuses.append(status)
        self.reasons.append(str(reason)[:20_000])
        self.details.append(dict(details))

    @property
    def final_status(self) -> CandidateStatus | None:
        return self.statuses[-1] if self.statuses else None


class _Rejected(RuntimeError):
    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = str(reason)[:20_000]


class OperatorDesignOrchestrator:
    """Execute one fixed design, validation, retention, memory, and audit flow."""

    def __init__(
        self,
        *,
        evidence_builder: EvidenceBundleBuilder,
        proposal_validator: ProposalValidator,
        compiler: OperatorCompiler,
        candidate_validator: FixedBudgetCandidateValidator,
        memory: MechanismMemory,
        registry: OperatorRegistry,
        llm_designer: LLMDesignerAdapter | None = None,
        research_agent_backend: ResearchAgentBackend | None = None,
        heuristic_designer: HeuristicDesigner | None = None,
        audit_store: AgentAuditStore | None = None,
        recorder: TrajectoryRecorder | None = None,
        domain_kit: DomainKit[Any, Any, Any] | None = None,
    ) -> None:
        self.evidence_builder = evidence_builder
        self.proposal_validator = proposal_validator
        self.compiler = compiler
        self.domain_kit = domain_kit or UAVDomainKit(compiler)
        for component in (evidence_builder, proposal_validator):
            component_kit = getattr(component, "domain_kit", None)
            if component_kit is not None and (
                component_kit.domain_id != self.domain_kit.domain_id
                or component_kit.ir_version != self.domain_kit.ir_version
            ):
                raise ValueError(
                    "orchestrator components must use the same domain and IR version"
                )
        self.candidate_validator = candidate_validator
        self.memory = memory
        self.registry = registry
        self.llm_designer = llm_designer
        self.research_agent_backend = research_agent_backend
        self.heuristic_designer = heuristic_designer
        self.audit_store = audit_store
        self.recorder = recorder

    def run(self, request: OperatorDesignRequest) -> OperatorDesignOrchestrationResult:
        """Run one candidate decision using only the supplied validation split."""

        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        bundle = self.evidence_builder.build(
            request.problem_summary,
            request.parent_operator_ids,
            request.design_budget,
            parent_profiles=request.parent_profiles,
            counterfactual_results=request.counterfactual_results,
            counterfactual_seed=request.counterfactual_seed,
        )
        self._verify_resolved_parents(request, bundle)
        bundle_id = self._persist_bundle(request, bundle)
        population = self._resolve_population(request, bundle)
        primary_parent = request.parent_operator_ids[0]
        agent_run_id = self._agent_run_id(request)
        provider_snapshots = self._provider_snapshots(request.design_mode, request.review_mode)
        self._reset_direct_provider_usage(request, provider_snapshots)
        tool_audits: list[tuple[ToolExecutionResult, Mapping[str, Any]]] = []
        research_result: ResearchAgentResult | None = None
        drafts: "OrderedDict[str, _CandidateDraft]" = OrderedDict()
        candidate_id = f"candidate_{request.request_id}_01"[:200]
        proposal: OperatorProposal | None = None
        review: OperatorReview | None = None
        validation_report: ValidationReport | None = None
        relation: Literal["structural_variant", "parameter_variant"] | None = None
        compiled_operator: PathOperator | None = None
        mechanism_id: str | None = None
        insight_id: int | None = None
        lineage_ids: list[int] = []
        rejection_stage: str | None = None
        rejection_reason: str | None = None
        warnings: list[str] = []

        try:
            if request.design_mode in {"single_agent", "multi_agent"}:
                agent_stage = request.design_mode
                research_result = self._run_research_agent(
                    request,
                    bundle,
                    agent_run_id,
                    tool_audits,
                )
                agent_run_id = research_result.agent_run_id
                self._ingest_agent_attempts(research_result, drafts)
                candidate_id = research_result.candidates[-1].candidate_id
                if research_result.bundle_hash != bundle.bundle_hash:
                    raise _Rejected(
                        agent_stage,
                        "research agent returned a result for a different evidence bundle",
                    )
                if research_result.selected_candidate_id is None or research_result.proposal is None:
                    reason = research_result.candidates[-1].rejection_reason or "agent produced no smoke-passed proposal"
                    raise _Rejected(agent_stage, reason)
                candidate_id = research_result.selected_candidate_id
                if candidate_id not in drafts:
                    candidate_id = research_result.candidates[-1].candidate_id
                    raise _Rejected(
                        agent_stage,
                        "research agent selected an unknown candidate id",
                    )
                proposal = research_result.proposal
                draft = drafts[candidate_id]
                if draft.final_status != CandidateStatus.SMOKE_PASSED:
                    raise _Rejected(
                        agent_stage,
                        "research agent selected a candidate that did not pass its smoke stage",
                    )
                if draft.proposal is None or (
                    stable_hash(draft.proposal.model_dump(mode="json", by_alias=True))
                    != stable_hash(proposal.model_dump(mode="json", by_alias=True))
                ):
                    raise _Rejected(
                        agent_stage,
                        "selected candidate proposal does not match the agent result proposal",
                    )
            else:
                draft = _CandidateDraft(candidate_id=candidate_id)
                drafts[candidate_id] = draft
                draft.append(CandidateStatus.PROPOSED, "candidate payload returned by explicit design arm")
                try:
                    raw = self._run_direct_designer(request, bundle)
                except _Rejected:
                    raise
                except Exception as exc:
                    raise _Rejected(
                        "design",
                        f"designer failed: {type(exc).__name__}: {exc}",
                    ) from exc
                draft.details[0]["raw_proposal"] = (
                    raw.model_dump(mode="json", by_alias=True)
                    if isinstance(raw, BaseModel)
                    else raw
                )
                try:
                    proposal = raw if isinstance(raw, OperatorProposal) else OperatorProposal.model_validate(raw)
                except (ValidationError, ValueError, TypeError) as exc:
                    raise _Rejected("schema", f"proposal schema validation failed: {type(exc).__name__}: {exc}") from exc
                draft.proposal = proposal
                proposal_payload = proposal.model_dump(mode="json", by_alias=True)
                draft.details[0]["proposal"] = proposal_payload
                draft.append(
                    CandidateStatus.SCHEMA_VALID,
                    "OperatorProposal Pydantic schema passed",
                    proposal=proposal_payload,
                )

            assert proposal is not None
            drafts[candidate_id].proposal = proposal
            if proposal.spec.name in self.registry or self.memory.get_mechanism(proposal.spec.name) is not None:
                raise _Rejected("hard_validation", f"operator name already exists: {proposal.spec.name}")

            try:
                static_review = self.proposal_validator.validate_and_review(
                    proposal,
                    bundle,
                    review_mode="none" if request.review_mode == "llm" else request.review_mode,
                )
            except Exception as exc:
                raise _Rejected("hard_validation", f"proposal evidence validation failed: {exc}") from exc
            if request.review_mode == "llm":
                if self.llm_designer is None:
                    raise _Rejected("review", "llm review mode requires LLMDesignerAdapter")
                try:
                    model_review = self.llm_designer.review_from_evidence(
                        bundle,
                        proposal,
                        request.llm_call_config,
                    )
                except Exception as exc:
                    raise _Rejected(
                        "review",
                        f"LLM review failed: {type(exc).__name__}: {exc}",
                    ) from exc
                # The model supplies review scores and concerns.  Structural
                # novelty and lineage are deterministic hard-rule outputs and
                # cannot be overridden by a reviewer response.
                review = model_review.model_copy(
                    update={
                        "novelty_score": static_review.novelty_score,
                        "lineage_relation": static_review.lineage_relation,
                        "topology_fingerprint": static_review.topology_fingerprint,
                    }
                )
            else:
                review = static_review
            if drafts[candidate_id].final_status == CandidateStatus.SCHEMA_VALID:
                drafts[candidate_id].append(
                    CandidateStatus.REVIEWED,
                    f"review decision={review.decision}",
                    review=review.model_dump(mode="json"),
                )
            relation = review.lineage_relation or "structural_variant"
            if request.review_mode != "none" and review.decision != "approve":
                concerns = "; ".join(review.concerns) or "review did not approve candidate"
                raise _Rejected("review", concerns)

            try:
                proposal_envelope = proposal.to_envelope(
                    candidate_id,
                    {
                        "candidate_specs": request.design_budget.max_candidate_specs,
                        "validation_instances": len(request.validation_environments),
                        "smoke_seeds": 3,
                    },
                )
                ensure_domain_compatibility(self.domain_kit, proposal_envelope)
                self.domain_kit.parse_ir(proposal_envelope.payload)
            except Exception as exc:
                raise _Rejected(
                    "proposal_envelope",
                    f"proposal envelope validation failed: {type(exc).__name__}: {exc}",
                ) from exc

            try:
                compiled_operator = self.domain_kit.compile(
                    self.domain_kit.parse_ir(proposal.spec)
                )
            except Exception as exc:
                raise _Rejected("compile", f"operator compilation failed: {type(exc).__name__}: {exc}") from exc
            if drafts[candidate_id].final_status == CandidateStatus.REVIEWED:
                drafts[candidate_id].append(
                    CandidateStatus.COMPILED,
                    "trusted DSL compiler passed",
                    compile={
                        "operator_name": str(compiled_operator.name),
                        "compiler": type(self.compiler).__name__,
                    },
                )

            try:
                smoke_failures = self.candidate_validator.contract_failures(
                    compiled_operator,
                    request.smoke_environment,
                    generation=request.generation,
                    candidate_index=request.candidate_index,
                )
            except Exception as exc:
                raise _Rejected("smoke", f"contract smoke raised {type(exc).__name__}: {exc}") from exc
            if smoke_failures:
                raise _Rejected("smoke", "contract smoke failed: " + "; ".join(smoke_failures))
            if drafts[candidate_id].final_status == CandidateStatus.COMPILED:
                drafts[candidate_id].append(
                    CandidateStatus.SMOKE_PASSED,
                    "independent contract smoke passed",
                    smoke={
                        "map_id": request.smoke_environment.map_id,
                        "map_content_hash": request.smoke_environment.content_hash,
                        "failures": [],
                    },
                )

            try:
                validation_report = self.candidate_validator.validate(
                    population,
                    primary_parent,
                    compiled_operator,
                    request.validation_environments,
                    generation=request.generation,
                    candidate_index=request.candidate_index,
                    recorder=self.recorder,
                    root_run_id=request.root_run_id,
                )
            except Exception as exc:
                raise _Rejected("validation", f"formal validation raised {type(exc).__name__}: {exc}") from exc
            drafts[candidate_id].append(
                CandidateStatus.VALIDATED,
                "fixed-budget validation split evaluation completed",
                validation_report=validation_report.model_dump(mode="json"),
            )
            if not validation_report.retained:
                raise _Rejected(
                    "retention",
                    "; ".join(validation_report.retention_reasons) or "candidate not retained",
                )

            try:
                mechanism_id, insight_id, lineage_ids = self._persist_acceptance(
                    request,
                    bundle,
                    proposal,
                    review,
                    validation_report,
                    compiled_operator,
                    candidate_id,
                    relation,
                )
            except Exception as exc:
                raise _Rejected("memory", f"accepted candidate persistence failed: {type(exc).__name__}: {exc}") from exc
            drafts[candidate_id].append(
                CandidateStatus.ACCEPTED,
                "retention gate passed and mechanism was persisted",
                mechanism_id=mechanism_id,
            )
        except _Rejected as rejected:
            rejection_stage = rejected.stage
            rejection_reason = rejected.reason
            draft = drafts.get(candidate_id)
            if draft is None:
                draft = _CandidateDraft(candidate_id=candidate_id)
                drafts[candidate_id] = draft
                draft.append(CandidateStatus.PROPOSED, "design attempt started")
            if draft.final_status not in {CandidateStatus.ACCEPTED, CandidateStatus.REJECTED}:
                draft.append(
                    CandidateStatus.REJECTED,
                    rejected.reason,
                    stage=rejected.stage,
                )

        rejection_ids = self._persist_rejection_evidence(
            request,
            bundle,
            drafts,
            selected_candidate_id=candidate_id,
        )
        accepted = drafts[candidate_id].final_status == CandidateStatus.ACCEPTED
        final_reason = (
            "candidate accepted by fixed validation retention policy"
            if accepted
            else rejection_reason or drafts[candidate_id].reasons[-1]
        )
        provider_records = self._provider_records(provider_snapshots)
        audit_event_ids = self._persist_audit(
            request=request,
            bundle=bundle,
            bundle_id=bundle_id,
            agent_run_id=agent_run_id,
            drafts=drafts,
            proposal=proposal,
            review=review,
            research_result=research_result,
            provider_records=provider_records,
            tool_audits=tool_audits,
            accepted=accepted,
            rejection_stage=rejection_stage,
            started_at=started_at,
            wall_time_ms=(time.perf_counter() - started_clock) * 1_000.0,
        )
        return OperatorDesignOrchestrationResult(
            request_id=request.request_id,
            experiment_id=request.experiment_id,
            agent_run_id=agent_run_id,
            candidate_id=candidate_id,
            operator_name=None if proposal is None else proposal.spec.name,
            final_status=drafts[candidate_id].final_status or CandidateStatus.REJECTED,
            outcome="accepted" if accepted else "rejected",
            rejection_stage=None if accepted else rejection_stage,
            reason=final_reason,
            bundle_id=bundle_id,
            bundle_hash=bundle.bundle_hash,
            proposal=proposal,
            review=review,
            validation_report=validation_report,
            lineage_relation=relation,
            compiled=compiled_operator is not None,
            smoke_passed=CandidateStatus.SMOKE_PASSED in drafts[candidate_id].statuses,
            retained=accepted,
            mechanism_id=mechanism_id,
            insight_id=insight_id,
            lineage_ids=lineage_ids,
            rejection_evidence_ids=rejection_ids,
            audit_event_ids=audit_event_ids,
            candidate_portfolio=(
                None
                if research_result is None or research_result.portfolio is None
                else research_result.portfolio.model_dump(mode="json")
            ),
            multi_agent_role_traces=(
                []
                if research_result is None
                else [
                    item.model_dump(mode="json")
                    for item in research_result.role_traces
                ]
            ),
            warnings=warnings,
        )

    @staticmethod
    def _agent_run_id(request: OperatorDesignRequest) -> str:
        return f"agent_{request.request_id}"[:200]

    @staticmethod
    def _verify_resolved_parents(
        request: OperatorDesignRequest,
        bundle: OperatorEvidenceBundle,
    ) -> None:
        resolved = {spec.name for spec in bundle.parent_specs}
        missing = [name for name in request.parent_operator_ids if name not in resolved]
        if missing:
            raise KeyError(f"evidence bundle did not resolve requested parents: {missing}")

    def _resolve_population(
        self,
        request: OperatorDesignRequest,
        bundle: OperatorEvidenceBundle,
    ) -> list[PathOperator]:
        names = request.population_operator_names or list(self.registry.names())
        population: list[PathOperator] = []
        for name in names:
            population.append(self.registry.get(name))
        specs = {spec.name: spec for spec in bundle.parent_specs}
        for parent in request.parent_operator_ids:
            if not any(str(operator.name) == parent for operator in population):
                population.append(
                    self.domain_kit.compile(self.domain_kit.parse_ir(specs[parent]))
                )
        if not any(str(operator.name) == request.parent_operator_ids[0] for operator in population):
            raise ValueError("primary parent is absent from validation population")
        return population

    def _persist_bundle(
        self,
        request: OperatorDesignRequest,
        bundle: OperatorEvidenceBundle,
    ) -> str:
        bundle_id = f"bundle_{bundle.bundle_hash[:32]}"
        if self.audit_store is None:
            return bundle_id
        existing = self.audit_store.get_evidence_bundle(bundle_id)
        if existing is not None:
            if existing.bundle_hash != bundle.bundle_hash:
                raise ValueError("content-addressed evidence bundle id collision")
            return bundle_id
        payload = bundle.model_dump(mode="json", exclude={"bundle_hash"})
        self.audit_store.record_evidence_bundle(
            EvidenceBundleRecord(
                bundle_id=bundle_id,
                experiment_id=request.experiment_id,
                run_id=request.root_run_id,
                bundle=payload,
                bundle_hash=bundle.bundle_hash,
                metadata={
                    "request_id": request.request_id,
                    "parent_operator_ids": request.parent_operator_ids,
                    "design_mode": request.design_mode,
                },
            )
        )
        return bundle_id

    def _run_direct_designer(
        self,
        request: OperatorDesignRequest,
        bundle: OperatorEvidenceBundle,
    ) -> Any:
        if request.design_mode in {"llm_single", "llm_staged"}:
            if self.llm_designer is None:
                raise _Rejected("design", "explicit LLM design mode requires LLMDesignerAdapter")
            return self.llm_designer.propose_from_evidence(
                bundle,
                mode="single_call" if request.design_mode == "llm_single" else "staged",
                call_config=request.llm_call_config,
            )
        if request.design_mode == "heuristic":
            if self.heuristic_designer is None:
                raise _Rejected(
                    "design",
                    "heuristic compatibility mode requires an explicitly injected HeuristicDesigner",
                )
            insights = [
                insight
                for parent in request.parent_operator_ids
                for insight in self.memory.get_insights(parent, limit=8)
            ]
            return self.heuristic_designer.propose(
                request.problem_summary,
                list(bundle.parent_specs),
                list(bundle.parent_profiles),
                insights,
                [item.model_dump(mode="json") for item in bundle.representative_success_cases],
                [item.model_dump(mode="json") for item in bundle.representative_failure_cases],
            )
        raise _Rejected("design", f"unsupported direct design mode: {request.design_mode}")

    def _run_research_agent(
        self,
        request: OperatorDesignRequest,
        bundle: OperatorEvidenceBundle,
        agent_run_id: str,
        tool_audits: list[tuple[ToolExecutionResult, Mapping[str, Any]]],
    ) -> ResearchAgentResult:
        if self.research_agent_backend is None:
            raise _Rejected(
                request.design_mode,
                f"{request.design_mode} mode requires ResearchAgentBackend",
            )
        validator_config = getattr(self.candidate_validator, "config", None)
        map_config = getattr(validator_config, "maps", None)
        grid_resolution = float(getattr(map_config, "grid_resolution", 4.0))
        smoke_path = initialize_path(
            request.smoke_environment,
            grid_resolution=grid_resolution,
        )
        context = AgentToolContext(
            bundle=bundle,
            domain_kit=self.domain_kit,
            memory=self.memory,
            smoke_fixture=SmokeTestFixture(
                environment=request.smoke_environment,
                path=smoke_path,
            ),
        )

        def capture(result: ToolExecutionResult, arguments: Mapping[str, Any]) -> None:
            tool_audits.append((result, dict(arguments)))

        try:
            return self.research_agent_backend.run(
                context,
                budget=request.research_agent_budget,
                call_config=request.llm_call_config,
                audit_sink=capture,
                agent_run_id=agent_run_id,
            )
        except _Rejected:
            raise
        except Exception as exc:
            raise _Rejected(
                request.design_mode,
                f"research agent failed: {type(exc).__name__}: {exc}",
            ) from exc

    @staticmethod
    def _ingest_agent_attempts(
        result: ResearchAgentResult,
        drafts: "OrderedDict[str, _CandidateDraft]",
    ) -> None:
        is_multi_agent = result.backend == "deterministic_mock_multi_agent"
        label = "multi-agent" if is_multi_agent else "single-agent"
        reasons = {
            CandidateStatus.PROPOSED: f"{label} proposed candidate",
            CandidateStatus.SCHEMA_VALID: f"{label} schema validation passed",
            CandidateStatus.REVIEWED: f"{label} rule review completed",
            CandidateStatus.COMPILED: f"{label} compile tool passed",
            CandidateStatus.SMOKE_PASSED: f"{label} bounded smoke tool passed",
        }
        for attempt in result.candidates:
            draft = _CandidateDraft(candidate_id=attempt.candidate_id, proposal=attempt.proposal)
            for status in attempt.status_history:
                reason = (
                    attempt.rejection_reason
                    if status == CandidateStatus.REJECTED
                    else reasons.get(status, f"{label} state {status.value}")
                )
                # Candidate revision lineage belongs in the append-only audit
                # even when the original proposal is rejected before compile.
                details: dict[str, Any] = {
                    "supersedes_candidate_id": attempt.supersedes_candidate_id
                }
                if status in {CandidateStatus.PROPOSED, CandidateStatus.SCHEMA_VALID}:
                    details["proposal"] = (
                        None
                        if attempt.proposal is None
                        else attempt.proposal.model_dump(mode="json", by_alias=True)
                    )
                elif status == CandidateStatus.REVIEWED:
                    details["review"] = (
                        None if attempt.review is None else attempt.review.model_dump(mode="json")
                    )
                elif status == CandidateStatus.COMPILED:
                    details["compile"] = (
                        None
                        if attempt.compile_result is None
                        else attempt.compile_result.model_dump(mode="json")
                    )
                elif status == CandidateStatus.SMOKE_PASSED:
                    details["smoke"] = (
                        None
                        if attempt.smoke_result is None
                        else attempt.smoke_result.model_dump(mode="json")
                    )
                elif status == CandidateStatus.REJECTED:
                    details.update(
                        stage="multi_agent" if is_multi_agent else "single_agent",
                        review=None
                        if attempt.review is None
                        else attempt.review.model_dump(mode="json"),
                        compile=None
                        if attempt.compile_result is None
                        else attempt.compile_result.model_dump(mode="json"),
                        smoke=None
                        if attempt.smoke_result is None
                        else attempt.smoke_result.model_dump(mode="json"),
                    )
                draft.append(status, reason or f"{label} state {status.value}", **details)
            drafts[attempt.candidate_id] = draft

    def _persist_acceptance(
        self,
        request: OperatorDesignRequest,
        bundle: OperatorEvidenceBundle,
        proposal: OperatorProposal,
        review: OperatorReview,
        report: ValidationReport,
        compiled: PathOperator,
        candidate_id: str,
        relation: Literal["structural_variant", "parameter_variant"],
    ) -> tuple[str, int, list[int]]:
        for parent_spec in bundle.parent_specs:
            if self.memory.get_mechanism(parent_spec.name) is None:
                self.memory.add_mechanism(
                    mechanism_id=parent_spec.name,
                    name=parent_spec.name,
                    description=parent_spec.description,
                    definition=parent_spec.model_dump(mode="json"),
                    status="active",
                    tags=["parent"],
                    metadata={"source": "resolved_parent"},
                )
        spec = proposal.spec
        mechanism_id = self.memory.add_mechanism(
            mechanism_id=spec.name,
            name=spec.name,
            description=spec.description,
            definition=spec.model_dump(mode="json"),
            status="active",
            score=report.mean_gain,
            evidence_count=len(bundle.evidence_ids()),
            success_rate=report.candidate_feasibility_rate,
            tags=["evolved", relation, f"generation:{request.generation}"],
            metadata={
                "candidate_id": candidate_id,
                "request_id": request.request_id,
                "bundle_hash": bundle.bundle_hash,
                "review": review.model_dump(mode="json"),
                "validation_report": report.model_dump(mode="json"),
            },
        )
        lineage_ids = [
            self.memory.add_lineage(
                parent,
                mechanism_id,
                relation=relation,
                metadata={
                    "candidate_id": candidate_id,
                    "bundle_hash": bundle.bundle_hash,
                    "retained": True,
                },
            )
            for parent in spec.parent_operators
        ]
        confidence = min(1.0, max(0.0, len(report.outcomes) / 20.0))
        if report.evidence_level == "statistical":
            confidence = 1.0
        insight_id = self.memory.add_insight(
            operator_id=mechanism_id,
            insight_type="improvement_hypothesis",
            evidence={
                "bundle_hash": bundle.bundle_hash,
                "used_evidence_ids": proposal.used_evidence_ids,
                "validation_report": report.model_dump(mode="json"),
            },
            confidence=confidence,
            applicable_context={
                "expected": None
                if proposal.hypothesis is None
                else proposal.hypothesis.expected_effective_context
            },
            failure_context={"target_failure_modes": proposal.target_failure_modes},
        )
        self.registry.register(compiled)
        return mechanism_id, insight_id, lineage_ids

    def _persist_rejection_evidence(
        self,
        request: OperatorDesignRequest,
        bundle: OperatorEvidenceBundle,
        drafts: "OrderedDict[str, _CandidateDraft]",
        *,
        selected_candidate_id: str,
    ) -> list[int]:
        identifiers: list[int] = []
        for candidate_id, draft in drafts.items():
            if draft.final_status != CandidateStatus.REJECTED:
                continue
            proposal = draft.proposal
            operator_id = proposal.spec.name if proposal is not None else candidate_id
            identifiers.append(
                self.memory.add_failure_mode(
                    "candidate_rejected",
                    operator_id=operator_id,
                    count=1,
                    severity=1.0,
                    context={
                        "stage": draft.details[-1].get(
                            "stage",
                            request.design_mode
                            if request.design_mode in {"single_agent", "multi_agent"}
                            else "design",
                        ),
                        "generation": request.generation,
                        "selected_candidate": candidate_id == selected_candidate_id,
                    },
                    evidence=[
                        {
                            "bundle_hash": bundle.bundle_hash,
                            "status_history": [status.value for status in draft.statuses],
                            "reason": draft.reasons[-1],
                        }
                    ],
                    metadata={
                        "candidate_id": candidate_id,
                        "request_id": request.request_id,
                        "proposal": None
                        if proposal is None
                        else proposal.model_dump(mode="json", by_alias=True),
                    },
                )
            )
        return identifiers

    def _provider_snapshots(
        self,
        mode: DesignMode,
        review_mode: ReviewMode,
    ) -> list[tuple[Any, int]]:
        providers: list[Any] = []
        if mode in {"llm_single", "llm_staged"} and self.llm_designer is not None:
            providers.append(getattr(self.llm_designer, "provider", None))
        elif mode in {"single_agent", "multi_agent"} and self.research_agent_backend is not None:
            providers.append(getattr(self.research_agent_backend, "provider", None))
        if review_mode == "llm" and self.llm_designer is not None:
            providers.append(getattr(self.llm_designer, "provider", None))
        unique = [
            provider
            for index, provider in enumerate(providers)
            if provider is not None
            and all(provider is not earlier for earlier in providers[:index])
        ]
        return [
            (provider, len(getattr(provider, "call_records", ())))
            for provider in unique
        ]

    def _reset_direct_provider_usage(
        self,
        request: OperatorDesignRequest,
        snapshots: Sequence[tuple[Any, int]],
    ) -> None:
        agent_provider = (
            getattr(self.research_agent_backend, "provider", None)
            if request.design_mode in {"single_agent", "multi_agent"}
            else None
        )
        for provider, _ in snapshots:
            # The research backend owns one reset at the start of its whole
            # revision loop, so the orchestrator must not split that budget.
            if provider is agent_provider:
                continue
            reset_usage = getattr(provider, "reset_usage", None)
            if callable(reset_usage):
                reset_usage()

    @staticmethod
    def _provider_records(snapshots: Sequence[tuple[Any, int]]) -> list[Any]:
        records: list[Any] = []
        for provider, offset in snapshots:
            records.extend(list(getattr(provider, "call_records", ()))[offset:])
        return records

    def _persist_multi_agent_audit(
        self,
        *,
        request: OperatorDesignRequest,
        research_result: ResearchAgentResult,
        bundle_id: str,
        provider_call_ids: Mapping[str, str],
        started_at: datetime,
    ) -> None:
        if self.audit_store is None:
            return
        portfolio = research_result.portfolio
        traces = list(research_result.role_traces)
        failure_reason = next(
            (trace.error for trace in reversed(traces) if trace.error),
            None,
        ) or next(
            (
                attempt.rejection_reason
                for attempt in research_result.candidates
                if attempt.rejection_reason
            ),
            None,
        )
        multi_run_id = self._bounded_audit_id(
            research_result.agent_run_id,
            "multi",
            0,
            "no_portfolio" if portfolio is None else portfolio.portfolio_id,
        )
        payload = (
            None if portfolio is None else portfolio.canonical_payload(include_id=True)
        )
        budget = request.research_agent_budget
        usage = research_result.usage
        self.audit_store.record_multi_agent_run(
            MultiAgentRunRecord(
                multi_agent_run_id=multi_run_id,
                agent_run_id=research_result.agent_run_id,
                coordinator_version="multi_agent_coordinator_v1",
                bundle_id=bundle_id,
                bundle_hash=research_result.bundle_hash,
                budget=AuditAgentBudget(
                    max_steps=budget.max_turns,
                    max_tool_calls=budget.max_tool_calls,
                    max_llm_calls=4,
                    max_tokens=request.llm_call_config.max_total_tokens,
                ),
                usage=AuditAgentUsage(
                    steps=usage.turns,
                    tool_calls=usage.tool_calls,
                    llm_calls=len(research_result.provider_call_ids),
                    tokens=usage.total_tokens,
                ),
                portfolio_id=None if portfolio is None else portfolio.portfolio_id,
                portfolio=payload,
                portfolio_hash=None if portfolio is None else portfolio.portfolio_hash,
                selected_candidate_id=(
                    None if portfolio is None else portfolio.selected_candidate_id
                ),
                selection_reason=(
                    failure_reason or "multi-agent failed before portfolio selection"
                    if portfolio is None
                    else portfolio.selection_reason
                ),
                status="failed" if portfolio is None else "completed",
                error=failure_reason if portfolio is None else None,
                metadata={
                    "formal_validation_exposed_as_tool": False,
                    "portfolio_created": portfolio is not None,
                    "candidate_count": len(research_result.candidates),
                },
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
        )
        if portfolio is not None and payload is not None:
            self.audit_store.record_candidate_portfolio(
                CandidatePortfolioRecord(
                    portfolio_id=portfolio.portfolio_id,
                    multi_agent_run_id=multi_run_id,
                    bundle_hash=research_result.bundle_hash,
                    portfolio=payload,
                    portfolio_hash=portfolio.portfolio_hash,
                    selected_candidate_id=portfolio.selected_candidate_id,
                    selection_reason=portfolio.selection_reason,
                )
            )
        portfolio_candidates = {
            item.candidate_id: item for item in portfolio.candidates
        } if portfolio is not None else {}
        for sequence, trace in enumerate(traces):
            candidate = (
                None
                if trace.candidate_id is None
                else portfolio_candidates.get(trace.candidate_id)
            )
            if portfolio is None:
                action = trace.action
                output_summary = {
                    "output_model": trace.output_model,
                    "provider_output_available": trace.output_hash is not None,
                    "status": trace.status,
                }
            elif trace.role == "diagnoser":
                action = "diagnose"
                output_summary: Any = portfolio.diagnosis.model_dump(mode="json")
            elif trace.role in {"exploitation_designer", "exploration_designer"}:
                action = "design"
                output_summary = (
                    None
                    if candidate is None
                    else candidate.proposal.model_dump(mode="json", by_alias=True)
                )
            else:
                action = "review"
                output_summary = portfolio.critic_report.model_dump(mode="json")
            self.audit_store.record_multi_agent_role_event(
                MultiAgentRoleEventRecord(
                    role_event_id=self._bounded_audit_id(
                        multi_run_id, "role", sequence, trace.role
                    ),
                    multi_agent_run_id=multi_run_id,
                    sequence=sequence,
                    agent_role=trace.role,
                    action=action,
                    candidate_id=trace.candidate_id,
                    prompt_version=trace.prompt_version,
                    prompt_hash=trace.prompt_hash,
                    provider_call_id=(
                        None
                        if trace.provider_call_id is None
                            else provider_call_ids.get(trace.provider_call_id)
                    ),
                    input_hash=trace.input_hash,
                    output_hash=trace.output_hash,
                    input_summary={
                        "bundle_hash": research_result.bundle_hash,
                        "candidate_id": trace.candidate_id,
                        "output_model": trace.output_model,
                    },
                    output_summary=output_summary,
                    tokens=trace.usage.total_tokens,
                    latency_ms=trace.latency_ms,
                    status=trace.status,
                    error=trace.error,
                )
            )
        self.audit_store.record_multi_agent_role_event(
            MultiAgentRoleEventRecord(
                role_event_id=self._bounded_audit_id(
                    multi_run_id, "role", len(traces), "coordinator"
                ),
                multi_agent_run_id=multi_run_id,
                sequence=len(traces),
                agent_role="coordinator",
                action="select",
                input_summary={
                    "portfolio_hash": (
                        None if portfolio is None else portfolio.portfolio_hash
                    ),
                    "scores": (
                        {}
                        if portfolio is None
                        else {
                            item.candidate_id: item.portfolio_score
                            for item in portfolio.candidates
                        }
                    ),
                },
                output_summary={
                    "selected_candidate_id": (
                        None if portfolio is None else portfolio.selected_candidate_id
                    ),
                    "selection_reason": (
                        failure_reason or "multi-agent failed before portfolio selection"
                        if portfolio is None
                        else portfolio.selection_reason
                    ),
                },
                status="failed" if portfolio is None else "succeeded",
                error=failure_reason if portfolio is None else None,
            )
        )

    def _persist_audit(
        self,
        *,
        request: OperatorDesignRequest,
        bundle: OperatorEvidenceBundle,
        bundle_id: str,
        agent_run_id: str,
        drafts: "OrderedDict[str, _CandidateDraft]",
        proposal: OperatorProposal | None,
        review: OperatorReview | None,
        research_result: ResearchAgentResult | None,
        provider_records: Sequence[Any],
        tool_audits: Sequence[tuple[ToolExecutionResult, Mapping[str, Any]]],
        accepted: bool,
        rejection_stage: str | None,
        started_at: datetime,
        wall_time_ms: float,
    ) -> list[str]:
        if self.audit_store is None:
            return []
        token_input = sum(int(getattr(record.usage, "input_tokens", 0)) for record in provider_records)
        token_output = sum(int(getattr(record.usage, "output_tokens", 0)) for record in provider_records)
        if research_result is not None:
            usage = research_result.usage
            steps = usage.turns
            tool_count = usage.tool_calls
            llm_count = max(len(provider_records), len(research_result.provider_call_ids))
            tokens = usage.input_tokens + usage.output_tokens
            sdk_trace_id = research_result.sdk_trace_id
            provider_name = research_result.backend
        else:
            steps = max(1, len(provider_records))
            tool_count = len(tool_audits)
            llm_count = len(provider_records)
            tokens = token_input + token_output
            sdk_trace_id = None
            provider_name = (
                str(getattr(provider_records[0], "provider", "local"))
                if provider_records
                else request.design_mode
            )
        research_budget = request.research_agent_budget
        call_budget = request.llm_call_config
        run = AgentRunRecord(
            agent_run_id=agent_run_id,
            experiment_id=request.experiment_id,
            provider=provider_name,
            mode=request.design_mode,
            budget=AuditAgentBudget(
                max_steps=research_budget.max_turns,
                max_tool_calls=research_budget.max_tool_calls,
                max_llm_calls=research_budget.max_candidate_specs * 2,
                max_tokens=call_budget.max_total_tokens,
                max_wall_time_ms=call_budget.timeout_seconds * 1_000.0,
            ),
            usage=AuditAgentUsage(
                steps=steps,
                tool_calls=tool_count,
                llm_calls=llm_count,
                tokens=tokens,
                wall_time_ms=max(0.0, wall_time_ms),
            ),
            local_trace_id=request.request_id,
            sdk_trace_id=sdk_trace_id,
            status="completed" if accepted or rejection_stage != "design" else "failed",
            error=None if accepted else rejection_stage,
            metadata={
                "bundle_id": bundle_id,
                "bundle_hash": bundle.bundle_hash,
                "formal_validation_exposed_as_tool": False,
                "candidate_ids": list(drafts),
                "portfolio_id": (
                    None
                    if research_result is None or research_result.portfolio is None
                    else research_result.portfolio.portfolio_id
                ),
            },
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
        self.audit_store.record_agent_run(run)

        provider_call_ids: dict[str, str] = {}
        for index, record in enumerate(provider_records):
            output_model = str(getattr(record, "output_model", "structured_output"))
            prompt_version = str(getattr(record, "prompt_version", None) or "")
            try:
                template = get_prompt_template(prompt_version)
            except KeyError:
                template = (
                    DIAGNOSER_V1
                    if output_model == "DiagnosisReport"
                    else REVIEWER_V1
                    if output_model == "OperatorReview"
                    else DESIGNER_V1
                )
            response: Any = proposal
            resolved_candidate_id = next(reversed(drafts))
            role_trace = None
            if research_result is not None:
                role_trace = next(
                    (
                        item
                        for item in research_result.role_traces
                        if item.provider_call_id == getattr(record, "call_id", None)
                    ),
                    None,
                )
                if role_trace is not None and role_trace.candidate_id is not None:
                    resolved_candidate_id = role_trace.candidate_id
            if output_model == "DiagnosisReport":
                response = (
                    research_result.portfolio.diagnosis
                    if research_result is not None and research_result.portfolio is not None
                    else None if proposal is None or proposal.diagnosis is None else proposal.diagnosis
                )
            elif output_model == "OperatorReview":
                response = review
            elif (
                output_model == "OperatorProposal"
                and research_result is not None
                and research_result.portfolio is not None
                and role_trace is not None
            ):
                portfolio_candidate = next(
                    (
                        item
                        for item in research_result.portfolio.candidates
                        if item.candidate_id == role_trace.candidate_id
                    ),
                    None,
                )
                response = None if portfolio_candidate is None else portfolio_candidate.proposal
            elif (
                output_model == "PortfolioCritique"
                and research_result is not None
                and research_result.portfolio is not None
            ):
                response = research_result.portfolio.critic_report
            provider_status = str(getattr(record, "status", "provider_error"))
            response_id = getattr(record, "response_id", None)
            persisted_call_id = self._bounded_audit_id(
                agent_run_id, "llm", index, getattr(record, "call_id", index)
            )
            llm_record_fields: dict[str, Any] = {
                "call_id": persisted_call_id,
                "experiment_id": request.experiment_id,
                "agent_run_id": agent_run_id,
                "candidate_id": resolved_candidate_id,
                "bundle_id": bundle_id,
                "provider": str(getattr(record, "provider", provider_name)),
                "model": str(getattr(record, "model", None) or "unknown"),
                "prompt_version": prompt_version or template.version,
                "prompt": {
                    "system_prompt": template.system_text,
                    "provider_prompt_hash": getattr(record, "prompt_hash", None),
                    "request_hash": getattr(record, "request_hash", None),
                    "response_id": response_id,
                    "bundle_id": bundle_id,
                    "output_model": output_model,
                },
                "response": response,
                "usage": ModelUsage(
                    input_tokens=int(getattr(record.usage, "input_tokens", 0)),
                    output_tokens=int(getattr(record.usage, "output_tokens", 0)),
                    total_tokens=int(getattr(record.usage, "total_tokens", 0)),
                ),
                "retries": int(getattr(record, "retry_count", 0)),
                "latency_ms": float(getattr(record, "latency_ms", 0.0)),
                "status": "succeeded" if provider_status == "success" else "failed",
                "error": getattr(record, "error", None),
            }
            # Audit schema v2 promotes the provider response id to a dedicated
            # column.  Keeping it in the prompt metadata preserves it when an
            # older audit schema is injected during a rolling upgrade.
            if "response_id" in AuditLLMCallRecord.model_fields:
                llm_record_fields["response_id"] = response_id
            self.audit_store.record_llm_call(
                AuditLLMCallRecord.model_validate(llm_record_fields)
            )
            raw_call_id = getattr(record, "call_id", None)
            if raw_call_id is not None:
                provider_call_ids[str(raw_call_id)] = persisted_call_id

        if research_result is not None and request.design_mode == "multi_agent":
            self._persist_multi_agent_audit(
                request=request,
                research_result=research_result,
                bundle_id=bundle_id,
                provider_call_ids=provider_call_ids,
                started_at=started_at,
            )

        for result, arguments in tool_audits:
            if not result.authorized:
                authorization = AuthorizationDecision.DENIED
            elif result.tool_name in {"compile_operator_spec", "run_operator_smoke_test"}:
                authorization = AuthorizationDecision.NOT_REQUIRED
            else:
                authorization = AuthorizationDecision.READ_ONLY
            status = {
                "ok": "succeeded",
                "timeout": "timeout",
                "unauthorized": "denied",
                "budget_exceeded": "failed",
                "error": "failed",
            }[result.status]
            self.audit_store.record_tool_call(
                AgentToolCallRecord(
                    tool_call_id=self._bounded_audit_id(
                        agent_run_id, "tool", result.sequence, result.tool_name
                    ),
                    agent_run_id=agent_run_id,
                    sequence=result.sequence,
                    tool_name=result.tool_name,
                    authorization=authorization,
                    arguments=dict(arguments),
                    result={
                        "payload": result.payload,
                        "payload_json": result.payload_json,
                        "authorized": result.authorized,
                    },
                    latency_ms=result.latency_ms,
                    status=status,  # type: ignore[arg-type]
                    error=result.error,
                )
            )

        event_ids: list[str] = []
        for draft in drafts.values():
            for status, reason, details in zip(draft.statuses, draft.reasons, draft.details):
                event = self.audit_store.record_candidate_event(
                    CandidateEventRecord(
                        candidate_id=draft.candidate_id,
                        status=status,
                        reason=reason,
                        agent_run_id=agent_run_id,
                        evidence_bundle_id=bundle_id,
                        details=details,
                    )
                )
                event_ids.append(event.event_id)
        return event_ids

    @staticmethod
    def _bounded_audit_id(agent_run_id: str, kind: str, index: object, suffix: object) -> str:
        raw = f"{agent_run_id}:{kind}:{index}:{suffix}"
        if len(raw) <= 200:
            return raw
        return f"{raw[:160]}:{stable_hash(raw)[:32]}"


__all__ = [
    "DesignMode",
    "OperatorDesignOrchestrationResult",
    "OperatorDesignOrchestrator",
    "OperatorDesignRequest",
    "ReviewMode",
]
