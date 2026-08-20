"""Phase-8 evidence, LLM, agent, validation, demo, and ablation workflows.

These functions keep three boundaries explicit:

* proposal workflows stop after deterministic review;
* research-agent workflows stop after bounded compile/smoke tools;
* only ``validate_candidate_workflow`` and ``OperatorDesignOrchestrator``
  receive a validation split and make retention decisions.

Held-out test maps are consumed only by the post-retention section of
``agent_demo_workflow``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..agents.audit import (
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
from ..agents.designer_base import OperatorProposal
from ..agents.design_models import CandidateStatus
from ..agents.evidence import DesignBudget, EvidenceBundleBuilder, OperatorEvidenceBundle
from ..agents.heuristic_designer import HeuristicDesigner
from ..agents.llm_designer import LLMDesignerAdapter
from ..agents.multi_agent import DeterministicMockMultiAgent
from ..agents.orchestrator import (
    OperatorDesignOrchestrator,
    OperatorDesignRequest,
)
from ..agents.prompts import DESIGNER_V1, DIAGNOSER_V1, get_prompt_template
from ..agents.proposal_validation import ProposalValidator
from ..agents.providers import (
    LLMCallConfig,
    LLMConfigurationError,
    LLMProvider,
    MockLLMProvider,
    OpenAIProvider,
)
from ..agents.research_agent import (
    DeterministicMockResearchAgent,
    OpenAIAgentsResearchAgent,
)
from ..agents.tools import (
    AgentBudget,
    AgentToolContext,
    SmokeTestFixture,
    ToolExecutionResult,
)
from ..config import ExperimentConfig
from ..environment import Environment2D
from ..evolution.candidate_validator import FixedBudgetCandidateValidator
from ..memory import MechanismMemory
from ..operators.compiler import OperatorCompiler
from ..operators.registry import OperatorRegistry, build_manual_operator_registry
from ..path.evaluator import PathEvaluator
from ..path.initializer import initialize_path
from ..path.models import ObjectiveWeights
from ..reproducibility import canonical_json
from ..runtime import RunPaths
from ..trajectory import TrajectoryRecorder
from .agent_evidence import (
    _counterfactual_results,
    build_evidence_for_run,
    select_evidence_parents,
)
from .common import ensure_dataset, update_latest, write_csv, write_json
from .diagnose import run_diagnosis_workflow
from .run_search import run_search_workflow


ProviderName = Literal["mock", "openai"]
ProposalMode = Literal["single_call", "staged"]
AgentMode = Literal["single_agent", "multi_agent"]


def _resolve_agent_mode(
    config: ExperimentConfig,
    requested: AgentMode | str | None,
) -> AgentMode:
    value = requested or (
        config.agent.designer_mode
        if config.agent.designer_mode in {"single_agent", "multi_agent"}
        else "single_agent"
    )
    if value not in {"single_agent", "multi_agent"}:
        raise ValueError(f"unknown agent mode: {value}")
    return value  # type: ignore[return-value]


def create_llm_provider(
    provider: ProviderName | str,
    *,
    model: str | None = None,
) -> LLMProvider:
    """Create an explicit provider without cross-arm fallback."""

    if provider == "mock":
        return MockLLMProvider()
    if provider != "openai":
        raise ValueError(f"unknown LLM provider: {provider}")
    resolved_model = model or os.getenv("UOE_LLM_MODEL")
    api_key = os.getenv("UOE_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not resolved_model:
        raise LLMConfigurationError(
            "OpenAI provider requires UOE_LLM_MODEL or an explicit model"
        )
    if not api_key:
        raise LLMConfigurationError(
            "OpenAI provider requires OPENAI_API_KEY or UOE_LLM_API_KEY"
        )
    return OpenAIProvider(api_key=api_key, model=resolved_model)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _llm_config(config: ExperimentConfig, model: str | None = None) -> LLMCallConfig:
    return LLMCallConfig(
        model=model or os.getenv("UOE_LLM_MODEL"),
        **config.agent.llm_call.model_dump(mode="python"),
    )


def _agent_budget(config: ExperimentConfig) -> AgentBudget:
    return AgentBudget.model_validate(config.agent.agent_budget.model_dump(mode="python"))


def _design_budget(config: ExperimentConfig) -> DesignBudget:
    return DesignBudget.model_validate(config.agent.design_budget.model_dump(mode="python"))


def _evaluator(config: ExperimentConfig) -> PathEvaluator:
    return PathEvaluator(ObjectiveWeights.model_validate(config.objective.model_dump()))


def _database(run_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(run_dir)
    database = directory / "experiment.sqlite"
    if not database.exists():
        raise FileNotFoundError(f"experiment database not found: {database}")
    return directory, database


def _persist_bundle(
    audit: AgentAuditStore,
    bundle: OperatorEvidenceBundle,
    *,
    experiment_id: str,
    run_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    bundle_id = f"bundle_{bundle.bundle_hash[:32]}"
    existing = audit.get_evidence_bundle(bundle_id)
    if existing is not None:
        if existing.bundle_hash != bundle.bundle_hash:
            raise ValueError("content-addressed evidence bundle collision")
        return bundle_id
    audit.record_evidence_bundle(
        EvidenceBundleRecord(
            bundle_id=bundle_id,
            experiment_id=experiment_id,
            run_id=run_id,
            bundle=bundle.model_dump(mode="json", exclude={"bundle_hash"}),
            bundle_hash=bundle.bundle_hash,
            metadata=dict(metadata or {}),
        )
    )
    return bundle_id


def _bundle_counts(bundle: OperatorEvidenceBundle) -> dict[str, int]:
    return {
        "parents": len(bundle.parent_specs),
        "effective_contexts": len(bundle.effective_contexts),
        "failure_contexts": len(bundle.failure_contexts),
        "failure_modes": len(bundle.failure_modes),
        "synergies": len(bundle.synergy_evidence),
        "counterfactuals": len(bundle.counterfactual_evidence),
        "success_cases": len(bundle.representative_success_cases),
        "failure_cases": len(bundle.representative_failure_cases),
        "total_evidence": len(bundle.evidence_ids()),
    }


def build_evidence_workflow(
    config: ExperimentConfig,
    run_dir: str | Path,
    *,
    parent_operator_ids: Sequence[str] | None = None,
    train_maps: Sequence[Environment2D] = (),
    problem_summary: str = (
        "Improve fixed-budget UAV path search using computed trajectory, "
        "mechanism-memory, and bounded counterfactual evidence."
    ),
    registry: OperatorRegistry | None = None,
) -> dict[str, Any]:
    """Build, canonically write, and audit one evidence bundle for a run."""

    directory, database = _database(run_dir)
    operator_registry = registry or build_manual_operator_registry()
    bundle = build_evidence_for_run(
        config,
        database,
        parent_operator_ids=parent_operator_ids,
        train_maps=train_maps,
        problem_summary=problem_summary,
        registry=operator_registry,
    )
    with AgentAuditStore(database) as audit:
        bundle_id = _persist_bundle(
            audit,
            bundle,
            experiment_id=config.name,
            run_id=directory.name,
            metadata={"workflow": "build-evidence"},
        )
    canonical_path = directory / "evidence_bundle.canonical.json"
    canonical_path.write_text(
        canonical_json(bundle.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    report = {
        "bundle_id": bundle_id,
        "bundle_hash": bundle.bundle_hash,
        "counts": _bundle_counts(bundle),
        "limitations": bundle.limitations,
        "canonical_json": canonical_json(bundle.model_dump(mode="json")),
        "bundle": bundle.model_dump(mode="json"),
        "canonical_path": str(canonical_path),
    }
    write_json(directory / "evidence_summary.json", report)
    return report


def _audit_provider_calls(
    audit: AgentAuditStore,
    records: Sequence[Any],
    *,
    experiment_id: str,
    agent_run_id: str,
    candidate_id: str,
    bundle_id: str,
    proposal: OperatorProposal | None,
    research_result: Any | None = None,
) -> dict[str, str]:
    persisted_ids: dict[str, str] = {}
    for index, record in enumerate(records):
        output_model = str(getattr(record, "output_model", "OperatorProposal"))
        prompt_version = str(getattr(record, "prompt_version", None) or "")
        try:
            template = get_prompt_template(prompt_version)
        except KeyError:
            template = DIAGNOSER_V1 if output_model == "DiagnosisReport" else DESIGNER_V1
        response: Any = proposal
        resolved_candidate_id = candidate_id
        if output_model == "DiagnosisReport":
            response = (
                research_result.portfolio.diagnosis
                if research_result is not None and research_result.portfolio is not None
                else None if proposal is None else proposal.diagnosis
            )
        elif research_result is not None and research_result.portfolio is not None:
            trace = next(
                (
                    item
                    for item in research_result.role_traces
                    if item.provider_call_id == getattr(record, "call_id", None)
                ),
                None,
            )
            if trace is not None and trace.candidate_id is not None:
                resolved_candidate_id = trace.candidate_id
            if output_model == "OperatorProposal" and trace is not None:
                portfolio_candidate = next(
                    (
                        item
                        for item in research_result.portfolio.candidates
                        if item.candidate_id == trace.candidate_id
                    ),
                    None,
                )
                response = None if portfolio_candidate is None else portfolio_candidate.proposal
            elif output_model == "PortfolioCritique":
                response = research_result.portfolio.critic_report
        provider_status = str(getattr(record, "status", "provider_error"))
        usage = getattr(record, "usage", None)
        persisted_call_id = f"{agent_run_id}:llm:{index}"
        audit.record_llm_call(
            AuditLLMCallRecord(
                call_id=persisted_call_id,
                experiment_id=experiment_id,
                agent_run_id=agent_run_id,
                candidate_id=resolved_candidate_id,
                bundle_id=bundle_id,
                provider=str(getattr(record, "provider", "unknown")),
                model=str(getattr(record, "model", None) or "unknown"),
                response_id=getattr(record, "response_id", None),
                prompt_version=prompt_version or template.version,
                prompt={
                    "system_prompt": template.system_text,
                    "provider_prompt_hash": getattr(record, "prompt_hash", None),
                    "request_hash": getattr(record, "request_hash", None),
                    "output_model": output_model,
                },
                response=response,
                usage=ModelUsage(
                    input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                    total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                ),
                retries=int(getattr(record, "retry_count", 0) or 0),
                latency_ms=float(getattr(record, "latency_ms", 0.0) or 0.0),
                status="succeeded" if provider_status == "success" else "failed",
                error=getattr(record, "error", None),
            )
        )
        raw_call_id = getattr(record, "call_id", None)
        if raw_call_id is not None:
            persisted_ids[str(raw_call_id)] = persisted_call_id
    return persisted_ids


def _audit_tool_calls(
    audit: AgentAuditStore,
    agent_run_id: str,
    calls: Sequence[tuple[ToolExecutionResult, Mapping[str, Any]]],
) -> None:
    for result, arguments in calls:
        authorization = (
            AuthorizationDecision.DENIED
            if not result.authorized
            else AuthorizationDecision.NOT_REQUIRED
            if result.tool_name in {"compile_operator_spec", "run_operator_smoke_test"}
            else AuthorizationDecision.READ_ONLY
        )
        audit.record_tool_call(
            AgentToolCallRecord(
                tool_call_id=f"{agent_run_id}:tool:{result.sequence}",
                agent_run_id=agent_run_id,
                sequence=result.sequence,
                tool_name=result.tool_name,
                authorization=authorization,
                arguments=dict(arguments),
                result={"payload": result.payload, "authorized": result.authorized},
                latency_ms=result.latency_ms,
                status={
                    "ok": "succeeded",
                    "error": "failed",
                    "timeout": "timeout",
                    "unauthorized": "denied",
                    "budget_exceeded": "failed",
                }[result.status],
                error=result.error,
            )
        )


def _audit_multi_agent_result(
    audit: AgentAuditStore,
    result: Any,
    *,
    config: ExperimentConfig,
    agent_run_id: str,
    bundle_id: str,
    provider_call_ids: Mapping[str, str],
    started_at: datetime,
) -> None:
    """Persist the replayable portfolio and ordered role trace, if present."""

    portfolio = getattr(result, "portfolio", None)
    traces = list(getattr(result, "role_traces", ()))
    if getattr(result, "backend", "") != "deterministic_mock_multi_agent":
        return
    failure_reason = next(
        (trace.error for trace in reversed(traces) if trace.error),
        None,
    ) or next(
        (
            attempt.rejection_reason
            for attempt in result.candidates
            if attempt.rejection_reason
        ),
        None,
    )
    multi_run_id = f"multi_{agent_run_id}"[:200]
    portfolio_payload = (
        None if portfolio is None else portfolio.canonical_payload(include_id=True)
    )
    budget = AuditAgentBudget(
        max_steps=config.agent.agent_budget.max_turns,
        max_tool_calls=config.agent.agent_budget.max_tool_calls,
        max_llm_calls=4,
        max_tokens=config.agent.llm_call.max_total_tokens,
    )
    usage = AuditAgentUsage(
        steps=result.usage.turns,
        tool_calls=result.usage.tool_calls,
        llm_calls=len(result.provider_call_ids),
        tokens=result.usage.total_tokens,
    )
    audit.record_multi_agent_run(
        MultiAgentRunRecord(
            multi_agent_run_id=multi_run_id,
            agent_run_id=agent_run_id,
            coordinator_version="multi_agent_coordinator_v1",
            bundle_id=bundle_id,
            bundle_hash=result.bundle_hash,
            budget=budget,
            usage=usage,
            portfolio_id=None if portfolio is None else portfolio.portfolio_id,
            portfolio=portfolio_payload,
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
                "candidate_count": len(result.candidates),
                "portfolio_created": portfolio is not None,
            },
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
    )
    if portfolio is not None and portfolio_payload is not None:
        audit.record_candidate_portfolio(
            CandidatePortfolioRecord(
                portfolio_id=portfolio.portfolio_id,
                multi_agent_run_id=multi_run_id,
                bundle_hash=result.bundle_hash,
                portfolio=portfolio_payload,
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
        audit.record_multi_agent_role_event(
            MultiAgentRoleEventRecord(
                role_event_id=f"{multi_run_id}:role:{sequence}",
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
                    "bundle_hash": result.bundle_hash,
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

    audit.record_multi_agent_role_event(
        MultiAgentRoleEventRecord(
            role_event_id=f"{multi_run_id}:role:select",
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
def propose_operator_workflow(
    config: ExperimentConfig,
    run_dir: str | Path,
    *,
    provider: ProviderName | str | None = None,
    mode: ProposalMode = "single_call",
    parent_operator_ids: Sequence[str] | None = None,
    train_maps: Sequence[Environment2D] = (),
    candidate_id: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate and hard-review one proposal without compiling or validating it."""

    directory, database = _database(run_dir)
    selected_provider = provider or config.agent.provider
    llm_provider = create_llm_provider(selected_provider, model=model)
    bundle = build_evidence_for_run(
        config,
        database,
        parent_operator_ids=parent_operator_ids,
        train_maps=train_maps,
    )
    adapter = LLMDesignerAdapter(provider=llm_provider)
    validator = ProposalValidator()
    candidate = candidate_id or _new_id("candidate_proposal")
    agent_run_id = _new_id("proposal_run")
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    proposal: OperatorProposal | None = None
    review: Any = None
    error: str | None = None

    with AgentAuditStore(database) as audit:
        bundle_id = _persist_bundle(
            audit,
            bundle,
            experiment_id=config.name,
            run_id=directory.name,
            metadata={"workflow": "propose-operator", "mode": mode},
        )
        try:
            proposal = adapter.propose_from_evidence(
                bundle,
                mode=mode,
                call_config=_llm_config(config, model),
            )
            review = validator.validate_and_review(
                proposal,
                bundle,
                review_mode="none" if config.agent.review_mode == "none" else "rule_based",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        records = list(getattr(llm_provider, "call_records", ()))
        input_tokens = sum(record.usage.input_tokens for record in records)
        output_tokens = sum(record.usage.output_tokens for record in records)
        audit.record_agent_run(
            AgentRunRecord(
                agent_run_id=agent_run_id,
                experiment_id=config.name,
                provider=str(selected_provider),
                mode=f"llm_{mode}",
                budget=AuditAgentBudget(
                    max_steps=2 if mode == "staged" else 1,
                    max_llm_calls=2 if mode == "staged" else 1,
                    max_tokens=config.agent.llm_call.max_total_tokens,
                ),
                usage=AuditAgentUsage(
                    steps=len(records),
                    llm_calls=len(records),
                    tokens=input_tokens + output_tokens,
                    wall_time_ms=(time.perf_counter() - started_clock) * 1_000.0,
                ),
                local_trace_id=agent_run_id,
                status="failed" if error else "completed",
                error=error,
                metadata={
                    "bundle_id": bundle_id,
                    "candidate_id": candidate,
                    "compile_executed": False,
                    "formal_validation_executed": False,
                },
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
        )
        audit.record_candidate_event(
            CandidateEventRecord(
                candidate_id=candidate,
                status=CandidateStatus.PROPOSED,
                reason="structured provider design attempt started",
                agent_run_id=agent_run_id,
                evidence_bundle_id=bundle_id,
                details={"mode": mode, "provider": selected_provider},
            )
        )
        if error is None and proposal is not None and review is not None:
            audit.record_candidate_event(
                CandidateEventRecord(
                    candidate_id=candidate,
                    status=CandidateStatus.SCHEMA_VALID,
                    reason="OperatorProposal schema and evidence references passed",
                    agent_run_id=agent_run_id,
                    evidence_bundle_id=bundle_id,
                    details={"proposal": proposal.model_dump(mode="json", by_alias=True)},
                )
            )
            audit.record_candidate_event(
                CandidateEventRecord(
                    candidate_id=candidate,
                    status=CandidateStatus.REVIEWED,
                    reason=f"deterministic review decision={review.decision}",
                    agent_run_id=agent_run_id,
                    evidence_bundle_id=bundle_id,
                    details={"review": review.model_dump(mode="json")},
                )
            )
        else:
            audit.record_candidate_event(
                CandidateEventRecord(
                    candidate_id=candidate,
                    status=CandidateStatus.REJECTED,
                    reason=error or "proposal generation failed",
                    agent_run_id=agent_run_id,
                    evidence_bundle_id=bundle_id,
                    details={"stage": "proposal_or_review"},
                )
            )
        _audit_provider_calls(
            audit,
            records,
            experiment_id=config.name,
            agent_run_id=agent_run_id,
            candidate_id=candidate,
            bundle_id=bundle_id,
            proposal=proposal,
        )

    report = {
        "candidate_id": candidate,
        "agent_run_id": agent_run_id,
        "bundle_id": bundle_id,
        "bundle_hash": bundle.bundle_hash,
        "status": CandidateStatus.REJECTED.value if error else CandidateStatus.REVIEWED.value,
        "proposal": None if proposal is None else proposal.model_dump(mode="json", by_alias=True),
        "review": None if review is None else review.model_dump(mode="json"),
        "error": error,
        "compile_executed": False,
        "formal_validation_executed": False,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    write_json(directory / f"proposal_{candidate}.json", report)
    return report


def run_agent_workflow(
    config: ExperimentConfig,
    run_dir: str | Path,
    *,
    provider: ProviderName | str | None = None,
    agent_mode: AgentMode | str | None = None,
    parent_operator_ids: Sequence[str] | None = None,
    train_maps: Sequence[Environment2D] = (),
    smoke_environment: Environment2D | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Run a whitelisted research agent through compile and smoke only."""

    directory, database = _database(run_dir)
    selected_provider = provider or config.agent.provider
    selected_mode = _resolve_agent_mode(config, agent_mode)
    if selected_mode == "multi_agent" and selected_provider != "mock":
        raise ValueError("multi_agent is offline-only and requires provider='mock'")
    available_train = list(train_maps)
    if smoke_environment is None:
        if not available_train:
            available_train = list(ensure_dataset(config)["train"])
        if not available_train:
            raise ValueError("run-agent requires a smoke environment")
        smoke_environment = available_train[0]
    registry = build_manual_operator_registry()
    bundle = build_evidence_for_run(
        config,
        database,
        parent_operator_ids=parent_operator_ids,
        train_maps=available_train,
        registry=registry,
    )
    compiler = OperatorCompiler(config.dsl)
    tool_audits: list[tuple[ToolExecutionResult, Mapping[str, Any]]] = []
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    agent_run_id = _new_id("research_run")

    with (
        MechanismMemory(database) as memory,
        AgentAuditStore(database) as audit,
    ):
        bundle_id = _persist_bundle(
            audit,
            bundle,
            experiment_id=config.name,
            run_id=directory.name,
            metadata={"workflow": "run-agent"},
        )
        context = AgentToolContext(
            bundle=bundle,
            compiler=compiler,
            memory=memory,
            smoke_fixture=SmokeTestFixture(
                smoke_environment,
                initialize_path(
                    smoke_environment,
                    grid_resolution=config.maps.grid_resolution,
                ),
            ),
        )

        def capture(result: ToolExecutionResult, arguments: Mapping[str, Any]) -> None:
            tool_audits.append((result, dict(arguments)))

        provider_records: Sequence[Any] = ()
        if selected_provider == "mock":
            structured_provider = create_llm_provider("mock", model=model)
            backend = (
                DeterministicMockMultiAgent(structured_provider)
                if selected_mode == "multi_agent"
                else DeterministicMockResearchAgent(structured_provider)
            )
        elif selected_provider == "openai":
            # Fail explicitly on missing key/model before attempting the SDK.
            create_llm_provider("openai", model=model)
            structured_provider = None
            backend = OpenAIAgentsResearchAgent(
                remote_tracing=config.agent.remote_tracing,
                trace_include_sensitive_data=config.agent.trace_include_sensitive_data,
            )
        else:
            raise ValueError(f"unknown LLM provider: {selected_provider}")

        result = backend.run(
            context,
            budget=_agent_budget(config),
            call_config=_llm_config(config, model),
            audit_sink=capture,
            agent_run_id=agent_run_id,
        )
        if structured_provider is not None:
            provider_records = list(structured_provider.call_records)

        audit.record_agent_run(
            AgentRunRecord(
                agent_run_id=agent_run_id,
                experiment_id=config.name,
                provider=str(selected_provider),
                mode=selected_mode,
                budget=AuditAgentBudget(
                    max_steps=config.agent.agent_budget.max_turns,
                    max_tool_calls=config.agent.agent_budget.max_tool_calls,
                    max_llm_calls=config.agent.agent_budget.max_candidate_specs * 2,
                    max_tokens=config.agent.llm_call.max_total_tokens,
                ),
                usage=AuditAgentUsage(
                    steps=result.usage.turns,
                    tool_calls=result.usage.tool_calls,
                    llm_calls=len(provider_records),
                    tokens=result.usage.total_tokens,
                    wall_time_ms=(time.perf_counter() - started_clock) * 1_000.0,
                ),
                local_trace_id=agent_run_id,
                sdk_trace_id=result.sdk_trace_id,
                status="completed",
                metadata={
                    "bundle_id": bundle_id,
                    "candidate_ids": [item.candidate_id for item in result.candidates],
                    "portfolio_id": (
                        None if result.portfolio is None else result.portfolio.portfolio_id
                    ),
                    "formal_validation_executed": False,
                },
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
        )

        provider_call_ids = _audit_provider_calls(
            audit,
            provider_records,
            experiment_id=config.name,
            agent_run_id=agent_run_id,
            candidate_id=result.selected_candidate_id or result.candidates[-1].candidate_id,
            bundle_id=bundle_id,
            proposal=result.proposal,
            research_result=result,
        )
        _audit_tool_calls(audit, agent_run_id, tool_audits)
        _audit_multi_agent_result(
            audit,
            result,
            config=config,
            agent_run_id=agent_run_id,
            bundle_id=bundle_id,
            provider_call_ids=provider_call_ids,
            started_at=started_at,
        )

        for attempt in result.candidates:
            for status in attempt.status_history:
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
                audit.record_candidate_event(
                    CandidateEventRecord(
                        candidate_id=attempt.candidate_id,
                        status=status,
                        reason=(
                            attempt.rejection_reason
                            if status == CandidateStatus.REJECTED
                            else f"{selected_mode.replace('_', '-')} transition {status.value}"
                        )
                        or status.value,
                        agent_run_id=agent_run_id,
                        evidence_bundle_id=bundle_id,
                        details=details,
                    )
                )

    report = result.model_dump(mode="json")
    report.update(
        {
            "bundle_id": bundle_id,
            "agent_mode": selected_mode,
            "formal_validation_executed": False,
            "smoke_map_id": smoke_environment.map_id,
        }
    )
    write_json(directory / "agent_result.json", report)
    return report


def _proposal_from_candidate_events(
    audit: AgentAuditStore,
    candidate_id: str,
) -> tuple[OperatorProposal, str, str | None, CandidateStatus]:
    events = audit.list_candidate_events(candidate_id)
    if not events:
        raise KeyError(f"candidate not found in audit database: {candidate_id}")
    proposal_payload: Any = None
    bundle_id: str | None = None
    agent_run_id: str | None = None
    for event in events:
        bundle_id = event.evidence_bundle_id or bundle_id
        agent_run_id = event.agent_run_id or agent_run_id
        value = event.details.get("proposal")
        if value:
            proposal_payload = value
    if proposal_payload is None:
        raise ValueError(f"candidate audit has no reconstructable proposal: {candidate_id}")
    if bundle_id is None:
        raise ValueError(f"candidate audit has no evidence bundle reference: {candidate_id}")
    return (
        OperatorProposal.model_validate(proposal_payload),
        bundle_id,
        agent_run_id,
        events[-1].status,
    )


def _bundle_from_audit(audit: AgentAuditStore, bundle_id: str) -> OperatorEvidenceBundle:
    record = audit.get_evidence_bundle(bundle_id)
    if record is None:
        raise KeyError(f"evidence bundle not found: {bundle_id}")
    return OperatorEvidenceBundle.model_validate(
        {**record.bundle, "bundle_hash": record.bundle_hash or ""}
    )


def _persist_validation_memory(
    memory: MechanismMemory,
    *,
    proposal: OperatorProposal,
    bundle: OperatorEvidenceBundle,
    candidate_id: str,
    report: Any,
    retained: bool,
    relation: str | None,
) -> dict[str, Any]:
    parent_ids = list(proposal.spec.parent_operators)
    if not retained:
        failure_id = memory.add_failure_mode(
            "candidate_retention_rejected",
            operator_id=proposal.spec.name,
            count=1,
            context={"candidate_id": candidate_id, "split": "validation"},
            evidence=[report.model_dump(mode="json")],
            metadata={"bundle_hash": bundle.bundle_hash},
        )
        return {"mechanism_id": None, "lineage_ids": [], "failure_id": failure_id}

    parent_specs = {spec.name: spec for spec in bundle.parent_specs}
    for parent_id in parent_ids:
        if memory.get_mechanism(parent_id) is None and parent_id in parent_specs:
            parent = parent_specs[parent_id]
            memory.add_mechanism(
                parent_id,
                parent.model_dump(mode="json"),
                name=parent.name,
                description=parent.description,
                tags=["parent"],
                metadata={"source": "candidate_validation"},
            )
    mechanism_id = memory.add_mechanism(
        proposal.spec.name,
        proposal.spec.model_dump(mode="json"),
        name=proposal.spec.name,
        description=proposal.spec.description,
        score=float(report.mean_gain),
        evidence_count=len(bundle.evidence_ids()),
        success_rate=float(report.candidate_feasibility_rate),
        tags=["evolved", relation or "structural_variant"],
        metadata={
            "candidate_id": candidate_id,
            "bundle_hash": bundle.bundle_hash,
            "validation_report": report.model_dump(mode="json"),
        },
    )
    lineage_ids = [
        memory.add_lineage(
            parent_id,
            mechanism_id,
            relation=relation or "structural_variant",
            metadata={"candidate_id": candidate_id, "retained": True},
        )
        for parent_id in parent_ids
    ]
    insight_id = memory.add_insight(
        operator_id=mechanism_id,
        insight_type="improvement_hypothesis",
        evidence={
            "used_evidence_ids": proposal.used_evidence_ids,
            "validation_report": report.model_dump(mode="json"),
        },
        confidence=min(1.0, len(report.outcomes) / 20.0),
        applicable_context={
            "expected": None
            if proposal.hypothesis is None
            else proposal.hypothesis.expected_effective_context
        },
        failure_context={"target_failure_modes": proposal.target_failure_modes},
    )
    return {
        "mechanism_id": mechanism_id,
        "lineage_ids": lineage_ids,
        "insight_id": insight_id,
    }


def validate_candidate_workflow(
    config: ExperimentConfig,
    run_dir: str | Path,
    candidate_id: str,
    validation_maps: Sequence[Environment2D],
    *,
    forbidden_map_hashes: Iterable[str] = (),
) -> dict[str, Any]:
    """Reconstruct and formally validate one audited candidate.

    Only a concrete validation sequence is accepted.  Passing a dataset
    mapping is rejected so a caller cannot accidentally expose the test split.
    ``forbidden_map_hashes`` provides an additional explicit held-out guard.
    """

    if isinstance(validation_maps, Mapping):
        raise TypeError("validate-candidate accepts only a validation split sequence")
    environments = list(validation_maps)
    if not environments:
        raise ValueError("validation split must not be empty")
    forbidden = {str(value) for value in forbidden_map_hashes}
    leaked = [item.map_id for item in environments if item.content_hash in forbidden]
    if leaked:
        raise ValueError(f"held-out test maps cannot enter candidate retention: {leaked}")

    directory, database = _database(run_dir)
    registry = build_manual_operator_registry()
    compiler = OperatorCompiler(config.dsl)
    evaluator = _evaluator(config)
    fixed_validator = FixedBudgetCandidateValidator(config, evaluator)

    with (
        AgentAuditStore(database) as audit,
        MechanismMemory(database) as memory,
        TrajectoryRecorder(database) as recorder,
    ):
        proposal, bundle_id, agent_run_id, current = _proposal_from_candidate_events(
            audit, candidate_id
        )
        if current in {CandidateStatus.ACCEPTED, CandidateStatus.REJECTED}:
            raise ValueError(f"candidate is already terminal: {current.value}")
        bundle = _bundle_from_audit(audit, bundle_id)
        review = ProposalValidator().validate_and_review(
            proposal,
            bundle,
            review_mode="none" if config.agent.review_mode == "none" else "rule_based",
        )
        if config.agent.review_mode != "none" and review.decision != "approve":
            audit.record_candidate_event(
                CandidateEventRecord(
                    candidate_id=candidate_id,
                    status=CandidateStatus.REJECTED,
                    reason="candidate review did not approve formal validation",
                    agent_run_id=agent_run_id,
                    evidence_bundle_id=bundle_id,
                    details={"review": review.model_dump(mode="json")},
                )
            )
            raise ValueError("candidate review did not approve formal validation")

        try:
            compiled = compiler.compile(proposal.spec)
        except Exception as exc:
            audit.record_candidate_event(
                CandidateEventRecord(
                    candidate_id=candidate_id,
                    status=CandidateStatus.REJECTED,
                    reason=f"compile failed: {type(exc).__name__}: {exc}",
                    agent_run_id=agent_run_id,
                    evidence_bundle_id=bundle_id,
                    details={"stage": "compile"},
                )
            )
            raise
        if current == CandidateStatus.REVIEWED:
            audit.record_candidate_event(
                CandidateEventRecord(
                    candidate_id=candidate_id,
                    status=CandidateStatus.COMPILED,
                    reason="trusted DSL compiler passed",
                    agent_run_id=agent_run_id,
                    evidence_bundle_id=bundle_id,
                    details={"operator_name": compiled.name},
                )
            )
            current = CandidateStatus.COMPILED

        smoke_failures = fixed_validator.contract_failures(
            compiled,
            environments[0],
            generation=1,
            candidate_index=0,
        )
        if smoke_failures:
            audit.record_candidate_event(
                CandidateEventRecord(
                    candidate_id=candidate_id,
                    status=CandidateStatus.REJECTED,
                    reason="contract smoke failed: " + "; ".join(smoke_failures),
                    agent_run_id=agent_run_id,
                    evidence_bundle_id=bundle_id,
                    details={"stage": "smoke", "failures": smoke_failures},
                )
            )
            memory.add_failure_mode(
                "candidate_smoke_rejected",
                operator_id=proposal.spec.name,
                evidence=smoke_failures,
                metadata={"candidate_id": candidate_id},
            )
            report = {
                "candidate_id": candidate_id,
                "status": CandidateStatus.REJECTED.value,
                "retained": False,
                "smoke_failures": smoke_failures,
                "validation_map_ids": [],
                "test_split_accessed": False,
            }
            write_json(directory / f"candidate_validation_{candidate_id}.json", report)
            return report
        if current == CandidateStatus.COMPILED:
            audit.record_candidate_event(
                CandidateEventRecord(
                    candidate_id=candidate_id,
                    status=CandidateStatus.SMOKE_PASSED,
                    reason="independent contract smoke passed",
                    agent_run_id=agent_run_id,
                    evidence_bundle_id=bundle_id,
                    details={"map_id": environments[0].map_id},
                )
            )

        parent_id = proposal.spec.parent_operators[0]
        population = list(registry.values())
        if parent_id not in registry:
            parent_spec = next(
                (spec for spec in bundle.parent_specs if spec.name == parent_id),
                None,
            )
            if parent_spec is None:
                raise KeyError(f"candidate parent is unavailable: {parent_id}")
            parent_compiled = compiler.compile(parent_spec)
            registry.register(parent_compiled)
            population.append(parent_compiled)
        validation_report = fixed_validator.validate(
            population,
            parent_id,
            compiled,
            environments,
            generation=1,
            candidate_index=0,
            recorder=recorder,
            root_run_id=directory.name,
        )
        audit.record_candidate_event(
            CandidateEventRecord(
                candidate_id=candidate_id,
                status=CandidateStatus.VALIDATED,
                reason="fixed-budget validation split evaluation completed",
                agent_run_id=agent_run_id,
                evidence_bundle_id=bundle_id,
                details={
                    "validation_report": validation_report.model_dump(mode="json"),
                    "validation_map_hashes": [item.content_hash for item in environments],
                    "test_split_accessed": False,
                },
            )
        )
        memory_result = _persist_validation_memory(
            memory,
            proposal=proposal,
            bundle=bundle,
            candidate_id=candidate_id,
            report=validation_report,
            retained=validation_report.retained,
            relation=review.lineage_relation,
        )
        final_status = (
            CandidateStatus.ACCEPTED
            if validation_report.retained
            else CandidateStatus.REJECTED
        )
        audit.record_candidate_event(
            CandidateEventRecord(
                candidate_id=candidate_id,
                status=final_status,
                reason=(
                    "candidate retained and mechanism memory updated"
                    if validation_report.retained
                    else "; ".join(validation_report.retention_reasons)
                    or "candidate rejected by retention gate"
                ),
                agent_run_id=agent_run_id,
                evidence_bundle_id=bundle_id,
                details=memory_result,
            )
        )

    output = {
        "candidate_id": candidate_id,
        "operator_name": proposal.spec.name,
        "status": final_status.value,
        "retained": validation_report.retained,
        "validation_report": validation_report.model_dump(mode="json"),
        "validation_map_ids": [item.map_id for item in environments],
        "validation_map_hashes": [item.content_hash for item in environments],
        "test_split_accessed": False,
        "memory": memory_result,
    }
    write_json(directory / f"candidate_validation_{candidate_id}.json", output)
    return output


def agent_demo_workflow(
    config: ExperimentConfig,
    *,
    provider: ProviderName | str = "mock",
    agent_mode: AgentMode | str | None = None,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    paths: RunPaths | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Run the complete offline Phase-8 demonstration in a fresh run directory."""

    selected_mode = _resolve_agent_mode(config, agent_mode)
    if selected_mode == "multi_agent" and provider != "mock":
        raise ValueError("multi_agent is offline-only and requires provider='mock'")

    run_paths = paths or RunPaths.create(
        config,
        "agent-demo",
        run_id=run_id,
        run_dir=run_dir,
    )
    dataset = ensure_dataset(config)
    train_maps = list(dataset["train"])
    validation_maps = list(dataset["validation"])
    test_maps = list(dataset["test"])
    if not train_maps or not validation_maps or not test_maps:
        raise ValueError("agent-demo requires non-empty train, validation, and test splits")

    search_summary = run_search_workflow(config, run_paths, train_maps)
    diagnosis_warning: str | None = None
    try:
        diagnosis = run_diagnosis_workflow(
            config,
            run_paths.result_dir,
            figure_dir=run_paths.figure_dir,
        )
    except Exception as exc:
        # Very small smoke budgets can leave every delayed-reward aggregate
        # null; diagnosis and memory are already persisted before the optional
        # legacy chart renderer sees those null bars.
        diagnosis_path = run_paths.result_dir / "diagnosis.json"
        if not diagnosis_path.exists():
            raise
        diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
        diagnosis_warning = f"diagnostic figures incomplete: {type(exc).__name__}: {exc}"
    registry = build_manual_operator_registry()
    baseline_population = list(registry.values())
    evaluator = _evaluator(config)
    compiler = OperatorCompiler(config.dsl)
    fixed_validator = FixedBudgetCandidateValidator(config, evaluator)
    selected_provider = create_llm_provider(provider, model=model)
    if provider == "mock":
        research_backend = (
            DeterministicMockMultiAgent(selected_provider)
            if selected_mode == "multi_agent"
            else DeterministicMockResearchAgent(selected_provider)
        )
    elif provider == "openai":
        research_backend = OpenAIAgentsResearchAgent(
            remote_tracing=config.agent.remote_tracing,
            trace_include_sensitive_data=config.agent.trace_include_sensitive_data,
        )
    else:
        raise ValueError(f"unknown LLM provider: {provider}")

    with (
        MechanismMemory(run_paths.database) as memory,
        TrajectoryRecorder(run_paths.database) as recorder,
        AgentAuditStore(run_paths.database) as audit,
    ):
        parents = select_evidence_parents(memory, registry, limit=1)
        counterfactual, counterfactual_seed = _counterfactual_results(
            config,
            memory,
            recorder,
            registry,
            parents[0],
            train_maps,
        )
        evidence_builder = EvidenceBundleBuilder(
            memory,
            registry,
            recorder=recorder,
            minimum_reliable_samples=config.diagnostics.minimum_context_samples,
        )
        orchestrator = OperatorDesignOrchestrator(
            evidence_builder=evidence_builder,
            proposal_validator=ProposalValidator(),
            compiler=compiler,
            candidate_validator=fixed_validator,
            memory=memory,
            registry=registry,
            llm_designer=LLMDesignerAdapter(provider=selected_provider),
            research_agent_backend=research_backend,
            heuristic_designer=HeuristicDesigner(),
            audit_store=audit,
            recorder=recorder,
        )
        request = OperatorDesignRequest(
            request_id=f"{run_paths.run_id}-candidate-0"[:120],
            experiment_id=config.name,
            root_run_id=run_paths.run_id,
            problem_summary=(
                "Improve fixed-budget UAV path search from measured trajectory, "
                "mechanism-memory, and compact counterfactual evidence."
            ),
            parent_operator_ids=parents,
            smoke_environment=train_maps[0],
            validation_environments=validation_maps,
            design_mode=selected_mode,
            review_mode=(
                "none" if config.agent.review_mode == "none" else "rule_based"
            ),
            generation=1,
            candidate_index=0,
            population_operator_names=list(registry.names()),
            counterfactual_results=counterfactual,
            counterfactual_seed=counterfactual_seed,
            design_budget=_design_budget(config),
            llm_call_config=_llm_config(config, model),
            research_agent_budget=_agent_budget(config),
        )
        orchestration = orchestrator.run(request)
        evidence_record = audit.get_evidence_bundle(orchestration.bundle_id)
        if evidence_record is None:
            raise RuntimeError("orchestrator evidence bundle was not persisted")
        evidence_payload = {
            **evidence_record.bundle,
            "bundle_hash": evidence_record.bundle_hash,
        }
        evidence_path = run_paths.result_dir / "agent_evidence_bundle.canonical.json"
        evidence_path.write_text(
            canonical_json(evidence_payload) + "\n",
            encoding="utf-8",
        )

        test_comparison: dict[str, Any]
        if orchestration.retained and orchestration.operator_name is not None:
            candidate_operator = registry.get(orchestration.operator_name)
            candidate_population = list(baseline_population)
            parent_index = next(
                index
                for index, operator in enumerate(candidate_population)
                if str(operator.name) == parents[0]
            )
            candidate_population[parent_index] = candidate_operator
            test_config = config.model_copy(
                update={
                    "search": config.search.model_copy(
                        update={"validation_iterations": config.search.test_iterations}
                    )
                }
            )
            test_validator = FixedBudgetCandidateValidator(test_config, evaluator)
            test_outcomes = test_validator.compare(
                baseline_population,
                candidate_population,
                test_maps,
                generation=1,
                candidate_index=0,
                recorder=recorder,
                root_run_id=f"{run_paths.run_id}-heldout-test",
            )
            test_comparison = {
                "executed": True,
                "after_retention": True,
                "outcomes": [item.model_dump(mode="json") for item in test_outcomes],
                "mean_gain": (
                    sum(item.gain for item in test_outcomes) / len(test_outcomes)
                    if test_outcomes
                    else 0.0
                ),
            }
        else:
            test_comparison = {
                "executed": False,
                "after_retention": True,
                "reason": "candidate was not retained on validation split",
                "outcomes": [],
            }

        candidate_events = audit.list_candidate_events(orchestration.candidate_id)
        llm_calls = audit.list_llm_calls(orchestration.agent_run_id)
        tool_calls = audit.list_tool_calls(orchestration.agent_run_id)
        multi_agent_runs = audit.list_multi_agent_runs(orchestration.agent_run_id)
        portfolios = [
            item
            for multi_run in multi_agent_runs
            for item in audit.list_candidate_portfolios(multi_run.multi_agent_run_id)
        ]
        role_events = [
            item
            for multi_run in multi_agent_runs
            for item in audit.list_multi_agent_role_events(multi_run.multi_agent_run_id)
        ]
        mechanism = (
            None
            if orchestration.mechanism_id is None
            else memory.get_mechanism(orchestration.mechanism_id)
        )
        lineage = (
            []
            if orchestration.mechanism_id is None
            else memory.get_lineage(orchestration.mechanism_id)
        )
        memory_summary = {
            "mechanism": None
            if mechanism is None
            else mechanism.model_dump(mode="json"),
            "lineage": [item.model_dump(mode="json") for item in lineage],
            "mechanism_count": len(memory.list_mechanisms()),
            "rejection_evidence_ids": orchestration.rejection_evidence_ids,
        }
        audit_summary = {
            "agent_run_id": orchestration.agent_run_id,
            "candidate_events": [item.model_dump(mode="json") for item in candidate_events],
            "llm_calls": [item.model_dump(mode="json") for item in llm_calls],
            "tool_calls": [item.model_dump(mode="json") for item in tool_calls],
            "multi_agent_runs": [
                item.model_dump(mode="json") for item in multi_agent_runs
            ],
            "candidate_portfolios": [
                item.model_dump(mode="json") for item in portfolios
            ],
            "multi_agent_role_events": [
                item.model_dump(mode="json") for item in role_events
            ],
        }

    split_guard = {
        "train_map_ids": [item.map_id for item in train_maps],
        "validation_map_ids": [item.map_id for item in validation_maps],
        "test_map_ids": [item.map_id for item in test_maps],
        "retention_map_hashes": [item.content_hash for item in validation_maps],
        "heldout_test_hashes": [item.content_hash for item in test_maps],
        "test_split_used_for_retention": False,
        "test_evaluation_after_retention": bool(test_comparison["executed"]),
    }
    report = {
        "run_id": run_paths.run_id,
        "agent_mode": selected_mode,
        "run_dir": str(run_paths.result_dir),
        "database": str(run_paths.database),
        "search": search_summary,
        "diagnosis": {
            "trace_count": diagnosis["trace_count"],
            "operator_profile_count": len(diagnosis["operator_profiles"]),
            "synergy_count": len(diagnosis["synergies"]),
            "warning": diagnosis_warning,
        },
        "parent_operator_ids": parents,
        "counterfactual": {
            "seed": counterfactual_seed,
            "count": len(counterfactual),
            "results": [item.model_dump(mode="json") for item in counterfactual],
        },
        "evidence_bundle": {
            "bundle_id": orchestration.bundle_id,
            "bundle_hash": orchestration.bundle_hash,
            "canonical_path": str(evidence_path),
        },
        "orchestration": orchestration.model_dump(mode="json"),
        "test_comparison": test_comparison,
        "split_guard": split_guard,
        "audit": audit_summary,
        "memory": memory_summary,
    }
    write_json(run_paths.result_dir / "agent_demo.json", report)
    update_latest(config, run_paths.run_id, run_paths.result_dir)
    return report


def run_agent_ablations_workflow(
    config: ExperimentConfig,
    *,
    provider: ProviderName | str = "mock",
    paths: RunPaths,
    model: str | None = None,
    validation_maps: Sequence[Environment2D] | None = None,
    train_maps: Sequence[Environment2D] | None = None,
    parent_operator_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run six compact design ablations under shared CRN validation inputs."""

    if isinstance(validation_maps, Mapping):
        raise TypeError("ablations require the validation split, not a dataset mapping")
    dataset = None
    if validation_maps is None or train_maps is None:
        dataset = ensure_dataset(config)
    validation = list(
        validation_maps if validation_maps is not None else dataset["validation"]  # type: ignore[index]
    )
    if not validation:
        raise ValueError("ablations require at least one validation map")
    available_train = list(
        train_maps if train_maps is not None else dataset["train"]  # type: ignore[index]
    )
    if not available_train:
        raise ValueError("ablations require a train/smoke map")

    if not paths.database.exists():
        run_search_workflow(config, paths, available_train)
        try:
            run_diagnosis_workflow(
                config,
                paths.result_dir,
                figure_dir=paths.figure_dir,
            )
        except Exception:
            # As in agent-demo, a tiny all-null delayed-reward chart may fail
            # after diagnosis and memory were already persisted.
            if not (paths.result_dir / "diagnosis.json").exists():
                raise
    directory, database = _database(paths.result_dir)
    registry = build_manual_operator_registry()
    bundle = build_evidence_for_run(
        config,
        database,
        parent_operator_ids=parent_operator_ids,
        train_maps=available_train,
        registry=registry,
    )
    parent_id = bundle.parent_specs[0].name
    population = list(registry.values())
    compiler = OperatorCompiler(config.dsl)
    evaluator = _evaluator(config)
    fixed_validator = FixedBudgetCandidateValidator(config, evaluator)
    proposal_validator = ProposalValidator()
    rows: list[dict[str, Any]] = []
    validation_reports: dict[str, dict[str, Any]] = {}

    def empty_runtime_metrics() -> dict[str, Any]:
        return {
            "median_parent_operator_runtime_ms": None,
            "median_candidate_operator_runtime_ms": None,
            "median_operator_runtime_reduction": None,
            "candidate_operator_call_count": None,
            "candidate_operator_changed_call_count": None,
            "candidate_operator_accepted_call_count": None,
            "candidate_effective_call_rate": None,
            "candidate_operator_acceptance_rate": None,
            "runtime_evidence_eligible": None,
            "runtime_evidence_reason": None,
        }

    def evaluate_arm(
        arm: str,
        proposal: OperatorProposal,
        *,
        token_input: int,
        token_output: int,
        hard_review: bool,
        evidence_scope: str,
    ) -> None:
        try:
            review = (
                proposal_validator.validate_and_review(proposal, bundle)
                if hard_review
                else None
            )
            if review is not None and review.decision != "approve":
                raise ValueError("rule review requested revision")
            compiled = compiler.compile(proposal.spec)
            validation_report = fixed_validator.validate(
                population,
                parent_id,
                compiled,
                validation,
                generation=1,
                candidate_index=0,
                recorder=None,
                root_run_id=f"{directory.name}-ablation-shared",
            )
            validation_reports[arm] = validation_report.model_dump(mode="json")
            rows.append(
                {
                    "arm": arm,
                    "status": "validated",
                    "candidate": proposal.spec.name,
                    "retained": validation_report.retained,
                    "mean_gain": validation_report.mean_gain,
                    "win_rate": validation_report.win_rate,
                    "candidate_feasibility_rate": validation_report.candidate_feasibility_rate,
                    "median_runtime_reduction": validation_report.median_runtime_reduction,
                    "median_parent_operator_runtime_ms": validation_report.median_parent_operator_runtime_ms,
                    "median_candidate_operator_runtime_ms": validation_report.median_candidate_operator_runtime_ms,
                    "median_operator_runtime_reduction": validation_report.median_operator_runtime_reduction,
                    "candidate_operator_call_count": validation_report.candidate_operator_call_count,
                    "candidate_operator_changed_call_count": validation_report.candidate_operator_changed_call_count,
                    "candidate_operator_accepted_call_count": validation_report.candidate_operator_accepted_call_count,
                    "candidate_effective_call_rate": validation_report.candidate_effective_call_rate,
                    "candidate_operator_acceptance_rate": validation_report.candidate_operator_acceptance_rate,
                    "runtime_evidence_eligible": validation_report.runtime_evidence_eligible,
                    "runtime_evidence_reason": validation_report.runtime_evidence_reason,
                    "input_tokens": token_input,
                    "output_tokens": token_output,
                    "total_tokens": token_input + token_output,
                    "evidence_scope": evidence_scope,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "arm": arm,
                    "status": "rejected",
                    "candidate": proposal.spec.name,
                    "retained": False,
                    "mean_gain": None,
                    "win_rate": None,
                    "candidate_feasibility_rate": None,
                    "median_runtime_reduction": None,
                    **empty_runtime_metrics(),
                    "input_tokens": token_input,
                    "output_tokens": token_output,
                    "total_tokens": token_input + token_output,
                    "evidence_scope": evidence_scope,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    heuristic = HeuristicDesigner().propose(
        bundle.problem_summary,
        list(bundle.parent_specs),
        list(bundle.parent_profiles),
        [],
        [item.model_dump(mode="json") for item in bundle.representative_success_cases],
        [item.model_dump(mode="json") for item in bundle.representative_failure_cases],
    )
    evaluate_arm(
        "heuristic",
        heuristic,
        token_input=0,
        token_output=0,
        hard_review=False,
        evidence_scope="deterministic_profile_score",
    )

    for arm, mode, scope in (
        ("score_only_llm", "single_call", "single structured total-score call"),
        ("diagnostic_llm", "staged", "structured diagnosis without agent tools"),
        (
            "diagnosis_memory_llm",
            "staged",
            "structured diagnosis plus mechanism-memory bundle",
        ),
    ):
        arm_provider = create_llm_provider(provider, model=model)
        adapter = LLMDesignerAdapter(provider=arm_provider)
        try:
            proposal = adapter.propose_from_evidence(
                bundle,
                mode=mode,  # type: ignore[arg-type]
                call_config=_llm_config(config, model),
            )
            records = list(getattr(arm_provider, "call_records", ()))
            evaluate_arm(
                arm,
                proposal,
                token_input=sum(item.usage.input_tokens for item in records),
                token_output=sum(item.usage.output_tokens for item in records),
                hard_review=True,
                evidence_scope=scope,
            )
        except Exception as exc:
            records = list(getattr(arm_provider, "call_records", ()))
            rows.append(
                {
                    "arm": arm,
                    "status": "rejected",
                    "candidate": None,
                    "retained": False,
                    "mean_gain": None,
                    "win_rate": None,
                    "candidate_feasibility_rate": None,
                    "median_runtime_reduction": None,
                    **empty_runtime_metrics(),
                    "input_tokens": sum(item.usage.input_tokens for item in records),
                    "output_tokens": sum(item.usage.output_tokens for item in records),
                    "total_tokens": sum(item.usage.total_tokens for item in records),
                    "evidence_scope": scope,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if provider == "mock":
        agent_provider = create_llm_provider("mock", model=model)
        agent_backend = DeterministicMockResearchAgent(agent_provider)
        with MechanismMemory(database) as memory:
            agent_result = agent_backend.run(
                AgentToolContext(
                    bundle=bundle,
                    compiler=compiler,
                    memory=memory,
                    smoke_fixture=SmokeTestFixture(
                        available_train[0],
                        initialize_path(
                            available_train[0],
                            grid_resolution=config.maps.grid_resolution,
                        ),
                    ),
                ),
                budget=_agent_budget(config),
                call_config=_llm_config(config, model),
                agent_run_id=f"ablation_agent_{bundle.bundle_hash[:12]}",
            )
        if agent_result.proposal is not None:
            evaluate_arm(
                "single_agent",
                agent_result.proposal,
                token_input=agent_result.usage.input_tokens,
                token_output=agent_result.usage.output_tokens,
                hard_review=True,
                evidence_scope="diagnosis, memory, and whitelisted tools",
            )
        else:
            rows.append(
                {
                    "arm": "single_agent",
                    "status": "rejected",
                    "candidate": None,
                    "retained": False,
                    "mean_gain": None,
                    "win_rate": None,
                    "candidate_feasibility_rate": None,
                    "median_runtime_reduction": None,
                    **empty_runtime_metrics(),
                    "input_tokens": agent_result.usage.input_tokens,
                    "output_tokens": agent_result.usage.output_tokens,
                    "total_tokens": agent_result.usage.total_tokens,
                    "evidence_scope": "diagnosis, memory, and whitelisted tools",
                    "error": agent_result.candidates[-1].rejection_reason,
                }
            )

        multi_provider = create_llm_provider("mock", model=model)
        multi_backend = DeterministicMockMultiAgent(multi_provider)
        with MechanismMemory(database) as memory:
            multi_result = multi_backend.run(
                AgentToolContext(
                    bundle=bundle,
                    compiler=compiler,
                    memory=memory,
                    smoke_fixture=SmokeTestFixture(
                        available_train[0],
                        initialize_path(
                            available_train[0],
                            grid_resolution=config.maps.grid_resolution,
                        ),
                    ),
                ),
                budget=_agent_budget(config),
                call_config=_llm_config(config, model),
                agent_run_id=f"ablation_multi_{bundle.bundle_hash[:12]}",
            )
        if multi_result.proposal is not None:
            evaluate_arm(
                "multi_agent",
                multi_result.proposal,
                token_input=multi_result.usage.input_tokens,
                token_output=multi_result.usage.output_tokens,
                hard_review=True,
                evidence_scope="shared diagnosis, two designers, critic, and portfolio",
            )
        else:
            rows.append(
                {
                    "arm": "multi_agent",
                    "status": "rejected",
                    "candidate": None,
                    "retained": False,
                    "mean_gain": None,
                    "win_rate": None,
                    "candidate_feasibility_rate": None,
                    "median_runtime_reduction": None,
                    **empty_runtime_metrics(),
                    "input_tokens": multi_result.usage.input_tokens,
                    "output_tokens": multi_result.usage.output_tokens,
                    "total_tokens": multi_result.usage.total_tokens,
                    "evidence_scope": "shared diagnosis, two designers, critic, and portfolio",
                    "error": "; ".join(
                        item.rejection_reason or "no eligible portfolio candidate"
                        for item in multi_result.candidates
                    ),
                }
            )
    else:
        # Explicit configuration check occurs even though live ablations are
        # normally skipped in offline CI.
        create_llm_provider("openai", model=model)
        rows.append(
            {
                "arm": "single_agent",
                "status": "not_run",
                "candidate": None,
                "retained": False,
                "mean_gain": None,
                "win_rate": None,
                "candidate_feasibility_rate": None,
                "median_runtime_reduction": None,
                **empty_runtime_metrics(),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "evidence_scope": "OpenAI Agents SDK live arm",
                "error": "use run-agent for the live SDK arm",
            }
        )
        rows.append(
            {
                "arm": "multi_agent",
                "status": "not_run",
                "candidate": None,
                "retained": False,
                "mean_gain": None,
                "win_rate": None,
                "candidate_feasibility_rate": None,
                "median_runtime_reduction": None,
                **empty_runtime_metrics(),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "evidence_scope": "offline deterministic mock multi-agent only",
                "error": "multi_agent requires provider=mock",
            }
        )

    report = {
        "bundle_hash": bundle.bundle_hash,
        "parent_operator_id": parent_id,
        "shared_validation_map_ids": [item.map_id for item in validation],
        "shared_validation_map_hashes": [item.content_hash for item in validation],
        "shared_generation": 1,
        "shared_candidate_index": 0,
        "shared_validation_iterations": config.search.validation_iterations,
        "runtime_validation_repetitions": config.evolution.runtime_validation_repetitions,
        "min_runtime_effective_call_rate": config.evolution.min_runtime_effective_call_rate,
        "test_split_accessed": False,
        "arms": rows,
        "validation_reports": validation_reports,
        "token_summary": {
            row["arm"]: int(row["total_tokens"] or 0) for row in rows
        },
    }
    write_json(directory / "agent_ablations.json", report)
    write_csv(directory / "agent_ablations.csv", rows)
    return report


__all__ = [
    "agent_demo_workflow",
    "build_evidence_workflow",
    "create_llm_provider",
    "propose_operator_workflow",
    "run_agent_ablations_workflow",
    "run_agent_workflow",
    "validate_candidate_workflow",
]
