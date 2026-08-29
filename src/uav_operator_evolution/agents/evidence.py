"""Compact, content-addressed evidence for LLM and agent operator design."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from operator_evolution_core.proposal import DomainKit, ensure_domain_compatibility

from ..diagnosis.counterfactual import CounterfactualResult
from ..domain.uav_kit import UAVDomainKit, UAV_IR_VERSION
from ..memory import MechanismMemory
from ..operators.registry import OperatorRegistry
from ..operators.specs import OperatorSpec
from ..reproducibility import canonical_json, stable_hash
from ..trajectory import OperatorTrace, TrajectoryRecorder


class EvidenceModel(BaseModel):
    """Strict JSON-native base for every model-facing evidence object."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvidenceItem(EvidenceModel):
    evidence_id: str = Field(pattern=r"^(ctx|fail|syn|cf|case)_[0-9a-f]{24}$")
    source_refs: list[str] = Field(default_factory=list, max_length=64)
    sample_count: int = Field(default=0, ge=0)
    effect_size: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    low_confidence: bool = True

    @field_validator("source_refs")
    @classmethod
    def stable_source_refs(cls, values: list[str]) -> list[str]:
        return sorted({str(value) for value in values})


class ContextEvidence(EvidenceItem):
    operator_id: str
    classification: Literal["effective", "failure"]
    context: dict[str, Any] = Field(default_factory=dict)
    mean_reward: float | None = None
    baseline_reward: float | None = None


class FailureEvidence(EvidenceItem):
    operator_id: str
    failure_mode: str
    severity: float = Field(default=1.0, ge=0.0)
    context: dict[str, Any] = Field(default_factory=dict)


class SynergyEvidence(EvidenceItem):
    first_operator: str
    second_operator: str
    score: float
    context: dict[str, Any] = Field(default_factory=dict)
    evidence_type: Literal["association"] = "association"


class CounterfactualEvidence(EvidenceItem):
    operator_id: str
    compared_operators: list[str] = Field(default_factory=list)
    mean_reward: float | None = None
    mean_advantage: float | None = None
    feasible_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    error_count: int = Field(default=0, ge=0)
    batch_id: str | None = None
    seed: int | None = Field(default=None, ge=0)


class CaseSummary(EvidenceItem):
    operator_id: str
    outcome: Literal["success", "failure"]
    case_id: str | None = None
    trace_id: int | None = Field(default=None, ge=1)
    run_id: str | None = None
    map_id: str | None = None
    iteration: int | None = Field(default=None, ge=0)
    context: dict[str, Any] = Field(default_factory=dict)
    before_objective: float | None = None
    candidate_objective: float | None = None
    accepted_objective: float | None = None
    before_feasible: bool | None = None
    candidate_feasible: bool | None = None
    accepted_feasible: bool | None = None
    immediate_reward: float | None = None
    delayed_rewards: dict[int, float | None] = Field(default_factory=dict)
    accepted: bool | None = None
    acceptance_reason: str | None = None
    error: str | None = None
    runtime_ms: float | None = Field(default=None, ge=0.0)


class DesignBudget(EvidenceModel):
    max_parent_specs: int = Field(default=4, ge=1, le=8)
    max_context_evidence: int = Field(default=8, ge=0, le=64)
    max_failure_evidence: int = Field(default=8, ge=0, le=64)
    max_synergy_evidence: int = Field(default=8, ge=0, le=64)
    max_counterfactual_evidence: int = Field(default=8, ge=0, le=64)
    max_success_cases: int = Field(default=3, ge=0, le=20)
    max_failure_cases: int = Field(default=3, ge=0, le=20)
    max_bundle_chars: int = Field(default=60_000, ge=2_000, le=1_000_000)
    max_candidate_specs: int = Field(default=1, ge=1, le=8)


class OperatorEvidenceBundle(EvidenceModel):
    bundle_version: str = "1"
    bundle_hash: str = ""
    problem_summary: str = Field(min_length=1, max_length=4_000)
    parent_specs: list[OperatorSpec] = Field(min_length=1, max_length=4)
    parent_profiles: list[dict[str, Any]] = Field(default_factory=list, max_length=4)
    effective_contexts: list[ContextEvidence] = Field(default_factory=list)
    failure_contexts: list[ContextEvidence] = Field(default_factory=list)
    failure_modes: list[FailureEvidence] = Field(default_factory=list)
    synergy_evidence: list[SynergyEvidence] = Field(default_factory=list)
    counterfactual_evidence: list[CounterfactualEvidence] = Field(default_factory=list)
    representative_success_cases: list[CaseSummary] = Field(default_factory=list)
    representative_failure_cases: list[CaseSummary] = Field(default_factory=list)
    existing_operator_names: list[str] = Field(default_factory=list)
    allowed_primitives: dict[str, list[str]] = Field(default_factory=dict)
    design_budget: DesignBudget = Field(default_factory=DesignBudget)
    limitations: list[str] = Field(default_factory=list)

    @property
    def domain_id(self) -> str:
        """Implicit binding for pre-envelope UAV artifacts (not serialized)."""

        return UAVDomainKit.domain_id

    @property
    def ir_version(self) -> str:
        """Implicit ``uav-v1`` binding without changing legacy bundle hashes."""

        return UAV_IR_VERSION

    @model_validator(mode="after")
    def canonicalize_and_hash(self) -> "OperatorEvidenceBundle":
        self.parent_specs.sort(key=lambda item: item.name)
        self.parent_profiles.sort(key=lambda item: str(item.get("operator_id", item.get("operator_name", ""))))
        for field_name in (
            "effective_contexts",
            "failure_contexts",
            "failure_modes",
            "synergy_evidence",
            "counterfactual_evidence",
            "representative_success_cases",
            "representative_failure_cases",
        ):
            getattr(self, field_name).sort(key=lambda item: item.evidence_id)
        object.__setattr__(
            self, "existing_operator_names", sorted(set(self.existing_operator_names))
        )
        object.__setattr__(
            self,
            "allowed_primitives",
            {key: sorted(set(values)) for key, values in sorted(self.allowed_primitives.items())},
        )
        object.__setattr__(self, "limitations", sorted(set(self.limitations)))
        items = list(self.iter_evidence())
        identifiers = [item.evidence_id for item in items]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evidence IDs must be unique inside a bundle")
        expected = stable_hash(self.model_dump(mode="json", exclude={"bundle_hash"}))
        if self.bundle_hash and self.bundle_hash != expected:
            raise ValueError("bundle_hash does not match canonical bundle content")
        object.__setattr__(self, "bundle_hash", expected)
        return self

    def iter_evidence(self) -> Iterable[EvidenceItem]:
        for field_name in (
            "effective_contexts",
            "failure_contexts",
            "failure_modes",
            "synergy_evidence",
            "counterfactual_evidence",
            "representative_success_cases",
            "representative_failure_cases",
        ):
            yield from getattr(self, field_name)

    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.iter_evidence())

    def allowed_primitive_names(self) -> tuple[str, ...]:
        return tuple(
            name for category in sorted(self.allowed_primitives) for name in self.allowed_primitives[category]
        )


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _evidence_id(prefix: str, semantic_payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{stable_hash(_json_value(semantic_payload))[:24]}"


def _confidence(sample_count: int, minimum_samples: int) -> tuple[float, bool]:
    if minimum_samples <= 0:
        return 1.0, False
    confidence = min(1.0, sample_count / minimum_samples)
    return confidence, sample_count < minimum_samples


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return {}


def _profile_value(profile: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in profile:
            return profile[name]
    return default


class EvidenceBundleBuilder:
    """Build bounded evidence without sending raw trajectory databases to a model."""

    def __init__(
        self,
        memory: MechanismMemory,
        registry: OperatorRegistry,
        *,
        recorder: TrajectoryRecorder | None = None,
        minimum_reliable_samples: int = 3,
        domain_kit: DomainKit[Any, Any, Any] | None = None,
    ) -> None:
        if minimum_reliable_samples < 1:
            raise ValueError("minimum_reliable_samples must be positive")
        self.memory = memory
        self.registry = registry
        self.recorder = recorder
        self.minimum_reliable_samples = int(minimum_reliable_samples)
        self.domain_kit = domain_kit or UAVDomainKit()

    def build(
        self,
        problem_summary: str,
        parent_operator_ids: Sequence[str],
        design_budget: DesignBudget | None = None,
        *,
        parent_profiles: Sequence[Mapping[str, Any] | Any] | None = None,
        counterfactual_results: Sequence[CounterfactualResult | Mapping[str, Any]] = (),
        counterfactual_seed: int | None = None,
    ) -> OperatorEvidenceBundle:
        budget = design_budget or DesignBudget()
        parents = list(dict.fromkeys(str(item) for item in parent_operator_ids))
        if not parents:
            raise ValueError("at least one parent operator is required")
        if len(parents) > budget.max_parent_specs:
            raise ValueError("parent operator count exceeds DesignBudget")

        specs = [self._resolve_spec(name) for name in parents]
        profiles = self._resolve_profiles(parents, parent_profiles)
        limitations: list[str] = []

        effective, failed_contexts = self._context_evidence(profiles, budget)
        if not effective:
            limitations.append("no effective context evidence was available")
        if not failed_contexts:
            limitations.append("no failure context evidence was available")
        failures = self._failure_evidence(parents, budget)
        if not failures:
            limitations.append("no persisted failure-mode evidence was available")
        synergies = self._synergy_evidence(parents, budget)
        if not synergies:
            limitations.append("no persisted synergy evidence was available")
        success_cases, failure_cases = self._case_evidence(parents, profiles, budget)
        if not success_cases:
            limitations.append("no representative success cases were available")
        if not failure_cases:
            limitations.append("no representative failure cases were available")
        counterfactual = self._counterfactual_evidence(
            counterfactual_results, budget, counterfactual_seed
        )
        if not counterfactual:
            limitations.append("counterfactual evidence was not supplied")

        existing = set(self.registry.names())
        existing.update(record.mechanism_id for record in self.memory.list_mechanisms(status=None))
        bundle = OperatorEvidenceBundle(
            problem_summary=problem_summary,
            parent_specs=specs,
            parent_profiles=profiles,
            effective_contexts=effective,
            failure_contexts=failed_contexts,
            failure_modes=failures,
            synergy_evidence=synergies,
            counterfactual_evidence=counterfactual,
            representative_success_cases=success_cases,
            representative_failure_cases=failure_cases,
            existing_operator_names=sorted(existing),
            allowed_primitives={
                key: list(values)
                for key, values in self.domain_kit.capability_catalog().items()
            },
            design_budget=budget,
            limitations=limitations,
        )
        ensure_domain_compatibility(
            self.domain_kit,
            bundle,
            allow_legacy_unversioned=True,
        )
        if len(canonical_json(bundle.model_dump(mode="json"))) > budget.max_bundle_chars:
            raise ValueError("canonical evidence bundle exceeds max_bundle_chars")
        return bundle

    def _resolve_spec(self, operator_id: str) -> OperatorSpec:
        if operator_id in self.registry:
            operator = self.registry.get(operator_id)
            spec = getattr(operator, "spec", None)
            if spec is not None:
                return self.domain_kit.parse_ir(spec)
        mechanism = self.memory.get_mechanism(operator_id)
        if mechanism is not None and mechanism.definition:
            return self.domain_kit.parse_ir(mechanism.definition)
        builtin = self.domain_kit.builtin_ir(operator_id)
        if builtin is not None:
            return builtin
        raise KeyError(f"operator specification not found: {operator_id}")

    def _resolve_profiles(
        self,
        parents: Sequence[str],
        supplied: Sequence[Mapping[str, Any] | Any] | None,
    ) -> list[dict[str, Any]]:
        supplied_by_id: dict[str, dict[str, Any]] = {}
        for item in supplied or ():
            payload = _mapping(_json_value(item))
            name = str(_profile_value(payload, "operator_id", "operator_name", default=""))
            if name:
                supplied_by_id[name] = payload
        output: list[dict[str, Any]] = []
        for parent in parents:
            payload = supplied_by_id.get(parent)
            profile_id: int | None = None
            if payload is None:
                rows = self.memory.get_operator_profiles(parent, limit=1)
                if rows:
                    payload = dict(rows[0].profile)
                    profile_id = rows[0].profile_id
            if payload is None:
                mechanism = self.memory.get_mechanism(parent)
                profile = mechanism.metadata.get("profile") if mechanism is not None else None
                payload = dict(profile) if isinstance(profile, Mapping) else {}
            compact = self._compact_profile(parent, payload)
            if profile_id is not None:
                compact["source_profile_id"] = profile_id
            output.append(compact)
        return output

    @staticmethod
    def _compact_profile(operator_id: str, profile: Mapping[str, Any]) -> dict[str, Any]:
        allowed = (
            "attempts", "total_calls", "acceptances", "acceptance_rate", "successes",
            "success_rate", "mean_immediate_reward", "average_immediate_reward",
            "mean_delayed_rewards", "average_delayed_reward", "delayed_sample_counts",
            "feasibility_rate", "mean_runtime_ms", "average_runtime_ms", "failure_modes",
            "representative_success_ids", "representative_failure_ids",
            "effective_context_groups", "failure_context_groups", "effective_contexts",
            "failure_contexts",
        )
        result: dict[str, Any] = {"operator_id": operator_id}
        for key in allowed:
            if key in profile:
                value = _json_value(profile[key])
                if key in {
                    "effective_context_groups",
                    "failure_context_groups",
                    "effective_contexts",
                    "failure_contexts",
                } and isinstance(value, list):
                    value = value[:8]
                elif key in {
                    "representative_success_ids",
                    "representative_failure_ids",
                } and isinstance(value, list):
                    value = value[:3]
                elif key == "failure_modes" and isinstance(value, dict):
                    value = dict(
                        sorted(value.items(), key=lambda item: (-int(item[1]), str(item[0])))[:16]
                    )
                result[key] = value
        return result

    def _context_evidence(
        self, profiles: Sequence[dict[str, Any]], budget: DesignBudget
    ) -> tuple[list[ContextEvidence], list[ContextEvidence]]:
        output: dict[str, list[ContextEvidence]] = {"effective": [], "failure": []}
        for profile in profiles:
            operator_id = str(profile["operator_id"])
            baseline = _number(_profile_value(profile, "mean_immediate_reward", "average_immediate_reward"))
            profile_id = profile.get("source_profile_id", operator_id)
            for classification, names in (
                ("effective", ("effective_context_groups", "effective_contexts")),
                ("failure", ("failure_context_groups", "failure_contexts")),
            ):
                raw: list[Any] = []
                for name in names:
                    if isinstance(profile.get(name), list):
                        raw = list(profile[name])
                        break
                for item in raw:
                    payload = _mapping(item)
                    context = _mapping(payload.get("context", payload))
                    for statistic in ("calls", "sample_count", "average_reward", "mean_reward", "effect_size"):
                        context.pop(statistic, None)
                    count = int(payload.get("calls", payload.get("sample_count", 0)) or 0)
                    mean_reward = _number(payload.get("average_reward", payload.get("mean_reward")))
                    effect = _number(payload.get("effect_size"))
                    if effect is None and mean_reward is not None and baseline is not None:
                        effect = mean_reward - baseline
                    confidence, low = _confidence(count, self.minimum_reliable_samples)
                    semantic = {
                        "operator_id": operator_id,
                        "classification": classification,
                        "context": context,
                        "sample_count": count,
                        "effect_size": effect,
                        "mean_reward": mean_reward,
                        "baseline_reward": baseline,
                    }
                    output[classification].append(
                        ContextEvidence(
                            evidence_id=_evidence_id("ctx", semantic),
                            source_refs=[f"profile:{profile_id}"],
                            sample_count=count,
                            effect_size=effect,
                            confidence=confidence,
                            low_confidence=low,
                            operator_id=operator_id,
                            classification=classification,  # type: ignore[arg-type]
                            context=context,
                            mean_reward=mean_reward,
                            baseline_reward=baseline,
                        )
                    )
        effective = self._dedupe(output["effective"])[: budget.max_context_evidence]
        failed = self._dedupe(output["failure"])[: budget.max_context_evidence]
        return effective, failed

    def _failure_evidence(
        self, parents: Sequence[str], budget: DesignBudget
    ) -> list[FailureEvidence]:
        items: list[FailureEvidence] = []
        for parent in parents:
            for row in self.memory.get_failure_modes(parent, limit=budget.max_failure_evidence * 4):
                confidence, low = _confidence(row.count, self.minimum_reliable_samples)
                semantic = {
                    "operator_id": parent,
                    "failure_mode": row.mode,
                    "severity": row.severity,
                    "context": row.context,
                    "sample_count": row.count,
                }
                items.append(
                    FailureEvidence(
                        evidence_id=_evidence_id("fail", semantic),
                        source_refs=[f"failure:{row.failure_id}"],
                        sample_count=row.count,
                        effect_size=-float(row.severity),
                        confidence=confidence,
                        low_confidence=low,
                        operator_id=parent,
                        failure_mode=row.mode,
                        severity=row.severity,
                        context=row.context,
                    )
                )
        return self._dedupe(items)[: budget.max_failure_evidence]

    def _synergy_evidence(
        self, parents: Sequence[str], budget: DesignBudget
    ) -> list[SynergyEvidence]:
        items: list[SynergyEvidence] = []
        for parent in parents:
            for row in self.memory.get_synergies(operator_id=parent, limit=budget.max_synergy_evidence * 4):
                confidence, low = _confidence(row.sample_count, self.minimum_reliable_samples)
                semantic = {
                    "first_operator": row.first_operator,
                    "second_operator": row.second_operator,
                    "score": row.score,
                    "context": row.context,
                    "sample_count": row.sample_count,
                }
                items.append(
                    SynergyEvidence(
                        evidence_id=_evidence_id("syn", semantic),
                        source_refs=[f"synergy:{row.synergy_id}"],
                        sample_count=row.sample_count,
                        effect_size=row.score,
                        confidence=confidence,
                        low_confidence=low,
                        first_operator=row.first_operator,
                        second_operator=row.second_operator,
                        score=row.score,
                        context=row.context,
                    )
                )
        items = self._dedupe(items)
        items.sort(key=lambda item: (-abs(item.score), item.evidence_id))
        return items[: budget.max_synergy_evidence]

    def _case_evidence(
        self,
        parents: Sequence[str],
        profiles: Sequence[dict[str, Any]],
        budget: DesignBudget,
    ) -> tuple[list[CaseSummary], list[CaseSummary]]:
        success: list[CaseSummary] = []
        failure: list[CaseSummary] = []
        profile_by_id = {str(item["operator_id"]): item for item in profiles}
        for parent in parents:
            for outcome, target, limit in (
                ("success", success, budget.max_success_cases),
                ("failure", failure, budget.max_failure_cases),
            ):
                for row in self.memory.get_relevant_cases(
                    operator_id=parent, outcome=outcome, limit=limit
                ):
                    target.append(self._case_from_memory(row, parent, outcome))
                remaining = limit - len(target)
                if remaining <= 0 or self.recorder is None:
                    continue
                profile = profile_by_id.get(parent, {})
                key = "representative_success_ids" if outcome == "success" else "representative_failure_ids"
                for trace_id in list(profile.get(key, []))[:remaining]:
                    trace = self.recorder.get_trace(int(trace_id))
                    if trace is not None:
                        target.append(self._case_from_trace(trace, outcome))
        return (
            self._dedupe(success)[: budget.max_success_cases],
            self._dedupe(failure)[: budget.max_failure_cases],
        )

    def _case_from_trace(self, trace: OperatorTrace, outcome: str) -> CaseSummary:
        context = {
            "map_difficulty": trace.map_difficulty,
            "phase": trace.context.get("phase"),
            "search_phase": trace.context.get("search_phase"),
            "stagnation_count": trace.context.get("search_features", {}).get("stagnation_count")
            if isinstance(trace.context.get("search_features"), Mapping)
            else None,
        }
        semantic = {
            "operator_id": trace.operator_id,
            "trace_id": trace.trace_id,
            "outcome": outcome,
            "before": trace.before_objective,
            "candidate": trace.candidate_objective,
            "accepted": trace.accepted_objective,
        }
        return CaseSummary(
            evidence_id=_evidence_id("case", semantic),
            source_refs=[f"trace:{trace.trace_id}"],
            sample_count=1,
            effect_size=trace.immediate_reward,
            confidence=min(1.0, 1 / self.minimum_reliable_samples),
            low_confidence=self.minimum_reliable_samples > 1,
            operator_id=trace.operator_id,
            outcome=outcome,  # type: ignore[arg-type]
            trace_id=trace.trace_id,
            run_id=trace.run_id,
            map_id=trace.map_id,
            iteration=trace.iteration,
            context={key: value for key, value in context.items() if value is not None},
            before_objective=trace.before_objective,
            candidate_objective=trace.candidate_objective,
            accepted_objective=trace.accepted_objective,
            before_feasible=trace.before_feasible,
            candidate_feasible=trace.candidate_feasible,
            accepted_feasible=trace.accepted_feasible,
            immediate_reward=trace.immediate_reward,
            delayed_rewards=trace.delayed_rewards,
            accepted=trace.accepted,
            acceptance_reason=trace.acceptance_reason,
            error=trace.error,
            runtime_ms=trace.runtime_ms,
        )

    def _case_from_memory(self, row: Any, operator_id: str, outcome: str) -> CaseSummary:
        result = _mapping(row.result)
        semantic = {
            "operator_id": operator_id,
            "case_id": row.case_id,
            "outcome": outcome,
            "score": row.score,
            "context": row.context,
        }
        return CaseSummary(
            evidence_id=_evidence_id("case", semantic),
            source_refs=[f"case:{row.case_id}"],
            sample_count=1,
            effect_size=row.score,
            confidence=min(1.0, 1 / self.minimum_reliable_samples),
            low_confidence=self.minimum_reliable_samples > 1,
            operator_id=operator_id,
            outcome=outcome,  # type: ignore[arg-type]
            case_id=row.case_id,
            context=_mapping(row.context),
            before_objective=_number(result.get("before_objective")),
            candidate_objective=_number(result.get("candidate_objective")),
            accepted_objective=_number(result.get("accepted_objective")),
            immediate_reward=_number(result.get("immediate_reward", row.score)),
            accepted=result.get("accepted") if isinstance(result.get("accepted"), bool) else None,
            error=str(result["error"]) if result.get("error") else None,
            runtime_ms=_number(result.get("runtime_ms")),
        )

    def _counterfactual_evidence(
        self,
        supplied: Sequence[CounterfactualResult | Mapping[str, Any]],
        budget: DesignBudget,
        seed: int | None,
    ) -> list[CounterfactualEvidence]:
        rows = [
            item if isinstance(item, CounterfactualResult) else CounterfactualResult.model_validate(item)
            for item in supplied
        ]
        if not rows:
            return []
        names = sorted({row.operator_id for row in rows})
        # Runtime is useful raw evaluator telemetry but is deliberately not
        # semantic evidence: wall-clock jitter must not change evidence IDs or
        # a content-addressed bundle.  Sort the stable row projection as well,
        # because provider/tool callers need not preserve evaluator order.
        semantic_rows = [
            {
                "state_index": row.state_index,
                "source_trace_id": row.source_trace_id,
                "operator_id": row.operator_id,
                "before_objective": row.before_objective,
                "candidate_objective": row.candidate_objective,
                "reward": row.reward,
                "advantage": row.advantage,
                "feasible": row.feasible,
                "error": row.error,
            }
            for row in rows
        ]
        semantic_rows.sort(key=canonical_json)
        batch_id = f"cfbatch_{stable_hash({'seed': seed, 'rows': semantic_rows})[:24]}"
        groups: dict[str, list[CounterfactualResult]] = defaultdict(list)
        for row in rows:
            groups[row.operator_id].append(row)
        output: list[CounterfactualEvidence] = []
        for operator_id, group in groups.items():
            rewards = [float(row.reward) for row in group if row.reward is not None]
            advantages = [float(row.advantage) for row in group if row.advantage is not None]
            feasible = [bool(row.feasible) for row in group if row.feasible is not None]
            sample_count = len(group)
            mean_reward = fmean(rewards) if rewards else None
            mean_advantage = fmean(advantages) if advantages else None
            confidence, low = _confidence(sample_count, self.minimum_reliable_samples)
            semantic = {
                "operator_id": operator_id,
                "compared_operators": names,
                "sample_count": sample_count,
                "mean_reward": mean_reward,
                "mean_advantage": mean_advantage,
                "feasible_rate": fmean(feasible) if feasible else None,
                "error_count": sum(row.error is not None for row in group),
                "batch_id": batch_id,
                "seed": seed,
            }
            refs = [f"trace:{row.source_trace_id}" for row in group if row.source_trace_id is not None]
            output.append(
                CounterfactualEvidence(
                    evidence_id=_evidence_id("cf", semantic),
                    source_refs=[batch_id, *refs],
                    sample_count=sample_count,
                    effect_size=mean_advantage if mean_advantage is not None else mean_reward,
                    confidence=confidence,
                    low_confidence=low,
                    operator_id=operator_id,
                    compared_operators=names,
                    mean_reward=mean_reward,
                    mean_advantage=mean_advantage,
                    feasible_rate=fmean(feasible) if feasible else None,
                    error_count=sum(row.error is not None for row in group),
                    batch_id=batch_id,
                    seed=seed,
                )
            )
        output.sort(key=lambda item: item.evidence_id)
        return output[: budget.max_counterfactual_evidence]

    @staticmethod
    def _dedupe(items: Sequence[Any]) -> list[Any]:
        unique: dict[str, Any] = {}
        for item in items:
            current = unique.get(item.evidence_id)
            if current is None:
                unique[item.evidence_id] = item
                continue
            if current.model_dump(mode="json", exclude={"source_refs"}) != item.model_dump(
                mode="json", exclude={"source_refs"}
            ):
                raise ValueError(f"evidence hash collision: {item.evidence_id}")
            current.source_refs = sorted(set(current.source_refs) | set(item.source_refs))
        return sorted(unique.values(), key=lambda item: item.evidence_id)


__all__ = [
    "CaseSummary",
    "ContextEvidence",
    "CounterfactualEvidence",
    "DesignBudget",
    "EvidenceBundleBuilder",
    "FailureEvidence",
    "OperatorEvidenceBundle",
    "SynergyEvidence",
]
