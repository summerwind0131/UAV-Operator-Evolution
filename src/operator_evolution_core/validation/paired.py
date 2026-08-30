"""Safety contracts and pre-registered paired candidate retention rules."""

from __future__ import annotations

from typing import AbstractSet, Literal, Protocol

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from .fitness import FitnessPolicy


class RetentionConfig(Protocol):
    min_global_gain: float
    min_specialist_gain: float
    min_feasibility_gain: float
    min_runtime_reduction: float
    min_runtime_effective_call_rate: float
    require_bootstrap_ci: bool


class ValidationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PairedOutcome(ValidationModel):
    # ``map_id`` remains the UAV v1 serialization name. New domains use the
    # non-serialized ``instance_id`` compatibility property below.
    map_id: str = Field(validation_alias=AliasChoices("map_id", "instance_id"))
    # ``difficulty`` remains the UAV v1 serialization name. Generic callers
    # use ``context_label`` and never need to know the legacy field spelling.
    difficulty: str = Field(
        validation_alias=AliasChoices("difficulty", "context_label")
    )
    parent_best_cost: float
    candidate_best_cost: float
    parent_feasible: bool
    candidate_feasible: bool
    parent_runtime_ms: float = Field(ge=0)
    candidate_runtime_ms: float = Field(ge=0)
    runtime_repetitions: int = Field(1, ge=1)
    parent_runtime_samples_ms: list[float] = Field(default_factory=list)
    candidate_runtime_samples_ms: list[float] = Field(default_factory=list)
    timing_order: list[Literal["parent_first", "candidate_first"]] = Field(
        default_factory=list
    )
    parent_operator_runtime_ms: float = Field(0.0, ge=0)
    candidate_operator_runtime_ms: float = Field(0.0, ge=0)
    parent_operator_runtime_samples_ms: list[float] = Field(default_factory=list)
    candidate_operator_runtime_samples_ms: list[float] = Field(default_factory=list)
    parent_operator_call_count: int = Field(0, ge=0)
    candidate_operator_call_count: int = Field(0, ge=0)
    parent_operator_changed_call_count: int = Field(0, ge=0)
    candidate_operator_changed_call_count: int = Field(0, ge=0)
    parent_operator_accepted_call_count: int = Field(0, ge=0)
    candidate_operator_accepted_call_count: int = Field(0, ge=0)

    @model_validator(mode="after")
    def repeated_timing_and_call_counts_are_consistent(self) -> "PairedOutcome":
        sample_fields = (
            "parent_runtime_samples_ms",
            "candidate_runtime_samples_ms",
            "parent_operator_runtime_samples_ms",
            "candidate_operator_runtime_samples_ms",
        )
        for field_name in sample_fields:
            samples = getattr(self, field_name)
            if samples and len(samples) != self.runtime_repetitions:
                raise ValueError(
                    f"{field_name} must contain exactly runtime_repetitions samples"
                )
            if any(value < 0 for value in samples):
                raise ValueError(f"{field_name} cannot contain negative values")
        if self.timing_order and len(self.timing_order) != self.runtime_repetitions:
            raise ValueError(
                "timing_order must contain exactly runtime_repetitions entries"
            )
        for prefix in ("parent", "candidate"):
            calls = getattr(self, f"{prefix}_operator_call_count")
            changed = getattr(self, f"{prefix}_operator_changed_call_count")
            accepted = getattr(self, f"{prefix}_operator_accepted_call_count")
            if changed > calls or accepted > calls:
                raise ValueError(
                    f"{prefix} changed/accepted operator counts cannot exceed calls"
                )
        return self

    @property
    def gain(self) -> float:
        denominator = max(abs(self.parent_best_cost), 1e-12)
        return (self.parent_best_cost - self.candidate_best_cost) / denominator

    @property
    def instance_id(self) -> str:
        return self.map_id

    @property
    def context_label(self) -> str:
        return self.difficulty

    @property
    def candidate_effective_call_rate(self) -> float:
        if self.candidate_operator_call_count == 0:
            return 0.0
        return (
            self.candidate_operator_changed_call_count
            / self.candidate_operator_call_count
        )

    @property
    def candidate_operator_acceptance_rate(self) -> float:
        if self.candidate_operator_call_count == 0:
            return 0.0
        return (
            self.candidate_operator_accepted_call_count
            / self.candidate_operator_call_count
        )


class ValidationReport(ValidationModel):
    parent_operator: str
    candidate_operator: str
    safety_passed: bool
    safety_failures: list[str] = Field(default_factory=list)
    outcomes: list[PairedOutcome] = Field(default_factory=list)
    mean_gain: float = 0.0
    win_rate: float = 0.0
    parent_feasibility_rate: float = 0.0
    candidate_feasibility_rate: float = 0.0
    median_runtime_reduction: float = 0.0
    median_parent_operator_runtime_ms: float = 0.0
    median_candidate_operator_runtime_ms: float = 0.0
    median_operator_runtime_reduction: float = 0.0
    candidate_operator_call_count: int = 0
    candidate_operator_changed_call_count: int = 0
    candidate_operator_accepted_call_count: int = 0
    candidate_effective_call_rate: float = 0.0
    candidate_operator_acceptance_rate: float = 0.0
    runtime_evidence_eligible: bool = False
    runtime_evidence_reason: str = "runtime evidence not evaluated"
    specialist_gain: float = 0.0
    bootstrap_ci: tuple[float, float] | None = None
    retained: bool = False
    retention_reasons: list[str] = Field(default_factory=list)
    evidence_level: str = "exploratory"


def paired_bootstrap_ci(
    gains: list[float], seed: int, samples: int = 2000
) -> tuple[float, float] | None:
    if len(gains) < 2:
        return None
    rng = np.random.default_rng(seed)
    values = np.asarray(gains, dtype=float)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def decide_retention(
    parent_operator: str,
    candidate_operator: str,
    outcomes: list[PairedOutcome],
    config: RetentionConfig,
    *,
    safety_passed: bool = True,
    safety_failures: list[str] | None = None,
    bootstrap_seed: int = 0,
    fitness_policy: FitnessPolicy | str = FitnessPolicy.UAV_LEGACY_V1,
    specialist_contexts: AbstractSet[str] = frozenset(),
) -> ValidationReport:
    """Apply effect-size gates; held-out test data must never be passed here."""

    failures = list(safety_failures or [])
    if not outcomes:
        reasons = (
            ["failed safety gate", *failures]
            if not safety_passed
            else ["no validation outcomes"]
        )
        return ValidationReport(
            parent_operator=parent_operator,
            candidate_operator=candidate_operator,
            safety_passed=safety_passed,
            safety_failures=failures,
            retained=False,
            retention_reasons=reasons,
        )

    gains = [outcome.gain for outcome in outcomes]
    mean_gain = float(np.mean(gains))
    win_rate = float(np.mean([gain > 0 for gain in gains]))
    parent_feasible = float(np.mean([outcome.parent_feasible for outcome in outcomes]))
    candidate_feasible = float(np.mean([outcome.candidate_feasible for outcome in outcomes]))
    parent_runtime = float(np.median([outcome.parent_runtime_ms for outcome in outcomes]))
    candidate_runtime = float(np.median([outcome.candidate_runtime_ms for outcome in outcomes]))
    runtime_reduction = (parent_runtime - candidate_runtime) / max(parent_runtime, 1e-12)
    parent_operator_runtime = float(
        np.median([outcome.parent_operator_runtime_ms for outcome in outcomes])
    )
    candidate_operator_runtime = float(
        np.median([outcome.candidate_operator_runtime_ms for outcome in outcomes])
    )
    operator_runtime_reduction = (
        (parent_operator_runtime - candidate_operator_runtime)
        / max(parent_operator_runtime, 1e-12)
        if parent_operator_runtime > 0
        else 0.0
    )
    candidate_calls = sum(
        outcome.candidate_operator_call_count for outcome in outcomes
    )
    candidate_changed_calls = sum(
        outcome.candidate_operator_changed_call_count for outcome in outcomes
    )
    candidate_accepted_calls = sum(
        outcome.candidate_operator_accepted_call_count for outcome in outcomes
    )
    candidate_effective_call_rate = (
        candidate_changed_calls / candidate_calls if candidate_calls else 0.0
    )
    candidate_acceptance_rate = (
        candidate_accepted_calls / candidate_calls if candidate_calls else 0.0
    )
    runtime_evidence_eligible = bool(
        candidate_calls > 0
        and candidate_effective_call_rate
        >= config.min_runtime_effective_call_rate
    )
    if candidate_calls == 0:
        runtime_evidence_reason = "candidate operator was never called"
    elif candidate_changed_calls == 0:
        runtime_evidence_reason = (
            f"candidate operator made no effective path changes (0/{candidate_calls} calls)"
        )
    elif not runtime_evidence_eligible:
        runtime_evidence_reason = (
            "candidate effective call rate "
            f"{candidate_effective_call_rate:.6f} is below the pre-registered minimum "
            f"{config.min_runtime_effective_call_rate:.6f}"
        )
    else:
        runtime_evidence_reason = (
            "candidate effective call rate passed "
            f"({candidate_changed_calls}/{candidate_calls} calls changed the path)"
        )
    hard = [
        outcome.gain
        for outcome in outcomes
        if outcome.context_label in specialist_contexts
    ]
    specialist_gain = float(np.mean(hard)) if hard else 0.0
    ci = paired_bootstrap_ci(gains, bootstrap_seed) if config.require_bootstrap_ci else None

    feasibility_drop = parent_feasible - candidate_feasible
    global_gate = mean_gain >= config.min_global_gain and win_rate >= 0.60 and feasibility_drop <= 0.02
    specialist_gate = (
        len(hard) >= 2
        and specialist_gain >= config.min_specialist_gain
        and float(np.mean([gain > 0 for gain in hard])) >= 2 / 3
        and mean_gain >= -0.01
        and feasibility_drop <= 0.02
    )
    feasibility_gate = (
        candidate_feasible - parent_feasible >= config.min_feasibility_gain and mean_gain >= -0.02
    )
    runtime_effect_gate = (
        runtime_reduction >= config.min_runtime_reduction
        and mean_gain >= -0.01
        and feasibility_drop <= 0.02
    )
    selected_policy = FitnessPolicy(fitness_policy)
    runtime_gate = bool(
        selected_policy is FitnessPolicy.UAV_LEGACY_V1
        and runtime_effect_gate
        and runtime_evidence_eligible
    )
    reasons: list[str] = []
    if global_gate:
        reasons.append("global paired gain")
    if specialist_gate:
        reasons.append("difficult-context specialization")
    if feasibility_gate:
        reasons.append("feasibility improvement")
    if runtime_gate:
        reasons.append("runtime reduction")

    statistical_gate = not config.require_bootstrap_ci or (ci is not None and ci[0] > 0)
    retained = safety_passed and bool(reasons) and statistical_gate
    if not safety_passed:
        reasons = ["failed safety gate", *failures]
    elif config.require_bootstrap_ci and not statistical_gate:
        reasons.append("bootstrap confidence interval includes zero")
    elif not reasons:
        if (
            selected_policy is FitnessPolicy.UAV_LEGACY_V1
            and runtime_effect_gate
            and not runtime_evidence_eligible
        ):
            reasons.append(f"runtime evidence ineligible: {runtime_evidence_reason}")
        else:
            reasons.append("no pre-registered effect threshold met")

    return ValidationReport(
        parent_operator=parent_operator,
        candidate_operator=candidate_operator,
        safety_passed=safety_passed,
        safety_failures=failures,
        outcomes=outcomes,
        mean_gain=mean_gain,
        win_rate=win_rate,
        parent_feasibility_rate=parent_feasible,
        candidate_feasibility_rate=candidate_feasible,
        median_runtime_reduction=runtime_reduction,
        median_parent_operator_runtime_ms=parent_operator_runtime,
        median_candidate_operator_runtime_ms=candidate_operator_runtime,
        median_operator_runtime_reduction=operator_runtime_reduction,
        candidate_operator_call_count=candidate_calls,
        candidate_operator_changed_call_count=candidate_changed_calls,
        candidate_operator_accepted_call_count=candidate_accepted_calls,
        candidate_effective_call_rate=candidate_effective_call_rate,
        candidate_operator_acceptance_rate=candidate_acceptance_rate,
        runtime_evidence_eligible=runtime_evidence_eligible,
        runtime_evidence_reason=runtime_evidence_reason,
        specialist_gain=specialist_gain,
        bootstrap_ci=ci,
        retained=retained,
        retention_reasons=reasons,
        evidence_level="statistical" if config.require_bootstrap_ci and statistical_gate else "exploratory",
    )


__all__ = [
    "PairedOutcome",
    "RetentionConfig",
    "ValidationReport",
    "decide_retention",
    "paired_bootstrap_ci",
]
