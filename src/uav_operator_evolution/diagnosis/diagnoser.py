"""Evidence-backed profiles of operator behavior."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ..trajectory import OperatorTrace, TrajectoryRecorder


class _DiagnosticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class OperatorProfile(_DiagnosticModel):
    """Aggregate evidence for an operator in an optional context group."""

    operator_id: str
    operator_family: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    attempts: int = Field(ge=1)
    acceptances: int = Field(ge=0)
    acceptance_rate: float = Field(ge=0.0, le=1.0)
    successes: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    mean_immediate_reward: float | None = None
    median_immediate_reward: float | None = None
    std_immediate_reward: float | None = None
    min_immediate_reward: float | None = None
    max_immediate_reward: float | None = None
    mean_delayed_rewards: dict[int, float] = Field(default_factory=dict)
    delayed_sample_counts: dict[int, int] = Field(default_factory=dict)
    delayed_positive_counts: dict[int, int] = Field(default_factory=dict)
    feasibility_gain_rate: float | None = None
    feasibility_rate: float | None = None
    mean_runtime_ms: float = Field(ge=0.0)
    failure_modes: dict[str, int] = Field(default_factory=dict)
    representative_success_ids: list[int] = Field(default_factory=list)
    representative_failure_ids: list[int] = Field(default_factory=list)
    effective_context_groups: list[dict[str, Any]] = Field(default_factory=list)
    failure_context_groups: list[dict[str, Any]] = Field(default_factory=list)
    synergy_relation_records: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return self.attempts

    @property
    def accepted_count(self) -> int:
        return self.acceptances

    @property
    def mean_reward(self) -> float | None:
        return self.mean_immediate_reward

    @property
    def group(self) -> dict[str, Any]:
        return self.context

    @computed_field
    @property
    def operator_name(self) -> str:
        """Visualization-compatible spelling of :attr:`operator_id`."""

        return self.operator_id

    @computed_field
    @property
    def total_calls(self) -> int:
        return self.attempts

    @computed_field
    @property
    def immediate_improvement_rate(self) -> float:
        return self.success_rate

    @computed_field
    @property
    def delayed_improvement_rate(self) -> float | None:
        observed = sum(self.delayed_sample_counts.values())
        if not observed:
            return None
        return sum(self.delayed_positive_counts.values()) / observed

    @computed_field
    @property
    def average_immediate_reward(self) -> float | None:
        return self.mean_immediate_reward

    @computed_field
    @property
    def average_delayed_reward(self) -> float | None:
        observed = sum(self.delayed_sample_counts.values())
        if not observed:
            return None
        weighted_total = sum(
            mean * self.delayed_sample_counts.get(horizon, 0)
            for horizon, mean in self.mean_delayed_rewards.items()
        )
        return weighted_total / observed

    @computed_field
    @property
    def reward_std(self) -> float | None:
        return self.std_immediate_reward

    @computed_field
    @property
    def average_runtime_ms(self) -> float:
        return self.mean_runtime_ms

    @computed_field
    @property
    def effective_contexts(self) -> list[dict[str, Any]]:
        return [dict(context) for context in self.effective_context_groups]

    @computed_field
    @property
    def failure_contexts(self) -> list[dict[str, Any]]:
        return [dict(context) for context in self.failure_context_groups]

    @computed_field
    @property
    def representative_success_cases(self) -> list[int]:
        return list(self.representative_success_ids)

    @computed_field
    @property
    def representative_failure_cases(self) -> list[int]:
        return list(self.representative_failure_ids)

    @computed_field
    @property
    def synergy_relations(self) -> list[dict[str, Any]]:
        return [dict(relation) for relation in self.synergy_relation_records]


class SequentialSynergy(_DiagnosticModel):
    """Change in a follow-up operator's reward after another operator."""

    first_operator: str
    second_operator: str
    context: dict[str, Any] = Field(default_factory=dict)
    occurrences: int = Field(ge=1)
    mean_followup_reward: float
    mean_sequence_reward: float
    baseline_followup_reward: float
    synergy: float
    acceptance_rate: float = Field(ge=0.0, le=1.0)

    @property
    def operator_a(self) -> str:
        return self.first_operator

    @property
    def operator_b(self) -> str:
        return self.second_operator

    @property
    def score(self) -> float:
        return self.synergy

    @property
    def sample_count(self) -> int:
        return self.occurrences

    @computed_field
    @property
    def reward_delta(self) -> float:
        return self.synergy

    @computed_field
    @property
    def synergy_score(self) -> float:
        return self.synergy

    @computed_field
    @property
    def relation(self) -> str:
        return f"{self.first_operator}->{self.second_operator}"


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nested(value: Any, dotted_name: str) -> Any:
    current = value
    for component in dotted_name.split("."):
        if isinstance(current, Mapping):
            if component not in current:
                return None
            current = current[component]
        else:
            if not hasattr(current, component):
                return None
            current = getattr(current, component)
    return current


def _group_value(trace: OperatorTrace, name: str) -> Any:
    if name.startswith("context."):
        return _nested(trace.context, name.removeprefix("context."))
    direct = _nested(trace, name)
    if direct is not None:
        return direct
    return _nested(trace.context, name)


def _hashable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


class OperatorDiagnoser:
    """Build operator profiles from traces or directly from a recorder."""

    def __init__(
        self,
        recorder: TrajectoryRecorder | None = None,
        *,
        minimum_context_samples: int = 1,
        representative_cases: int = 3,
        group_by: Sequence[str] | str | None = None,
    ) -> None:
        if minimum_context_samples < 1:
            raise ValueError("minimum_context_samples must be at least one")
        if representative_cases < 0:
            raise ValueError("representative_cases cannot be negative")
        self.recorder = recorder
        self.minimum_context_samples = int(minimum_context_samples)
        self.representative_cases = int(representative_cases)
        self.group_by = self._normalise_group_by(group_by)

    @staticmethod
    def _normalise_group_by(group_by: Sequence[str] | str | None) -> tuple[str, ...]:
        if group_by is None:
            return ()
        if isinstance(group_by, str):
            return (group_by,)
        return tuple(str(name) for name in group_by)

    def _traces(
        self, traces: Iterable[OperatorTrace | Mapping[str, Any]] | None
    ) -> list[OperatorTrace]:
        if traces is None:
            if self.recorder is None:
                raise ValueError("traces are required when no recorder was configured")
            return self.recorder.list_traces()
        return [
            trace
            if isinstance(trace, OperatorTrace)
            else OperatorTrace.model_validate(trace)
            for trace in traces
        ]

    def diagnose(
        self,
        traces: Iterable[OperatorTrace | Mapping[str, Any]] | None = None,
        *,
        group_by: Sequence[str] | str | None = None,
    ) -> list[OperatorProfile]:
        """Aggregate traces by operator and optional context fields."""

        items = self._traces(traces)
        fields = self.group_by if group_by is None else self._normalise_group_by(group_by)
        groups: dict[tuple[str, tuple[str, ...]], list[OperatorTrace]] = defaultdict(list)
        contexts: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        for trace in items:
            context = {name: _group_value(trace, name) for name in fields}
            key = (trace.operator_id, tuple(_hashable(context[name]) for name in fields))
            groups[key].append(trace)
            contexts[key] = context

        profiles = [
            self._profile(operator_id, contexts[key], group)
            for key, group in groups.items()
            for operator_id in [key[0]]
            # The threshold applies to context slices.  An overall operator
            # profile remains useful even when it has only a few observations.
            if not fields or len(group) >= self.minimum_context_samples
        ]
        # Sequence relations are also attached to their source profile for the
        # legacy report schema, while remaining available as typed standalone
        # results through ``analyze_synergies``.
        relations = compute_sequential_synergies(
            items,
            min_samples=self.minimum_context_samples,
            group_by=fields,
        )
        enriched: list[OperatorProfile] = []
        for profile in profiles:
            relevant = [
                relation.model_dump(mode="json")
                for relation in relations
                if relation.first_operator == profile.operator_id
                and relation.context == profile.context
            ]
            enriched.append(
                profile.model_copy(update={"synergy_relation_records": relevant})
            )
        return sorted(
            enriched,
            key=lambda profile: (profile.operator_id, _hashable(profile.context)),
        )

    diagnose_grouped = diagnose

    def _profile(
        self,
        operator_id: str,
        context: dict[str, Any],
        traces: list[OperatorTrace],
    ) -> OperatorProfile:
        immediate = [
            value
            for trace in traces
            if (value := _finite(trace.immediate_reward)) is not None
        ]
        successes = sum(value > 0.0 for value in immediate)
        delayed: dict[int, list[float]] = defaultdict(list)
        for trace in traces:
            for horizon, raw_value in trace.delayed_rewards.items():
                value = _finite(raw_value)
                if value is not None:
                    delayed[int(horizon)].append(value)

        feasibility_observed = [
            trace
            for trace in traces
            if trace.before_feasible is not None and trace.accepted_feasible is not None
        ]
        feasibility_gains = sum(
            trace.before_feasible is False and trace.accepted_feasible is True
            for trace in feasibility_observed
        )
        failures: Counter[str] = Counter()
        for trace in traces:
            if trace.error:
                failures[trace.error] += 1
            elif not trace.accepted:
                failures[trace.acceptance_reason or "rejected"] += 1
            elif trace.immediate_reward is not None and trace.immediate_reward <= 0:
                failures["non_improving"] += 1

        by_reward = sorted(
            traces,
            key=lambda trace: (
                _finite(trace.immediate_reward)
                if _finite(trace.immediate_reward) is not None
                else float("-inf")
            ),
            reverse=True,
        )
        success_ids = [
            int(trace.trace_id)
            for trace in by_reward
            if trace.trace_id is not None
            and trace.immediate_reward is not None
            and trace.immediate_reward > 0
        ][: self.representative_cases]
        failure_ids = [
            int(trace.trace_id)
            for trace in reversed(by_reward)
            if trace.trace_id is not None
            and (not trace.accepted or (trace.immediate_reward or 0.0) <= 0.0)
        ][: self.representative_cases]
        families = [trace.operator_family for trace in traces if trace.operator_family]
        family = Counter(families).most_common(1)[0][0] if families else None
        effective_contexts, failure_contexts = self._classify_contexts(traces, context)
        return OperatorProfile(
            operator_id=operator_id,
            operator_family=family,
            context=context,
            attempts=len(traces),
            acceptances=sum(trace.accepted for trace in traces),
            acceptance_rate=sum(trace.accepted for trace in traces) / len(traces),
            successes=successes,
            success_rate=successes / len(traces),
            mean_immediate_reward=_mean(immediate),
            median_immediate_reward=statistics.median(immediate) if immediate else None,
            std_immediate_reward=statistics.pstdev(immediate) if immediate else None,
            min_immediate_reward=min(immediate) if immediate else None,
            max_immediate_reward=max(immediate) if immediate else None,
            mean_delayed_rewards={
                horizon: statistics.fmean(values)
                for horizon, values in sorted(delayed.items())
            },
            delayed_sample_counts={
                horizon: len(values) for horizon, values in sorted(delayed.items())
            },
            delayed_positive_counts={
                horizon: sum(value > 0.0 for value in values)
                for horizon, values in sorted(delayed.items())
            },
            feasibility_gain_rate=(
                feasibility_gains / len(feasibility_observed)
                if feasibility_observed
                else None
            ),
            feasibility_rate=(
                sum(trace.accepted_feasible is True for trace in feasibility_observed)
                / len(feasibility_observed)
                if feasibility_observed
                else None
            ),
            mean_runtime_ms=statistics.fmean(trace.runtime_ms for trace in traces),
            failure_modes=dict(failures.most_common()),
            representative_success_ids=success_ids,
            representative_failure_ids=failure_ids,
            effective_context_groups=effective_contexts,
            failure_context_groups=failure_contexts,
        )

    def _classify_contexts(
        self,
        traces: Sequence[OperatorTrace],
        grouped_context: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Find simple context slices with positive or non-positive mean reward."""

        slices: dict[tuple[str, str], tuple[dict[str, Any], list[float]]] = {}
        for trace in traces:
            values: dict[str, Any] = dict(trace.context)
            if trace.map_difficulty is not None:
                values.setdefault("map_difficulty", trace.map_difficulty)
            for name, value in values.items():
                reward = _finite(trace.immediate_reward)
                if reward is None:
                    continue
                key = (str(name), _hashable(value))
                if key not in slices:
                    slices[key] = ({str(name): value}, [])
                slices[key][1].append(reward)

        # Explicitly grouped profiles should still expose their group even if
        # the source trace did not duplicate it in ``trace.context``.
        if grouped_context and not slices:
            rewards = [
                reward
                for trace in traces
                if (reward := _finite(trace.immediate_reward)) is not None
            ]
            if rewards:
                slices[("__group__", _hashable(grouped_context))] = (
                    dict(grouped_context),
                    rewards,
                )

        effective: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for context, rewards in slices.values():
            if len(rewards) < self.minimum_context_samples:
                continue
            target = effective if statistics.fmean(rewards) > 0.0 else failures
            target.append(context)
        sort_key = lambda value: _hashable(value)
        return sorted(effective, key=sort_key), sorted(failures, key=sort_key)

    def analyze_synergies(
        self,
        traces: Iterable[OperatorTrace | Mapping[str, Any]] | None = None,
        *,
        min_samples: int = 1,
        max_gap: int = 1,
        group_by: Sequence[str] | str | None = None,
        reward_horizon: int | None = None,
    ) -> list[SequentialSynergy]:
        return compute_sequential_synergies(
            self._traces(traces),
            min_samples=min_samples,
            max_gap=max_gap,
            group_by=self.group_by if group_by is None else group_by,
            reward_horizon=reward_horizon,
        )

    sequential_synergies = analyze_synergies


Diagnoser = OperatorDiagnoser


def _reward(trace: OperatorTrace, horizon: int | None) -> float | None:
    return _finite(
        trace.immediate_reward if horizon is None else trace.delayed_rewards.get(horizon)
    )


def compute_sequential_synergies(
    traces: Iterable[OperatorTrace | Mapping[str, Any]],
    *,
    min_samples: int = 1,
    max_gap: int = 1,
    group_by: Sequence[str] | str | None = None,
    reward_horizon: int | None = None,
) -> list[SequentialSynergy]:
    """Measure whether B performs unusually well when it follows A.

    ``synergy`` is the mean reward of B after A minus B's mean reward over all
    observed contexts in the same input.  Sequences never cross run, episode, or
    map boundaries.
    """

    if min_samples < 1 or max_gap < 1:
        raise ValueError("min_samples and max_gap must be positive")
    items = [
        trace
        if isinstance(trace, OperatorTrace)
        else OperatorTrace.model_validate(trace)
        for trace in traces
    ]
    fields = OperatorDiagnoser._normalise_group_by(group_by)
    baselines: dict[tuple[str, tuple[str, ...]], list[float]] = defaultdict(list)
    for trace in items:
        value = _reward(trace, reward_horizon)
        if value is not None:
            context_key = tuple(
                _hashable(_group_value(trace, name)) for name in fields
            )
            baselines[(trace.operator_id, context_key)].append(value)

    trajectories: dict[tuple[str, str | None, str], list[OperatorTrace]] = defaultdict(list)
    for trace in items:
        trajectories[(trace.run_id, trace.episode_id, trace.map_id)].append(trace)
    for trajectory in trajectories.values():
        trajectory.sort(key=lambda trace: (trace.iteration, trace.timestamp, trace.trace_id or 0))

    observations: dict[
        tuple[str, str, tuple[str, ...]], list[tuple[float, float, bool]]
    ] = defaultdict(list)
    contexts: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for trajectory in trajectories.values():
        for second_position, second in enumerate(trajectory):
            followup = _reward(second, reward_horizon)
            if followup is None:
                continue
            for gap in range(1, max_gap + 1):
                first_position = second_position - gap
                if first_position < 0:
                    break
                first = trajectory[first_position]
                first_reward = _reward(first, reward_horizon) or 0.0
                context = {name: _group_value(second, name) for name in fields}
                key = (
                    first.operator_id,
                    second.operator_id,
                    tuple(_hashable(context[name]) for name in fields),
                )
                observations[key].append((followup, first_reward + followup, second.accepted))
                contexts[key] = context

    results: list[SequentialSynergy] = []
    for key, values in observations.items():
        if len(values) < min_samples:
            continue
        first_operator, second_operator, _ = key
        baseline_values = baselines.get((second_operator, key[2]), [])
        if not baseline_values:
            continue
        followups = [value[0] for value in values]
        baseline = statistics.fmean(baseline_values)
        results.append(
            SequentialSynergy(
                first_operator=first_operator,
                second_operator=second_operator,
                context=contexts[key],
                occurrences=len(values),
                mean_followup_reward=statistics.fmean(followups),
                mean_sequence_reward=statistics.fmean(value[1] for value in values),
                baseline_followup_reward=baseline,
                synergy=statistics.fmean(followups) - baseline,
                acceptance_rate=sum(value[2] for value in values) / len(values),
            )
        )
    return sorted(
        results,
        key=lambda result: (-result.synergy, result.first_operator, result.second_operator),
    )
