"""Models used to record an operator decision without losing search state."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any, Mapping

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


def _number(value: Any) -> float | None:
    """Return a finite float when *value* represents one."""

    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _state_value(state: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in state:
            return state[name]
    metrics = state.get("metrics")
    if isinstance(metrics, Mapping):
        for name in names:
            if name in metrics:
                return metrics[name]
    return None


class OperatorTrace(BaseModel):
    """One attempted application of a search operator.

    ``before_state``, ``candidate_state`` and ``accepted_state`` deliberately use
    JSON-native dictionaries.  A planner can therefore retain its full path,
    constraint and feature representation without this persistence layer knowing
    planner-specific types.  Frequently queried values are duplicated in typed
    fields and inferred from the snapshots when omitted.
    """

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        populate_by_name=True,
    )

    trace_id: int | None = Field(default=None, ge=1)
    run_id: str = "default"
    episode_id: str | None = None
    map_id: str = ""
    map_difficulty: str | None = None
    iteration: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    seed: int | None = Field(default=None, ge=0)

    operator_id: str = Field(
        default="unknown",
        validation_alias=AliasChoices("operator_id", "operator", "operator_name"),
    )
    operator_family: str | None = None
    operator_version: str | None = None
    operator_params: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "operator_params", "operator_parameters", "parameters", "params"
        ),
    )
    context: dict[str, Any] = Field(default_factory=dict)

    before_state: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("before_state", "state_before", "before"),
    )
    candidate_state: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("candidate_state", "state_candidate", "candidate"),
    )
    accepted_state: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("accepted_state", "state_accepted", "after_state", "after"),
    )

    before_objective: float | None = Field(
        default=None,
        validation_alias=AliasChoices("before_objective", "objective_before", "cost_before"),
    )
    candidate_objective: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "candidate_objective", "objective_candidate", "cost_candidate"
        ),
    )
    accepted_objective: float | None = Field(
        default=None,
        validation_alias=AliasChoices("accepted_objective", "objective_after", "cost_after"),
    )
    before_components: dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("before_components", "components_before"),
    )
    candidate_components: dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("candidate_components", "components_candidate"),
    )
    accepted_components: dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("accepted_components", "components_after"),
    )
    before_feasible: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("before_feasible", "feasible_before"),
    )
    candidate_feasible: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("candidate_feasible", "feasible_candidate"),
    )
    accepted_feasible: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("accepted_feasible", "feasible_after"),
    )

    accepted: bool = Field(
        default=False,
        validation_alias=AliasChoices("accepted", "is_accepted", "was_accepted"),
    )
    acceptance_reason: str | None = None
    acceptance_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    temperature: float | None = Field(default=None, ge=0.0)
    immediate_reward: float | None = Field(
        default=None,
        validation_alias=AliasChoices("immediate_reward", "reward"),
    )
    delayed_rewards: dict[int, float | None] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("delayed_rewards", "delayed_reward"),
    )
    runtime_ms: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("runtime_ms", "execution_time_ms", "elapsed_ms"),
    )
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def complete_snapshots_and_summaries(self) -> "OperatorTrace":
        # The accepted snapshot means the state the search actually carries
        # forward, not merely the proposed candidate.
        if not self.accepted_state:
            source = self.candidate_state if self.accepted else self.before_state
            object.__setattr__(self, "accepted_state", dict(source))

        snapshots = (
            ("before", self.before_state),
            ("candidate", self.candidate_state),
            ("accepted", self.accepted_state),
        )
        for prefix, snapshot in snapshots:
            objective_name = f"{prefix}_objective"
            if getattr(self, objective_name) is None:
                value = _state_value(snapshot, "objective", "cost", "score", "total_cost")
                object.__setattr__(self, objective_name, _number(value))

            components_name = f"{prefix}_components"
            if not getattr(self, components_name):
                value = _state_value(snapshot, "objective_components", "components")
                if isinstance(value, Mapping):
                    cleaned = {
                        str(key): number
                        for key, raw in value.items()
                        if (number := _number(raw)) is not None
                    }
                    object.__setattr__(self, components_name, cleaned)

            feasible_name = f"{prefix}_feasible"
            if getattr(self, feasible_name) is None:
                value = _state_value(snapshot, "feasible", "is_feasible")
                if value is not None:
                    object.__setattr__(self, feasible_name, bool(value))

        if self.immediate_reward is None:
            before = self.before_objective
            after = self.accepted_objective
            if before is not None and after is not None:
                # Objectives are costs: positive reward means improvement.
                object.__setattr__(self, "immediate_reward", before - after)
        return self

    @property
    def state_before(self) -> dict[str, Any]:
        """Compatibility spelling used by a few planner integrations."""

        return self.before_state

    @property
    def state_candidate(self) -> dict[str, Any]:
        return self.candidate_state

    @property
    def state_accepted(self) -> dict[str, Any]:
        return self.accepted_state

    @property
    def operator_name(self) -> str:
        return self.operator_id

    @property
    def objective_before(self) -> float | None:
        return self.before_objective

    @property
    def objective_candidate(self) -> float | None:
        return self.candidate_objective

    @property
    def objective_after(self) -> float | None:
        return self.accepted_objective

    @property
    def reward(self) -> float | None:
        return self.immediate_reward

    def delayed_reward(self, horizon: int) -> float | None:
        """Return delayed reward at *horizon*, or ``None`` when censored."""

        return self.delayed_rewards.get(int(horizon))
