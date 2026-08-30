"""Typed rows returned by :mod:`operator_evolution_core.memory`."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryRecord(BaseModel):
    """A Pydantic record with lightweight mapping compatibility."""

    model_config = ConfigDict(extra="allow", populate_by_name=True, validate_assignment=True)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> Iterator[str]:
        return iter(self.model_dump().keys())

    def items(self) -> Any:
        return self.model_dump().items()


class MechanismRecord(MemoryRecord):
    mechanism_id: str = Field(validation_alias=AliasChoices("mechanism_id", "id"))
    name: str
    description: str = ""
    definition: Any = Field(default_factory=dict)
    status: str = "active"
    score: float = 0.0
    evidence_count: int = Field(default=0, ge=0)
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def id(self) -> str:
        return self.mechanism_id


class OperatorHistoryRecord(MemoryRecord):
    history_id: int | None = Field(default=None, ge=1)
    mechanism_id: str | None = None
    operator_id: str
    run_id: str | None = None
    trace_id: int | None = None
    accepted: bool | None = None
    immediate_reward: float | None = None
    delayed_rewards: dict[int, float | None] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def id(self) -> int | None:
        return self.history_id


class FailureModeRecord(MemoryRecord):
    failure_id: int | None = Field(default=None, ge=1)
    mechanism_id: str | None = None
    operator_id: str | None = None
    mode: str
    count: int = Field(default=1, ge=1)
    severity: float = Field(default=1.0, ge=0.0)
    context: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def id(self) -> int | None:
        return self.failure_id


class SynergyRecord(MemoryRecord):
    synergy_id: int | None = Field(default=None, ge=1)
    first_operator: str
    second_operator: str
    score: float
    sample_count: int = Field(default=1, ge=1)
    context: dict[str, Any] = Field(default_factory=dict)
    mechanism_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def id(self) -> int | None:
        return self.synergy_id


class CaseRecord(MemoryRecord):
    case_id: str = Field(validation_alias=AliasChoices("case_id", "id"))
    mechanism_id: str | None = None
    operator_id: str | None = None
    outcome: str = "unknown"
    score: float | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    state: Any = Field(default_factory=dict)
    action: Any = Field(default_factory=dict)
    result: Any = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def id(self) -> str:
        return self.case_id


class LineageRecord(MemoryRecord):
    lineage_id: int | None = Field(default=None, ge=1)
    parent_id: str
    child_id: str
    relation: str = "derived_from"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    depth: int = Field(default=1, ge=1)

    @property
    def id(self) -> int | None:
        return self.lineage_id


InsightType = Literal[
    "effective_mechanism",
    "failure_mode",
    "synergy",
    "tradeoff",
    "generalization_issue",
    "improvement_hypothesis",
]


class MechanismInsight(MemoryRecord):
    """Structured interpretation that remains linked to computed evidence."""

    insight_id: int | None = Field(default=None, ge=1)
    operator_id: str
    insight_type: InsightType
    evidence: Any = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    applicable_context: dict[str, Any] = Field(default_factory=dict)
    failure_context: dict[str, Any] = Field(default_factory=dict)
    source_profile_id: str | int | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def id(self) -> int | None:
        return self.insight_id


class OperatorProfileRecord(MemoryRecord):
    """Immutable snapshot of one computed operator profile."""

    profile_id: int | None = Field(default=None, ge=1)
    operator_id: str
    run_id: str | None = None
    generation: int = Field(default=0, ge=0)
    profile: dict[str, Any]
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def id(self) -> int | None:
        return self.profile_id
