"""Domain-neutral mechanism records for controlled cross-domain transfer."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


MechanismTag = Literal["repair", "diversify", "intensify", "rollback"]
OrdinalLevel = Literal["low", "medium", "high", "unknown"]


class AbstractMechanismContextV1(BaseModel):
    """Coarse context that cannot carry raw domain feature values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_pressure: OrdinalLevel = "unknown"
    stagnation: OrdinalLevel = "unknown"
    diversity: OrdinalLevel = "unknown"
    feasibility: Literal["feasible", "mixed", "infeasible", "unknown"] = "unknown"
    phase: Literal["early", "middle", "late", "unknown"] = "unknown"


class ExpectedMechanismEffectV1(BaseModel):
    """Direction-only expectation, deliberately independent of domain metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feasibility: Literal["improve", "preserve", "risk", "unknown"] = "unknown"
    cost: Literal["improve", "preserve", "risk", "unknown"] = "unknown"
    diversity: Literal["increase", "preserve", "decrease", "unknown"] = "unknown"
    locality: Literal["local", "global", "mixed", "unknown"] = "unknown"


class MechanismRecordV1(BaseModel):
    """Content-addressed semantic mechanism evidence without domain IR or code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["mechanism-record-v1"] = "mechanism-record-v1"
    mechanism_id: str = Field(pattern=r"^mechanism-v1-[0-9a-f]{16}$")
    source_domain_id: str = Field(min_length=1, max_length=128)
    mechanism_tags: tuple[MechanismTag, ...] = Field(min_length=1, max_length=4)
    trigger_context: AbstractMechanismContextV1
    expected_effect: ExpectedMechanismEffectV1
    failure_modes: tuple[str, ...] = Field(default=(), max_length=8)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=16)
    evidence_strength: float = Field(ge=0.0, le=1.0)
    evidence_sample_count: int = Field(ge=1)
    evidence_splits: tuple[Literal["train", "validation"], ...] = Field(
        min_length=1,
        max_length=2,
    )
    bank_run_id: str = Field(min_length=1, max_length=128)
    bank_master_seed: int = Field(ge=0, le=2**63 - 1)
    source_operator_id: str = Field(min_length=1, max_length=128)
    source_code_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    source_population_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if len(set(self.mechanism_tags)) != len(self.mechanism_tags):
            raise ValueError("mechanism_tags must be unique")
        if len(set(self.failure_modes)) != len(self.failure_modes):
            raise ValueError("failure_modes must be unique")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("evidence_refs must be unique")
        if len(set(self.evidence_splits)) != len(self.evidence_splits):
            raise ValueError("evidence_splits must be unique")
        expected_hash = mechanism_record_provenance_hash(self)
        if self.provenance_hash != expected_hash:
            raise ValueError("provenance_hash does not match the mechanism payload")
        expected_id = f"mechanism-v1-{expected_hash[:16]}"
        if self.mechanism_id != expected_id:
            raise ValueError("mechanism_id does not match provenance_hash")
        return self


def _identity_payload(record: MechanismRecordV1 | dict[str, object]) -> dict[str, object]:
    if isinstance(record, MechanismRecordV1):
        payload = record.model_dump(mode="json")
    else:
        payload = dict(record)
    payload.pop("mechanism_id", None)
    payload.pop("provenance_hash", None)
    return payload


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def mechanism_record_provenance_hash(
    record: MechanismRecordV1 | dict[str, object],
) -> str:
    canonical = json.dumps(
        _jsonable(_identity_payload(record)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def create_mechanism_record_v1(**payload: object) -> MechanismRecordV1:
    """Create a record while deriving both immutable identity fields."""

    raw = dict(payload)
    raw.setdefault("schema_version", "mechanism-record-v1")
    provenance_hash = mechanism_record_provenance_hash(raw)
    raw["provenance_hash"] = provenance_hash
    raw["mechanism_id"] = f"mechanism-v1-{provenance_hash[:16]}"
    return MechanismRecordV1.model_validate(raw)


def abstract_context_similarity(
    query: AbstractMechanismContextV1,
    candidate: AbstractMechanismContextV1,
) -> float:
    """Exact similarity over known ordinal fields; unknown values add no evidence."""

    query_values = query.model_dump()
    candidate_values = candidate.model_dump()
    comparable = [key for key, value in query_values.items() if value != "unknown"]
    if not comparable:
        return 0.0
    matches = sum(
        1
        for key in comparable
        if candidate_values[key] != "unknown" and candidate_values[key] == query_values[key]
    )
    return matches / len(comparable)


def retrieve_top4_mechanisms_v1(
    records: list[MechanismRecordV1] | tuple[MechanismRecordV1, ...],
    context: AbstractMechanismContextV1,
) -> tuple[MechanismRecordV1, ...]:
    """Fixed top-4 retrieval: context, evidence strength, then mechanism ID."""

    ranked = sorted(
        records,
        key=lambda record: (
            -abstract_context_similarity(context, record.trigger_context),
            -record.evidence_strength,
            record.mechanism_id,
        ),
    )
    return tuple(ranked[:4])
