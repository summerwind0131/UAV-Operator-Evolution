"""Versioned, domain-neutral candidate proposal envelopes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PayloadT = TypeVar("PayloadT")


def proposal_canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def proposal_hash(value: Any) -> str:
    return hashlib.sha256(proposal_canonical_json(value).encode("utf-8")).hexdigest()


class ProposalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class ProposalBudgetDeclaration(ProposalModel):
    """Auditable hard limits declared before a candidate is evaluated."""

    limits: dict[str, int | float] = Field(min_length=1, max_length=32)

    @field_validator("limits")
    @classmethod
    def finite_non_negative_limits(
        cls, values: Mapping[str, int | float]
    ) -> dict[str, int | float]:
        output: dict[str, int | float] = {}
        for raw_name, value in values.items():
            name = str(raw_name)
            if not name or len(name) > 100:
                raise ValueError("budget limit names must contain 1-100 characters")
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError("budget limits must be finite non-negative numbers")
            output[name] = value
        return dict(sorted(output.items()))


class CandidateProposalEnvelope(ProposalModel, Generic[PayloadT]):
    """Content-addressed metadata around one domain-owned typed IR payload."""

    schema_version: Literal["proposal-envelope-v1"] = "proposal-envelope-v1"
    envelope_hash: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    candidate_id: str = Field(
        min_length=1,
        max_length=200,
    )
    domain_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9._-]*$",
    )
    ir_version: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9._-]*$",
    )
    parent_ids: list[str] = Field(min_length=1, max_length=16)
    evidence_refs: list[str] = Field(default_factory=list, max_length=256)
    design_rationale: str = Field(min_length=1, max_length=20_000)
    budget_declaration: ProposalBudgetDeclaration
    payload: PayloadT

    @model_validator(mode="after")
    def canonicalize_and_hash(self) -> "CandidateProposalEnvelope[PayloadT]":
        parents = list(dict.fromkeys(str(item) for item in self.parent_ids))
        if len(parents) != len(self.parent_ids) or any(not item for item in parents):
            raise ValueError("parent_ids must contain unique non-empty identifiers")
        evidence = sorted(set(str(item) for item in self.evidence_refs))
        if any(not item for item in evidence):
            raise ValueError("evidence_refs cannot contain empty identifiers")
        object.__setattr__(self, "parent_ids", parents)
        object.__setattr__(self, "evidence_refs", evidence)
        payload = self.model_dump(mode="json", exclude={"envelope_hash"})
        expected = proposal_hash(payload)
        if self.envelope_hash and self.envelope_hash != expected:
            raise ValueError("envelope_hash does not match proposal-envelope-v1 content")
        object.__setattr__(self, "envelope_hash", expected)
        return self


__all__ = [
    "CandidateProposalEnvelope",
    "ProposalBudgetDeclaration",
    "proposal_canonical_json",
    "proposal_hash",
]
