"""Strict data models for evidence-grounded operator design decisions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DesignModel(BaseModel):
    """Base model shared by all non-executable LLM/agent outputs."""

    model_config = ConfigDict(extra="forbid", strict=True)


class DiagnosisClaim(DesignModel):
    claim: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=32)
    confidence: float = Field(ge=0.0, le=1.0)
    alternative_explanation: str = Field(min_length=1, max_length=2000)


class DiagnosisReport(DesignModel):
    parent_operator: str = Field(min_length=1, max_length=200)
    effective_mechanisms: list[DiagnosisClaim] = Field(default_factory=list, max_length=16)
    failure_modes: list[DiagnosisClaim] = Field(default_factory=list, max_length=16)
    useful_synergies: list[DiagnosisClaim] = Field(default_factory=list, max_length=16)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=16)


class DesignHypothesis(DesignModel):
    hypothesis: str = Field(min_length=1, max_length=2000)
    target_failure_mode: str = Field(min_length=1, max_length=1000)
    expected_mechanism: str = Field(min_length=1, max_length=2000)
    expected_effective_context: str = Field(min_length=1, max_length=2000)
    possible_side_effects: list[str] = Field(default_factory=list, max_length=16)
    evidence_ids: list[str] = Field(min_length=1, max_length=32)


class OperatorReview(DesignModel):
    decision: Literal["approve", "revise", "reject"]
    evidence_alignment_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    safety_score: float = Field(ge=0.0, le=1.0)
    testability_score: float = Field(ge=0.0, le=1.0)
    concerns: list[str] = Field(default_factory=list, max_length=32)
    required_revisions: list[str] = Field(default_factory=list, max_length=32)
    lineage_relation: Literal["structural_variant", "parameter_variant"] | None = None
    topology_fingerprint: str | None = None


class CandidateStatus(str, Enum):
    PROPOSED = "PROPOSED"
    SCHEMA_VALID = "SCHEMA_VALID"
    REVIEWED = "REVIEWED"
    COMPILED = "COMPILED"
    SMOKE_PASSED = "SMOKE_PASSED"
    VALIDATED = "VALIDATED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class OperatorDesignResult(DesignModel):
    """Serializable orchestration result without executable operator objects."""

    candidate_id: str = Field(min_length=1, max_length=200)
    status: CandidateStatus
    proposal: dict[str, Any] | None = None
    review: OperatorReview | None = None
    rejection_reason: str | None = None
    supersedes_candidate_id: str | None = None


__all__ = [
    "CandidateStatus",
    "DesignHypothesis",
    "DiagnosisClaim",
    "DiagnosisReport",
    "OperatorDesignResult",
    "OperatorReview",
]
