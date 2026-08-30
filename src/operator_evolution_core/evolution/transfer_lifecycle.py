"""Auditable evidence selection for the registered three-arm transfer study."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..memory import (
    AbstractMechanismContextV1,
    MechanismBankV1,
    retrieve_top4_mechanisms_v1,
)


TransferArmV1 = Literal["scratch", "same-domain", "cross-domain"]


class TransferEvidenceSelectionV1(BaseModel):
    """Immutable receipt for one candidate's abstract evidence retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["transfer-evidence-selection-v1"] = (
        "transfer-evidence-selection-v1"
    )
    arm: TransferArmV1
    target_domain_id: str = Field(min_length=1, max_length=128)
    source_domain_id: str | None = None
    source_bank_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    query_context: AbstractMechanismContextV1
    mechanism_ids: tuple[str, ...] = Field(default=(), max_length=4)


class TransferCandidateLifecycleV1(BaseModel):
    """Domain-neutral validation result for one redesigned candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["transfer-candidate-lifecycle-v1"] = (
        "transfer-candidate-lifecycle-v1"
    )
    generation: int = Field(ge=0)
    candidate_index: int = Field(ge=0)
    parent_operator_id: str = Field(min_length=1)
    candidate_operator_id: str = Field(min_length=1)
    evidence: TransferEvidenceSelectionV1
    smoke_passed: bool
    validation_outcomes: int = Field(ge=0)
    mean_gain: float
    parent_feasibility_rate: float = Field(ge=0.0, le=1.0)
    candidate_feasibility_rate: float = Field(ge=0.0, le=1.0)
    candidate_effective_call_rate: float = Field(ge=0.0, le=1.0)
    candidate_acceptance_rate: float = Field(ge=0.0, le=1.0)
    retained: bool
    retention_reasons: tuple[str, ...]


class TransferArmLifecycleV1(BaseModel):
    """Sealed validation-only outcome for one target-domain design arm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["transfer-arm-lifecycle-v1"] = (
        "transfer-arm-lifecycle-v1"
    )
    target_domain_id: str = Field(min_length=1)
    arm: TransferArmV1
    master_seed: int = Field(ge=0)
    search_calls: int = Field(gt=0)
    generations: int = Field(gt=0)
    candidates_per_generation: int = Field(gt=0)
    validation_instances: int = Field(gt=0)
    runtime_repetitions: int = Field(gt=0)
    initial_population_ids: tuple[str, ...] = Field(min_length=1)
    final_population_ids: tuple[str, ...] = Field(min_length=1)
    candidates: tuple[TransferCandidateLifecycleV1, ...]
    test_instances_opened: Literal[False] = False
    remote_provider_calls: Literal[0] = 0


def select_transfer_evidence_v1(
    *,
    arm: TransferArmV1,
    target_domain_id: str,
    same_domain_bank: MechanismBankV1,
    cross_domain_bank: MechanismBankV1,
    context: AbstractMechanismContextV1,
) -> TransferEvidenceSelectionV1:
    """Resolve scratch/same/cross evidence without exposing domain IR.

    The function deliberately fixes the retrieval limit at four and validates
    the direction before reading any records.  Callers then resolve the IDs
    against the already-validated frozen bank.
    """

    if same_domain_bank.source_domain_id != target_domain_id:
        raise ValueError("same-domain bank does not belong to the target domain")
    if cross_domain_bank.source_domain_id == target_domain_id:
        raise ValueError("cross-domain bank must come from another domain")

    if arm == "scratch":
        return TransferEvidenceSelectionV1(
            arm=arm,
            target_domain_id=target_domain_id,
            query_context=context,
        )

    bank = same_domain_bank if arm == "same-domain" else cross_domain_bank
    records = retrieve_top4_mechanisms_v1(bank.records, context)
    return TransferEvidenceSelectionV1(
        arm=arm,
        target_domain_id=target_domain_id,
        source_domain_id=bank.source_domain_id,
        source_bank_hash=bank.bank_hash,
        query_context=context,
        mechanism_ids=tuple(record.mechanism_id for record in records),
    )


def transfer_candidate_context_v1(
    generation: int,
    candidate_index: int,
    *,
    generations: int,
) -> AbstractMechanismContextV1:
    """Create a deterministic, domain-neutral query from lifecycle position."""

    if generation < 0 or candidate_index < 0:
        raise ValueError("generation and candidate_index must be non-negative")
    if generations <= 0 or generation >= generations:
        raise ValueError("generation must fall within the registered lifecycle")
    if generation == 0:
        phase = "early"
    elif generation == generations - 1:
        phase = "late"
    else:
        phase = "middle"
    stagnation = ("low", "medium", "high")[candidate_index % 3]
    return AbstractMechanismContextV1(
        constraint_pressure="medium",
        stagnation=stagnation,
        diversity="medium",
        feasibility="mixed",
        phase=phase,
    )


__all__ = [
    "TransferArmV1",
    "TransferArmLifecycleV1",
    "TransferCandidateLifecycleV1",
    "TransferEvidenceSelectionV1",
    "select_transfer_evidence_v1",
    "transfer_candidate_context_v1",
]
