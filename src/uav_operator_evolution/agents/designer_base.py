"""Common interface for data-driven operator designers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from operator_evolution_core.proposal import (
    CandidateProposalEnvelope,
    ProposalBudgetDeclaration,
)

from ..domain.adapters import UAV_DOMAIN_ID
from ..domain.uav_kit import UAV_IR_VERSION
from ..operators.specs import OperatorSpec
from .design_models import DesignHypothesis, DiagnosisReport


class OperatorProposal(BaseModel):
    """Validated design output containing data rather than executable code."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    specification: OperatorSpec = Field(
        validation_alias=AliasChoices("operator_spec", "specification"),
        serialization_alias="operator_spec",
    )
    design_rationale: str = Field(min_length=1)
    evidence_used: list[str] = Field(default_factory=list)
    target_failure_modes: list[str] = Field(default_factory=list)
    changes_from_parents: list[str] = Field(default_factory=list)
    expected_contexts: list[str] = Field(default_factory=list)
    potential_risks: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("expected_risks", "potential_risks"),
        serialization_alias="expected_risks",
    )
    evidence_level: str = "exploratory"
    diagnosis: DiagnosisReport | None = None
    hypothesis: DesignHypothesis | None = None
    expected_advantages: list[str] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)

    @property
    def spec(self) -> OperatorSpec:
        """Compatibility alias used by callers and documentation."""

        return self.specification

    @property
    def operator_spec(self) -> OperatorSpec:
        """Phase-8 alias while preserving the historical field name."""

        return self.specification

    @property
    def expected_risks(self) -> list[str]:
        """Phase-8 alias while preserving ``potential_risks`` callers."""

        return self.potential_risks

    @property
    def domain_id(self) -> str:
        """Implicit binding for legacy UAV proposal JSON."""

        return UAV_DOMAIN_ID

    @property
    def ir_version(self) -> str:
        """Implicit binding for legacy proposals without serialized metadata."""

        return UAV_IR_VERSION

    def to_envelope(
        self,
        candidate_id: str,
        budget_declaration: ProposalBudgetDeclaration
        | Mapping[str, int | float],
    ) -> CandidateProposalEnvelope[OperatorSpec]:
        budget = (
            budget_declaration
            if isinstance(budget_declaration, ProposalBudgetDeclaration)
            else ProposalBudgetDeclaration(limits=dict(budget_declaration))
        )
        return CandidateProposalEnvelope[OperatorSpec](
            candidate_id=candidate_id,
            domain_id=self.domain_id,
            ir_version=self.ir_version,
            parent_ids=list(self.spec.parent_operators),
            evidence_refs=list(self.used_evidence_ids),
            design_rationale=self.design_rationale,
            budget_declaration=budget,
            payload=self.spec,
        )


@runtime_checkable
class OperatorDesigner(Protocol):
    """Designer contract shared by heuristic and optional LLM adapters."""

    def propose(
        self,
        problem_description: str,
        parent_specs: list[OperatorSpec],
        parent_profiles: list[Any],
        memory_context: list[Any],
        success_cases: list[dict[str, Any]],
        failure_cases: list[dict[str, Any]],
    ) -> OperatorProposal:
        ...
