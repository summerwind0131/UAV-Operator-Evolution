"""Versioned proposal envelopes and domain capability protocols."""

from .domain_kit import (
    DomainCompatibilityError,
    DomainKit,
    DomainSmokeReport,
    ensure_domain_compatibility,
    flattened_capabilities,
)
from .models import (
    CandidateProposalEnvelope,
    ProposalBudgetDeclaration,
    proposal_canonical_json,
    proposal_hash,
)

__all__ = [
    "CandidateProposalEnvelope",
    "DomainCompatibilityError",
    "DomainKit",
    "DomainSmokeReport",
    "ProposalBudgetDeclaration",
    "ensure_domain_compatibility",
    "flattened_capabilities",
    "proposal_canonical_json",
    "proposal_hash",
]
