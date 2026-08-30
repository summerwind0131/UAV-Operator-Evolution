"""Generic evolution lifecycle capabilities and dependency injection."""

from .contracts import (
    EvolutionArtifactSink,
    EvolutionManagerDependencies,
    EvolutionSplitCapabilities,
    NullEvolutionArtifactSink,
    PopulationFreezeReceipt,
    PopulationSeed,
    population_fingerprint,
)
from .transfer import (
    MechanismTransferPreregistrationV1,
    TransferBudgetV1,
    TransferStatisticsV1,
)
from .transfer_lifecycle import (
    TransferArmV1,
    TransferArmLifecycleV1,
    TransferCandidateLifecycleV1,
    TransferEvidenceSelectionV1,
    select_transfer_evidence_v1,
    transfer_candidate_context_v1,
)

__all__ = [
    "EvolutionArtifactSink",
    "EvolutionManagerDependencies",
    "EvolutionSplitCapabilities",
    "NullEvolutionArtifactSink",
    "PopulationFreezeReceipt",
    "PopulationSeed",
    "population_fingerprint",
    "MechanismTransferPreregistrationV1",
    "TransferBudgetV1",
    "TransferStatisticsV1",
    "TransferArmV1",
    "TransferArmLifecycleV1",
    "TransferCandidateLifecycleV1",
    "TransferEvidenceSelectionV1",
    "select_transfer_evidence_v1",
    "transfer_candidate_context_v1",
]
