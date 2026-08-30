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
]
