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

__all__ = [
    "EvolutionArtifactSink",
    "EvolutionManagerDependencies",
    "EvolutionSplitCapabilities",
    "NullEvolutionArtifactSink",
    "PopulationFreezeReceipt",
    "PopulationSeed",
    "population_fingerprint",
]
