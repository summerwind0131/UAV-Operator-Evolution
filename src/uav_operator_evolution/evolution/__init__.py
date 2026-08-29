"""Candidate validation, fitness ranking, and population evolution."""

from operator_evolution_core.evolution import (
    EvolutionManagerDependencies,
    EvolutionSplitCapabilities,
    PopulationFreezeReceipt,
    PopulationSeed,
)

from .candidate_validator import FixedBudgetCandidateValidator

from .fitness import compute_fitness
from .validation import PairedOutcome, ValidationReport, decide_retention

__all__ = [
    "EvolutionManagerDependencies",
    "EvolutionSplitCapabilities",
    "FixedBudgetCandidateValidator",
    "PairedOutcome",
    "PopulationFreezeReceipt",
    "PopulationSeed",
    "ValidationReport",
    "compute_fitness",
    "decide_retention",
]
