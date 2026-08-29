"""Compatibility imports for versioned core fitness policies."""

from operator_evolution_core.validation.fitness import (
    DETERMINISTIC_FITNESS_WEIGHTS,
    FITNESS_WEIGHTS,
    FitnessPolicy,
    compute_fitness,
)

__all__ = [
    "DETERMINISTIC_FITNESS_WEIGHTS",
    "FITNESS_WEIGHTS",
    "FitnessPolicy",
    "compute_fitness",
]

