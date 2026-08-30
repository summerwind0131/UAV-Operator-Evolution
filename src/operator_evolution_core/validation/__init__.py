"""Generic paired validation, statistics, and versioned fitness policies."""

from .fitness import (
    DETERMINISTIC_FITNESS_WEIGHTS,
    FITNESS_WEIGHTS,
    FitnessPolicy,
    compute_fitness,
)
from .paired import (
    PairedOutcome,
    RetentionConfig,
    ValidationReport,
    decide_retention,
    paired_bootstrap_ci,
)
from .schedule import (
    CRNSeed,
    SlotReplacement,
    abba_timing_order,
    build_crn_seed_schedule,
    replace_population_slot,
)
from .validator import ArmMeasurement, GenericPairedCandidateValidator

__all__ = [
    "ArmMeasurement",
    "CRNSeed",
    "DETERMINISTIC_FITNESS_WEIGHTS",
    "FITNESS_WEIGHTS",
    "FitnessPolicy",
    "GenericPairedCandidateValidator",
    "PairedOutcome",
    "RetentionConfig",
    "SlotReplacement",
    "ValidationReport",
    "abba_timing_order",
    "build_crn_seed_schedule",
    "compute_fitness",
    "decide_retention",
    "paired_bootstrap_ci",
    "replace_population_slot",
]
