"""Operator profiling, sequence analysis, and small counterfactual probes."""

from .counterfactual import CounterfactualEvaluator, CounterfactualResult
from .diagnoser import (
    Diagnoser,
    OperatorDiagnoser,
    OperatorProfile,
    SequentialSynergy,
    compute_sequential_synergies,
)
from .features import UAV_FEATURE_CATALOG

__all__ = [
    "CounterfactualEvaluator",
    "CounterfactualResult",
    "Diagnoser",
    "OperatorDiagnoser",
    "OperatorProfile",
    "SequentialSynergy",
    "compute_sequential_synergies",
    "UAV_FEATURE_CATALOG",
]
