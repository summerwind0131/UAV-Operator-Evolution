"""Operator profiling, sequence analysis, and small counterfactual probes."""

from .counterfactual import CounterfactualEvaluator, CounterfactualResult
from .diagnoser import (
    Diagnoser,
    OperatorDiagnoser,
    OperatorProfile,
    SequentialSynergy,
    compute_sequential_synergies,
)

__all__ = [
    "CounterfactualEvaluator",
    "CounterfactualResult",
    "Diagnoser",
    "OperatorDiagnoser",
    "OperatorProfile",
    "SequentialSynergy",
    "compute_sequential_synergies",
]
