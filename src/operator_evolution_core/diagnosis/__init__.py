"""Domain-independent operator profiling and sequence diagnostics."""

from .diagnoser import (
    Diagnoser,
    OperatorDiagnoser,
    OperatorProfile,
    SequentialSynergy,
    compute_sequential_synergies,
)
from .features import FeatureCatalog

__all__ = [
    "Diagnoser",
    "FeatureCatalog",
    "OperatorDiagnoser",
    "OperatorProfile",
    "SequentialSynergy",
    "compute_sequential_synergies",
]

