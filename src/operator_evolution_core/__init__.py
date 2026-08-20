"""Experimental domain-independent contracts for operator evolution.

The package remains internal to the UAV repository until a second domain has
validated the interfaces.  Its modules must never import UAV implementation
types.
"""

from .contracts import (
    DatasetSplit,
    DomainAdapter,
    Evaluator,
    FeatureExtractor,
    Initializer,
    InstanceRef,
    ObjectiveEvaluation,
    SearchContextView,
    SolutionCodec,
    SolutionGuard,
    TraceEncoder,
)

__all__ = [
    "DatasetSplit",
    "DomainAdapter",
    "Evaluator",
    "FeatureExtractor",
    "Initializer",
    "InstanceRef",
    "ObjectiveEvaluation",
    "SearchContextView",
    "SolutionCodec",
    "SolutionGuard",
    "TraceEncoder",
]
