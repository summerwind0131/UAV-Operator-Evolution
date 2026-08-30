"""Minimal contracts shared by future domain adapters."""

from .domain import (
    DomainAdapter,
    Evaluator,
    FeatureExtractor,
    Initializer,
    SearchContextView,
    SolutionCodec,
    SolutionGuard,
    TraceEncoder,
)
from .models import DatasetSplit, InstanceRef, ObjectiveEvaluation

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
