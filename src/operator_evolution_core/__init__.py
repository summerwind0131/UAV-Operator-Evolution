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
from .search import (
    AcceptancePolicy,
    BlockRandomRoundRobinScheduler,
    GenericSearchKernel,
    OperatorOutcome,
    OperatorScheduler,
    SearchBudget,
    SearchContext,
    SearchOperator,
    SearchResult,
    SearchStep,
    SimulatedAnnealingAcceptance,
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
    "AcceptancePolicy",
    "BlockRandomRoundRobinScheduler",
    "GenericSearchKernel",
    "OperatorOutcome",
    "OperatorScheduler",
    "SearchBudget",
    "SearchContext",
    "SearchOperator",
    "SearchResult",
    "SearchStep",
    "SimulatedAnnealingAcceptance",
]
