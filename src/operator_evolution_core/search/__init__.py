"""Public surface of the experimental domain-independent search core."""

from .kernel import GenericSearchKernel, StepCallback
from .models import (
    OperatorOutcome,
    SearchBudget,
    SearchContext,
    SearchResult,
    SearchStep,
)
from .policies import BlockRandomRoundRobinScheduler, SimulatedAnnealingAcceptance
from .protocols import AcceptancePolicy, OperatorScheduler, SearchOperator

__all__ = [
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
    "StepCallback",
]

