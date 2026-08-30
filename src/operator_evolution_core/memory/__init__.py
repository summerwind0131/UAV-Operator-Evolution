"""Domain-independent persistent mechanism and evidence memory."""

from .models import (
    CaseRecord,
    FailureModeRecord,
    LineageRecord,
    MechanismInsight,
    MechanismRecord,
    OperatorHistoryRecord,
    OperatorProfileRecord,
    SynergyRecord,
)
from .store import MechanismMemory

Mechanism = MechanismRecord
OperatorHistory = OperatorHistoryRecord
FailureMode = FailureModeRecord
Synergy = SynergyRecord
MemoryCase = CaseRecord

__all__ = [
    "CaseRecord",
    "FailureMode",
    "FailureModeRecord",
    "LineageRecord",
    "Mechanism",
    "MechanismInsight",
    "MechanismMemory",
    "MechanismRecord",
    "MemoryCase",
    "OperatorHistory",
    "OperatorHistoryRecord",
    "OperatorProfileRecord",
    "Synergy",
    "SynergyRecord",
]
