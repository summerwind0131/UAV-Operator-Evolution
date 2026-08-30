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
from .transfer import (
    AbstractMechanismContextV1,
    ExpectedMechanismEffectV1,
    MechanismBankV1,
    MechanismRecordV1,
    abstract_context_similarity,
    create_mechanism_record_v1,
    create_mechanism_bank_v1,
    mechanism_bank_hash,
    mechanism_record_provenance_hash,
    retrieve_top4_mechanisms_v1,
)

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
    "MechanismRecordV1",
    "MechanismBankV1",
    "AbstractMechanismContextV1",
    "ExpectedMechanismEffectV1",
    "abstract_context_similarity",
    "create_mechanism_record_v1",
    "create_mechanism_bank_v1",
    "mechanism_bank_hash",
    "mechanism_record_provenance_hash",
    "retrieve_top4_mechanisms_v1",
    "MemoryCase",
    "OperatorHistory",
    "OperatorHistoryRecord",
    "OperatorProfileRecord",
    "Synergy",
    "SynergyRecord",
]
