from __future__ import annotations

from operator_evolution_core.memory import (
    MechanismMemory as CoreMechanismMemory,
    MechanismRecord as CoreMechanismRecord,
)
from uav_operator_evolution.memory import (
    MechanismMemory as UAVMechanismMemory,
    MechanismRecord as UAVMechanismRecord,
)


def test_uav_memory_imports_are_identity_compatible_core_facades() -> None:
    assert UAVMechanismMemory is CoreMechanismMemory
    assert UAVMechanismRecord is CoreMechanismRecord


def test_core_memory_accepts_domain_neutral_operator_evidence() -> None:
    with CoreMechanismMemory(":memory:") as memory:
        mechanism_id = memory.add_mechanism(
            "jssp-adjacent-swap",
            {"selector": "critical_block", "transform": "swap"},
            name="JSSP adjacent swap",
            tags=["jssp", "operator"],
            metadata={"domain_id": "jssp", "ir_version": "jssp-v1"},
        )
        stored = memory.get_mechanism(mechanism_id)

    assert stored is not None
    assert stored.metadata["domain_id"] == "jssp"
