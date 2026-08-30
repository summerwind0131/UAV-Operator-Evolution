from __future__ import annotations

import pytest

from uav_operator_evolution.environment import MapGenerator
from uav_operator_evolution.transfer import (
    UAVMechanismBankConfig,
    build_uav_mechanism_bank,
)


def test_uav_bank_uses_explicit_disjoint_splits_and_emits_no_domain_ir() -> None:
    generator = MapGenerator(20260902, grid_resolution=4.0, generation_attempts=20)
    train = [generator.generate_map("train", 0, "sparse")]
    validation = [generator.generate_map("validation", 0, "medium")]
    bank = build_uav_mechanism_bank(
        train,
        validation,
        bank_master_seeds=(2026090201,),
        source_code_commit="4bb54dc067a592a766e649b51751e91cd8c6d888",
        config=UAVMechanismBankConfig(
            train_calls=8,
            validation_calls=8,
            train_instances=1,
            validation_instances=1,
        ),
    )

    assert bank.source_domain_id == "uav-path-planning-2d"
    assert len(bank.records) == 8
    serialized = bank.model_dump_json()
    for forbidden in ("selection_strategy", "transformations", "primitive", "uav-v1"):
        assert forbidden not in serialized


def test_uav_bank_rejects_semantically_overlapping_capabilities() -> None:
    environment = MapGenerator(7).generate_map("train", 0, "sparse")
    with pytest.raises(ValueError, match="overlap"):
        build_uav_mechanism_bank(
            [environment],
            [environment],
            bank_master_seeds=(1,),
            source_code_commit="4bb54dc",
            config=UAVMechanismBankConfig(
                train_calls=8,
                validation_calls=8,
                train_instances=1,
                validation_instances=1,
            ),
        )
