from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from operator_evolution_core.contracts import InstanceRef, ObjectiveEvaluation
from uav_operator_evolution.reproducibility import stable_hash


ROOT = Path(__file__).resolve().parents[1]


def test_instance_ref_is_strict_content_addressed_and_json_stable() -> None:
    reference = InstanceRef(
        domain_id="example-domain",
        instance_id="train-001",
        split="train",
        difficulty="small",
        content_hash="a" * 64,
        metadata={"seed": 7, "shape": [3, 4]},
    )

    restored = InstanceRef.model_validate_json(reference.model_dump_json())

    assert restored == reference
    assert stable_hash(restored.model_dump(mode="json")) == stable_hash(
        reference.model_dump(mode="json")
    )
    with pytest.raises(ValidationError):
        InstanceRef.model_validate(
            {**reference.model_dump(), "content_hash": "not-a-sha256"}
        )
    with pytest.raises(ValidationError):
        InstanceRef.model_validate({**reference.model_dump(), "unexpected": True})


def test_core_contracts_reject_nonfinite_or_negative_constraint_values() -> None:
    with pytest.raises(ValidationError, match="finite"):
        ObjectiveEvaluation(
            scalar_cost=math.inf,
            components={"cost": 1.0},
            feasible=True,
        )
    with pytest.raises(ValidationError, match="non-negative"):
        ObjectiveEvaluation(
            scalar_cost=1.0,
            components={"cost": 1.0},
            feasible=False,
            violations={"capacity": -1.0},
        )
    with pytest.raises(ValidationError, match="finite"):
        InstanceRef(
            domain_id="example-domain",
            instance_id="bad-metadata",
            split="validation",
            content_hash="b" * 64,
            metadata={"nested": {"value": math.nan}},
        )


def test_experimental_core_has_no_domain_implementation_imports() -> None:
    core_root = ROOT / "src" / "operator_evolution_core"
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(core_root.rglob("*.py"))
    )
    assert "uav_operator_evolution" not in sources
    assert "jssp_operator_evolution" not in sources
