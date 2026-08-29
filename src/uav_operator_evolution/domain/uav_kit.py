"""UAV-owned ``uav-v1`` IR kit for generic proposal infrastructure."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import JsonValue

from operator_evolution_core.proposal import DomainSmokeReport, proposal_hash

from ..environment import Environment2D
from ..operators.catalog import manual_operator_specs
from ..operators.compiler import CompiledOperator, OperatorCompiler
from ..operators.specs import OperatorSpec, primitive_catalog
from ..path import PathEvaluator
from ..path.models import ObjectiveWeights, Path
from ..search.context import SearchContext
from .adapters import UAV_DOMAIN_ID

UAV_IR_VERSION = "uav-v1"


@dataclass(slots=True, frozen=True)
class UAVSmokeFixture:
    environment: Environment2D
    path: Path
    seeds: tuple[int, ...] = (0, 1, 2)


def _condition_topology(condition: Any) -> dict[str, Any] | None:
    if condition is None:
        return None
    return {"feature": condition.feature, "operator": condition.operator}


class UAVDomainKit:
    """Trusted parser/compiler/smoke boundary for the bounded UAV DSL."""

    domain_id = UAV_DOMAIN_ID
    ir_version = UAV_IR_VERSION
    accepts_legacy_unversioned = True

    def __init__(self, compiler: OperatorCompiler | None = None) -> None:
        self.compiler = compiler or OperatorCompiler()

    def parse_ir(self, payload: Any) -> OperatorSpec:
        return (
            payload
            if isinstance(payload, OperatorSpec)
            else OperatorSpec.model_validate(payload)
        )

    def serialize_ir(self, ir: OperatorSpec) -> JsonValue:
        return self.parse_ir(ir).model_dump(mode="json")

    def capability_catalog(self) -> dict[str, tuple[str, ...]]:
        return {key: tuple(values) for key, values in primitive_catalog().items()}

    def compile(self, ir: OperatorSpec) -> CompiledOperator:
        return self.compiler.compile(self.parse_ir(ir))

    def smoke(
        self,
        ir: OperatorSpec,
        fixture: UAVSmokeFixture,
    ) -> DomainSmokeReport:
        if not isinstance(fixture, UAVSmokeFixture):
            raise TypeError("uav-v1 smoke requires UAVSmokeFixture")
        compiled = self.compile(ir)
        original = list(fixture.path)
        evaluator = PathEvaluator(ObjectiveWeights())
        evaluation = evaluator.evaluate(original, fixture.environment)
        context = SearchContext(
            iteration=0,
            max_iterations=1,
            current_evaluation=evaluation,
            best_evaluation=evaluation,
        )
        failures: list[str] = []
        successes = 0
        for seed in fixture.seeds:
            path_argument = list(original)
            result = compiled.apply(
                path_argument,
                fixture.environment,
                np.random.default_rng(seed),
                context,
            )
            candidate = list(result.path)
            if path_argument != original:
                failures.append(f"seed {seed}: input mutation")
            if not 2 <= len(candidate) <= compiled.limits.max_waypoints:
                failures.append(f"seed {seed}: invalid waypoint count")
            elif candidate[0] != original[0] or candidate[-1] != original[-1]:
                failures.append(f"seed {seed}: endpoint changed")
            if any(
                not math.isfinite(float(value))
                for point in candidate
                for value in point
            ):
                failures.append(f"seed {seed}: non-finite coordinate")
            if result.success:
                successes += 1
        return DomainSmokeReport(
            domain_id=self.domain_id,
            ir_version=self.ir_version,
            smoke_passed=not failures,
            seeds_tested=len(fixture.seeds),
            successful_calls=successes,
            failures=failures,
        )

    def capability_usage(self, ir: OperatorSpec) -> tuple[str, ...]:
        spec = self.parse_ir(ir)
        result = [spec.selection_strategy.kind]
        result.extend(step.kind for step in spec.transformations)
        if spec.repair_strategy is not None:
            result.append(spec.repair_strategy.kind)
            result.extend(
                step.kind for step in spec.repair_strategy.transformations
            )
        if spec.fallback_strategy is not None:
            result.append(spec.fallback_strategy.kind)
        return tuple(result)

    def topology_payload(self, ir: OperatorSpec) -> dict[str, Any]:
        spec = self.parse_ir(ir)
        repair = spec.repair_strategy
        return {
            "conditions": [
                _condition_topology(condition)
                for condition in spec.applicability_conditions
            ],
            "selection": spec.selection_strategy.kind,
            "transformations": [
                {"kind": step.kind, "when": _condition_topology(step.when)}
                for step in spec.transformations
            ],
            "repair": (
                None
                if repair is None
                else {
                    "kind": repair.kind,
                    "transformations": [
                        {
                            "kind": step.kind,
                            "when": _condition_topology(step.when),
                        }
                        for step in repair.transformations
                    ],
                }
            ),
            "fallback": (
                None
                if spec.fallback_strategy is None
                else spec.fallback_strategy.kind
            ),
        }

    def topology_fingerprint(self, ir: OperatorSpec) -> str:
        return proposal_hash(self.topology_payload(ir))

    def behavior_payload(self, ir: OperatorSpec) -> dict[str, Any]:
        payload = self.parse_ir(ir).model_dump(mode="json")
        for field in (
            "name",
            "description",
            "parent_operators",
            "expected_mechanism",
            "target_failure_modes",
        ):
            payload.pop(field, None)
        return payload

    def behavior_fingerprint(self, ir: OperatorSpec) -> str:
        return proposal_hash(self.behavior_payload(ir))

    def static_safety_score(self, ir: OperatorSpec) -> float:
        fallback = self.parse_ir(ir).fallback_strategy
        return (
            1.0
            if fallback is not None and fallback.kind == "rollback_on_failure"
            else 0.8
        )

    def ir_name(self, ir: OperatorSpec) -> str:
        return self.parse_ir(ir).name

    def ir_parent_ids(self, ir: OperatorSpec) -> tuple[str, ...]:
        return tuple(self.parse_ir(ir).parent_operators)

    def builtin_ir(self, operator_id: str) -> OperatorSpec | None:
        return manual_operator_specs().get(str(operator_id))


__all__ = ["UAVDomainKit", "UAVSmokeFixture", "UAV_IR_VERSION"]
