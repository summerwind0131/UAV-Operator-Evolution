"""JSSP-owned proposal capability boundary for the ``jssp-v1`` IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import JsonValue

from operator_evolution_core.proposal import DomainSmokeReport, proposal_hash
from operator_evolution_core.search import SearchContext

from ..adapter import JSSP_DOMAIN_ID
from ..models import JobShopInstance, JobShopSolution
from .compiler import CompiledJSSPOperator, JSSPOperatorCompiler
from .population import initial_operator_specs
from .specs import JSSP_IR_VERSION, JSSPOperatorSpec, capability_catalog


@dataclass(frozen=True, slots=True)
class JSSPSmokeFixture:
    instance: JobShopInstance
    solution: JobShopSolution
    seeds: tuple[int, ...] = (0, 1, 2)


class JSSPDomainKit:
    domain_id = JSSP_DOMAIN_ID
    ir_version = JSSP_IR_VERSION
    accepts_legacy_unversioned = False

    def __init__(self, compiler: JSSPOperatorCompiler | None = None) -> None:
        self.compiler = compiler or JSSPOperatorCompiler()

    def parse_ir(self, payload: Any) -> JSSPOperatorSpec:
        return payload if isinstance(payload, JSSPOperatorSpec) else JSSPOperatorSpec.model_validate(payload)

    def serialize_ir(self, ir: JSSPOperatorSpec) -> JsonValue:
        return self.parse_ir(ir).model_dump(mode="json")

    def capability_catalog(self) -> dict[str, tuple[str, ...]]:
        return capability_catalog()

    def compile(self, ir: JSSPOperatorSpec) -> CompiledJSSPOperator:
        return self.compiler.compile(self.parse_ir(ir))

    def smoke(
        self,
        ir: JSSPOperatorSpec,
        fixture: JSSPSmokeFixture,
    ) -> DomainSmokeReport:
        if not isinstance(fixture, JSSPSmokeFixture):
            raise TypeError("jssp-v1 smoke requires JSSPSmokeFixture")
        compiled = self.compile(ir)
        original = tuple(fixture.solution.sequence)
        failures: list[str] = []
        successes = 0
        for seed in fixture.seeds:
            argument = JobShopSolution(sequence=original)
            result = compiled.apply(
                argument,
                fixture.instance,
                np.random.default_rng(seed),
                SearchContext(),
            )
            if argument.sequence != original:
                failures.append(f"seed {seed}: input mutation")
            if sorted(result.solution.sequence) != sorted(original):
                failures.append(f"seed {seed}: multiplicity changed")
            if len(result.solution.sequence) != fixture.instance.operation_count:
                failures.append(f"seed {seed}: invalid sequence length")
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

    def capability_usage(self, ir: JSSPOperatorSpec) -> tuple[str, ...]:
        spec = self.parse_ir(ir)
        return (spec.selector.kind, spec.transform.kind, spec.repair.kind)

    def topology_fingerprint(self, ir: JSSPOperatorSpec) -> str:
        spec = self.parse_ir(ir)
        return proposal_hash(
            {
                "selector": spec.selector.kind,
                "transform": spec.transform.kind,
                "repair": spec.repair.kind,
            }
        )

    def behavior_fingerprint(self, ir: JSSPOperatorSpec) -> str:
        payload = self.parse_ir(ir).model_dump(mode="json")
        for field in ("operator_id", "name", "description", "parent_ids"):
            payload.pop(field, None)
        return proposal_hash(payload)

    def static_safety_score(self, ir: JSSPOperatorSpec) -> float:
        return 1.0 if self.parse_ir(ir).repair.kind == "multiplicity_guard" else 0.0

    def ir_name(self, ir: JSSPOperatorSpec) -> str:
        return self.parse_ir(ir).name

    def ir_parent_ids(self, ir: JSSPOperatorSpec) -> tuple[str, ...]:
        return tuple(self.parse_ir(ir).parent_ids)

    def builtin_ir(self, operator_id: str) -> JSSPOperatorSpec | None:
        return next(
            (spec for spec in initial_operator_specs() if spec.operator_id == operator_id),
            None,
        )


__all__ = ["JSSPDomainKit", "JSSPSmokeFixture"]
