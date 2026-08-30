"""Fixed eight-slot P0 used for JSSP qualification experiments."""

from __future__ import annotations

from .compiler import CompiledJSSPOperator, JSSPOperatorCompiler
from .specs import JSSPOperatorSpec


def initial_operator_specs() -> tuple[JSSPOperatorSpec, ...]:
    raw = (
        ("random-adjacent-swap", "Random adjacent swap", "random_adjacent", "swap"),
        ("random-two-point-swap", "Random two-point swap", "random_pair", "swap"),
        ("bounded-insertion", "Bounded insertion", "bounded_pair", "insert"),
        ("bounded-reversal", "Bounded reversal", "bounded_pair", "reverse"),
        (
            "critical-block-adjacent-swap",
            "Critical-block adjacent swap",
            "critical_block_adjacent",
            "swap",
        ),
        (
            "critical-block-endpoint-swap",
            "Critical-block endpoint swap",
            "critical_block_endpoints",
            "swap",
        ),
        (
            "bottleneck-block-insertion",
            "Bottleneck block insertion",
            "bottleneck_block",
            "insert",
        ),
        (
            "high-idle-gap-relocation",
            "High idle-gap relocation",
            "high_idle_gap",
            "insert",
        ),
    )
    return tuple(
        JSSPOperatorSpec.model_validate(
            {
                "operator_id": operator_id,
                "name": name,
                "description": (
                    f"Bounded jssp-v1 operator using {selector} and {transform}."
                ),
                "selector": {
                    "kind": selector,
                    "max_distance": 16,
                    "max_attempts": 8,
                },
                "transform": {"kind": transform, "max_segment_length": 32},
                "repair": {"kind": "multiplicity_guard"},
            }
        )
        for operator_id, name, selector, transform in raw
    )


def initial_operator_population(
    compiler: JSSPOperatorCompiler | None = None,
) -> tuple[CompiledJSSPOperator, ...]:
    active = compiler or JSSPOperatorCompiler()
    return tuple(active.compile(spec) for spec in initial_operator_specs())


__all__ = ["initial_operator_population", "initial_operator_specs"]
