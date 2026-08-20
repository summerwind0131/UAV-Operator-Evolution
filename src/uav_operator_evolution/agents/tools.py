"""Whitelisted, bounded tools available to the operator research agent.

The dispatcher deliberately exposes compact typed evidence and the trusted DSL
compiler.  It has no filesystem, shell, network, arbitrary Python, or formal
candidate-validation tool.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..environment import Environment2D
from ..memory import MechanismMemory
from ..operators.compiler import OperatorCompiler
from ..operators.specs import OperatorSpec
from ..path import PathEvaluator
from ..path.models import ObjectiveWeights, Path
from ..reproducibility import canonical_json
from ..search.context import SearchContext
from .evidence import OperatorEvidenceBundle


class AgentToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AgentBudget(AgentToolModel):
    max_turns: int = Field(6, ge=1, le=32)
    max_tool_calls: int = Field(12, ge=0, le=128)
    max_candidate_specs: int = Field(2, ge=1, le=8)
    max_revisions: int = Field(1, ge=0, le=4)
    max_smoke_tests: int = Field(2, ge=0, le=16)


class AgentUsage(AgentToolModel):
    turns: int = Field(0, ge=0)
    tool_calls: int = Field(0, ge=0)
    candidate_specs: int = Field(0, ge=0)
    revisions: int = Field(0, ge=0)
    smoke_tests: int = Field(0, ge=0)
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class AgentBudgetExceeded(RuntimeError):
    """Raised before an action would exceed a local hard budget."""


class AgentBudgetController:
    def __init__(self, budget: AgentBudget | None = None) -> None:
        self.budget = budget or AgentBudget()
        self.usage = AgentUsage()

    def _increment(self, field: str, limit_field: str, amount: int = 1) -> None:
        current = int(getattr(self.usage, field))
        limit = int(getattr(self.budget, limit_field))
        if current + amount > limit:
            raise AgentBudgetExceeded(f"{field} budget exceeded ({current + amount}>{limit})")
        self.usage = self.usage.model_copy(update={field: current + amount})

    def start_turn(self) -> None:
        self._increment("turns", "max_turns")

    def register_tool(self, *, smoke: bool = False) -> None:
        self._increment("tool_calls", "max_tool_calls")
        if smoke:
            self._increment("smoke_tests", "max_smoke_tests")

    def register_candidate(self, *, revision: bool = False) -> None:
        self._increment("candidate_specs", "max_candidate_specs")
        if revision:
            self._increment("revisions", "max_revisions")

    def add_tokens(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.usage = self.usage.model_copy(
            update={
                "input_tokens": self.usage.input_tokens + max(0, int(input_tokens)),
                "output_tokens": self.usage.output_tokens + max(0, int(output_tokens)),
            }
        )


class OperatorIdInput(AgentToolModel):
    operator_id: str = Field(min_length=1, max_length=200)


class CaseQueryInput(OperatorIdInput):
    outcome: Literal["success", "failure"] | None = None
    limit: int = Field(3, ge=1, le=3)


class LineageQueryInput(OperatorIdInput):
    direction: Literal["ancestors", "descendants", "both"] = "both"
    max_depth: int = Field(4, ge=1, le=8)


class EmptyToolInput(AgentToolModel):
    pass


class OperatorSpecInput(AgentToolModel):
    operator_spec: OperatorSpec


class ToolExecutionResult(AgentToolModel):
    tool_name: str
    status: Literal["ok", "error", "timeout", "unauthorized", "budget_exceeded"]
    authorized: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_json: str = "{}"
    error: str | None = None
    latency_ms: float = Field(ge=0.0)
    sequence: int = Field(ge=1)


class ToolAuditSink(Protocol):
    def __call__(self, result: ToolExecutionResult, arguments: Mapping[str, Any]) -> None: ...


@dataclass(slots=True, frozen=True)
class SmokeTestFixture:
    environment: Environment2D
    path: Path
    seeds: tuple[int, ...] = (0, 1, 2)


@dataclass(slots=True)
class AgentToolContext:
    bundle: OperatorEvidenceBundle
    compiler: OperatorCompiler
    memory: MechanismMemory | None = None
    smoke_fixture: SmokeTestFixture | None = None


@dataclass(slots=True, frozen=True)
class _ToolDefinition:
    input_model: type[AgentToolModel]
    handler: Callable[[AgentToolModel], dict[str, Any]]


AUTHORIZED_TOOL_NAMES = (
    "get_operator_profile",
    "get_failure_modes",
    "get_synergies",
    "get_relevant_cases",
    "get_lineage",
    "get_counterfactual_results",
    "get_allowed_primitives",
    "get_parent_operator_spec",
    "compile_operator_spec",
    "run_operator_smoke_test",
)


def _compact(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _compact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value]
    return value


class AgentToolDispatcher:
    """Dispatch exactly ten read/compile/smoke tools under a shared budget."""

    def __init__(
        self,
        context: AgentToolContext,
        budget: AgentBudgetController,
        *,
        audit_sink: ToolAuditSink | None = None,
        max_result_chars: int = 12_000,
        tool_timeout_ms: float = 1_000.0,
    ) -> None:
        self.context = context
        self.budget = budget
        self.audit_sink = audit_sink
        self.max_result_chars = max(256, int(max_result_chars))
        self.tool_timeout_ms = max(0.1, float(tool_timeout_ms))
        self._sequence = 0
        self._tools: dict[str, _ToolDefinition] = {
            "get_operator_profile": _ToolDefinition(OperatorIdInput, self._profile),
            "get_failure_modes": _ToolDefinition(OperatorIdInput, self._failures),
            "get_synergies": _ToolDefinition(OperatorIdInput, self._synergies),
            "get_relevant_cases": _ToolDefinition(CaseQueryInput, self._cases),
            "get_lineage": _ToolDefinition(LineageQueryInput, self._lineage),
            "get_counterfactual_results": _ToolDefinition(OperatorIdInput, self._counterfactuals),
            "get_allowed_primitives": _ToolDefinition(EmptyToolInput, self._primitives),
            "get_parent_operator_spec": _ToolDefinition(OperatorIdInput, self._parent_spec),
            "compile_operator_spec": _ToolDefinition(OperatorSpecInput, self._compile),
            "run_operator_smoke_test": _ToolDefinition(OperatorSpecInput, self._smoke),
        }
        if tuple(self._tools) != AUTHORIZED_TOOL_NAMES:
            raise AssertionError("agent tool registry does not match the fixed whitelist")

    @property
    def authorized_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def execute(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> ToolExecutionResult:
        self._sequence += 1
        started = time.perf_counter()
        arguments = dict(arguments or {})
        definition = self._tools.get(tool_name)
        if definition is None:
            result = self._result(tool_name, "unauthorized", False, started, error="tool is not authorized")
            self._audit(result, arguments)
            return result
        try:
            self.budget.register_tool(smoke=tool_name == "run_operator_smoke_test")
        except AgentBudgetExceeded as exc:
            result = self._result(tool_name, "budget_exceeded", True, started, error=str(exc))
            self._audit(result, arguments)
            return result
        try:
            validated = definition.input_model.model_validate(arguments)
            payload = definition.handler(validated)
            elapsed = (time.perf_counter() - started) * 1_000.0
            if elapsed > self.tool_timeout_ms:
                result = self._result(
                    tool_name,
                    "timeout",
                    True,
                    started,
                    error=f"tool deadline exceeded ({elapsed:.3f} ms)",
                )
            else:
                result = self._result(tool_name, "ok", True, started, payload=payload)
        except Exception as exc:  # fail closed at the tool boundary
            result = self._result(
                tool_name,
                "error",
                True,
                started,
                error=f"{type(exc).__name__}: {exc}",
            )
        self._audit(result, arguments)
        return result

    def _result(
        self,
        name: str,
        status: Literal["ok", "error", "timeout", "unauthorized", "budget_exceeded"],
        authorized: bool,
        started: float,
        *,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> ToolExecutionResult:
        compact = _compact(payload or {})
        encoded = canonical_json(compact)
        if len(encoded) > self.max_result_chars:
            compact = {
                "truncated": True,
                "original_chars": len(encoded),
                "preview": encoded[: self.max_result_chars - 96],
            }
            encoded = canonical_json(compact)
        return ToolExecutionResult(
            tool_name=name,
            status=status,
            authorized=authorized,
            payload=compact,
            payload_json=encoded,
            error=error,
            latency_ms=(time.perf_counter() - started) * 1_000.0,
            sequence=self._sequence,
        )

    def _audit(self, result: ToolExecutionResult, arguments: Mapping[str, Any]) -> None:
        if self.audit_sink is not None:
            self.audit_sink(result, arguments)

    def _profile(self, query: AgentToolModel) -> dict[str, Any]:
        assert isinstance(query, OperatorIdInput)
        rows = [
            profile for profile in self.context.bundle.parent_profiles
            if str(profile.get("operator_id")) == query.operator_id
        ]
        return {"operator_id": query.operator_id, "profiles": rows[:1]}

    def _failures(self, query: AgentToolModel) -> dict[str, Any]:
        assert isinstance(query, OperatorIdInput)
        rows = [
            item.model_dump(mode="json") for item in self.context.bundle.failure_modes
            if item.operator_id == query.operator_id
        ]
        return {"operator_id": query.operator_id, "failure_modes": rows[:8]}

    def _synergies(self, query: AgentToolModel) -> dict[str, Any]:
        assert isinstance(query, OperatorIdInput)
        rows = [
            item.model_dump(mode="json") for item in self.context.bundle.synergy_evidence
            if query.operator_id in {item.first_operator, item.second_operator}
        ]
        return {"operator_id": query.operator_id, "synergies": rows[:8]}

    def _cases(self, query: AgentToolModel) -> dict[str, Any]:
        assert isinstance(query, CaseQueryInput)
        rows = [
            *self.context.bundle.representative_success_cases,
            *self.context.bundle.representative_failure_cases,
        ]
        selected = [
            item.model_dump(mode="json") for item in rows
            if item.operator_id == query.operator_id
            and (query.outcome is None or item.outcome == query.outcome)
        ][: query.limit]
        return {"operator_id": query.operator_id, "cases": selected}

    def _lineage(self, query: AgentToolModel) -> dict[str, Any]:
        assert isinstance(query, LineageQueryInput)
        if self.context.memory is None:
            return {"operator_id": query.operator_id, "lineage": [], "limitation": "memory unavailable"}
        rows = self.context.memory.get_lineage(
            query.operator_id, direction=query.direction, max_depth=query.max_depth
        )
        return {"operator_id": query.operator_id, "lineage": [_compact(row) for row in rows[:32]]}

    def _counterfactuals(self, query: AgentToolModel) -> dict[str, Any]:
        assert isinstance(query, OperatorIdInput)
        rows = [
            item.model_dump(mode="json") for item in self.context.bundle.counterfactual_evidence
            if item.operator_id == query.operator_id
        ]
        return {"operator_id": query.operator_id, "counterfactual_results": rows[:8]}

    def _primitives(self, query: AgentToolModel) -> dict[str, Any]:
        assert isinstance(query, EmptyToolInput)
        return {"allowed_primitives": self.context.bundle.allowed_primitives}

    def _parent_spec(self, query: AgentToolModel) -> dict[str, Any]:
        assert isinstance(query, OperatorIdInput)
        spec = next(
            (item for item in self.context.bundle.parent_specs if item.name == query.operator_id),
            None,
        )
        if spec is None:
            raise KeyError(f"operator is not a parent in this evidence bundle: {query.operator_id}")
        return {"operator_spec": spec.model_dump(mode="json")}

    def _compile(self, query: AgentToolModel) -> dict[str, Any]:
        assert isinstance(query, OperatorSpecInput)
        compiled = self.context.compiler.compile(query.operator_spec)
        return {
            "compiled": True,
            "operator_name": compiled.name,
            "parent_operators": list(compiled.parent_operator_ids),
        }

    def _smoke(self, query: AgentToolModel) -> dict[str, Any]:
        assert isinstance(query, OperatorSpecInput)
        fixture = self.context.smoke_fixture
        if fixture is None:
            raise RuntimeError("smoke fixture unavailable")
        compiled = self.context.compiler.compile(query.operator_spec)
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
            if any(not math.isfinite(float(value)) for point in candidate for value in point):
                failures.append(f"seed {seed}: non-finite coordinate")
            if result.success:
                successes += 1
        return {
            "smoke_passed": not failures,
            "seeds_tested": len(fixture.seeds),
            "successful_calls": successes,
            "failures": failures,
        }


__all__ = [
    "AUTHORIZED_TOOL_NAMES",
    "AgentBudget",
    "AgentBudgetController",
    "AgentBudgetExceeded",
    "AgentToolContext",
    "AgentToolDispatcher",
    "AgentUsage",
    "SmokeTestFixture",
    "ToolExecutionResult",
]
