from __future__ import annotations

import time

from uav_operator_evolution.agents.evidence import DesignBudget, OperatorEvidenceBundle
from uav_operator_evolution.agents.tools import (
    AUTHORIZED_TOOL_NAMES,
    AgentBudget,
    AgentBudgetController,
    AgentToolContext,
    AgentToolDispatcher,
    SmokeTestFixture,
)
from uav_operator_evolution.environment import Environment2D
from uav_operator_evolution.operators.catalog import manual_operator_specs
from uav_operator_evolution.operators.compiler import OperatorCompiler
from uav_operator_evolution.operators.specs import primitive_catalog


def _bundle() -> OperatorEvidenceBundle:
    parent = manual_operator_specs()["segment_shift"]
    return OperatorEvidenceBundle(
        problem_summary="Improve one bounded operator.",
        parent_specs=[parent],
        parent_profiles=[{"operator_id": parent.name, "attempts": 4}],
        existing_operator_names=[parent.name],
        allowed_primitives={key: list(values) for key, values in primitive_catalog().items()},
        design_budget=DesignBudget(),
        limitations=["fixture has no trajectory evidence"],
    )


def _dispatcher(*, max_calls: int = 12, timeout_ms: float = 1000.0):
    environment = Environment2D(
        map_id="smoke", width=20, height=20, start=(1, 1), goal=(19, 19), obstacles=[]
    )
    budget = AgentBudgetController(
        AgentBudget(max_tool_calls=max_calls, max_smoke_tests=2)
    )
    dispatcher = AgentToolDispatcher(
        AgentToolContext(
            bundle=_bundle(),
            compiler=OperatorCompiler(),
            smoke_fixture=SmokeTestFixture(environment, [(1, 1), (8, 9), (19, 19)]),
        ),
        budget,
        tool_timeout_ms=timeout_ms,
    )
    return dispatcher, budget


def test_fixed_whitelist_rejects_arbitrary_tools() -> None:
    dispatcher, budget = _dispatcher()
    assert dispatcher.authorized_names == AUTHORIZED_TOOL_NAMES
    result = dispatcher.execute("run_shell", {"command": "whoami"})
    assert result.status == "unauthorized"
    assert result.authorized is False
    assert budget.usage.tool_calls == 0


def test_tool_budget_and_compact_counterfactual_boundary() -> None:
    dispatcher, budget = _dispatcher(max_calls=1)
    first = dispatcher.execute("get_counterfactual_results", {"operator_id": "segment_shift"})
    second = dispatcher.execute("get_allowed_primitives", {})
    assert first.status == "ok"
    assert first.payload == {
        "operator_id": "segment_shift",
        "counterfactual_results": [],
    }
    assert "path" not in first.payload_json
    assert second.status == "budget_exceeded"
    assert budget.usage.tool_calls == 1


def test_compile_and_smoke_use_trusted_dsl() -> None:
    dispatcher, budget = _dispatcher()
    spec = manual_operator_specs()["segment_shift"].model_dump(mode="json")
    compiled = dispatcher.execute("compile_operator_spec", {"operator_spec": spec})
    smoke = dispatcher.execute("run_operator_smoke_test", {"operator_spec": spec})
    assert compiled.status == "ok" and compiled.payload["compiled"] is True
    assert smoke.status == "ok" and smoke.payload["smoke_passed"] is True
    assert budget.usage.smoke_tests == 1


def test_elapsed_tool_timeout_is_fail_closed(monkeypatch) -> None:
    dispatcher, _ = _dispatcher(timeout_ms=1.0)
    original = dispatcher._tools["get_allowed_primitives"]

    def slow(query):
        time.sleep(0.01)
        return original.handler(query)

    monkeypatch.setitem(
        dispatcher._tools,
        "get_allowed_primitives",
        type(original)(original.input_model, slow),
    )
    result = dispatcher.execute("get_allowed_primitives", {})
    assert result.status == "timeout"
    assert result.payload == {}
