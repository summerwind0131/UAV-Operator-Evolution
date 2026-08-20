from __future__ import annotations

import numpy as np
import pytest

from uav_operator_evolution.config import DSLConfig
from uav_operator_evolution.environment.environment import Environment2D
from uav_operator_evolution.operators.compiler import OperatorCompilationError, OperatorCompiler
from uav_operator_evolution.operators.specs import OperatorSpec
from uav_operator_evolution.path.evaluator import PathEvaluator
from uav_operator_evolution.search.context import SearchContext


def environment() -> Environment2D:
    return Environment2D(width=20, height=20, start=(1, 1), goal=(19, 19), safety_distance=0)


def test_compiler_executes_bounded_composite_without_mutating_input() -> None:
    spec = OperatorSpec.model_validate(
        {
            "name": "ShortcutThenSmooth",
            "description": "test composite",
            "selection_strategy": {"kind": "select_long_segment"},
            "transformations": [{"kind": "shortcut_segment"}, {"kind": "smooth_segment"}],
            "expected_mechanism": "shorten and smooth",
        }
    )
    operator = OperatorCompiler().compile(spec)
    path = [(1.0, 1.0), (2.0, 10.0), (10.0, 12.0), (19.0, 19.0)]
    original = list(path)
    evaluation = PathEvaluator().evaluate(path, environment())
    result = operator.apply(
        path,
        environment(),
        np.random.default_rng(4),
        SearchContext(current_evaluation=evaluation, best_evaluation=evaluation),
    )
    assert path == original
    assert result.path[0] == original[0] and result.path[-1] == original[-1]
    assert len(result.path) <= len(path)


def test_compiler_enforces_runtime_configuration_limits() -> None:
    spec = OperatorSpec.model_validate(
        {
            "name": "TooManySteps",
            "description": "test",
            "selection_strategy": {"kind": "select_long_segment"},
            "transformations": [{"kind": "smooth_segment"}, {"kind": "smooth_segment"}],
            "expected_mechanism": "test",
        }
    )
    with pytest.raises(OperatorCompilationError):
        OperatorCompiler(DSLConfig(max_transformations=1)).compile(spec)

