from __future__ import annotations

from pathlib import Path

from uav_operator_evolution.afl_uav.coordinator import AFLUAVCoordinator
from uav_operator_evolution.afl_uav.artifact import AUTHORITATIVE_OBJECTIVE
from uav_operator_evolution.afl_uav.buffer import SolverBuffer
from uav_operator_evolution.afl_uav.mock_solver import (
    afl_uav_mock_factory,
    assemble_solver_source,
)
from uav_operator_evolution.afl_uav.models import (
    AFLUAVLimits,
    CodeStageDraft,
    ConstraintDefinition,
    SolverRevision,
    UAVProblemDescription,
)
from uav_operator_evolution.afl_uav.validation import (
    CodeStageValidator,
    GeneratedCodePolicy,
    STAGE_CONTRACTS,
    TaskDescriptionValidator,
    validate_complete_solver_source,
)
from uav_operator_evolution.agents.providers import LLMCallConfig, MockLLMProvider
from uav_operator_evolution.environment.environment import Environment2D
from uav_operator_evolution.environment.obstacles import CircleObstacle
from uav_operator_evolution.path.evaluator import PathEvaluator
from uav_operator_evolution.path.models import ObjectiveWeights


def _environment() -> Environment2D:
    return Environment2D(
        map_id="afl-uav-test",
        width=20.0,
        height=20.0,
        start=(2.0, 2.0),
        goal=(18.0, 18.0),
        obstacles=[CircleObstacle(center=(10.0, 10.0), radius=2.0)],
        safety_distance=0.75,
        difficulty="medium",
        seed=17,
    )


def _coordinator(
    factory=afl_uav_mock_factory,
    *,
    solver_buffer: SolverBuffer | None = None,
) -> AFLUAVCoordinator:
    return AFLUAVCoordinator(
        provider=MockLLMProvider(factory=factory),
        call_config=LLMCallConfig(
            max_output_tokens=16_384,
            max_total_tokens=250_000,
        ),
        evaluator=PathEvaluator(ObjectiveWeights()),
        limits=AFLUAVLimits(execution_timeout_seconds=10.0),
        solver_buffer=solver_buffer,
    )


def test_task_description_hard_contract_rejects_missing_input() -> None:
    description = UAVProblemDescription(
        problem_type="STATIC_2D_UAV_PATH_PLANNING",
        description="Plan a bounded static path.",
        constraints=[
            ConstraintDefinition(
                constraint_id=name,
                abbreviation=name[:2].upper(),
                explanation=f"Enforce {name}.",
            )
            for name in (
                "endpoint_preservation",
                "map_bounds",
                "obstacle_clearance",
                "finite_coordinates",
                "waypoint_limit",
            )
        ],
        inputs=["width"],
        output="Return a feasible path.",
        objective="Minimize length, collision, smoothness, risk, and waypoint terms.",
        source_hash="0" * 64,
    )
    issues = TaskDescriptionValidator().validate(description, "0" * 64)
    assert "missing required input: height" in issues


def test_authoritative_audit_objective_names_all_trusted_terms() -> None:
    lowered = AUTHORITATIVE_OBJECTIVE.lower()
    assert all(
        term in lowered
        for term in ("length", "collision", "smoothness", "risk", "waypoint")
    )


def test_constraint_label_accepts_descriptive_non_executable_name() -> None:
    constraint = ConstraintDefinition(
        constraint_id="endpoint_preservation",
        abbreviation="endpoint_preservation",
        explanation="Keep the first and last waypoints fixed.",
    )
    assert constraint.abbreviation == "endpoint_preservation"


def test_generated_code_policy_rejects_unsafe_import() -> None:
    issues = GeneratedCodePolicy().validate(
        "import subprocess\nsubprocess.run(['whoami'])\n",
        max_source_chars=10_000,
    )
    assert "import is not allowed: subprocess" in issues
    assert "name is forbidden: subprocess" in issues


def test_stage_contracts_reject_cross_stage_budget_and_geometry_errors() -> None:
    validator = CodeStageValidator()
    contracts = {contract.name: contract for contract in STAGE_CONTRACTS}

    geometry_issues = validator.validate(
        contracts["geometry"],
        "def segment_collision_free(start, end, problem):\n"
        "    problem['_metrics']['collision_checks'] += 1\n"
        "    return not problem['risk_zones']\n"
        "\n"
        "def marker():\n"
        "    return ('circle', 'rectangle')\n",
        set(),
    )
    assert "risk zones are soft costs and must not be treated as obstacles" in geometry_issues

    read_issues = validator.validate(
        contracts["read_problem"],
        "def read_problem(path):\n"
        "    problem = {'circle': 1, 'rectangle': 1}\n"
        "    problem['seed'] = data.get('seed', 0)\n"
        "    return problem\n",
        set(),
    )
    assert "read_problem must obtain seed from the environment object" in read_issues

    cost_issues = validator.validate(
        contracts["cost"],
        "def path_cost(path, problem):\n"
        "    problem['_metrics']['objective_evaluations'] += 1\n"
        "    return 0.0\n",
        set(),
    )
    assert "path_cost collision tests must use the counted collision checker" in cost_issues

    repair_issues = validator.validate(
        contracts["repair"],
        "def repair(original, candidate, problem, rng):\n"
        "    path_cost(candidate, problem)\n"
        "    return candidate\n",
        set(),
    )
    assert "repair must not consume hidden objective evaluations" in repair_issues
    assert "repair must verify candidate segments continuously" in repair_issues

    validation_issues = validator.validate(
        contracts["validation"],
        "def validate_path(path, problem):\n    return True\n",
        set(),
    )
    assert "validate_path must verify every segment continuously" in validation_issues

    main_issues = validator.validate(
        contracts["main"],
        "def main():\n"
        "    problem['_metrics']['objective_evaluations'] += 1\n",
        set(),
    )
    assert (
        "main must not mutate individual counters maintained by worker functions"
        in main_issues
    )


def test_complete_reference_solver_satisfies_all_static_stage_contracts() -> None:
    assert validate_complete_solver_source(assemble_solver_source()) == []


def test_complete_solver_rejects_public_signature_drift() -> None:
    source = assemble_solver_source().replace(
        "def destroy(path, rng):",
        "def destroy(path, rng, problem):",
        1,
    )
    assert "destroy must use exact parameters ['path', 'rng']" in (
        validate_complete_solver_source(source)
    )


def test_read_problem_accepts_direct_preservation_of_raw_obstacles() -> None:
    validator = CodeStageValidator()
    contract = {item.name: item for item in STAGE_CONTRACTS}["read_problem"]
    source = (
        "def read_problem(path):\n"
        "    with open(path) as handle:\n"
        "        data = json.load(handle)\n"
        "    env = data['environment']\n"
        "    return {'obstacles': env.get('obstacles', []), "
        "'seed': env.get('seed', 0)}\n"
    )
    assert "read_problem must preserve both circle and rectangle obstacles" not in (
        validator.validate(contract, source, set())
    )


def test_complete_solver_rejects_undefined_function_calls() -> None:
    source = assemble_solver_source().replace(
        "print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))",
        "print(missing_risk_helper(result))",
        1,
    )
    assert "complete solver calls undefined functions: missing_risk_helper" in (
        validate_complete_solver_source(source)
    )


def test_mock_afl_uav_pipeline_generates_and_validates_solver(tmp_path: Path) -> None:
    result = _coordinator().run(
        run_id="afl-uav-test",
        environment=_environment(),
        objective_weights=ObjectiveWeights(),
        output_dir=tmp_path,
        iterations=24,
        grid_resolution=2.0,
        max_waypoints=128,
    )
    assert result.status == "success"
    assert result.external_validation is not None
    assert result.external_validation.passed is True
    assert result.external_validation.evaluation is not None
    assert result.external_validation.evaluation.feasible is True
    assert result.execution_attempts == 1
    assert len(result.role_events) == 18
    assert set(result.code_stage_revisions.values()) == {0}
    assert Path(result.solver_path).is_file()


def test_code_roles_receive_compact_prior_interfaces_not_accumulated_source(
    tmp_path: Path,
) -> None:
    captured_payloads: list[dict[str, object]] = []

    def capturing_factory(output_model, user_payload):
        captured_payloads.append(dict(user_payload))
        return afl_uav_mock_factory(output_model, user_payload)

    provider = MockLLMProvider(factory=capturing_factory)
    coordinator = AFLUAVCoordinator(
        provider=provider,
        call_config=LLMCallConfig(max_output_tokens=16_384, max_total_tokens=250_000),
        evaluator=PathEvaluator(ObjectiveWeights()),
        limits=AFLUAVLimits(execution_timeout_seconds=10.0),
    )

    coordinator.run(
        run_id="afl-uav-compact-context-test",
        environment=_environment(),
        objective_weights=ObjectiveWeights(),
        output_dir=tmp_path,
        execute_generated=False,
    )

    description_payloads = [
        payload
        for payload in captured_payloads
        if payload.get("role")
        in {"description_generation", "description_judgment", "description_revision"}
    ]
    assert description_payloads
    assert all(
        payload["description_contract"]["required_inputs"]
        == [
            "width",
            "height",
            "start",
            "goal",
            "obstacles",
            "risk_zones",
            "safety_distance",
            "seed",
            "objective_weights",
            "grid_resolution",
            "max_waypoints",
        ]
        for payload in description_payloads
    )

    code_payloads = [
        payload
        for payload in captured_payloads
        if payload.get("role")
        in {"code_generation", "code_judgment", "code_revision"}
    ]
    assert code_payloads
    assert all("accumulated_source" not in payload for payload in code_payloads)
    assert all(
        payload["solver_header_contract"]["preimported_modules"]
        == ["argparse", "heapq", "json", "math", "random"]
        for payload in code_payloads
    )
    assert all(payload["trusted_preconditions"] for payload in code_payloads)
    generation_payloads = [
        payload for payload in code_payloads if payload["role"] == "code_generation"
    ]
    interface_counts = [len(payload["available_interfaces"]) for payload in generation_payloads]
    assert interface_counts[0] == 0
    assert interface_counts == sorted(interface_counts)
    cost_interfaces = {
        interface["function_name"]
        for interface in generation_payloads[2]["available_interfaces"]
    }
    assert "segment_collision_free" in cost_interfaces
    assert "_segment_box_fraction" in cost_interfaces
    assert all(
        "source" not in interface
        for payload in generation_payloads
        for interface in payload["available_interfaces"]
    )
    main_requirements = generation_payloads[-1]["stage_requirements"]
    assert any("Never perform more objective evaluations" in item for item in main_requirements)
    assert any("do not define nested functions" in item for item in main_requirements)


def test_execution_error_invokes_eaa_and_complete_revision(tmp_path: Path) -> None:
    def failing_once_factory(output_model, user_payload):
        if output_model is CodeStageDraft and user_payload.get("function_name") == "main":
            return CodeStageDraft(
                function_name="main",
                source=(
                    "def main() -> None:\n"
                    "    raise RuntimeError('intentional first execution failure')\n\n"
                    "if __name__ == '__main__':\n"
                    "    main()"
                ),
                rationale="Exercise the bounded EAA revision loop.",
            )
        if output_model is SolverRevision:
            return SolverRevision(
                source=assemble_solver_source(),
                changes=["Replace the failing main stage with the complete solver."],
            )
        return afl_uav_mock_factory(output_model, user_payload)

    result = _coordinator(failing_once_factory).run(
        run_id="afl-uav-eaa-test",
        environment=_environment(),
        objective_weights=ObjectiveWeights(),
        output_dir=tmp_path,
        iterations=8,
        grid_resolution=2.0,
        max_waypoints=128,
    )
    assert result.status == "success"
    assert result.execution_attempts == 2
    assert [event.role for event in result.role_events[-2:]] == [
        "error_analysis_agent",
        "revision_agent",
    ]


def test_validated_solver_buffer_skips_code_roles_but_revalidates(
    tmp_path: Path,
) -> None:
    solver_buffer = SolverBuffer(tmp_path / "buffer")
    first = _coordinator(solver_buffer=solver_buffer).run(
        run_id="afl-uav-buffer-first",
        environment=_environment(),
        objective_weights=ObjectiveWeights(),
        output_dir=tmp_path / "first",
        iterations=8,
        grid_resolution=2.0,
        max_waypoints=128,
    )
    second = _coordinator(solver_buffer=solver_buffer).run(
        run_id="afl-uav-buffer-second",
        environment=_environment(),
        objective_weights=ObjectiveWeights(),
        output_dir=tmp_path / "second",
        iterations=8,
        grid_resolution=2.0,
        max_waypoints=128,
    )

    assert first.status == "success"
    assert first.buffer_hit is False
    assert len(first.role_events) == 18
    assert second.status == "success"
    assert second.buffer_hit is True
    assert second.buffer_key == first.buffer_key
    assert second.solver_hash == first.solver_hash
    assert len(second.role_events) == 2
    assert second.execution_attempts == 1
    assert second.external_validation is not None
    assert second.external_validation.passed is True
