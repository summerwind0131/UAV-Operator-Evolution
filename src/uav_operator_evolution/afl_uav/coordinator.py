"""Bounded three-subtask, four-role AFL reproduction for UAV path planning."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from ..agents.prompts import PromptTemplate
from ..agents.providers import LLMCallConfig, LLMProvider, LLMProviderError
from ..environment.environment import Environment2D
from ..path.evaluator import PathEvaluator
from ..path.models import ObjectiveWeights
from ..reproducibility import canonical_json, stable_hash
from .buffer import SolverBuffer
from .mock_solver import SOLVER_HEADER
from .models import (
    AFLUAVLimits,
    AFLUAVRunResult,
    CodeJudgment,
    CodeStageDraft,
    DescriptionJudgment,
    ErrorAnalysis,
    RoleEvent,
    SolverRevision,
    UAVProblemDescription,
    UAVSolverInstance,
)
from .prompts import (
    CODE_GENERATOR_V4,
    CODE_JUDGE_V4,
    CODE_REVISION_V4,
    COMPLETE_CODE_REVISION_V1,
    DESCRIPTION_GENERATOR_V2,
    DESCRIPTION_JUDGE_V2,
    DESCRIPTION_REVISION_V2,
    ERROR_ANALYSIS_V1,
)
from .runner import GeneratedSolverRunner
from .validation import (
    CodeStageValidator,
    GeneratedCodePolicy,
    REQUIRED_CONSTRAINTS,
    REQUIRED_INPUTS,
    STAGE_CONTRACTS,
    TaskDescriptionValidator,
    validate_solver_output,
)


OutputModel = TypeVar("OutputModel", bound=BaseModel)


class AFLUAVCoordinator:
    """Implement the paper's description, code, and solution subtasks with hard budgets."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        call_config: LLMCallConfig,
        evaluator: PathEvaluator,
        limits: AFLUAVLimits | None = None,
        runner: GeneratedSolverRunner | None = None,
        solver_buffer: SolverBuffer | None = None,
    ) -> None:
        self.provider = provider
        self.call_config = call_config
        self.evaluator = evaluator
        self.limits = limits or AFLUAVLimits()
        self.runner = runner or GeneratedSolverRunner()
        self.solver_buffer = solver_buffer
        self.description_validator = TaskDescriptionValidator()
        self.stage_validator = CodeStageValidator()
        self.code_policy = GeneratedCodePolicy()
        self._events: list[RoleEvent] = []

    @property
    def role_events(self) -> tuple[RoleEvent, ...]:
        """Expose the current append-only role audit, including failed runs."""

        return tuple(self._events)

    def run(
        self,
        *,
        run_id: str,
        environment: Environment2D,
        objective_weights: ObjectiveWeights,
        output_dir: str | Path,
        iterations: int = 100,
        grid_resolution: float = 4.0,
        max_waypoints: int = 128,
        execute_generated: bool = True,
    ) -> AFLUAVRunResult:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self._events = []
        instance = UAVSolverInstance(
            environment=environment,
            objective_weights=objective_weights,
            grid_resolution=grid_resolution,
            max_waypoints=max_waypoints,
        )
        instance_payload = instance.model_dump(mode="json")
        source_hash = stable_hash(instance_payload)
        instance_path = destination / "afl_uav_instance.json"
        self._write_json(instance_path, instance_payload)

        description, description_revisions = self._describe(instance_payload, source_hash)
        self._write_json(
            destination / "problem_description.json",
            description.model_dump(mode="json"),
        )
        buffer_key = SolverBuffer.key(description, instance)
        cached_source = None if self.solver_buffer is None else self.solver_buffer.load(buffer_key)
        buffer_hit = cached_source is not None
        if cached_source is None:
            solver_source, stage_revisions = self._generate_solver(description, instance_payload)
        else:
            solver_source = cached_source
            stage_revisions = {contract.name: 0 for contract in STAGE_CONTRACTS}
        policy_issues = self.code_policy.validate(
            solver_source,
            max_source_chars=self.limits.max_source_chars,
        )
        if policy_issues:
            raise RuntimeError("generated solver failed static policy: " + "; ".join(policy_issues))
        solver_path = destination / "generated_uav_solver.py"
        solver_path.write_text(solver_source, encoding="utf-8")
        output_path = destination / "solver_output.json"

        if not execute_generated:
            return self._result(
                run_id=run_id,
                status="generated_only",
                description=description,
                description_revisions=description_revisions,
                stage_revisions=stage_revisions,
                solver_path=solver_path,
                solver_source=solver_source,
                buffer_key=buffer_key,
                buffer_hit=buffer_hit,
                instance_path=instance_path,
                output_path=output_path,
                execution_attempts=0,
            )

        execution = None
        external = None
        failure_reason: str | None = None
        attempts = 0
        for revision_index in range(self.limits.max_execution_revisions + 1):
            attempts += 1
            execution = self.runner.execute(
                solver_path=solver_path,
                source=solver_source,
                instance_path=instance_path,
                output_path=output_path,
                iterations=iterations,
                timeout_seconds=self.limits.execution_timeout_seconds,
                max_source_chars=self.limits.max_source_chars,
            )
            if execution.status == "success":
                external = validate_solver_output(
                    execution.output_payload,
                    environment,
                    self.evaluator,
                    max_waypoints=max_waypoints,
                )
                if external.passed:
                    return self._result(
                        run_id=run_id,
                        status="success",
                        description=description,
                        description_revisions=description_revisions,
                        stage_revisions=stage_revisions,
                        solver_path=solver_path,
                        solver_source=solver_source,
                        buffer_key=buffer_key,
                        buffer_hit=buffer_hit,
                        instance_path=instance_path,
                        output_path=output_path,
                        execution_attempts=attempts,
                        execution=execution,
                        external_validation=external,
                    )
                failure_reason = "external validation failed: " + "; ".join(external.failures)
            else:
                failure_reason = execution.error or execution.status
            if revision_index >= self.limits.max_execution_revisions:
                break
            analysis = self._call(
                role="error_analysis_agent",
                action="analyze_execution_failure",
                template=ERROR_ANALYSIS_V1,
                output_model=ErrorAnalysis,
                user_payload={
                    "problem_description": description.model_dump(mode="json"),
                    "instance": instance_payload,
                    "solver_source": solver_source,
                    "execution": execution.model_dump(mode="json"),
                    "external_validation": (
                        None if external is None else external.model_dump(mode="json")
                    ),
                    "failure_reason": failure_reason,
                },
            )
            revision = self._call(
                role="revision_agent",
                action="revise_complete_solver",
                template=COMPLETE_CODE_REVISION_V1,
                output_model=SolverRevision,
                user_payload={
                    "problem_description": description.model_dump(mode="json"),
                    "instance": instance_payload,
                    "solver_source": solver_source,
                    "error_analysis": analysis.model_dump(mode="json"),
                    "failure_reason": failure_reason,
                },
            )
            solver_source = revision.source.rstrip() + "\n"
            solver_path.write_text(solver_source, encoding="utf-8")

        return self._result(
            run_id=run_id,
            status="failed",
            description=description,
            description_revisions=description_revisions,
            stage_revisions=stage_revisions,
            solver_path=solver_path,
            solver_source=solver_source,
            buffer_key=buffer_key,
            buffer_hit=buffer_hit,
            instance_path=instance_path,
            output_path=output_path,
            execution_attempts=attempts,
            execution=execution,
            external_validation=external,
            failure_reason=failure_reason or "solver failed without a diagnostic message",
        )

    def _describe(
        self,
        instance_payload: dict[str, Any],
        source_hash: str,
    ) -> tuple[UAVProblemDescription, int]:
        description_contract = {
            "problem_type": "STATIC_2D_UAV_PATH_PLANNING",
            "source_hash": source_hash,
            "required_inputs": list(REQUIRED_INPUTS),
            "required_constraints": list(REQUIRED_CONSTRAINTS),
            "output_requirement": "Return a feasible start-to-goal path as ordered 2D waypoints.",
            "objective_semantics": (
                "The global objective_weights length, collision, smoothness, risk, and waypoint "
                "multiply their respective total terms. Each risk_zones item weight is only a "
                "local multiplier inside the risk-exposure term; it is not the global risk "
                "objective weight."
            ),
        }
        description = self._call(
            role="generation_agent",
            action="generate_problem_description",
            template=DESCRIPTION_GENERATOR_V2,
            output_model=UAVProblemDescription,
            user_payload={
                "role": "description_generation",
                "source_hash": source_hash,
                "raw_instance": instance_payload,
                "description_contract": description_contract,
            },
        )
        revisions = 0
        while True:
            deterministic_issues = self.description_validator.validate(description, source_hash)
            judgment = self._call(
                role="judgment_agent",
                action="judge_problem_description",
                template=DESCRIPTION_JUDGE_V2,
                output_model=DescriptionJudgment,
                user_payload={
                    "role": "description_judgment",
                    "source_hash": source_hash,
                    "raw_instance": instance_payload,
                    "description": description.model_dump(mode="json"),
                    "deterministic_issues": deterministic_issues,
                    "description_contract": description_contract,
                },
            )
            issues = [*deterministic_issues, *judgment.issues]
            if judgment.approved and not issues:
                return description, revisions
            if revisions >= self.limits.max_description_revisions:
                raise RuntimeError(
                    "problem description did not converge within budget: " + "; ".join(issues)
                )
            revisions += 1
            description = self._call(
                role="revision_agent",
                action="revise_problem_description",
                template=DESCRIPTION_REVISION_V2,
                output_model=UAVProblemDescription,
                user_payload={
                    "role": "description_revision",
                    "source_hash": source_hash,
                    "raw_instance": instance_payload,
                    "previous_description": description.model_dump(mode="json"),
                    "judgment": judgment.model_dump(mode="json"),
                    "deterministic_issues": deterministic_issues,
                    "description_contract": description_contract,
                },
            )

    def _generate_solver(
        self,
        description: UAVProblemDescription,
        instance_payload: dict[str, Any],
    ) -> tuple[str, dict[str, int]]:
        accumulated: list[str] = []
        existing_functions: set[str] = set()
        revisions_by_stage: dict[str, int] = {}
        solver_header_contract = {
            "preimported_modules": ["argparse", "heapq", "json", "math", "random"],
            "integration": (
                "The fixed header and all approved stage fragments are concatenated into one "
                "Python module in stage order."
            ),
        }
        trusted_preconditions = [
            "The instance schema and field types are checked by read_problem.",
            "Map width and height are finite and positive.",
            "Start and goal are finite, in bounds, satisfy obstacle safety clearance, and differ.",
            "Benchmark instances have at least one collision-free start-to-goal route.",
            "grid_resolution is positive and max_waypoints is at least two.",
        ]
        available_interfaces: list[dict[str, str]] = []
        for contract in STAGE_CONTRACTS:
            stage_interfaces = [dict(item) for item in available_interfaces]
            revisions = 0
            draft = self._call(
                role="generation_agent",
                action=f"generate_{contract.name}",
                template=CODE_GENERATOR_V4,
                output_model=CodeStageDraft,
                user_payload={
                    "role": "code_generation",
                    "function_name": contract.name,
                    "stage_id": contract.name,
                    "required_function_name": contract.primary_function,
                    "signature": contract.signature,
                    "purpose": contract.purpose,
                    "stage_requirements": self._stage_requirements(contract.name),
                    "problem_description": description.model_dump(mode="json"),
                    "raw_instance": instance_payload,
                    "available_interfaces": stage_interfaces,
                    "solver_header_contract": solver_header_contract,
                    "trusted_preconditions": trusted_preconditions,
                },
            )
            while True:
                deterministic_issues = []
                if draft.function_name != contract.name:
                    deterministic_issues.append(
                        f"draft stage {draft.function_name} does not match {contract.name}"
                    )
                deterministic_issues.extend(
                    self.stage_validator.validate(contract, draft.source, existing_functions)
                )
                judgment = self._call(
                    role="judgment_agent",
                    action=f"judge_{contract.name}",
                    template=CODE_JUDGE_V4,
                    output_model=CodeJudgment,
                    user_payload={
                        "role": "code_judgment",
                        "function_name": contract.name,
                        "stage_id": contract.name,
                        "required_function_name": contract.primary_function,
                        "signature": contract.signature,
                        "purpose": contract.purpose,
                        "stage_requirements": self._stage_requirements(contract.name),
                        "problem_description": description.model_dump(mode="json"),
                        "available_interfaces": stage_interfaces,
                        "solver_header_contract": solver_header_contract,
                        "trusted_preconditions": trusted_preconditions,
                        "stage_source": draft.source,
                        "deterministic_issues": deterministic_issues,
                    },
                )
                issues = [*deterministic_issues, *judgment.issues]
                if judgment.approved and not issues:
                    break
                if revisions >= self.limits.max_code_revisions_per_stage:
                    raise RuntimeError(
                        f"code stage {contract.name} did not converge within budget: "
                        + "; ".join(issues)
                    )
                revisions += 1
                draft = self._call(
                    role="revision_agent",
                    action=f"revise_{contract.name}",
                    template=CODE_REVISION_V4,
                    output_model=CodeStageDraft,
                    user_payload={
                        "role": "code_revision",
                        "function_name": contract.name,
                        "stage_id": contract.name,
                        "required_function_name": contract.primary_function,
                        "signature": contract.signature,
                        "purpose": contract.purpose,
                        "stage_requirements": self._stage_requirements(contract.name),
                        "problem_description": description.model_dump(mode="json"),
                        "available_interfaces": stage_interfaces,
                        "solver_header_contract": solver_header_contract,
                        "trusted_preconditions": trusted_preconditions,
                        "previous_stage": draft.model_dump(mode="json"),
                        "judgment": judgment.model_dump(mode="json"),
                        "deterministic_issues": deterministic_issues,
                    },
                )
            accumulated.append(draft.source.strip())
            tree = ast.parse(draft.source)
            stage_functions = [
                node for node in tree.body if isinstance(node, ast.FunctionDef)
            ]
            existing_functions.update(node.name for node in stage_functions)
            for node in stage_functions:
                available_interfaces.append(
                    {
                        "stage_id": contract.name,
                        "function_name": node.name,
                        "signature": self._function_signature(node),
                        "purpose": (
                            contract.purpose
                            if node.name == contract.primary_function
                            else f"Approved helper declared by the {contract.name} stage."
                        ),
                    }
                )
            revisions_by_stage[contract.name] = revisions
        return self._assemble(accumulated), revisions_by_stage

    @staticmethod
    def _function_signature(node: ast.FunctionDef) -> str:
        rendered = ast.unparse(node)
        return rendered.splitlines()[0].strip()

    @staticmethod
    def _stage_requirements(stage_name: str) -> list[str]:
        common = [
            "Use only names provided by the solver header, function arguments, local variables, "
            "and available_interfaces.",
            "Keep all loops bounded by finite input sizes, iteration limits, or evaluation limits.",
            "Never use global or nonlocal; shared counters live only in problem['_metrics'].",
        ]
        counter_requirements = {
            "read_problem": [
                "The raw instance has top-level environment, objective_weights, grid_resolution, "
                "and max_waypoints. The random seed is environment.seed, not a top-level field.",
                "Preserve every obstacle kind: circle has kind/center/radius; rectangle has "
                "kind/min_x/min_y/max_x/max_y. Keep risk_zones separate from obstacles.",
            ],
            "geometry": [
                "At the beginning of every segment_collision_free call, if problem contains "
                "a _metrics dict, increment its collision_checks integer exactly once.",
                "Hard obstacles include both circles and axis-aligned rectangles. Enforce safety "
                "clearance continuously along the entire segment for both kinds.",
                "Risk zones are soft objective regions, NEVER hard obstacles; do not inspect "
                "risk_zones in segment_collision_free.",
                "For rectangle slab intersection, a parallel coordinate outside its slab "
                "means this rectangle is missed; a parallel coordinate inside its slab must "
                "continue to the other coordinate test. Never skip that remaining test.",
            ],
            "cost": [
                "At the beginning of every path_cost call, if problem contains a _metrics dict, "
                "increment its objective_evaluations integer exactly once.",
                "Match the trusted objective: polyline length; collision penalty by colliding "
                "segments; sum of (turn_angle/pi)^2; weighted segment length inside rectangular "
                "risk zones; and max(0, len(path)-2) intermediate waypoints. Apply global weights "
                "exactly once. Risk zones are not collisions.",
                "For risk rectangle clipping, vertical and horizontal segments inside the "
                "parallel slab must still be clipped by the other coordinate. Only a truly "
                "zero-length segment contributes zero risk length unconditionally.",
            ],
            "initial": [
                "For deterministic grid search, increment problem['_metrics']['node_expansions'] "
                "once for every node removed from the search frontier when _metrics exists.",
                "A neighbor edge is admissible if and only if segment_collision_free(current, "
                "neighbor, problem) returns True. Skip it when that call returns False; never "
                "invert this condition.",
                "Return a feasible path when the trusted connected-instance precondition holds; "
                "raise a clear RuntimeError instead of returning a known-infeasible fallback.",
            ],
            "repair": [
                "Do not call path_cost: repair has no evaluation-budget argument and therefore "
                "must not consume hidden objective evaluations. Main evaluates once afterward.",
                "Check every candidate segment with segment_collision_free and roll back to a "
                "copied original path if any hard constraint fails.",
                "If repair performs a graph/grid search, increment node_expansions once per node "
                "removed from its frontier; otherwise do not fabricate expansion counts.",
            ],
            "destroy": [
                "Use the exact destroy(path, rng) signature. Preserve start and goal and "
                "return at least two points. Do not reference problem or call geometry APIs "
                "that require a problem argument; repair handles feasibility afterward.",
            ],
            "validation": [
                "After endpoint, finiteness, bounds, and waypoint checks, call "
                "segment_collision_free for every consecutive path segment. Risk-zone exposure "
                "must not make a path infeasible.",
            ],
        }
        if stage_name != "main":
            return [*common, *counter_requirements.get(stage_name, [])]
        return [
            *common,
            "Return a top-level function defined exactly as def main(): with no parameters. "
            "Do not return only an if __name__ guard; the trusted assembler adds that guard.",
            "Accept CLI flags --path, --iteration, --output, --seed, and --max-evaluations.",
            "Read the instance with read_problem and use the explicit seed for deterministic RNG.",
            "Immediately after reading, initialize problem['_metrics'] with integer zeros for "
            "objective_evaluations, collision_checks, and node_expansions. Earlier functions "
            "update these counters; main must report them and must not guess or duplicate them.",
            "Use direct sequential statements and ordinary local variables; do not define nested "
            "functions or use global/nonlocal counter closures.",
            "Never increment or assign an individual metric in main: path_cost, "
            "segment_collision_free, and search functions already maintain the counters.",
            "Never perform more objective evaluations than --max-evaluations; include the initial "
            "path_cost call in that count and check the counter before every later path_cost call.",
            "Write one JSON object to --output and print the same object to stdout.",
            "On success include status='success', path, initial_cost, best_cost, iterations, seed, "
            "objective_evaluations, collision_checks, and node_expansions.",
            "Call validate_path before accepting the initial or candidate path. If initial_path is "
            "infeasible, raise RuntimeError; never output None or an infeasible fallback as success.",
            "Keep path endpoints fixed, retain a copied best feasible path, validate it once more "
            "before output, and return the best feasible path found so far.",
        ]

    @staticmethod
    def _assemble(stages: list[str]) -> str:
        suffix = "\n\n\n".join(stage.strip() for stage in stages if stage.strip())
        source = SOLVER_HEADER + ("\n\n\n" + suffix if suffix else "") + "\n"
        tree = ast.parse(source)
        has_guard = any(
            isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Name) and child.id == "__name__"
                for child in ast.walk(node.test)
            )
            for node in tree.body
        )
        if not has_guard:
            source = source.rstrip() + "\n\n\nif __name__ == '__main__':\n    main()\n"
        return source

    def _call(
        self,
        *,
        role: str,
        action: str,
        template: PromptTemplate,
        output_model: type[OutputModel],
        user_payload: dict[str, Any],
    ) -> OutputModel:
        try:
            result = self.provider.generate_structured(
                system_prompt=template.system_prompt,
                user_payload=user_payload,
                output_model=output_model,
                config=self.call_config,
                prompt_version=template.version,
                prompt_hash=template.prompt_hash,
            )
        except LLMProviderError as exc:
            record = exc.record
            self._events.append(
                RoleEvent(
                    sequence=len(self._events) + 1,
                    role=role,  # type: ignore[arg-type]
                    action=action,
                    prompt_version=template.version,
                    output_model=output_model.__name__,
                    provider_call_id=None if record is None else record.call_id,
                    status="failed",
                    error=str(exc)[:2_000],
                )
            )
            raise
        record = self.provider.call_records[-1] if self.provider.call_records else None
        self._events.append(
            RoleEvent(
                sequence=len(self._events) + 1,
                role=role,  # type: ignore[arg-type]
                action=action,
                prompt_version=template.version,
                output_model=output_model.__name__,
                provider_call_id=None if record is None else record.call_id,
                status="succeeded",
            )
        )
        return result

    def _result(
        self,
        *,
        run_id: str,
        status: str,
        description: UAVProblemDescription,
        description_revisions: int,
        stage_revisions: dict[str, int],
        solver_path: Path,
        solver_source: str,
        buffer_key: str,
        buffer_hit: bool,
        instance_path: Path,
        output_path: Path,
        execution_attempts: int,
        execution: Any = None,
        external_validation: Any = None,
        failure_reason: str | None = None,
    ) -> AFLUAVRunResult:
        result = AFLUAVRunResult(
            run_id=run_id,
            status=status,  # type: ignore[arg-type]
            problem_description=description,
            description_revisions=description_revisions,
            code_stage_revisions=stage_revisions,
            solver_path=str(solver_path.resolve()),
            solver_hash=stable_hash({"source": solver_source}),
            buffer_key=buffer_key,
            buffer_hit=buffer_hit,
            instance_path=str(instance_path.resolve()),
            output_path=str(output_path.resolve()),
            execution_attempts=execution_attempts,
            execution=execution,
            external_validation=external_validation,
            role_events=list(self._events),
            provider_calls=[
                record.model_dump(mode="json") for record in self.provider.call_records
            ],
            failure_reason=failure_reason,
        )
        if (
            result.status == "success"
            and result.external_validation is not None
            and result.external_validation.passed
            and self.solver_buffer is not None
        ):
            self.solver_buffer.store(
                buffer_key,
                solver_source,
                metadata={
                    "problem_type": description.problem_type,
                    "run_id": run_id,
                    "solver_hash": result.solver_hash,
                },
            )
        return result

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")


__all__ = ["AFLUAVCoordinator"]
