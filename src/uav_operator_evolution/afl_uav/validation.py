"""Deterministic contract checks that cannot be overridden by AFL agents."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..environment.environment import Environment2D
from ..path.evaluator import PathEvaluator
from ..path.models import copy_and_validate_path
from .models import ExternalValidation, FunctionName, UAVProblemDescription


REQUIRED_INPUTS = (
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
)

REQUIRED_CONSTRAINTS = (
    "endpoint_preservation",
    "map_bounds",
    "obstacle_clearance",
    "finite_coordinates",
    "waypoint_limit",
)


@dataclass(frozen=True, slots=True)
class StageContract:
    name: FunctionName
    primary_function: str
    signature: str
    purpose: str


STAGE_CONTRACTS: tuple[StageContract, ...] = (
    StageContract(
        "read_problem",
        "read_problem",
        "read_problem(path: str) -> dict",
        "Parse and validate the complete AFL-UAV JSON instance without fabricating fields.",
    ),
    StageContract(
        "geometry",
        "segment_collision_free",
        "segment_collision_free(start, end, problem) -> bool",
        "Provide bounded continuous collision checks for circles and rectangles.",
    ),
    StageContract(
        "cost",
        "path_cost",
        "path_cost(path, problem) -> float",
        "Evaluate length, collision, smoothness, risk, and waypoint terms.",
    ),
    StageContract(
        "initial",
        "initial_path",
        "initial_path(problem) -> list[list[float]]",
        "Construct a feasible start-to-goal path using a deterministic grid search.",
    ),
    StageContract(
        "destroy",
        "destroy",
        "destroy(path, rng) -> list[list[float]]",
        "Remove a bounded internal path segment while preserving endpoints.",
    ),
    StageContract(
        "repair",
        "repair",
        "repair(original, candidate, problem, rng) -> list[list[float]]",
        "Return a feasible repaired candidate or safely roll back to the original path.",
    ),
    StageContract(
        "validation",
        "validate_path",
        "validate_path(path, problem) -> bool",
        "Check endpoints, finite coordinates, bounds, waypoint limit, and clearance.",
    ),
    StageContract(
        "main",
        "main",
        "main() -> None",
        "Parse CLI arguments, obey seed/time/evaluation contracts, improve the path, "
        "and report path plus objective, collision, and expansion counters as JSON.",
    ),
)

EXPECTED_PARAMETERS: dict[str, list[str]] = {
    "read_problem": ["path"],
    "segment_collision_free": ["start", "end", "problem"],
    "path_cost": ["path", "problem"],
    "initial_path": ["problem"],
    "destroy": ["path", "rng"],
    "repair": ["original", "candidate", "problem", "rng"],
    "validate_path": ["path", "problem"],
    "main": [],
}


class TaskDescriptionValidator:
    """Check the typed D(G) representation against non-negotiable UAV semantics."""

    def validate(self, description: UAVProblemDescription, source_hash: str) -> list[str]:
        issues: list[str] = []
        if description.source_hash != source_hash:
            issues.append("source_hash does not match the raw solver instance")
        if description.problem_type != "STATIC_2D_UAV_PATH_PLANNING":
            issues.append("problem_type must be STATIC_2D_UAV_PATH_PLANNING")
        constraint_ids = {item.constraint_id for item in description.constraints}
        for required in REQUIRED_CONSTRAINTS:
            if required not in constraint_ids:
                issues.append(f"missing required constraint: {required}")
        inputs = set(description.inputs)
        for required in REQUIRED_INPUTS:
            if required not in inputs:
                issues.append(f"missing required input: {required}")
        unknown = sorted(inputs - set(REQUIRED_INPUTS))
        if unknown:
            issues.append("description invents unavailable inputs: " + ", ".join(unknown))
        output = description.output.lower()
        if "path" not in output or "feasible" not in output:
            issues.append("output must require a feasible path")
        objective = description.objective.lower()
        for term in ("length", "collision", "smoothness", "risk", "waypoint"):
            if term not in objective:
                issues.append(f"objective omits the {term} term")
        return issues


class CodeStageValidator:
    """Validate one generated function stage before it enters accumulated source."""

    def validate(
        self,
        contract: StageContract,
        source: str,
        existing_functions: set[str],
    ) -> list[str]:
        issues: list[str] = []
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return [f"syntax error: {exc.msg} at line {exc.lineno}"]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        if imports:
            issues.append("stage source must not contain imports")
        functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        if contract.primary_function not in functions:
            issues.append(f"stage must define {contract.primary_function}")
        else:
            primary = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == contract.primary_function
            )
            actual_parameters = [item.arg for item in primary.args.args]
            if (
                actual_parameters != EXPECTED_PARAMETERS[contract.primary_function]
                or primary.args.posonlyargs
                or primary.args.kwonlyargs
                or primary.args.vararg is not None
                or primary.args.kwarg is not None
            ):
                issues.append(
                    f"{contract.primary_function} must use exact parameters "
                    f"{EXPECTED_PARAMETERS[contract.primary_function]}"
                )
        duplicates = sorted(existing_functions.intersection(functions))
        if duplicates:
            issues.append("stage redefines accumulated functions: " + ", ".join(duplicates))
        if contract.name != "main":
            top_level_exec = [
                node
                for node in tree.body
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Expr))
            ]
            if top_level_exec:
                issues.append("non-main stages may only define functions")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                issues.append("global and nonlocal statements are forbidden")
                break
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        string_literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        if contract.name == "read_problem":
            preserves_raw_payload = any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "problem"
                    for target in node.targets
                )
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "json"
                and node.value.func.attr == "load"
                for node in ast.walk(tree)
            )
            preserves_raw_obstacles = any(
                (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value == "obstacles"
                )
                or (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and bool(node.args)
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "obstacles"
                )
                for node in ast.walk(tree)
            )
            if (
                not preserves_raw_payload
                and not preserves_raw_obstacles
                and not {"circle", "rectangle"}.issubset(string_literals)
            ):
                issues.append("read_problem must preserve both circle and rectangle obstacles")
            seed_from_environment = any(
                (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in {"env", "environment"}
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value == "seed"
                )
                or (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {"env", "environment"}
                    and node.func.attr == "get"
                    and bool(node.args)
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "seed"
                )
                for node in ast.walk(tree)
            )
            if not preserves_raw_payload and not seed_from_environment:
                issues.append("read_problem must obtain seed from the environment object")
        elif contract.name == "geometry":
            if "risk_zones" in string_literals:
                issues.append("risk zones are soft costs and must not be treated as obstacles")
            if not {"circle", "rectangle"}.issubset(string_literals):
                issues.append("geometry must handle both circle and rectangle obstacles")
            if "collision_checks" not in string_literals:
                issues.append("geometry must update the collision_checks metric")
        elif contract.name == "cost":
            if "objective_evaluations" not in string_literals:
                issues.append("path_cost must update the objective_evaluations metric")
            if "segment_collision_free" not in called_names:
                issues.append(
                    "path_cost collision tests must use the counted collision checker"
                )
        elif contract.name == "initial":
            if "node_expansions" not in string_literals:
                issues.append("initial_path must update the node_expansions metric")
        elif contract.name == "repair":
            if "path_cost" in called_names:
                issues.append("repair must not consume hidden objective evaluations")
            if "segment_collision_free" not in called_names:
                issues.append("repair must verify candidate segments continuously")
        elif contract.name == "validation":
            if "segment_collision_free" not in called_names:
                issues.append("validate_path must verify every segment continuously")
        elif contract.name == "main":
            nested_functions = [
                node.name
                for top_level in tree.body
                if isinstance(top_level, ast.FunctionDef)
                for node in ast.walk(top_level)
                if isinstance(node, ast.FunctionDef) and node is not top_level
            ]
            if nested_functions:
                issues.append("main must not define nested helper functions")
            counter_names = {"objective_evaluations", "collision_checks", "node_expansions"}
            for node in ast.walk(tree):
                target: ast.expr | None = None
                if isinstance(node, ast.AugAssign):
                    target = node.target
                elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                if target is None:
                    continue
                target_literals = {
                    child.value
                    for child in ast.walk(target)
                    if isinstance(child, ast.Constant) and isinstance(child.value, str)
                }
                if target_literals.intersection(counter_names):
                    issues.append(
                        "main must not mutate individual counters maintained by worker functions"
                    )
                    break
        return issues


class GeneratedCodePolicy:
    """Conservative AST policy; this is a guardrail, not an OS sandbox."""

    allowed_imports = {"argparse", "heapq", "json", "math", "random"}
    banned_calls = {"eval", "exec", "compile", "__import__", "input", "breakpoint"}
    banned_names = {"os", "subprocess", "socket", "pathlib", "shutil", "requests", "sys"}

    def validate(self, source: str, *, max_source_chars: int) -> list[str]:
        if len(source) > max_source_chars:
            return [f"source exceeds {max_source_chars} characters"]
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return [f"syntax error: {exc.msg} at line {exc.lineno}"]
        issues: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root not in self.allowed_imports:
                        issues.append(f"import is not allowed: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                issues.append("from-import statements are forbidden")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.banned_calls:
                    issues.append(f"call is forbidden: {node.func.id}")
            elif isinstance(node, ast.Name) and node.id in self.banned_names:
                issues.append(f"name is forbidden: {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                issues.append(f"dunder attribute access is forbidden: {node.attr}")
        return sorted(set(issues))


def validate_complete_solver_source(
    source: str,
    *,
    max_source_chars: int = 500_000,
) -> list[str]:
    """Apply policy and every primary stage contract to a complete candidate."""

    issues = GeneratedCodePolicy().validate(source, max_source_chars=max_source_chars)
    if issues:
        return issues
    tree = ast.parse(source)
    top_level_functions: dict[str, list[ast.FunctionDef]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            top_level_functions.setdefault(node.name, []).append(node)
    allowed_runtime_names = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "hash",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "open",
        "print",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "ValueError",
        "RuntimeError",
    }
    defined_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    undefined_calls = sorted(
        {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id not in defined_names
            and node.func.id not in allowed_runtime_names
        }
    )
    if undefined_calls:
        issues.append(
            "complete solver calls undefined functions: " + ", ".join(undefined_calls)
        )
    stage_validator = CodeStageValidator()
    for contract in STAGE_CONTRACTS:
        definitions = top_level_functions.get(contract.primary_function, [])
        if len(definitions) != 1:
            issues.append(
                f"complete solver must define {contract.primary_function} exactly once"
            )
            continue
        function = definitions[0]
        actual_parameters = [item.arg for item in function.args.args]
        if (
            actual_parameters != EXPECTED_PARAMETERS[contract.primary_function]
            or function.args.posonlyargs
            or function.args.kwonlyargs
            or function.args.vararg is not None
            or function.args.kwarg is not None
        ):
            issues.append(
                f"{contract.primary_function} must use exact parameters "
                f"{EXPECTED_PARAMETERS[contract.primary_function]}"
            )
        stage_source = ast.get_source_segment(source, definitions[0]) or ast.unparse(
            definitions[0]
        )
        issues.extend(stage_validator.validate(contract, stage_source, set()))
    has_main_guard = any(
        isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Name) and child.id == "__name__"
            for child in ast.walk(node.test)
        )
        for node in tree.body
    )
    if not has_main_guard:
        issues.append("complete solver must contain a __main__ CLI guard")
    return sorted(set(issues))


def validate_solver_output(
    payload: Mapping[str, Any] | None,
    environment: Environment2D,
    evaluator: PathEvaluator,
    *,
    max_waypoints: int,
) -> ExternalValidation:
    """Independently verify generated output using the trusted project evaluator."""

    failures: list[str] = []
    if payload is None:
        return ExternalValidation(passed=False, failures=["solver produced no JSON payload"])
    if payload.get("status") != "success":
        failures.append("solver status is not success")
    raw_path = payload.get("path")
    if not isinstance(raw_path, Sequence) or isinstance(raw_path, (str, bytes)):
        return ExternalValidation(passed=False, failures=[*failures, "path is missing or invalid"])
    try:
        path = copy_and_validate_path(raw_path)
    except (TypeError, ValueError) as exc:
        return ExternalValidation(passed=False, failures=[*failures, f"invalid path: {exc}"])
    if len(path) > max_waypoints:
        failures.append(f"path has {len(path)} waypoints; maximum is {max_waypoints}")
    if not all(math.isfinite(value) for point in path for value in point):
        failures.append("path contains non-finite coordinates")
    try:
        evaluation = evaluator.evaluate(path, environment)
    except (TypeError, ValueError) as exc:
        return ExternalValidation(
            passed=False,
            path=path,
            failures=[*failures, f"trusted evaluator rejected path: {exc}"],
        )
    if not evaluation.feasible:
        failures.append("trusted evaluator reports an infeasible path")

    def optional_number(name: str) -> float | None:
        value = payload.get(name)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            failures.append(f"{name} is not numeric")
            return None
        if not math.isfinite(number):
            failures.append(f"{name} is not finite")
            return None
        return number

    return ExternalValidation(
        passed=not failures,
        path=path,
        evaluation=evaluation,
        failures=failures,
        reported_initial_cost=optional_number("initial_cost"),
        reported_best_cost=optional_number("best_cost"),
    )


__all__ = [
    "CodeStageValidator",
    "GeneratedCodePolicy",
    "REQUIRED_CONSTRAINTS",
    "REQUIRED_INPUTS",
    "STAGE_CONTRACTS",
    "StageContract",
    "TaskDescriptionValidator",
    "validate_solver_output",
    "validate_complete_solver_source",
]
