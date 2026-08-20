"""Strict data models for the AFL-style UAV solver-generation experiment."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..environment.environment import Environment2D
from ..path.models import EvaluationResult, ObjectiveWeights


class AFLUAVModel(BaseModel):
    """Base model for replayable, JSON-native AFL-UAV artifacts."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ConstraintDefinition(AFLUAVModel):
    constraint_id: str = Field(min_length=1, max_length=100)
    # This is a human-readable artifact label, not an executable identifier.
    # Descriptive labels such as ``endpoint_preservation`` are legitimate and
    # slightly exceed the original arbitrary 20-character bound.
    abbreviation: str = Field(min_length=1, max_length=32)
    explanation: str = Field(min_length=1, max_length=1_000)


class UAVProblemDescription(AFLUAVModel):
    """Typed equivalent of the paper's D(G)={P,S,K,X,Y,Z}."""

    problem_type: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4_000)
    constraints: list[ConstraintDefinition] = Field(min_length=1, max_length=32)
    inputs: list[str] = Field(min_length=1, max_length=32)
    output: str = Field(min_length=1, max_length=2_000)
    objective: str = Field(min_length=1, max_length=2_000)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def unique_contract_entries(self) -> "UAVProblemDescription":
        constraint_ids = [item.constraint_id for item in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint identifiers must be unique")
        if len(self.inputs) != len(set(self.inputs)):
            raise ValueError("input names must be unique")
        return self


class DescriptionJudgment(AFLUAVModel):
    approved: bool
    explanation: str = Field(min_length=1, max_length=4_000)
    issues: list[str] = Field(default_factory=list, max_length=32)
    required_revisions: list[str] = Field(default_factory=list, max_length=32)


FunctionName = Literal[
    "read_problem",
    "geometry",
    "cost",
    "initial",
    "destroy",
    "repair",
    "validation",
    "main",
]


class CodeStageDraft(AFLUAVModel):
    function_name: FunctionName
    source: str = Field(min_length=1, max_length=80_000)
    rationale: str = Field(min_length=1, max_length=2_000)


class CodeJudgment(AFLUAVModel):
    approved: bool
    explanation: str = Field(min_length=1, max_length=4_000)
    issues: list[str] = Field(default_factory=list, max_length=32)
    required_revisions: list[str] = Field(default_factory=list, max_length=32)


class ErrorAnalysis(AFLUAVModel):
    cause: str = Field(min_length=1, max_length=4_000)
    suggestions: list[str] = Field(min_length=1, max_length=32)


class SolverRevision(AFLUAVModel):
    source: str = Field(min_length=1, max_length=500_000)
    changes: list[str] = Field(min_length=1, max_length=32)


PublicFunctionName = Literal[
    "read_problem",
    "segment_collision_free",
    "path_cost",
    "initial_path",
    "destroy",
    "repair",
    "validate_path",
    "main",
]


class FunctionSourceReplacement(AFLUAVModel):
    """One complete top-level function emitted by the Revision Agent."""

    function_name: PublicFunctionName
    source: str = Field(min_length=1, max_length=100_000)
    reason: str = Field(min_length=1, max_length=2_000)


class SolverPatch(AFLUAVModel):
    """A bounded set of AST-addressed replacements for a saved solver."""

    functions: list[FunctionSourceReplacement] = Field(default_factory=list, max_length=8)
    ensure_main_guard: bool = False


class UAVSolverInstance(AFLUAVModel):
    schema_version: Literal["afl-uav-instance-v1"] = "afl-uav-instance-v1"
    environment: Environment2D
    objective_weights: ObjectiveWeights
    grid_resolution: float = Field(default=4.0, gt=0.0, le=100.0)
    max_waypoints: int = Field(default=128, ge=2, le=10_000)


class RoleEvent(AFLUAVModel):
    sequence: int = Field(ge=1)
    role: Literal[
        "generation_agent",
        "judgment_agent",
        "revision_agent",
        "error_analysis_agent",
    ]
    action: str = Field(min_length=1, max_length=200)
    prompt_version: str = Field(min_length=1, max_length=200)
    output_model: str = Field(min_length=1, max_length=200)
    provider_call_id: str | None = None
    status: Literal["succeeded", "failed"]
    error: str | None = Field(default=None, max_length=2_000)


class ExecutionReport(AFLUAVModel):
    status: Literal[
        "success",
        "policy_rejected",
        "runtime_error",
        "timeout",
        "output_error",
    ]
    return_code: int | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    stdout: str = Field(default="", max_length=20_000)
    stderr: str = Field(default="", max_length=20_000)
    output_payload: dict[str, Any] | None = None
    error: str | None = Field(default=None, max_length=4_000)


class ExternalValidation(AFLUAVModel):
    passed: bool
    path: list[tuple[float, float]] | None = None
    evaluation: EvaluationResult | None = None
    failures: list[str] = Field(default_factory=list, max_length=64)
    reported_initial_cost: float | None = None
    reported_best_cost: float | None = None


class AFLUAVLimits(AFLUAVModel):
    max_description_revisions: int = Field(default=2, ge=0, le=10)
    max_code_revisions_per_stage: int = Field(default=2, ge=0, le=10)
    max_execution_revisions: int = Field(default=1, ge=0, le=5)
    execution_timeout_seconds: float = Field(default=20.0, gt=0.0, le=300.0)
    max_source_chars: int = Field(default=500_000, ge=1_000, le=2_000_000)


class AFLUAVRunResult(AFLUAVModel):
    run_id: str = Field(min_length=1, max_length=300)
    status: Literal["success", "generated_only", "failed"]
    problem_description: UAVProblemDescription
    description_revisions: int = Field(ge=0)
    code_stage_revisions: dict[str, int]
    solver_path: str
    solver_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    buffer_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    buffer_hit: bool
    instance_path: str
    output_path: str
    execution_attempts: int = Field(ge=0)
    execution: ExecutionReport | None = None
    external_validation: ExternalValidation | None = None
    role_events: list[RoleEvent]
    provider_calls: list[dict[str, Any]]
    failure_reason: str | None = Field(default=None, max_length=4_000)
    upstream_repository: str = "https://github.com/ZHANG-NI/AFL"
    upstream_commit: str = "602c6be26f98204e514adef982577a9d5d5c215f"
    upstream_license: str = "No license file observed at the audited commit"


__all__ = [
    "AFLUAVLimits",
    "AFLUAVRunResult",
    "CodeJudgment",
    "CodeStageDraft",
    "ConstraintDefinition",
    "DescriptionJudgment",
    "ErrorAnalysis",
    "ExecutionReport",
    "ExternalValidation",
    "FunctionName",
    "RoleEvent",
    "SolverPatch",
    "SolverRevision",
    "FunctionSourceReplacement",
    "PublicFunctionName",
    "UAVProblemDescription",
    "UAVSolverInstance",
]
