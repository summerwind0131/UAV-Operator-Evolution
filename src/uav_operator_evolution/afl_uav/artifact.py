"""Generate, approve, freeze, load, and audit reusable AFL-UAV solvers."""

from __future__ import annotations

import ast
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..agents.providers import (
    DeepSeekProvider,
    GeminiProvider,
    LLMCallConfig,
    LLMCallRecord,
    LLMProviderError,
    MockLLMProvider,
    OpenAIProvider,
    ProviderName,
)
from ..config import ExperimentConfig
from ..environment.environment import Environment2D
from ..path.evaluator import PathEvaluator
from ..path.models import ObjectiveWeights
from ..reproducibility import canonical_json, stable_hash
from .coordinator import AFLUAVCoordinator
from .mock_solver import afl_uav_mock_factory
from .models import (
    AFLUAVLimits,
    CodeJudgment,
    RoleEvent,
    SolverPatch,
    SolverRevision,
    UAVProblemDescription,
    UAVSolverInstance,
)
from .prompts import AUDITED_SOLVER_JUDGE_V1, AUDITED_SOLVER_PATCH_V1
from .prompts import AUDITED_DESCRIPTION_REVISION_V1
from .runner import GeneratedSolverRunner
from .validation import (
    GeneratedCodePolicy,
    REQUIRED_CONSTRAINTS,
    REQUIRED_INPUTS,
    TaskDescriptionValidator,
    validate_complete_solver_source,
    validate_solver_output,
)


COUNTER_NAMES = ("objective_evaluations", "collision_checks", "node_expansions")
REAL_PROVIDERS = {"openai", "deepseek", "gemini"}
AFL_PROVIDER_MODELS = {
    "openai": "gpt-4.1-2025-04-14",
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-2.5-pro",
}
QUALIFICATION_MAP_IDS = (
    "train-000-sparse-c4b92a431b",
    "train-001-dense-b533794b47",
    "train-002-corridor-594730e0c7",
    "train-003-clustered-fdcee6fa2f",
    "train-004-rooms_maze-8a9244e3f3",
    "train-005-mixed-5dee276124",
)

POST_GENERATION_AUDIT_ISSUES = (
    "The complete source must end with an executable __main__ guard that calls main().",
    "destroy receives only path and rng; it must not reference an undefined problem variable "
    "or perform collision checks that require problem.",
    "Rectangle segment collision must process both slabs when dx or dy is zero; do not "
    "continue past an obstacle merely because a parallel coordinate is inside its slab.",
    "Risk-zone clipping must correctly count vertical and horizontal segments inside a "
    "rectangle, while assigning zero exposure only to truly zero-length segments.",
    "repair must reference only variables defined in its own signature and scope when "
    "splicing a repaired segment back into the candidate path.",
    "destroy and repair must preserve at least the start and goal and handle every short "
    "path without indexing errors.",
    "main must check remaining objective-evaluation budget immediately before every "
    "path_cost call, including the initial and final candidate evaluations.",
    "Initial-path simplification must explicitly validate every resulting segment, including "
    "the final segment to goal, and fail rather than return a colliding shortcut.",
    "After read_problem returns the flat problem contract, every later function must read "
    "width, height, start, goal, and obstacles from flat problem keys, never problem.environment.",
    "Do not call an undefined risk helper. Compute rectangular risk-segment intersection "
    "inside path_cost or use a helper that is actually defined in the approved source.",
    "The initial grid search must connect arbitrary non-grid start and goal coordinates to "
    "visible grid nodes; random terminals are not guaranteed to lie on grid_resolution.",
    "Collision cost is one unit for each colliding path segment before applying the global "
    "collision weight; never use the colliding segment's Euclidean length as that unit.",
    "A degenerate smoothness turn contributes the trusted penalty of one, not zero.",
    "Unknown obstacle kinds must fail closed in geometry, never be ignored.",
    "Parse and preserve both circle and rectangle obstacle schemas; use environment.seed.",
    "Risk zones are soft objective regions and must never be treated as hard obstacles.",
    "validate_path and repair must continuously check every path segment, not only waypoints.",
    "repair must not call path_cost or consume hidden objective evaluations.",
    "path_cost alone increments objective_evaluations exactly once per call; main must not "
    "increment worker-maintained counters.",
    "The internal path objective must match the trusted length, colliding-segment, squared "
    "normalized turn, segment risk-exposure, and intermediate-waypoint terms.",
    "A zero-length segment has exactly zero risk exposure and zero added path length.",
    "All internal hard-collision tests, including those in path_cost, must call the counted "
    "segment_collision_free interface rather than an uncounted duplicate helper.",
    "Path simplification must not append the goal twice or create consecutive duplicate points.",
    "Reject unknown obstacle kinds rather than silently dropping hard obstacles.",
)

FULL_SOLVER_CONTRACT = {
    "required_functions": [
        "read_problem",
        "segment_collision_free",
        "path_cost",
        "initial_path",
        "destroy",
        "repair",
        "validate_path",
        "main",
    ],
    "allowed_imports": ["argparse", "heapq", "json", "math", "random"],
    "cli_flags": ["--path", "--iteration", "--output", "--seed", "--max-evaluations"],
    "output_fields": [
        "status",
        "path",
        "initial_cost",
        "best_cost",
        "iterations",
        "seed",
        "objective_evaluations",
        "collision_checks",
        "node_expansions",
    ],
    "obstacle_schemas": {
        "circle": ["kind", "center", "radius"],
        "rectangle": ["kind", "min_x", "min_y", "max_x", "max_y"],
    },
    "objective_terms": {
        "length": "total polyline length",
        "collision": "one trusted collision penalty per colliding path segment",
        "smoothness": "sum of (unsigned turn angle / pi)^2; degenerate turn penalty is 1",
        "risk": "sum of zone-weighted segment length inside each rectangular risk zone",
        "waypoint": "max(0, len(path)-2) intermediate waypoints",
    },
    "risk_semantics": (
        "continuous weighted segment length inside zones; no waypoint risk term; soft cost, "
        "never collision; zero-length segments have zero risk"
    ),
    "counter_owners": {
        "path_cost": "objective_evaluations",
        "segment_collision_free": "collision_checks",
        "grid frontier pops and successful repair-tree insertions": "node_expansions",
    },
    "counter_semantics": (
        "Every segment_collision_free call is a counted collision check and is allowed from "
        "initial search, repair, validation, and path_cost. Collision checks do not consume "
        "the objective-evaluation budget."
    ),
}

AUTHORITATIVE_OBJECTIVE = (
    "Minimize the configured weighted sum of total polyline length, one collision penalty "
    "per colliding path segment, the smoothness sum of squared normalized unsigned turn angles, "
    "zone-weighted path-segment length inside rectangular risk zones, and the number of "
    "intermediate waypoints. Risk zones are soft costs and never hard obstacles."
)


class CandidateUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_calls: int = Field(ge=0)
    retries: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class HumanSourceReview(BaseModel):
    """Explicit, hash-bound human decision made before restricted execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["approved_for_restricted_qualification"]
    approved_source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str = Field(min_length=1, max_length=200)
    overrode_llm_judgment: bool
    llm_judgment_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rationale: list[str] = Field(min_length=1, max_length=32)


class AFLSolverCandidate(BaseModel):
    """Generated source plus its complete, non-executable model audit trail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["afl-uav-solver-candidate-v1"] = (
        "afl-uav-solver-candidate-v1"
    )
    candidate_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    solver_filename: str = "candidate_solver.py"
    solver_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: ProviderName
    model: str | None = None
    provider_sdk_versions: dict[str, str]
    generated_from_split: Literal["train"] = "train"
    generated_from_map_id: str
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    objective_weights: ObjectiveWeights
    grid_resolution: float = Field(gt=0)
    max_waypoints: int = Field(ge=2)
    problem_description: UAVProblemDescription
    description_revisions: int = Field(ge=0)
    code_stage_revisions: dict[str, int]
    role_events: list[RoleEvent]
    provider_calls: list[dict[str, Any]]
    usage: CandidateUsage
    upstream_repository: str
    upstream_commit: str
    human_review: HumanSourceReview | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "AFLSolverCandidate":
        if Path(self.solver_filename).name != self.solver_filename:
            raise ValueError("solver_filename must be a local candidate filename")
        if self.provider in REAL_PROVIDERS and not self.model:
            raise ValueError("real solver candidates require an explicit model")
        if self.provider in REAL_PROVIDERS and self.model != AFL_PROVIDER_MODELS[self.provider]:
            raise ValueError("real solver candidate does not use the fixed provider model")
        if self.usage.logical_calls != len(self.provider_calls):
            raise ValueError("candidate logical-call count does not match provider audit")
        if self.usage.logical_calls > 56 or self.usage.total_tokens > 250_000:
            raise ValueError("candidate exceeds the fixed LLM call or token budget")
        records = [LLMCallRecord.model_validate(item) for item in self.provider_calls]
        if records and any(record.provider != self.provider for record in records):
            raise ValueError("candidate provider audit contains a different provider")
        if any(record.attempts > 3 for record in records):
            raise ValueError("candidate provider audit exceeds the retry limit")
        if _usage_from_calls(self.provider_calls) != self.usage:
            raise ValueError("candidate normalized usage does not match provider audit")
        if (
            self.human_review is not None
            and self.human_review.approved_source_hash != self.solver_hash
        ):
            raise ValueError("human review is not bound to the candidate source hash")
        return self


class QualificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    map_id: str
    difficulty: str
    status: str
    passed: bool
    duration_ms: float = Field(ge=0)
    objective_evaluations: int = Field(ge=0)
    collision_checks: int = Field(ge=0)
    node_expansions: int = Field(ge=0)
    total_cost: float | None = None
    failures: list[str] = Field(default_factory=list)


class AFLSolverArtifact(BaseModel):
    """A generated solver frozen after hash approval and trusted qualification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "afl-uav-solver-artifact-v1", "afl-uav-solver-artifact-v2"
    ] = "afl-uav-solver-artifact-v2"
    artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    planner_name: Literal["afl_uav"] = "afl_uav"
    solver_filename: str = "frozen_solver.py"
    solver_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    solver_cli_contract: Literal["afl-uav-solver-cli-v2"] = (
        "afl-uav-solver-cli-v2"
    )
    provider: ProviderName
    model: str | None = None
    research_claim_eligible: bool = False
    generated_from_split: Literal["train"] = "train"
    generated_from_map_id: str
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    objective_weights: ObjectiveWeights
    grid_resolution: float = Field(gt=0)
    max_waypoints: int = Field(ge=2)
    problem_description: UAVProblemDescription
    role_events: list[RoleEvent]
    provider_calls: list[dict[str, Any]]
    generation_execution_attempts: int = Field(ge=0)
    contract_smoke_counters: dict[str, int]
    upstream_repository: str
    upstream_commit: str
    candidate_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    candidate_source_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    approved_source_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    provider_sdk_versions: dict[str, str] = Field(default_factory=dict)
    llm_usage: CandidateUsage | None = None
    description_revisions: int = Field(default=0, ge=0)
    code_stage_revisions: dict[str, int] = Field(default_factory=dict)
    qualification_results: list[QualificationResult] = Field(default_factory=list)
    human_review: HumanSourceReview | None = None

    @model_validator(mode="after")
    def validate_portable_contract(self) -> "AFLSolverArtifact":
        if Path(self.solver_filename).name != self.solver_filename:
            raise ValueError("solver_filename must be a local artifact filename")
        if self.provider == "mock" and self.research_claim_eligible:
            raise ValueError("mock artifacts cannot be research-claim eligible")
        if set(self.contract_smoke_counters) != set(COUNTER_NAMES):
            raise ValueError("contract smoke must contain all three benchmark counters")
        if any(value < 0 for value in self.contract_smoke_counters.values()):
            raise ValueError("contract smoke counters must be non-negative")
        if self.schema_version == "afl-uav-solver-artifact-v2":
            if (
                self.candidate_id is None
                or self.candidate_source_hash != self.solver_hash
                or self.approved_source_hash != self.solver_hash
            ):
                raise ValueError("v2 artifacts require an approved candidate source hash")
            if not self.qualification_results or not all(
                item.passed for item in self.qualification_results
            ):
                raise ValueError("v2 artifacts require successful Train qualification")
            if self.llm_usage is None:
                raise ValueError("v2 artifacts require normalized LLM usage")
            if (
                self.human_review is not None
                and self.human_review.approved_source_hash != self.solver_hash
            ):
                raise ValueError("artifact human review is not bound to solver hash")
        if self.research_claim_eligible:
            if self.provider not in REAL_PROVIDERS or not self.model:
                raise ValueError("claim-eligible artifacts require a real provider and model")
            if self.model != AFL_PROVIDER_MODELS[self.provider]:
                raise ValueError("claim-eligible artifact does not use the fixed model")
            if self.schema_version != "afl-uav-solver-artifact-v2":
                raise ValueError("only v2 artifacts can be research-claim eligible")
            if not self.provider_calls or any(
                call.get("status") != "success" for call in self.provider_calls
            ):
                raise ValueError("claim-eligible artifacts require successful call audit")
            records = [
                LLMCallRecord.model_validate(item) for item in self.provider_calls
            ]
            if any(
                record.provider != self.provider
                or not record.model
                or not record.response_id
                for record in records
            ):
                raise ValueError("claim-eligible artifacts require a complete call audit")
            if tuple(item.map_id for item in self.qualification_results) != (
                QUALIFICATION_MAP_IDS
            ):
                raise ValueError(
                    "claim-eligible artifacts require the six fixed Train maps"
                )
        return self


def extract_solver_counters(
    payload: dict[str, Any],
    *,
    max_evaluations: int,
) -> dict[str, int]:
    """Read strict non-negative counters from one generated solver output."""

    counters: dict[str, int] = {}
    for name in COUNTER_NAMES:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"generated solver did not report valid {name}")
        counters[name] = value
    if counters["objective_evaluations"] > max_evaluations:
        raise ValueError("generated solver exceeded the objective-evaluation limit")
    return counters


def _artifact_identity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["schema_version"] == "afl-uav-solver-artifact-v1":
        return {
            "schema_version": payload["schema_version"],
            "planner_name": payload["planner_name"],
            "solver_hash": payload["solver_hash"],
            "solver_cli_contract": payload["solver_cli_contract"],
            "provider": payload["provider"],
            "model": payload.get("model"),
            "generated_from_split": payload["generated_from_split"],
            "generated_from_map_id": payload["generated_from_map_id"],
            "config_hash": payload["config_hash"],
            "objective_weights": payload["objective_weights"],
            "grid_resolution": payload["grid_resolution"],
            "max_waypoints": payload["max_waypoints"],
            "problem_description": payload["problem_description"],
            "upstream_repository": payload["upstream_repository"],
            "upstream_commit": payload["upstream_commit"],
        }
    return {
        key: value
        for key, value in payload.items()
        if key != "artifact_id" and not (key == "human_review" and value is None)
    }


def _candidate_identity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "candidate_id" and not (key == "human_review" and value is None)
    }


def _usage_from_calls(calls: list[dict[str, Any]]) -> CandidateUsage:
    def token(call: dict[str, Any], name: str) -> int:
        usage = call.get("usage")
        if not isinstance(usage, dict):
            return 0
        value = usage.get(name, 0)
        return max(0, int(value)) if isinstance(value, (int, float)) else 0

    return CandidateUsage(
        logical_calls=len(calls),
        retries=sum(max(0, int(call.get("retry_count", 0))) for call in calls),
        input_tokens=sum(token(call, "input_tokens") for call in calls),
        output_tokens=sum(token(call, "output_tokens") for call in calls),
        total_tokens=sum(token(call, "total_tokens") for call in calls),
    )


def _sdk_versions(provider: ProviderName) -> dict[str, str]:
    packages = {
        "openai": ("openai",),
        "deepseek": ("openai",),
        "gemini": ("google-genai",),
        "mock": (),
    }[provider]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _make_provider(provider: ProviderName, model: str | None):
    if provider == "mock":
        return MockLLMProvider(factory=afl_uav_mock_factory)
    if not model:
        raise ValueError("real AFL-UAV candidate generation requires an explicit model")
    if model != AFL_PROVIDER_MODELS[provider]:
        raise ValueError(
            f"{provider} AFL-UAV arm requires fixed model {AFL_PROVIDER_MODELS[provider]}"
        )
    if provider == "openai":
        return OpenAIProvider(model=model, allow_legacy_environment=False)
    if provider == "deepseek":
        return DeepSeekProvider(model=model)
    if provider == "gemini":
        return GeminiProvider(model=model)
    raise ValueError(f"unsupported AFL-UAV provider: {provider}")


def save_solver_candidate(
    candidate: AFLSolverCandidate,
    source: str,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if stable_hash({"source": source}) != candidate.solver_hash:
        raise ValueError("solver source does not match candidate solver_hash")
    (destination / candidate.solver_filename).write_text(source, encoding="utf-8")
    manifest_path = destination / "candidate.json"
    manifest_path.write_text(
        canonical_json(candidate.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_solver_candidate(
    path: str | Path,
    *,
    require_complete_contract: bool = True,
) -> tuple[AFLSolverCandidate, str, Path]:
    source = Path(path)
    manifest_path = source / "candidate.json" if source.is_dir() else source
    candidate = AFLSolverCandidate.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    solver_path = manifest_path.parent / candidate.solver_filename
    solver_source = solver_path.read_text(encoding="utf-8")
    if stable_hash({"source": solver_source}) != candidate.solver_hash:
        raise ValueError("AFL-UAV candidate solver hash does not match source")
    expected_id = stable_hash(
        _candidate_identity_payload(candidate.model_dump(mode="json"))
    )
    if candidate.candidate_id != expected_id:
        raise ValueError("AFL-UAV candidate identity hash does not match contents")
    issues = GeneratedCodePolicy().validate(solver_source, max_source_chars=500_000)
    if require_complete_contract:
        issues.extend(validate_complete_solver_source(solver_source))
    if issues:
        raise ValueError("AFL-UAV candidate violates code policy: " + "; ".join(issues))
    return candidate, solver_source, solver_path


def adopt_rejected_source_after_human_review(
    base_candidate_path: str | Path,
    failed_audit_path: str | Path,
    output_dir: str | Path,
    *,
    approved_source_hash: str,
) -> tuple[AFLSolverCandidate, Path]:
    """Promote an LLM-revised source after an explicit delegated human override.

    This never executes source.  It is intentionally limited to a failed post-audit
    directory whose provider calls all succeeded and whose source passes the complete
    deterministic contract.  The rejected LLM judgment remains hash-linked in the
    candidate rather than being hidden.
    """

    base, _, _ = load_solver_candidate(
        base_candidate_path,
        require_complete_contract=False,
    )
    audit = Path(failed_audit_path)
    if audit.is_file():
        audit = audit.parent
    failure_path = audit / "candidate_failure.json"
    source_path = audit / "rejected_candidate_solver.py"
    description_path = audit / "last_problem_description.json"
    judgment_path = audit / "last_judgment.json"
    for path in (failure_path, source_path, description_path, judgment_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
    source = source_path.read_text(encoding="utf-8")
    source_hash = stable_hash({"source": source})
    if approved_source_hash != source_hash:
        raise ValueError("human-approved hash does not match rejected audit source")
    if failure_payload.get("base_candidate_id") != base.candidate_id:
        raise ValueError("failed audit does not descend from the supplied base candidate")
    for field in ("provider", "model", "config_hash"):
        if failure_payload.get(field) != getattr(base, field):
            raise ValueError(f"failed audit {field} does not match base candidate")
    issues = validate_complete_solver_source(source)
    if issues:
        raise ValueError("human-reviewed source fails deterministic contract: " + "; ".join(issues))
    calls = failure_payload.get("provider_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("failed audit lacks provider calls")
    records = [LLMCallRecord.model_validate(item) for item in calls]
    if any(record.status != "success" for record in records):
        raise ValueError("human override cannot adopt a source with failed provider calls")
    if any(
        record.latency_ms > 60_000.0 * max(1, record.attempts) + 1.0
        for record in records
    ):
        raise ValueError("human override cannot adopt calls outside the strict deadline")
    usage = _usage_from_calls(calls)
    if usage.logical_calls > 56 or usage.total_tokens > 250_000:
        raise ValueError("human override cannot exceed the fixed LLM budgets")
    events = [RoleEvent.model_validate(item) for item in failure_payload.get("role_events", [])]
    description = UAVProblemDescription.model_validate_json(
        description_path.read_text(encoding="utf-8")
    )
    judgment = json.loads(judgment_path.read_text(encoding="utf-8"))
    review = HumanSourceReview(
        decision="approved_for_restricted_qualification",
        approved_source_hash=source_hash,
        reviewer="delegated_human_review_by_codex",
        overrode_llm_judgment=not bool(judgment.get("approved")),
        llm_judgment_hash=stable_hash(judgment),
        rationale=[
            "Complete AST policy and deterministic solver contract passed.",
            "The final LLM judgment mixed valid checks with source-contradicted findings; "
            "human review resolved those findings against the authoritative contract.",
            "Approval is restricted to CLI smoke and six fixed Train qualification maps; "
            "the shared trusted evaluator remains authoritative.",
        ],
    )
    payload = base.model_dump(mode="json")
    payload.update(
        {
            "candidate_id": "0" * 64,
            "solver_hash": source_hash,
            "problem_description": description.model_dump(mode="json"),
            "description_revisions": base.description_revisions
            + sum(event.action == "revise_problem_description_after_human_audit" for event in events),
            "code_stage_revisions": {
                **base.code_stage_revisions,
                "post_audit": sum(
                    event.action == "revise_complete_solver_after_human_audit"
                    and event.status == "succeeded"
                    for event in events
                ),
            },
            "role_events": [event.model_dump(mode="json") for event in events],
            "provider_calls": calls,
            "usage": usage.model_dump(mode="json"),
            "human_review": review.model_dump(mode="json"),
        }
    )
    payload["candidate_id"] = stable_hash(_candidate_identity_payload(payload))
    candidate = AFLSolverCandidate.model_validate(payload)
    return candidate, save_solver_candidate(candidate, source, output_dir)


def save_solver_artifact(
    artifact: AFLSolverArtifact,
    source: str,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if stable_hash({"source": source}) != artifact.solver_hash:
        raise ValueError("solver source does not match artifact solver_hash")
    (destination / artifact.solver_filename).write_text(source, encoding="utf-8")
    manifest_path = destination / "artifact.json"
    manifest_path.write_text(
        canonical_json(artifact.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_solver_artifact(
    path: str | Path,
) -> tuple[AFLSolverArtifact, str, Path]:
    source = Path(path)
    manifest_path = source / "artifact.json" if source.is_dir() else source
    artifact = AFLSolverArtifact.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    solver_path = manifest_path.parent / artifact.solver_filename
    solver_source = solver_path.read_text(encoding="utf-8")
    if stable_hash({"source": solver_source}) != artifact.solver_hash:
        raise ValueError("frozen AFL-UAV solver hash does not match artifact")
    expected_id = stable_hash(
        _artifact_identity_payload(artifact.model_dump(mode="json"))
    )
    if artifact.artifact_id != expected_id:
        raise ValueError("AFL-UAV artifact identity hash does not match contents")
    issues = GeneratedCodePolicy().validate(solver_source, max_source_chars=500_000)
    if issues:
        raise ValueError("frozen AFL-UAV solver violates code policy: " + "; ".join(issues))
    return artifact, solver_source, solver_path


def generate_solver_candidate(
    config: ExperimentConfig,
    environment: Environment2D,
    output_dir: str | Path,
    *,
    provider: ProviderName,
    model: str | None,
) -> tuple[AFLSolverCandidate, Path]:
    """Call AFL roles on one Train map and persist source without executing it."""

    destination = Path(output_dir)
    generation_dir = destination / "generation"
    llm_provider = _make_provider(provider, model)
    call_config = LLMCallConfig(
        model=model,
        timeout_seconds=60.0,
        max_retries=2,
        max_output_tokens=16_384,
        max_total_tokens=250_000,
        max_logical_calls=56,
    )
    weights = ObjectiveWeights.model_validate(config.objective.model_dump())
    coordinator = AFLUAVCoordinator(
        provider=llm_provider,
        call_config=call_config,
        evaluator=PathEvaluator(weights),
        limits=AFLUAVLimits(
            max_code_revisions_per_stage=4,
            execution_timeout_seconds=60.0,
            max_source_chars=500_000,
        ),
        solver_buffer=None,
    )
    try:
        result = coordinator.run(
            run_id=f"afl-uav-candidate-{environment.map_id}",
            environment=environment,
            objective_weights=weights,
            output_dir=generation_dir,
            iterations=100,
            grid_resolution=config.maps.grid_resolution,
            max_waypoints=config.dsl.max_waypoints,
            execute_generated=False,
        )
    except Exception as exc:
        destination.mkdir(parents=True, exist_ok=True)
        failure_path = destination / "candidate_failure.json"
        calls = [
            item.model_dump(mode="json")
            for item in getattr(llm_provider, "call_records", [])
        ]
        events = [
            item.model_dump(mode="json")
            for item in coordinator.role_events
        ]
        failure_path.write_text(
            canonical_json(
                {
                    "schema_version": "afl-uav-candidate-failure-v1",
                    "provider": provider,
                    "model": model,
                    "provider_sdk_versions": _sdk_versions(provider),
                    "generated_from_split": "train",
                    "generated_from_map_id": environment.map_id,
                    "config_hash": config.config_hash,
                    "status": "failed",
                    "failure_reason": str(exc)[:4_000],
                    "role_events": events,
                    "provider_calls": calls,
                    "usage": _usage_from_calls(calls).model_dump(mode="json"),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"AFL-UAV candidate generation failed; failure audit: {failure_path.resolve()}"
        ) from exc
    if result.status != "generated_only":
        destination.mkdir(parents=True, exist_ok=True)
        failure_path = destination / "candidate_failure.json"
        failure_payload = {
            "schema_version": "afl-uav-candidate-failure-v1",
            "provider": provider,
            "model": model,
            "provider_sdk_versions": _sdk_versions(provider),
            "generated_from_split": "train",
            "generated_from_map_id": environment.map_id,
            "config_hash": config.config_hash,
            "status": result.status,
            "failure_reason": result.failure_reason,
            "role_events": [
                item.model_dump(mode="json") for item in result.role_events
            ],
            "provider_calls": result.provider_calls,
            "usage": _usage_from_calls(result.provider_calls).model_dump(
                mode="json"
            ),
        }
        failure_path.write_text(
            canonical_json(failure_payload) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            (result.failure_reason or "AFL-UAV candidate generation failed")
            + f"; failure audit: {failure_path.resolve()}"
        )
    generated_path = Path(result.solver_path)
    solver_source = generated_path.read_text(encoding="utf-8")
    complete_issues = validate_complete_solver_source(solver_source)
    if complete_issues:
        destination.mkdir(parents=True, exist_ok=True)
        failure_path = destination / "candidate_failure.json"
        failure_path.write_text(
            canonical_json(
                {
                    "schema_version": "afl-uav-candidate-failure-v1",
                    "provider": provider,
                    "model": model,
                    "provider_sdk_versions": _sdk_versions(provider),
                    "generated_from_split": "train",
                    "generated_from_map_id": environment.map_id,
                    "config_hash": config.config_hash,
                    "status": "failed",
                    "failure_reason": (
                        "complete solver failed deterministic contract: "
                        + "; ".join(complete_issues)
                    )[:4_000],
                    "role_events": [
                        item.model_dump(mode="json") for item in result.role_events
                    ],
                    "provider_calls": result.provider_calls,
                    "usage": _usage_from_calls(result.provider_calls).model_dump(
                        mode="json"
                    ),
                    "executed": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"AFL-UAV candidate failed complete static contract; audit: {failure_path.resolve()}"
        )
    base_payload = {
        "schema_version": "afl-uav-solver-candidate-v1",
        "candidate_id": "0" * 64,
        "solver_filename": "candidate_solver.py",
        "solver_hash": stable_hash({"source": solver_source}),
        "provider": provider,
        "model": model,
        "provider_sdk_versions": _sdk_versions(provider),
        "generated_from_split": "train",
        "generated_from_map_id": environment.map_id,
        "config_hash": config.config_hash,
        "objective_weights": weights.model_dump(mode="json"),
        "grid_resolution": config.maps.grid_resolution,
        "max_waypoints": config.dsl.max_waypoints,
        "problem_description": result.problem_description.model_dump(mode="json"),
        "description_revisions": result.description_revisions,
        "code_stage_revisions": result.code_stage_revisions,
        "role_events": [item.model_dump(mode="json") for item in result.role_events],
        "provider_calls": result.provider_calls,
        "usage": _usage_from_calls(result.provider_calls).model_dump(mode="json"),
        "upstream_repository": result.upstream_repository,
        "upstream_commit": result.upstream_commit,
    }
    base_payload["candidate_id"] = stable_hash(_candidate_identity_payload(base_payload))
    candidate = AFLSolverCandidate.model_validate(base_payload)
    return candidate, save_solver_candidate(candidate, solver_source, destination)


def revise_solver_candidate_after_audit(
    config: ExperimentConfig,
    candidate_path: str | Path,
    output_dir: str | Path,
    *,
    audit_issues: tuple[str, ...] = POST_GENERATION_AUDIT_ISSUES,
) -> tuple[AFLSolverCandidate, Path]:
    """Revise a saved candidate after human audit without executing any source."""

    base, base_source, _ = load_solver_candidate(candidate_path)
    if base.provider == "mock" or not base.model:
        raise ValueError("post-audit revision requires a real provider candidate")
    remaining_calls = 56 - base.usage.logical_calls
    remaining_tokens = 250_000 - base.usage.total_tokens
    if remaining_calls < 3 or remaining_tokens <= 0:
        raise ValueError("candidate has insufficient remaining LLM budget for audit revision")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if base.config_hash != config.config_hash:
        raise ValueError("base candidate config hash does not match revision config")
    provider = _make_provider(base.provider, base.model)
    provider.continue_call_ids_after(base.usage.logical_calls)
    call_config = LLMCallConfig(
        model=base.model,
        timeout_seconds=60.0,
        max_retries=2,
        max_output_tokens=16_384,
        max_total_tokens=remaining_tokens,
        max_logical_calls=remaining_calls,
    )
    new_events: list[RoleEvent] = []

    def call(role: str, action: str, template: Any, output_model: Any, payload: Any):
        try:
            result = provider.generate_structured(
                system_prompt=template.system_prompt,
                user_payload=payload,
                output_model=output_model,
                config=call_config,
                prompt_version=template.version,
                prompt_hash=template.prompt_hash,
            )
        except LLMProviderError as exc:
            record = exc.record
            new_events.append(
                RoleEvent(
                    sequence=len(base.role_events) + len(new_events) + 1,
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
        record = provider.call_records[-1]
        new_events.append(
            RoleEvent(
                sequence=len(base.role_events) + len(new_events) + 1,
                role=role,  # type: ignore[arg-type]
                action=action,
                prompt_version=template.version,
                output_model=output_model.__name__,
                provider_call_id=record.call_id,
                status="succeeded",
            )
        )
        return result

    source = base_source
    description = base.problem_description
    judgment: CodeJudgment | None = None
    deterministic_issues = validate_complete_solver_source(source)
    failure_reason = "post-audit revision did not converge"
    try:
        description_contract = {
            "problem_type": "STATIC_2D_UAV_PATH_PLANNING",
            "source_hash": base.problem_description.source_hash,
            "required_inputs": list(REQUIRED_INPUTS),
            "required_constraints": list(REQUIRED_CONSTRAINTS),
            "output": "Return a feasible start-to-goal path as ordered 2D waypoints.",
            "authoritative_objective": AUTHORITATIVE_OBJECTIVE,
        }
        description = call(
            "revision_agent",
            "revise_problem_description_after_human_audit",
            AUDITED_DESCRIPTION_REVISION_V1,
            UAVProblemDescription,
            {
                "description_contract": description_contract,
                "previous_description": base.problem_description.model_dump(mode="json"),
            },
        )
        # These fields are benchmark constants, not model choices.  Normalize
        # them locally after structured parsing so harmless paraphrases cannot
        # consume repeated API calls or weaken the authoritative contract.
        description = description.model_copy(
            update={
                "problem_type": description_contract["problem_type"],
                "source_hash": description_contract["source_hash"],
                "inputs": description_contract["required_inputs"],
                "output": description_contract["output"],
                "objective": description_contract["authoritative_objective"],
            }
        )
        description_issues = TaskDescriptionValidator().validate(
            description,
            base.problem_description.source_hash,
        )
        if description.objective != AUTHORITATIVE_OBJECTIVE:
            description_issues.append("description must copy authoritative_objective exactly")
        if description.output != description_contract["output"]:
            description_issues.append("description must copy output exactly")
        if description_issues:
            raise RuntimeError("; ".join(description_issues))
        # A whole-solver audit may expose dependent geometry defects only after
        # an earlier patch.  Allow bounded convergence while the enclosing
        # 56-call and 250k-token budgets remain authoritative.
        max_revision_rounds = min(6, max(1, (remaining_calls - 1) // 2))
        for revision_index in range(max_revision_rounds):
            revision = call(
                "revision_agent",
                "revise_complete_solver_after_human_audit",
                AUDITED_SOLVER_PATCH_V1,
                SolverPatch,
                {
                    "base_candidate_id": base.candidate_id,
                    "problem_description": description.model_dump(mode="json"),
                    "full_solver_contract": FULL_SOLVER_CONTRACT,
                    "audit_issues": list(audit_issues),
                    "deterministic_issues": deterministic_issues,
                    "previous_judgment": (
                        None if judgment is None else judgment.model_dump(mode="json")
                    ),
                    "previous_source": source,
                },
            )
            patched = source
            names = [item.function_name for item in revision.functions]
            if len(names) != len(set(names)):
                raise RuntimeError("Revision Agent returned duplicate function replacements")
            embedded_main_guard = False
            for replacement in revision.functions:
                replacement_tree = ast.parse(replacement.source)
                replacement_functions = [
                    node
                    for node in replacement_tree.body
                    if isinstance(node, ast.FunctionDef)
                ]
                allowed_nodes: list[ast.stmt] = list(replacement_functions)
                if replacement.function_name == "main":
                    guards = [
                        node
                        for node in replacement_tree.body
                        if isinstance(node, ast.If)
                        and any(
                            isinstance(child, ast.Name) and child.id == "__name__"
                            for child in ast.walk(node.test)
                        )
                    ]
                    if len(guards) > 1:
                        raise RuntimeError("Revision Agent returned multiple __main__ guards")
                    if guards:
                        embedded_main_guard = True
                        allowed_nodes.extend(guards)
                if (
                    len(replacement_functions) != 1
                    or replacement_functions[0].name != replacement.function_name
                    or len(allowed_nodes) != len(replacement_tree.body)
                ):
                    raise RuntimeError(
                        "Revision Agent function patch must contain exactly one matching "
                        f"definition: {replacement.function_name}"
                    )
                current_tree = ast.parse(patched)
                current = [
                    node
                    for node in current_tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == replacement.function_name
                ]
                if len(current) != 1:
                    raise RuntimeError(
                        "Revision Agent function target must exist exactly once: "
                        f"{replacement.function_name}"
                    )
                lines = patched.splitlines(keepends=True)
                target = current[0]
                replacement_text = (
                    ast.get_source_segment(
                        replacement.source, replacement_functions[0]
                    )
                    or ast.unparse(replacement_functions[0])
                ).rstrip() + "\n"
                patched = "".join(
                    [
                        *lines[: target.lineno - 1],
                        replacement_text,
                        *lines[target.end_lineno :],
                    ]
                )
            if revision.ensure_main_guard or embedded_main_guard:
                patched_tree = ast.parse(patched)
                has_guard = any(
                    isinstance(node, ast.If)
                    and any(
                        isinstance(child, ast.Name) and child.id == "__name__"
                        for child in ast.walk(node.test)
                    )
                    for node in patched_tree.body
                )
                if not has_guard:
                    patched = patched.rstrip() + "\n\n\nif __name__ == '__main__':\n    main()\n"
            source = patched.rstrip() + "\n"
            deterministic_issues = validate_complete_solver_source(source)
            judgment = call(
                "judgment_agent",
                "judge_complete_solver_after_human_audit",
                AUDITED_SOLVER_JUDGE_V1,
                CodeJudgment,
                {
                    "base_candidate_id": base.candidate_id,
                    "problem_description": description.model_dump(mode="json"),
                    "full_solver_contract": FULL_SOLVER_CONTRACT,
                    "audit_issues": list(audit_issues),
                    "deterministic_issues": deterministic_issues,
                    "revised_source": source,
                },
            )
            combined_issues = [
                *deterministic_issues,
                *judgment.issues,
                *judgment.required_revisions,
            ]
            if not combined_issues:
                break
            failure_reason = (
                "; ".join(combined_issues)
                or judgment.explanation
                or "judgment rejected the source without a concrete issue"
            )[:4_000]
        else:
            raise RuntimeError(failure_reason)
    except Exception as exc:
        calls = [
            *base.provider_calls,
            *[record.model_dump(mode="json") for record in provider.call_records],
        ]
        failure_path = destination / "candidate_failure.json"
        (destination / "rejected_candidate_solver.py").write_text(
            source,
            encoding="utf-8",
        )
        if judgment is not None:
            (destination / "last_judgment.json").write_text(
                canonical_json(judgment.model_dump(mode="json")) + "\n",
                encoding="utf-8",
            )
        (destination / "last_problem_description.json").write_text(
            canonical_json(description.model_dump(mode="json")) + "\n",
            encoding="utf-8",
        )
        failure_path.write_text(
            canonical_json(
                {
                    "schema_version": "afl-uav-candidate-failure-v1",
                    "provider": base.provider,
                    "model": base.model,
                    "provider_sdk_versions": _sdk_versions(base.provider),
                    "generated_from_split": base.generated_from_split,
                    "generated_from_map_id": base.generated_from_map_id,
                    "config_hash": config.config_hash,
                    "status": "failed",
                    "failure_reason": str(exc)[:4_000],
                    "base_candidate_id": base.candidate_id,
                    "role_events": [
                        *[item.model_dump(mode="json") for item in base.role_events],
                        *[item.model_dump(mode="json") for item in new_events],
                    ],
                    "provider_calls": calls,
                    "usage": _usage_from_calls(calls).model_dump(mode="json"),
                    "executed": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"post-audit candidate revision failed; audit: {failure_path.resolve()}"
        ) from exc

    calls = [
        *base.provider_calls,
        *[record.model_dump(mode="json") for record in provider.call_records],
    ]
    payload = base.model_dump(mode="json")
    payload.update(
        {
            "candidate_id": "0" * 64,
            "solver_hash": stable_hash({"source": source}),
            "provider_sdk_versions": _sdk_versions(base.provider),
            "problem_description": description.model_dump(mode="json"),
            "description_revisions": base.description_revisions + 1,
            "code_stage_revisions": {
                **base.code_stage_revisions,
                "post_audit": base.code_stage_revisions.get("post_audit", 0)
                + sum(
                    event.action == "revise_complete_solver_after_human_audit"
                    for event in new_events
                ),
            },
            "role_events": [
                *[item.model_dump(mode="json") for item in base.role_events],
                *[item.model_dump(mode="json") for item in new_events],
            ],
            "provider_calls": calls,
            "usage": _usage_from_calls(calls).model_dump(mode="json"),
        }
    )
    payload["candidate_id"] = stable_hash(_candidate_identity_payload(payload))
    candidate = AFLSolverCandidate.model_validate(payload)
    return candidate, save_solver_candidate(candidate, source, destination)


def _execute_qualification(
    candidate: AFLSolverCandidate,
    solver_source: str,
    solver_path: Path,
    environment: Environment2D,
    output_dir: Path,
    *,
    iterations: int,
    timeout_seconds: float,
    max_evaluations: int,
) -> tuple[QualificationResult, dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    instance = UAVSolverInstance(
        environment=environment,
        objective_weights=candidate.objective_weights,
        grid_resolution=candidate.grid_resolution,
        max_waypoints=candidate.max_waypoints,
    )
    instance_path = output_dir / "instance.json"
    output_path = output_dir / "output.json"
    instance_path.write_text(
        canonical_json(instance.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    execution = GeneratedSolverRunner().execute(
        solver_path=solver_path,
        source=solver_source,
        instance_path=instance_path,
        output_path=output_path,
        iterations=iterations,
        timeout_seconds=timeout_seconds,
        max_source_chars=500_000,
        seed=environment.seed,
        max_evaluations=max_evaluations,
    )
    counters = {name: 0 for name in COUNTER_NAMES}
    failures: list[str] = []
    total_cost: float | None = None
    passed = False
    if execution.status == "success" and execution.output_payload is not None:
        try:
            counters = extract_solver_counters(
                execution.output_payload,
                max_evaluations=max_evaluations,
            )
            external = validate_solver_output(
                execution.output_payload,
                environment,
                PathEvaluator(candidate.objective_weights),
                max_waypoints=candidate.max_waypoints,
            )
            passed = external.passed
            failures.extend(external.failures)
            if external.evaluation is not None:
                total_cost = external.evaluation.total_cost
        except (TypeError, ValueError) as exc:
            failures.append(str(exc))
    else:
        failures.append(execution.error or execution.status)
    result = QualificationResult(
        map_id=environment.map_id,
        difficulty=environment.difficulty,
        status=execution.status,
        passed=passed,
        duration_ms=execution.duration_ms,
        total_cost=total_cost,
        failures=failures,
        **counters,
    )
    return result, execution.output_payload


def freeze_solver_candidate(
    config: ExperimentConfig,
    candidate_path: str | Path,
    qualification_environments: list[Environment2D],
    output_dir: str | Path,
    *,
    approved_source_hash: str,
    require_train_map_ids: bool = True,
) -> tuple[AFLSolverArtifact, Path]:
    """Execute only an explicitly approved source hash and freeze on success."""

    candidate, solver_source, candidate_solver_path = load_solver_candidate(candidate_path)
    if candidate.config_hash != config.config_hash:
        raise ValueError("candidate config hash does not match freeze configuration")
    if approved_source_hash != candidate.solver_hash:
        raise ValueError("approved source hash does not match candidate solver hash")
    if not qualification_environments:
        raise ValueError("at least one Train qualification map is required")
    map_ids = [environment.map_id for environment in qualification_environments]
    if len(map_ids) != len(set(map_ids)) or (
        require_train_map_ids
        and any(not map_id.startswith("train-") for map_id in map_ids)
    ):
        raise ValueError("qualification maps must be unique Train maps")

    destination = Path(output_dir)
    qualification_root = destination / "qualification"
    anchor = next(
        (
            item
            for item in qualification_environments
            if item.map_id == candidate.generated_from_map_id
        ),
        qualification_environments[0],
    )
    contract_result, contract_payload = _execute_qualification(
        candidate,
        solver_source,
        candidate_solver_path,
        anchor,
        qualification_root / "contract_smoke",
        iterations=16,
        timeout_seconds=60.0,
        max_evaluations=16,
    )
    if not contract_result.passed or contract_payload is None:
        raise RuntimeError(
            "candidate failed CLI-v2 contract smoke: "
            + "; ".join(contract_result.failures or [contract_result.status])
        )
    contract_counters = extract_solver_counters(contract_payload, max_evaluations=16)

    qualification_results: list[QualificationResult] = []
    for environment in qualification_environments:
        result, _ = _execute_qualification(
            candidate,
            solver_source,
            candidate_solver_path,
            environment,
            qualification_root / environment.map_id,
            iterations=min(256, config.planning_benchmark.max_objective_evaluations - 1),
            timeout_seconds=config.planning_benchmark.time_limit_seconds,
            max_evaluations=config.planning_benchmark.max_objective_evaluations,
        )
        qualification_results.append(result)
    failed = [item for item in qualification_results if not item.passed]
    if failed:
        details = "; ".join(
            f"{item.map_id}: {', '.join(item.failures) or item.status}" for item in failed
        )
        raise RuntimeError("candidate failed Train qualification: " + details)

    calls_successful = bool(candidate.provider_calls) and all(
        call.get("status") == "success" for call in candidate.provider_calls
    )
    # The SDK timeout is an I/O timeout and may not bound the complete response
    # wall clock.  Research eligibility also requires each recorded logical call
    # to fit within its per-attempt 60-second end-to-end deadline.
    calls_within_deadline = bool(candidate.provider_calls) and all(
        float(call.get("latency_ms", float("inf")))
        <= 60_000.0 * max(1, int(call.get("attempts", 1))) + 1.0
        for call in candidate.provider_calls
    )
    eligible = (
        candidate.provider in REAL_PROVIDERS
        and candidate.model == AFL_PROVIDER_MODELS[candidate.provider]
        and calls_successful
        and calls_within_deadline
        and all(item.passed for item in qualification_results)
        and tuple(item.map_id for item in qualification_results)
        == QUALIFICATION_MAP_IDS
    )
    base_payload = {
        "schema_version": "afl-uav-solver-artifact-v2",
        "artifact_id": "0" * 64,
        "planner_name": "afl_uav",
        "solver_filename": "frozen_solver.py",
        "solver_hash": candidate.solver_hash,
        "solver_cli_contract": "afl-uav-solver-cli-v2",
        "provider": candidate.provider,
        "model": candidate.model,
        "research_claim_eligible": eligible,
        "generated_from_split": "train",
        "generated_from_map_id": candidate.generated_from_map_id,
        "config_hash": candidate.config_hash,
        "objective_weights": candidate.objective_weights.model_dump(mode="json"),
        "grid_resolution": candidate.grid_resolution,
        "max_waypoints": candidate.max_waypoints,
        "problem_description": candidate.problem_description.model_dump(mode="json"),
        "role_events": [item.model_dump(mode="json") for item in candidate.role_events],
        "provider_calls": candidate.provider_calls,
        "generation_execution_attempts": 0,
        "contract_smoke_counters": contract_counters,
        "upstream_repository": candidate.upstream_repository,
        "upstream_commit": candidate.upstream_commit,
        "candidate_id": candidate.candidate_id,
        "candidate_source_hash": candidate.solver_hash,
        "approved_source_hash": approved_source_hash,
        "provider_sdk_versions": candidate.provider_sdk_versions,
        "llm_usage": candidate.usage.model_dump(mode="json"),
        "description_revisions": candidate.description_revisions,
        "code_stage_revisions": candidate.code_stage_revisions,
        "qualification_results": [
            item.model_dump(mode="json") for item in qualification_results
        ],
        "human_review": (
            None
            if candidate.human_review is None
            else candidate.human_review.model_dump(mode="json")
        ),
    }
    base_payload["artifact_id"] = stable_hash(_artifact_identity_payload(base_payload))
    artifact = AFLSolverArtifact.model_validate(base_payload)
    return artifact, save_solver_artifact(artifact, solver_source, destination)


def build_solver_artifact(
    config: ExperimentConfig,
    environment: Environment2D,
    output_dir: str | Path,
    *,
    provider: ProviderName = "mock",
    model: str | None = None,
    execute_untrusted_code: bool = False,
) -> tuple[AFLSolverArtifact, Path]:
    """Backward-compatible one-step helper restricted to deterministic mock code."""

    if provider != "mock":
        raise ValueError(
            "real providers require generate_solver_candidate followed by "
            "freeze_solver_candidate with an approved source hash"
        )
    destination = Path(output_dir)
    candidate, candidate_manifest = generate_solver_candidate(
        config,
        environment,
        destination / "candidate",
        provider="mock",
        model=model,
    )
    return freeze_solver_candidate(
        config,
        candidate_manifest,
        [environment],
        destination,
        approved_source_hash=candidate.solver_hash,
        require_train_map_ids=False,
    )


__all__ = [
    "AFLSolverArtifact",
    "AFLSolverCandidate",
    "AFL_PROVIDER_MODELS",
    "CandidateUsage",
    "QualificationResult",
    "QUALIFICATION_MAP_IDS",
    "POST_GENERATION_AUDIT_ISSUES",
    "build_solver_artifact",
    "extract_solver_counters",
    "freeze_solver_candidate",
    "generate_solver_candidate",
    "load_solver_artifact",
    "load_solver_candidate",
    "revise_solver_candidate_after_audit",
    "save_solver_artifact",
    "save_solver_candidate",
]
