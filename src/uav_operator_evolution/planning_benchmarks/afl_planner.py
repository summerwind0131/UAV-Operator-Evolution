"""Planner adapter for a frozen, agent-generated AFL-UAV solver."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..afl_uav.artifact import (
    AFLSolverArtifact,
    extract_solver_counters,
    load_solver_artifact,
)
from ..afl_uav.models import UAVSolverInstance
from ..afl_uav.runner import GeneratedSolverRunner
from ..path.models import copy_and_validate_path
from ..reproducibility import canonical_json
from .core import BudgetedEvaluator, PlannerResult, PlanningBudget


@dataclass
class FrozenAFLUAVPlanner:
    """Execute one solver frozen on Train under the common planning budget."""

    artifact_path: str | Path
    arm_id: str = "afl_uav"
    iteration_limit: int = 256
    name: str = field(default="afl_uav", init=False)
    stochastic: bool = field(default=True, init=False)
    research_claim_eligible: bool = field(default=False, init=False)
    artifact: AFLSolverArtifact = field(init=False, repr=False)
    solver_source: str = field(init=False, repr=False)
    solver_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        artifact, source, solver_path = load_solver_artifact(self.artifact_path)
        self.artifact = artifact
        self.solver_source = source
        self.solver_path = solver_path
        self.research_claim_eligible = artifact.research_claim_eligible

    def plan(
        self,
        problem: BudgetedEvaluator,
        budget: PlanningBudget,
        rng: np.random.Generator,
    ) -> PlannerResult:
        if problem.evaluator.weights != self.artifact.objective_weights:
            return problem.result(
                self.name,
                "error",
                message="artifact objective weights do not match benchmark evaluator",
                diagnostics={"artifact_id": self.artifact.artifact_id},
            )
        problem.check_time()
        solver_seed = int(rng.integers(0, (1 << 63) - 1))
        instance = UAVSolverInstance(
            environment=problem.environment,
            objective_weights=self.artifact.objective_weights,
            grid_resolution=self.artifact.grid_resolution,
            max_waypoints=self.artifact.max_waypoints,
        )
        diagnostics = {
            "arm_id": self.arm_id,
            "artifact_id": self.artifact.artifact_id,
            "solver_hash": self.artifact.solver_hash,
            "artifact_provider": self.artifact.provider,
            "artifact_model": self.artifact.model,
            "solver_seed": solver_seed,
        }
        with tempfile.TemporaryDirectory(prefix="afl_uav_planner_") as temporary:
            temporary_root = Path(temporary)
            instance_path = temporary_root / "instance.json"
            output_path = temporary_root / "output.json"
            instance_path.write_text(
                canonical_json(instance.model_dump(mode="json")) + "\n",
                encoding="utf-8",
            )
            remaining_time = max(
                1e-3,
                budget.time_limit_seconds - problem.elapsed_seconds,
            )
            execution = GeneratedSolverRunner().execute(
                solver_path=self.solver_path,
                source=self.solver_source,
                instance_path=instance_path,
                output_path=output_path,
                iterations=min(
                    self.iteration_limit,
                    max(0, budget.max_objective_evaluations - 1),
                ),
                timeout_seconds=remaining_time,
                max_source_chars=500_000,
                seed=solver_seed,
                max_evaluations=budget.max_objective_evaluations,
            )
        diagnostics.update(
            {
                "execution_status": execution.status,
                "execution_duration_ms": execution.duration_ms,
                "return_code": execution.return_code,
            }
        )
        if execution.status == "timeout":
            return problem.result(
                self.name,
                "timeout",
                message=execution.error or "generated solver timed out",
                diagnostics=diagnostics,
            )
        if execution.status != "success" or execution.output_payload is None:
            return problem.result(
                self.name,
                "error",
                message=execution.error or f"generated solver {execution.status}",
                diagnostics=diagnostics,
            )
        payload = execution.output_payload
        try:
            counters = extract_solver_counters(
                payload,
                max_evaluations=budget.max_objective_evaluations,
            )
            problem.record_external_counts(**counters)
            path = copy_and_validate_path(payload["path"])
        except (KeyError, TypeError, ValueError) as exc:
            return problem.result(
                self.name,
                "error",
                message=f"generated solver violated output contract: {exc}",
                diagnostics=diagnostics,
            )
        diagnostics.update(
            {
                "reported_initial_cost": payload.get("initial_cost"),
                "reported_best_cost": payload.get("best_cost"),
                "completed_iterations": payload.get("iterations"),
            }
        )
        status = (
            "timeout"
            if problem.elapsed_seconds >= budget.time_limit_seconds
            else "success"
        )
        return problem.result(
            self.name,
            status,
            path=path,
            message=(
                "solver completed at the wall-clock boundary"
                if status == "timeout"
                else ""
            ),
            diagnostics=diagnostics,
        )


__all__ = ["FrozenAFLUAVPlanner"]
