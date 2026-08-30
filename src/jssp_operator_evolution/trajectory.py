"""JSSP bridge into the shared three-state trajectory recorder."""

from __future__ import annotations

from typing import Any

from operator_evolution_core.search import SearchStep
from operator_evolution_core.trajectory import OperatorTrace, TrajectoryRecorder

from .adapter import create_jssp_domain_adapter
from .models import JobShopInstance, JobShopSolution


class JSSPTrajectorySink:
    def __init__(
        self,
        recorder: TrajectoryRecorder,
        *,
        run_id: str,
        instance: JobShopInstance,
        seed: int,
    ) -> None:
        self.recorder = recorder
        self.run_id = run_id
        self.instance = instance
        self.seed = int(seed)
        self.adapter = create_jssp_domain_adapter()

    def __call__(self, step: SearchStep[JobShopSolution], operator: Any) -> None:
        encoder = self.adapter.trace_encoder
        before = encoder.snapshot(
            step.solution_before,
            self.instance,
            step.evaluation_before,
            step.context_before,
        )
        candidate = encoder.snapshot(
            step.candidate_solution,
            self.instance,
            step.candidate_evaluation,
            step.context_before,
        )
        accepted = encoder.snapshot(
            step.current_solution_after,
            self.instance,
            step.current_evaluation_after,
            step.context_after,
        )
        spec = getattr(operator, "spec", None)
        self.recorder.record(
            OperatorTrace(
                run_id=self.run_id,
                instance_id=self.instance.instance_id,
                map_difficulty=(
                    f"{self.instance.job_count}x{self.instance.machines}"
                ),
                iteration=step.iteration,
                seed=self.seed,
                operator_id=step.operator_id,
                operator_family=(
                    getattr(getattr(spec, "selector", None), "kind", None)
                ),
                operator_version="jssp-v1",
                operator_params=(
                    spec.model_dump(mode="json") if spec is not None else {}
                ),
                context={
                    **step.context_before.as_features(),
                    "analysis": dict(before["features"]),
                    "instance_shape": (
                        f"{self.instance.job_count}x{self.instance.machines}"
                    ),
                    "source_family": self.instance.source_family,
                },
                before_state=before,
                candidate_state=candidate,
                accepted_state=accepted,
                accepted=step.accepted,
                acceptance_reason="accepted" if step.accepted else "rejected",
                temperature=step.temperature,
                immediate_reward=step.immediate_reward,
                runtime_ms=step.runtime_ms,
                error=(
                    None
                    if step.operator_outcome.success
                    else step.operator_outcome.failure_reason
                ),
                metadata=dict(step.operator_outcome.metadata),
            )
        )


__all__ = ["JSSPTrajectorySink"]
