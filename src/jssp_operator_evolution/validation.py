"""JSSP façade over generic CRN/ABBA paired candidate validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from operator_evolution_core.search import (
    BlockRandomRoundRobinScheduler,
    GenericSearchKernel,
    SearchBudget,
    SimulatedAnnealingAcceptance,
)
from operator_evolution_core.trajectory import TrajectoryRecorder
from operator_evolution_core.validation import (
    ArmMeasurement,
    FitnessPolicy,
    GenericPairedCandidateValidator,
    ValidationReport,
)

from .adapter import create_jssp_domain_adapter
from .models import JobShopInstance, JobShopSolution
from .operators import CompiledJSSPOperator
from .trajectory import JSSPTrajectorySink


def derive_jssp_seed(*parts: object) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


@dataclass(frozen=True, slots=True)
class JSSPRetentionConfig:
    min_global_gain: float = 0.01
    min_specialist_gain: float = 0.02
    min_feasibility_gain: float = 0.05
    min_runtime_reduction: float = 0.25
    min_runtime_effective_call_rate: float = 0.10
    require_bootstrap_ci: bool = False


class JSSPCandidateValidator:
    def __init__(
        self,
        *,
        search_calls: int = 240,
        master_seed: int = 20260823,
        runtime_repetitions: int = 2,
        retention_config: JSSPRetentionConfig | None = None,
        recorder: TrajectoryRecorder | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.adapter = create_jssp_domain_adapter()
        self.budget = SearchBudget(max_iterations=search_calls)
        self.recorder = recorder
        self.clock = clock
        self.generic = GenericPairedCandidateValidator[
            JobShopInstance, JobShopSolution, CompiledJSSPOperator
        ](
            adapter=self.adapter,
            budget=self.budget,
            retention_config=retention_config or JSSPRetentionConfig(),
            master_seed=master_seed,
            seed_deriver=derive_jssp_seed,
            arm_runner=self._run_arm,
            operator_id=lambda operator: operator.operator_id,
            instance_id=lambda instance: instance.instance_id,
            context_label=lambda instance: (
                f"{instance.job_count}x{instance.machines}"
            ),
            runtime_repetitions=runtime_repetitions,
            fitness_policy=FitnessPolicy.DETERMINISTIC_V2,
        )

    def validate(
        self,
        validation_instances: Sequence[JobShopInstance],
        operator_population: Sequence[CompiledJSSPOperator],
        parent_operator_id: str,
        candidate_operator: CompiledJSSPOperator,
        *,
        generation: int,
        candidate_index: int,
        root_run_id: str,
        safety_failures: Sequence[str] = (),
        persist_evidence: bool = False,
    ) -> ValidationReport:
        return self.generic.validate(
            validation_instances,
            operator_population,
            parent_operator_id,
            candidate_operator,
            generation=generation,
            candidate_index=candidate_index,
            root_run_id=root_run_id,
            safety_failures=safety_failures,
            persist_evidence=persist_evidence,
        )

    def _run_arm(
        self,
        population: Sequence[CompiledJSSPOperator],
        instance: JobShopInstance,
        initial: JobShopSolution,
        seed: int,
        target_id: str,
        generation: int,
        run_id: str,
        persist_evidence: bool,
    ) -> ArmMeasurement:
        del generation
        kernel = GenericSearchKernel(
            adapter=self.adapter,
            operators=population,
            scheduler=BlockRandomRoundRobinScheduler(),
            acceptance=SimulatedAnnealingAcceptance(),
            budget=self.budget,
            clock=self.clock,
        )
        callback = None
        if persist_evidence:
            if self.recorder is None:
                raise ValueError("persist_evidence requires a trajectory recorder")
            callback = JSSPTrajectorySink(
                self.recorder,
                run_id=run_id,
                instance=instance,
                seed=seed,
            )
        started = self.clock()
        result = kernel.run(
            instance,
            np.random.default_rng(seed),
            initial_solution=initial,
            on_step=callback,
        )
        total_runtime_ms = max(0.0, (self.clock() - started) * 1000.0)
        target_steps = [
            step for step in result.steps if step.operator_id == target_id
        ]
        return ArmMeasurement(
            best_cost=float(result.best_evaluation.scalar_cost),
            feasible=bool(result.best_evaluation.feasible),
            total_runtime_ms=total_runtime_ms,
            operator_runtime_ms=sum(step.runtime_ms for step in target_steps),
            operator_call_count=len(target_steps),
            operator_changed_call_count=sum(
                step.operator_outcome.success for step in target_steps
            ),
            operator_accepted_call_count=sum(step.accepted for step in target_steps),
        )


__all__ = [
    "JSSPCandidateValidator",
    "JSSPRetentionConfig",
    "derive_jssp_seed",
]
