"""End-to-end, fixed-budget operator evolution manager."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from operator_evolution_core.evolution import (
    EvolutionManagerDependencies,
    EvolutionSplitCapabilities,
    PopulationSeed,
    population_fingerprint,
)
from operator_evolution_core.validation import replace_population_slot

from ..agents.audit import AgentAuditStore
from ..agents.designer_base import OperatorProposal
from ..agents.evidence import DesignBudget, EvidenceBundleBuilder
from ..agents.heuristic_designer import HeuristicDesigner
from ..agents.llm_designer import LLMDesignerAdapter
from ..agents.multi_agent import DeterministicMockMultiAgent
from ..agents.orchestrator import (
    OperatorDesignOrchestrationResult,
    OperatorDesignOrchestrator,
    OperatorDesignRequest,
)
from ..agents.proposal_validation import ProposalValidator
from ..agents.providers import LLMCallConfig, MockLLMProvider, OpenAIProvider
from ..agents.research_agent import (
    DeterministicMockResearchAgent,
    OpenAIAgentsResearchAgent,
)
from ..agents.tools import AgentBudget as ResearchAgentBudget
from ..config import ExperimentConfig
from ..diagnosis.diagnoser import OperatorDiagnoser
from ..diagnosis.features import UAV_FEATURE_CATALOG
from ..domain.uav_adapter import UAVDomainAdapter
from ..domain.uav_kit import UAVDomainKit
from ..environment.environment import Environment2D
from ..memory import MechanismMemory
from ..operators.base import PathOperator
from ..operators.catalog import manual_operator_specs
from ..operators.compiler import OperatorCompiler
from ..operators.registry import OperatorRegistry, default_manual_operators
from ..operators.specs import OperatorSpec
from ..path.evaluator import PathEvaluator
from ..path.models import ObjectiveWeights
from ..reproducibility import derive_seed
from ..search.executor import SearchExecutor, SearchResult
from ..trajectory import TrajectoryRecorder
from .validation import PairedOutcome, ValidationReport
from .fitness import FitnessPolicy, compute_fitness
from .candidate_validator import FixedBudgetCandidateValidator


class EvolutionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunMetric(EvolutionModel):
    phase: str
    generation: int
    arm: str
    map_id: str
    difficulty: str
    best_cost: float
    final_cost: float
    feasible: bool
    runtime_ms: float


class GenerationSummary(EvolutionModel):
    generation: int
    population_before: list[str]
    population_after: list[str]
    profile_count: int
    synergy_count: int
    proposals: list[OperatorProposal] = Field(default_factory=list)
    validations: list[ValidationReport] = Field(default_factory=list)
    best_cost: float
    fitness: dict[str, float] = Field(default_factory=dict)


class EvolutionResult(EvolutionModel):
    run_id: str
    initial_population: list[str]
    final_population: list[str]
    generations: list[GenerationSummary]
    metrics: list[RunMetric]
    profiles: list[dict[str, Any]]
    synergies: list[dict[str, Any]]
    test_outcomes: list[PairedOutcome]
    trace_count: int
    retained_candidates: list[str]


@dataclass(slots=True)
class _Population:
    operators: list[PathOperator]
    specs: dict[str, OperatorSpec]

    def replace(self, parent_name: str, candidate: PathOperator, spec: OperatorSpec) -> bool:
        replacement = replace_population_slot(
            self.operators,
            parent_name,
            candidate,
            operator_id=lambda operator: str(operator.name),
        )
        if replacement is None:
            return False
        self.operators = list(replacement.population)
        self.specs.pop(parent_name, None)
        self.specs[str(candidate.name)] = spec
        return True


class OperatorEvolutionManager:
    """Evolve only operator internals under a fixed search protocol."""

    def __init__(
        self,
        config: ExperimentConfig,
        database_path: str | Path | None = None,
        *,
        jsonl_path: str | Path | None = None,
        designer: HeuristicDesigner | None = None,
        dependencies: EvolutionManagerDependencies[
            Environment2D,
            list[tuple[float, float]],
            PathOperator,
            OperatorSpec,
        ]
        | None = None,
    ) -> None:
        self.config = config
        self.database_path = (
            Path(database_path)
            if database_path is not None
            else Path(config.output.results_dir) / "operator_evolution.sqlite"
        )
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.compiler = OperatorCompiler(config.dsl)
        weights = ObjectiveWeights.model_validate(config.objective.model_dump())
        self.evaluator = PathEvaluator(weights)
        if dependencies is not None and designer is not None:
            raise ValueError("designer cannot be supplied both directly and via dependencies")
        if dependencies is None:
            domain_adapter = UAVDomainAdapter(
                self.evaluator,
                initializer_grid_resolution=config.maps.grid_resolution,
            )
            domain_kit = UAVDomainKit(self.compiler)

            def population_factory() -> PopulationSeed[PathOperator, OperatorSpec]:
                return PopulationSeed(
                    operators=tuple(default_manual_operators()),
                    ir_by_id=manual_operator_specs(),
                )

            dependencies = EvolutionManagerDependencies(
                domain_adapter=domain_adapter,
                domain_kit=domain_kit,
                population_factory=population_factory,
                candidate_validator=FixedBudgetCandidateValidator(
                    config, self.evaluator
                ),
                designer=designer or HeuristicDesigner(),
                orchestrator_factory=OperatorDesignOrchestrator,
            )
        self.dependencies = dependencies
        self.domain_adapter = dependencies.domain_adapter
        self.domain_kit = dependencies.domain_kit
        self.designer = dependencies.designer
        self.candidate_validator = dependencies.candidate_validator
        self.orchestrator_factory = dependencies.orchestrator_factory
        self.artifact_sink = dependencies.artifact_sink
        native_evaluator = getattr(self.domain_adapter.evaluator, "native_evaluator", None)
        if native_evaluator is not None:
            self.evaluator = native_evaluator
        native_compiler = getattr(self.domain_kit, "compiler", None)
        if native_compiler is not None:
            self.compiler = native_compiler
        self.last_population: list[PathOperator] = []
        self.initial_population: list[PathOperator] = []

    def run(
        self,
        datasets: Mapping[str, list[Environment2D]]
        | EvolutionSplitCapabilities[Environment2D],
        run_id: str,
    ) -> EvolutionResult:
        """Run P0→P1→P2 (or configured count), then one held-out comparison."""

        split_capabilities = (
            datasets
            if isinstance(datasets, EvolutionSplitCapabilities)
            else EvolutionSplitCapabilities.from_mapping(datasets)
        )
        train_environments = list(split_capabilities.open_train())
        validation_environments = list(split_capabilities.open_validation())
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_sink.emit(
            "run_started",
            {
                "run_id": run_id,
                "domain_id": self.domain_kit.domain_id,
                "ir_version": self.domain_kit.ir_version,
                "train_instances": len(train_environments),
                "validation_instances": len(validation_environments),
                "test_opened": False,
            },
        )
        seed_population = self.dependencies.population_factory()
        initial_operators = list(seed_population.operators)
        self.initial_population = list(initial_operators)
        population = _Population(initial_operators, dict(seed_population.ir_by_id))
        initial_names = [str(operator.name) for operator in initial_operators]
        initial_population_snapshot = list(initial_operators)
        metrics: list[RunMetric] = []
        summaries: list[GenerationSummary] = []
        all_profile_rows: list[dict[str, Any]] = []
        all_synergy_rows: list[dict[str, Any]] = []
        retained_names: list[str] = []

        with TrajectoryRecorder(self.database_path, self.jsonl_path) as recorder, MechanismMemory(
            self.database_path
        ) as memory:
            self._register_initial_memory(memory, population)
            for generation in range(self.config.evolution.generations):
                before_names = [str(operator.name) for operator in population.operators]
                train_metrics = self._run_population(
                    population.operators,
                    train_environments,
                    self.config.search.train_iterations,
                    recorder,
                    f"{run_id}-train-g{generation}",
                    generation,
                    phase="train",
                    arm=f"P{generation}",
                )
                metrics.extend(train_metrics)
                recorder.update_delayed_rewards(
                    self.config.diagnostics.delayed_horizons,
                    run_id=f"{run_id}-train-g{generation}",
                    baseline="before",
                )
                traces = recorder.list_traces(f"{run_id}-train-g{generation}")
                diagnoser = OperatorDiagnoser(
                    minimum_context_samples=1,
                    representative_cases=self.config.diagnostics.representative_cases,
                    feature_catalog=UAV_FEATURE_CATALOG,
                )
                profiles = diagnoser.diagnose(traces)
                synergies = diagnoser.analyze_synergies(
                    traces,
                    min_samples=max(1, self.config.diagnostics.minimum_context_samples),
                    reward_horizon=min(self.config.diagnostics.delayed_horizons),
                )
                profile_rows = [profile.model_dump(mode="json") for profile in profiles]
                synergy_rows = [relation.model_dump(mode="json") for relation in synergies]
                all_profile_rows.extend(profile_rows)
                all_synergy_rows.extend(synergy_rows)
                self._write_diagnostics_to_memory(memory, population, traces, profiles, synergies, generation)

                fitness_scores = self._fitness_scores(profiles)
                ranked_parent_names = self._rank_parents(profiles, before_names, fitness_scores)
                proposals: list[OperatorProposal] = []
                validations: list[ValidationReport] = []
                for candidate_index in range(self.config.evolution.candidates_per_generation):
                    parent_name = ranked_parent_names[candidate_index % len(ranked_parent_names)]
                    parent_spec = population.specs[parent_name]
                    parent_profile = next(
                        (profile for profile in profiles if profile.operator_id == parent_name),
                        profiles[0] if profiles else {},
                    )
                    parent_synergies = [
                        relation.model_dump(mode="python")
                        for relation in synergies
                        if relation.first_operator == parent_name
                    ]
                    profile_payload = (
                        parent_profile.model_copy(
                            update={"synergy_relations": parent_synergies}
                        )
                        if hasattr(parent_profile, "model_copy")
                        else parent_profile
                    )
                    relevant_memory = memory.get_failure_modes(parent_name, limit=5)
                    if self.config.agent.designer_mode != "heuristic":
                        orchestration, registered_candidate = self._run_phase8_candidate(
                            population,
                            parent_name,
                            profile_payload,
                            train_environments,
                            validation_environments,
                            generation,
                            candidate_index,
                            recorder,
                            memory,
                            run_id,
                        )
                        if orchestration.proposal is not None:
                            proposals.append(orchestration.proposal)
                        if orchestration.validation_report is not None:
                            validations.append(orchestration.validation_report)
                        if orchestration.retained and registered_candidate is not None:
                            candidate_operator, candidate_spec = registered_candidate
                            if population.replace(parent_name, candidate_operator, candidate_spec):
                                retained_names.append(str(candidate_operator.name))
                        continue
                    base_proposal = self.designer.propose(
                        "Improve fixed-budget UAV path search using computed trajectory evidence.",
                        [parent_spec],
                        [profile_payload],
                        relevant_memory,
                        [],
                        [],
                    )
                    candidate_name = f"G{generation + 1}C{candidate_index + 1}_{base_proposal.spec.name}"
                    candidate_spec = base_proposal.spec.model_copy(
                        update={
                            "name": candidate_name,
                            "parameters": {
                                **base_proposal.spec.parameters,
                                "generation": generation + 1,
                            },
                        }
                    )
                    proposal = base_proposal.model_copy(update={"specification": candidate_spec})
                    proposals.append(proposal)
                    candidate = self.domain_kit.compile(
                        self.domain_kit.parse_ir(candidate_spec)
                    )
                    report = self._validate_candidate(
                        population.operators,
                        parent_name,
                        candidate,
                        validation_environments,
                        generation,
                        candidate_index,
                        recorder,
                        run_id,
                    )
                    validations.append(report)
                    memory.add_mechanism(
                        candidate_name,
                        candidate_spec.model_dump(mode="json"),
                        name=candidate_name,
                        description=proposal.design_rationale,
                        status="active" if report.retained else "rejected",
                        score=report.mean_gain,
                        evidence_count=len(report.outcomes),
                        success_rate=report.win_rate,
                        tags=["operator", "generated", proposal.evidence_level],
                        metadata={
                            "generation": generation + 1,
                            "source": "heuristic_designer",
                            "code_version": "0.1.0",
                            "active_status": report.retained,
                            "creation_reason": proposal.design_rationale,
                            "validation": report.model_dump(mode="json"),
                            "evidence": proposal.evidence_used,
                        },
                        parent_ids=[parent_name],
                    )
                    memory.add_insight(
                        operator_id=candidate_name,
                        insight_type="improvement_hypothesis",
                        evidence={
                            "design_rationale": proposal.design_rationale,
                            "evidence_used": proposal.evidence_used,
                            "validation": report.model_dump(mode="json"),
                        },
                        confidence=min(1.0, len(report.outcomes) / 20.0),
                        applicable_context={"expected": proposal.expected_contexts},
                        failure_context={"targets": proposal.target_failure_modes},
                        source_profile_id=f"generation:{generation}",
                    )
                    if report.retained and population.replace(parent_name, candidate, candidate_spec):
                        retained_names.append(candidate_name)

                after_names = [str(operator.name) for operator in population.operators]
                summaries.append(
                    GenerationSummary(
                        generation=generation,
                        population_before=before_names,
                        population_after=after_names,
                        profile_count=len(profiles),
                        synergy_count=len(synergies),
                        proposals=proposals,
                        validations=validations,
                        best_cost=min((metric.best_cost for metric in train_metrics), default=float("inf")),
                        fitness=fitness_scores,
                    )
                )

                self.artifact_sink.emit(
                    "generation_completed",
                    {
                        "run_id": run_id,
                        "generation": generation,
                        "population_ids": after_names,
                        "retained_candidates": list(retained_names),
                        "test_opened": False,
                    },
                )

            final_names = [str(operator.name) for operator in population.operators]
            frozen_fingerprint = population_fingerprint(
                final_names,
                population.specs,
                self.domain_kit,
            )
            freeze_receipt = split_capabilities.freeze_population(
                final_names,
                frozen_fingerprint,
            )
            self.artifact_sink.emit(
                "population_frozen",
                {
                    "run_id": run_id,
                    "population_ids": final_names,
                    "population_fingerprint": frozen_fingerprint,
                    "freeze_receipt_id": freeze_receipt.receipt_id,
                    "test_opened": False,
                },
            )
            test_environments = list(split_capabilities.open_test(freeze_receipt))
            test_outcomes, test_metrics = self._compare_populations(
                initial_population_snapshot,
                population.operators,
                test_environments,
                self.config.search.test_iterations,
                phase="test",
                generation=self.config.evolution.generations,
                parent_arm="P0",
                candidate_arm=f"P{self.config.evolution.generations}",
                recorder=recorder,
                run_id_prefix=f"{run_id}-test",
            )
            metrics.extend(test_metrics)
            recorder.update_delayed_rewards(self.config.diagnostics.delayed_horizons)
            trace_count = len(recorder.list_traces())

        self.last_population = list(population.operators)

        self.artifact_sink.emit(
            "run_completed",
            {
                "run_id": run_id,
                "final_population": [
                    str(operator.name) for operator in population.operators
                ],
                "test_instances": len(test_outcomes),
                "test_opened": True,
            },
        )

        return EvolutionResult(
            run_id=run_id,
            initial_population=initial_names,
            final_population=[str(operator.name) for operator in population.operators],
            generations=summaries,
            metrics=metrics,
            profiles=all_profile_rows,
            synergies=all_synergy_rows,
            test_outcomes=test_outcomes,
            trace_count=trace_count,
            retained_candidates=retained_names,
        )

    def _executor(self, operators: list[PathOperator], iterations: int, recorder: TrajectoryRecorder | None = None) -> SearchExecutor:
        return SearchExecutor(
            operators,
            self.evaluator,
            max_iterations=iterations,
            temperature_start_ratio=self.config.search.temperature_start_ratio,
            temperature_end_ratio=self.config.search.temperature_end_ratio,
            recent_window=self.config.search.recent_window,
            initializer_grid_resolution=self.config.maps.grid_resolution,
            recorder=recorder,
        )

    def _run_population(
        self,
        operators: list[PathOperator],
        environments: Iterable[Environment2D],
        iterations: int,
        recorder: TrajectoryRecorder,
        search_run_id: str,
        generation: int,
        *,
        phase: str,
        arm: str,
    ) -> list[RunMetric]:
        rows: list[RunMetric] = []
        for map_index, environment in enumerate(environments):
            seed = derive_seed(self.config.seed, phase, generation, arm, map_index, environment.map_id)
            started = time.perf_counter()
            result = self._executor(operators, iterations, recorder).run(
                environment,
                np.random.default_rng(seed),
                run_id=search_run_id,
                generation=generation,
            )
            rows.append(self._metric(result, environment, phase, generation, arm, started))
        return rows

    @staticmethod
    def _metric(
        result: SearchResult,
        environment: Environment2D,
        phase: str,
        generation: int,
        arm: str,
        started: float,
    ) -> RunMetric:
        return RunMetric(
            phase=phase,
            generation=generation,
            arm=arm,
            map_id=environment.map_id,
            difficulty=environment.difficulty,
            best_cost=float(result.best_evaluation.total_cost),
            final_cost=float(result.final_evaluation.total_cost),
            feasible=bool(result.best_evaluation.feasible),
            runtime_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _fitness_scores(profiles: list[Any]) -> dict[str, float]:
        rows: list[dict[str, Any]] = []
        for profile in profiles:
            delayed_values = [float(value) for value in (profile.mean_delayed_rewards or {}).values()]
            immediate = float(profile.mean_immediate_reward or 0.0)
            delayed = max(delayed_values or [immediate])
            rows.append(
                {
                    "operator_name": profile.operator_id,
                    # Profiles contain reward rather than absolute per-operator
                    # terminal cost, so negative reward is the comparable cost proxy.
                    "cost": -immediate,
                    "feasible": float(profile.feasibility_rate or 0.0),
                    "delayed": delayed,
                    "worst_context": min(immediate, delayed),
                    "runtime": float(profile.mean_runtime_ms),
                }
            )
        return compute_fitness(rows, policy=FitnessPolicy.UAV_LEGACY_V1)

    @staticmethod
    def _rank_parents(
        profiles: list[Any], fallback: list[str], fitness_scores: dict[str, float]
    ) -> list[str]:
        def delayed(profile: Any) -> float:
            values = getattr(profile, "mean_delayed_rewards", {}) or {}
            return max([float(value) for value in values.values()] or [float(getattr(profile, "mean_immediate_reward", 0.0) or 0.0)])

        names = [
            profile.operator_id
            for profile in sorted(
                profiles,
                key=lambda item: (fitness_scores.get(item.operator_id, 0.0), delayed(item)),
                reverse=True,
            )
        ]
        names.extend(name for name in fallback if name not in names)
        return names or fallback

    def _run_phase8_candidate(
        self,
        population: _Population,
        parent_name: str,
        parent_profile: Any,
        train_environments: list[Environment2D],
        validation_environments: list[Environment2D],
        generation: int,
        candidate_index: int,
        recorder: TrajectoryRecorder,
        memory: MechanismMemory,
        root_run_id: str,
    ) -> tuple[
        OperatorDesignOrchestrationResult,
        tuple[PathOperator, OperatorSpec] | None,
    ]:
        """Run the Phase-8 design arm without exposing the held-out test split.

        The caller supplies train and validation environments explicitly.  This
        method intentionally has no dataset dictionary (and therefore no route
        by which a test environment can enter a retention decision).
        """

        smoke_environment = self._select_smoke_environment(
            train_environments, validation_environments, candidate_index
        )
        operator_registry = OperatorRegistry(population.operators)
        proposal_validator = ProposalValidator(self.domain_kit)
        provider = (
            MockLLMProvider()
            if self.config.agent.provider == "mock"
            else OpenAIProvider()
        )
        llm_designer = LLMDesignerAdapter(
            provider=provider,
            proposal_validator=proposal_validator,
        )
        research_backend = None
        if self.config.agent.designer_mode == "single_agent":
            if self.config.agent.provider == "mock":
                research_backend = DeterministicMockResearchAgent(
                    provider,
                    validator=proposal_validator,
                )
            else:
                research_backend = OpenAIAgentsResearchAgent(
                    validator=proposal_validator,
                    remote_tracing=self.config.agent.remote_tracing,
                    trace_include_sensitive_data=self.config.agent.trace_include_sensitive_data,
                )
        elif self.config.agent.designer_mode == "multi_agent":
            if self.config.agent.provider != "mock":
                raise ValueError(
                    "designer_mode='multi_agent' is offline-only and requires provider='mock'"
                )
            research_backend = DeterministicMockMultiAgent(
                provider,
                validator=proposal_validator,
            )

        design_mode = {
            "llm_single_call": "llm_single",
            "llm_staged": "llm_staged",
            "single_agent": "single_agent",
            "multi_agent": "multi_agent",
        }[self.config.agent.designer_mode]
        profile_payload = (
            parent_profile.model_dump(mode="json")
            if hasattr(parent_profile, "model_dump")
            else dict(parent_profile)
        )
        request_id = (
            f"evolution-g{generation + 1}-c{candidate_index + 1}-"
            f"{derive_seed(self.config.seed, root_run_id, generation, candidate_index):016x}"
        )
        request = OperatorDesignRequest(
            request_id=request_id,
            experiment_id=(root_run_id or "evolution")[:200],
            root_run_id=(root_run_id or "evolution")[:200],
            problem_summary=(
                "Improve fixed-budget UAV path search using computed trajectory evidence."
            ),
            parent_operator_ids=[parent_name],
            smoke_environment=smoke_environment,
            validation_environments=list(validation_environments),
            design_mode=design_mode,
            review_mode=self.config.agent.review_mode,
            generation=generation + 1,
            candidate_index=candidate_index,
            population_operator_names=[str(operator.name) for operator in population.operators],
            parent_profiles=[profile_payload],
            design_budget=DesignBudget.model_validate(
                self.config.agent.design_budget.model_dump(mode="python")
            ),
            llm_call_config=LLMCallConfig.model_validate(
                self.config.agent.llm_call.model_dump(mode="python")
            ),
            research_agent_budget=ResearchAgentBudget.model_validate(
                self.config.agent.agent_budget.model_dump(mode="python")
            ),
        )
        with AgentAuditStore(self.database_path) as audit_store:
            orchestrator = self.orchestrator_factory(
                evidence_builder=EvidenceBundleBuilder(
                    memory,
                    operator_registry,
                    recorder=recorder,
                    minimum_reliable_samples=max(
                        1, self.config.diagnostics.minimum_context_samples
                    ),
                    domain_kit=self.domain_kit,
                ),
                proposal_validator=proposal_validator,
                compiler=self.compiler,
                candidate_validator=self.candidate_validator,
                memory=memory,
                registry=operator_registry,
                llm_designer=llm_designer,
                research_agent_backend=research_backend,
                audit_store=audit_store,
                recorder=recorder,
                domain_kit=self.domain_kit,
            )
            result = orchestrator.run(request)

        registered_candidate: tuple[PathOperator, OperatorSpec] | None = None
        if result.retained and result.operator_name is not None and result.proposal is not None:
            registered_candidate = (
                operator_registry.get(result.operator_name),
                result.proposal.spec,
            )
        return result, registered_candidate

    @staticmethod
    def _select_smoke_environment(
        train_environments: list[Environment2D],
        validation_environments: list[Environment2D],
        candidate_index: int,
    ) -> Environment2D:
        validation_ids = {environment.map_id for environment in validation_environments}
        validation_hashes = {
            environment.content_hash for environment in validation_environments
        }
        eligible = [
            environment
            for environment in train_environments
            if environment.map_id not in validation_ids
            and environment.content_hash not in validation_hashes
        ]
        if not eligible:
            raise ValueError(
                "Phase-8 design requires a train smoke map distinct from every validation map"
            )
        return eligible[candidate_index % len(eligible)]

    def _validate_candidate(
        self,
        population: list[PathOperator],
        parent_name: str,
        candidate: PathOperator,
        environments: list[Environment2D],
        generation: int,
        candidate_index: int,
        recorder: TrajectoryRecorder,
        root_run_id: str,
    ) -> ValidationReport:
        return self.candidate_validator.validate(
            population,
            parent_name,
            candidate,
            environments,
            generation=generation,
            candidate_index=candidate_index,
            recorder=recorder,
            root_run_id=root_run_id,
        )

    def _compare_populations(
        self,
        parent: list[PathOperator],
        candidate: list[PathOperator],
        environments: Iterable[Environment2D],
        iterations: int,
        *,
        phase: str,
        generation: int,
        parent_arm: str,
        candidate_arm: str,
        recorder: TrajectoryRecorder | None = None,
        run_id_prefix: str | None = None,
    ) -> tuple[list[PairedOutcome], list[RunMetric]]:
        outcomes: list[PairedOutcome] = []
        metrics: list[RunMetric] = []
        for map_index, environment in enumerate(environments):
            seed = derive_seed(self.config.seed, "paired", phase, generation, map_index, environment.map_id)
            initial = self.domain_adapter.initializer.initialize(
                environment,
                np.random.default_rng(
                    derive_seed(
                        self.config.seed,
                        "paired-initial",
                        phase,
                        generation,
                        map_index,
                        environment.map_id,
                    )
                ),
            )
            started = time.perf_counter()
            parent_result = self._executor(parent, iterations, recorder).run(
                environment,
                np.random.default_rng(seed),
                initial_path=initial,
                recorder=recorder,
                run_id=(f"{run_id_prefix}-m{map_index}-parent" if run_id_prefix else "paired-parent"),
                generation=generation,
            )
            parent_runtime = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            candidate_result = self._executor(candidate, iterations, recorder).run(
                environment,
                np.random.default_rng(seed),
                initial_path=initial,
                recorder=recorder,
                run_id=(f"{run_id_prefix}-m{map_index}-candidate" if run_id_prefix else "paired-candidate"),
                generation=generation,
            )
            candidate_runtime = (time.perf_counter() - started) * 1000.0
            outcomes.append(
                PairedOutcome(
                    map_id=environment.map_id,
                    difficulty=environment.difficulty,
                    parent_best_cost=float(parent_result.best_evaluation.total_cost),
                    candidate_best_cost=float(candidate_result.best_evaluation.total_cost),
                    parent_feasible=bool(parent_result.best_evaluation.feasible),
                    candidate_feasible=bool(candidate_result.best_evaluation.feasible),
                    parent_runtime_ms=parent_runtime,
                    candidate_runtime_ms=candidate_runtime,
                )
            )
            metrics.extend(
                [
                    RunMetric(
                        phase=phase,
                        generation=generation,
                        arm=parent_arm,
                        map_id=environment.map_id,
                        difficulty=environment.difficulty,
                        best_cost=float(parent_result.best_evaluation.total_cost),
                        final_cost=float(parent_result.final_evaluation.total_cost),
                        feasible=bool(parent_result.best_evaluation.feasible),
                        runtime_ms=parent_runtime,
                    ),
                    RunMetric(
                        phase=phase,
                        generation=generation,
                        arm=candidate_arm,
                        map_id=environment.map_id,
                        difficulty=environment.difficulty,
                        best_cost=float(candidate_result.best_evaluation.total_cost),
                        final_cost=float(candidate_result.final_evaluation.total_cost),
                        feasible=bool(candidate_result.best_evaluation.feasible),
                        runtime_ms=candidate_runtime,
                    ),
                ]
            )
        return outcomes, metrics

    @staticmethod
    def _register_initial_memory(memory: MechanismMemory, population: _Population) -> None:
        for operator in population.operators:
            name = str(operator.name)
            memory.add_mechanism(
                name,
                population.specs[name].model_dump(mode="json"),
                name=name,
                description=population.specs[name].description,
                tags=["operator", "manual", "generation-0"],
                metadata={
                    "generation": 0,
                    "source": "manual",
                    "code_version": "0.1.0",
                    "creation_reason": "initial manual operator library",
                    "active_status": True,
                },
            )

    @staticmethod
    def _write_diagnostics_to_memory(
        memory: MechanismMemory,
        population: _Population,
        traces: list[Any],
        profiles: list[Any],
        synergies: list[Any],
        generation: int,
    ) -> None:
        for trace in traces:
            memory.record_operator_history(trace, mechanism_id=trace.operator_id if trace.operator_id in population.specs else None)
        for profile in profiles:
            spec = population.specs.get(profile.operator_id)
            memory.add_operator_profile(
                profile.model_dump(mode="json"),
                operator_id=profile.operator_id,
                run_id=traces[0].run_id if traces else None,
                generation=generation,
            )
            if spec is not None:
                delayed_values = list((profile.mean_delayed_rewards or {}).values())
                memory.add_mechanism(
                    profile.operator_id,
                    spec.model_dump(mode="json"),
                    name=profile.operator_id,
                    description=spec.description,
                    score=max(delayed_values or [profile.mean_immediate_reward or 0.0]),
                    evidence_count=profile.attempts,
                    success_rate=profile.success_rate,
                    tags=["operator", "profiled", f"generation-{generation}"],
                    metadata={"generation": generation, "profile": profile.model_dump(mode="json")},
                )
            for mode, count in profile.failure_modes.items():
                memory.add_failure_mode(
                    mode,
                    mechanism_id=profile.operator_id if spec is not None else None,
                    operator_id=profile.operator_id,
                    count=count,
                    evidence=profile.representative_failure_ids,
                    metadata={"generation": generation, "evidence_type": "association"},
                )
        for relation in synergies:
            memory.add_synergy(
                relation.first_operator,
                relation.second_operator,
                relation.synergy,
                sample_count=relation.occurrences,
                context=relation.context,
                metadata={"generation": generation, "evidence_type": "association"},
            )


__all__ = ["EvolutionResult", "GenerationSummary", "OperatorEvolutionManager", "RunMetric"]
