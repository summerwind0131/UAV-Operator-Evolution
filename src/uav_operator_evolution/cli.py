"""Command-line entry point for reproducible UAV operator-evolution experiments."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from .config import ExperimentConfig, load_config
from .experiments.common import (
    ensure_dataset,
    ensure_dataset_split,
    resolve_run_dir,
    update_latest,
)
from .experiments.diagnose import run_diagnosis_workflow
from .experiments.generate_maps import run_generate_maps
from .experiments.run_baselines import run_baselines_workflow
from .experiments.run_evolution import run_evolution_workflow
from .experiments.run_search import run_search_workflow
from .experiments.summarize import summarize_run
from .runtime import RunPaths, configure_logging

LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uav_operator_evolution.cli",
        description="Trajectory-informed operator evolution for UAV path planning",
    )
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def configured(name: str, help_text: str, *, run_options: bool = False) -> argparse.ArgumentParser:
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", default="configs/smoke.yaml", help="experiment YAML")
        if run_options:
            command.add_argument("--run-id", help="explicit experiment identifier")
            command.add_argument("--run-dir", help="explicit result directory")
        return command

    generate = configured("generate-maps", "generate deterministic train/validation/test maps")
    generate.add_argument("--force", action="store_true", help="replace generated files after validation")
    configured("run-search", "run the fixed P0 search and collect trajectories", run_options=True)
    diagnose = configured("diagnose", "diagnose an existing trajectory database")
    diagnose.add_argument("--run-dir", help="run containing experiment.sqlite; defaults to latest")
    configured("evolve", "run the configured operator generations", run_options=True)
    configured("demo", "run the complete offline research loop", run_options=True)
    configured("run-baselines", "run all six baseline arms", run_options=True)
    planner_benchmark = configured(
        "benchmark-planners",
        "run fair-budget UAV path-planning baselines",
        run_options=True,
    )
    planner_benchmark.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    planner_benchmark.add_argument(
        "--planners",
        nargs="+",
        help="planner ids; defaults to all configured planners",
    )
    planner_benchmark.add_argument("--maps-per-class", type=int)
    planner_benchmark.add_argument("--time-limit", type=float)
    planner_benchmark.add_argument("--max-evaluations", type=int)
    planner_benchmark.add_argument("--repetitions", type=int)
    planner_benchmark.add_argument(
        "--afl-artifact",
        action="append",
        metavar="ARM_ID=PATH",
        help="repeatable frozen AFL-UAV arm mapping, for example openai_gpt41=artifacts/run",
    )
    planner_benchmark.add_argument(
        "--evolutionary-afl-artifact",
        action="append",
        metavar="ARM_ID=PATH",
        help=(
            "repeatable offline Evolutionary AFL-UAV arm seeded by a frozen "
            "artifact; never calls the provider during planning"
        ),
    )
    summarize = configured("summarize", "summarize an existing run")
    summarize.add_argument("--run-dir", help="run directory; defaults to latest")

    build_evidence = configured(
        "build-evidence",
        "build and persist a bounded structured evidence bundle",
    )
    build_evidence.add_argument("--run-dir", help="run directory; defaults to latest")

    propose = configured(
        "propose-operator",
        "generate and review one structured LLM operator proposal",
    )
    propose.add_argument("--run-dir", help="run directory; defaults to latest")
    propose.add_argument("--provider", choices=("mock", "openai"), help="LLM provider")
    propose.add_argument(
        "--mode",
        choices=("single_call", "staged"),
        default="single_call",
        help="structured design protocol",
    )
    propose.add_argument("--model", help="explicit provider model")

    run_agent = configured(
        "run-agent",
        "run the bounded research-agent loop through compile and smoke",
    )
    run_agent.add_argument("--run-dir", help="run directory; defaults to latest")
    run_agent.add_argument("--provider", choices=("mock", "openai"), help="agent provider")
    run_agent.add_argument(
        "--agent-mode",
        choices=("single_agent", "multi_agent"),
        help="agent topology; defaults to the configured agent designer mode",
    )
    run_agent.add_argument("--model", help="explicit provider model")

    validate = configured(
        "validate-candidate",
        "validate an audited candidate on the validation split only",
    )
    validate.add_argument("--run-dir", help="run directory; defaults to latest")
    validate.add_argument("--candidate-id", required=True, help="audited candidate identifier")

    agent_demo = configured(
        "agent-demo",
        "run the complete offline Phase-8 agent experiment",
        run_options=True,
    )
    agent_demo.add_argument(
        "--provider",
        choices=("mock", "openai"),
        default="mock",
        help="agent provider (offline mock by default)",
    )
    agent_demo.add_argument(
        "--agent-mode",
        choices=("single_agent", "multi_agent"),
        help="agent topology; defaults to the configured agent designer mode",
    )
    agent_demo.add_argument("--model", help="explicit provider model")

    ablations = configured(
        "run-agent-ablations",
        "compare deterministic Phase-8 designer and agent arms",
        run_options=True,
    )
    ablations.add_argument(
        "--provider",
        choices=("mock", "openai"),
        default="mock",
        help="LLM provider",
    )
    ablations.add_argument("--model", help="explicit provider model")

    afl_uav = configured(
        "afl-uav-demo",
        "run the legacy offline mock AFL-UAV demo on one Train map",
        run_options=True,
    )
    afl_uav.add_argument(
        "--provider",
        choices=("mock",),
        default="mock",
        help="structured provider; offline mock by default",
    )
    afl_uav.add_argument("--model", help="explicit provider model")
    afl_uav.add_argument("--map", dest="map_path", help="optional Environment2D JSON map")
    afl_uav.add_argument(
        "--iteration",
        type=int,
        default=100,
        help="improvement iterations passed to the generated solver",
    )
    afl_uav.add_argument(
        "--execute-untrusted-code",
        action="store_true",
        help="explicitly execute real-model-generated Python; AST checks are not an OS sandbox",
    )
    build_afl_uav = configured(
        "build-afl-uav",
        "legacy one-step offline mock artifact builder",
        run_options=True,
    )
    build_afl_uav.add_argument(
        "--provider",
        choices=("mock",),
        default="mock",
    )
    build_afl_uav.add_argument("--model", help="explicit model for real generation")
    build_afl_uav.add_argument(
        "--map",
        dest="map_path",
        help="optional Train Environment2D JSON; defaults to the first Train map",
    )
    build_afl_uav.add_argument(
        "--execute-untrusted-code",
        action="store_true",
        help="allow execution of reviewed real-model-generated Python",
    )
    generate_candidate = configured(
        "generate-afl-uav-candidate",
        "generate and save AFL-UAV source without executing it",
        run_options=True,
    )
    generate_candidate.add_argument(
        "--provider",
        choices=("mock", "openai", "deepseek", "gemini"),
        required=True,
    )
    generate_candidate.add_argument(
        "--model",
        help="explicit model; required for every real provider",
    )
    generate_candidate.add_argument(
        "--map",
        dest="map_path",
        help="optional Train Environment2D JSON; defaults to the first Train map",
    )
    revise_candidate = configured(
        "revise-afl-uav-candidate",
        "revise a saved candidate after static audit without executing it",
        run_options=True,
    )
    revise_candidate.add_argument(
        "--candidate",
        required=True,
        help="base candidate directory or candidate.json",
    )
    adopt_candidate = configured(
        "adopt-afl-uav-audited-source",
        "hash-approve a failed post-audit source for restricted qualification without execution",
        run_options=True,
    )
    adopt_candidate.add_argument("--base-candidate", required=True)
    adopt_candidate.add_argument("--failed-audit", required=True)
    adopt_candidate.add_argument("--approve-source-hash", required=True)
    freeze_candidate = configured(
        "freeze-afl-uav",
        "execute and freeze only a hash-approved AFL-UAV candidate",
        run_options=True,
    )
    freeze_candidate.add_argument(
        "--candidate",
        required=True,
        help="candidate directory or candidate.json",
    )
    freeze_candidate.add_argument(
        "--approve-source-hash",
        required=True,
        help="exact SHA-256-like stable source hash shown by generation",
    )
    return parser


def _new_paths(config: ExperimentConfig, args: argparse.Namespace) -> RunPaths:
    return RunPaths.create(
        config,
        args.command,
        run_id=getattr(args, "run_id", None),
        run_dir=getattr(args, "run_dir", None),
    )


def _print(payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _parse_afl_artifacts(values: list[str] | None) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for value in values or []:
        arm_id, separator, path = value.partition("=")
        if not separator or not arm_id or not path:
            raise ValueError("--afl-artifact must use ARM_ID=PATH")
        if arm_id in artifacts:
            raise ValueError(f"duplicate AFL-UAV arm_id: {arm_id}")
        artifacts[arm_id] = path
    return artifacts


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(verbose=bool(args.verbose))
    LOGGER.info("command=%s config=%s hash=%s", args.command, config.name, config.config_hash[:8])

    if args.command == "generate-maps":
        manifest = run_generate_maps(config, overwrite=args.force)
        _print(
            {
                "manifest": str((config.output.data_dir / "manifest.json").resolve()),
                "manifest_hash": manifest.manifest_hash,
                "maps": len(manifest.maps),
                "splits": {
                    split: len(manifest.entries_for(split))
                    for split in ("train", "validation", "test")
                },
            }
        )
        return 0

    if args.command in {"diagnose", "summarize"}:
        directory = resolve_run_dir(config, args.run_dir)
        if args.command == "diagnose":
            report = run_diagnosis_workflow(
                config,
                directory,
                figure_dir=config.output.figures_dir / directory.name,
            )
            _print(
                {
                    "run_dir": str(directory.resolve()),
                    "trace_count": report["trace_count"],
                    "operators": len(report["operator_profiles"]),
                    "synergies": len(report["synergies"]),
                }
            )
        else:
            _print(summarize_run(directory))
        return 0

    if args.command in {
        "build-evidence",
        "propose-operator",
        "run-agent",
        "validate-candidate",
    }:
        from .experiments.agent_workflows import (
            build_evidence_workflow,
            propose_operator_workflow,
            run_agent_workflow,
            validate_candidate_workflow,
        )

        directory = resolve_run_dir(config, args.run_dir)
        datasets = ensure_dataset(config)
        if args.command == "build-evidence":
            _print(
                build_evidence_workflow(
                    config,
                    directory,
                    train_maps=datasets["train"],
                )
            )
        elif args.command == "propose-operator":
            _print(
                propose_operator_workflow(
                    config,
                    directory,
                    provider=args.provider,
                    mode=args.mode,
                    train_maps=datasets["train"],
                    model=args.model,
                )
            )
        elif args.command == "run-agent":
            _print(
                run_agent_workflow(
                    config,
                    directory,
                    provider=args.provider,
                    agent_mode=args.agent_mode,
                    train_maps=datasets["train"],
                    model=args.model,
                )
            )
        else:
            _print(
                validate_candidate_workflow(
                    config,
                    directory,
                    args.candidate_id,
                    datasets["validation"],
                    forbidden_map_hashes={item.content_hash for item in datasets["test"]},
                )
            )
        return 0

    if args.command in {"agent-demo", "run-agent-ablations"}:
        from .experiments.agent_workflows import (
            agent_demo_workflow,
            run_agent_ablations_workflow,
        )

        paths = _new_paths(config, args)
        configure_logging(paths.log_file, verbose=bool(args.verbose))
        if args.command == "agent-demo":
            report = agent_demo_workflow(
                config,
                provider=args.provider,
                agent_mode=args.agent_mode,
                paths=paths,
                model=args.model,
            )
        else:
            report = run_agent_ablations_workflow(
                config,
                provider=args.provider,
                paths=paths,
                model=args.model,
            )
        _print(report)
        return 0

    if args.command == "afl-uav-demo":
        from .experiments.afl_uav import load_afl_uav_environment, run_afl_uav_workflow

        paths = _new_paths(config, args)
        configure_logging(paths.log_file, verbose=bool(args.verbose))
        datasets = ensure_dataset(config)
        environment = load_afl_uav_environment(args.map_path, datasets["train"][0])
        report = run_afl_uav_workflow(
            config,
            paths,
            environment,
            provider=args.provider,
            model=args.model,
            iterations=args.iteration,
            execute_untrusted_code=bool(args.execute_untrusted_code),
        )
        _print(report)
        return 0

    if args.command == "generate-afl-uav-candidate":
        from .afl_uav.artifact import generate_solver_candidate
        from .experiments.afl_uav import load_afl_uav_environment

        if args.provider != "mock" and not args.model:
            raise ValueError("real AFL-UAV providers require an explicit --model")
        train_maps = ensure_dataset_split(config, "train")
        environment = load_afl_uav_environment(
            args.map_path,
            train_maps[0],
        )
        if all(
            environment.geometry_hash != candidate.geometry_hash
            for candidate in train_maps
        ):
            raise ValueError("AFL-UAV candidates may only be generated from Train maps")
        candidate_run_id = args.run_id or (
            f"afl-uav-candidate-{args.provider}-{environment.map_id}"
        )
        candidate_dir = (
            Path(args.run_dir)
            if args.run_dir is not None
            else config.output.results_dir
            / "afl_uav_candidates"
            / candidate_run_id
        )
        candidate, manifest_path = generate_solver_candidate(
            config,
            environment,
            candidate_dir,
            provider=args.provider,
            model=args.model,
        )
        _print(
            {
                "candidate_id": candidate.candidate_id,
                "candidate": str(manifest_path.resolve()),
                "candidate_source": str(
                    (manifest_path.parent / candidate.solver_filename).resolve()
                ),
                "source_hash_to_approve": candidate.solver_hash,
                "provider": candidate.provider,
                "model": candidate.model,
                "logical_calls": candidate.usage.logical_calls,
                "total_tokens": candidate.usage.total_tokens,
                "executed": False,
            }
        )
        return 0

    if args.command == "freeze-afl-uav":
        from .afl_uav.artifact import (
            QUALIFICATION_MAP_IDS,
            freeze_solver_candidate,
            load_solver_candidate,
        )

        train_maps = ensure_dataset_split(config, "train")
        train_by_id = {item.map_id: item for item in train_maps}
        missing = [map_id for map_id in QUALIFICATION_MAP_IDS if map_id not in train_by_id]
        if missing:
            raise ValueError(
                "fixed Train qualification maps are missing: " + ", ".join(missing)
            )
        candidate, _, _ = load_solver_candidate(args.candidate)
        artifact_run_id = args.run_id or (
            f"afl-uav-{candidate.provider}-{candidate.candidate_id[:12]}"
        )
        artifact_dir = (
            Path(args.run_dir)
            if args.run_dir is not None
            else config.output.results_dir
            / "afl_uav_artifacts"
            / artifact_run_id
        )
        artifact, manifest_path = freeze_solver_candidate(
            config,
            args.candidate,
            [train_by_id[map_id] for map_id in QUALIFICATION_MAP_IDS],
            artifact_dir,
            approved_source_hash=args.approve_source_hash,
        )
        _print(
            {
                "artifact_id": artifact.artifact_id,
                "artifact": str(manifest_path.resolve()),
                "approved_source_hash": artifact.approved_source_hash,
                "provider": artifact.provider,
                "model": artifact.model,
                "qualification_maps": len(artifact.qualification_results),
                "research_claim_eligible": artifact.research_claim_eligible,
            }
        )
        return 0

    if args.command == "revise-afl-uav-candidate":
        from .afl_uav.artifact import revise_solver_candidate_after_audit

        candidate_run_id = args.run_id or "afl-uav-post-audit-revision"
        candidate_dir = (
            Path(args.run_dir)
            if args.run_dir is not None
            else config.output.results_dir
            / "afl_uav_candidates"
            / candidate_run_id
        )
        candidate, manifest_path = revise_solver_candidate_after_audit(
            config,
            args.candidate,
            candidate_dir,
        )
        _print(
            {
                "candidate_id": candidate.candidate_id,
                "base_candidate": str(Path(args.candidate).resolve()),
                "candidate": str(manifest_path.resolve()),
                "candidate_source": str(
                    (manifest_path.parent / candidate.solver_filename).resolve()
                ),
                "source_hash_to_approve": candidate.solver_hash,
                "provider": candidate.provider,
                "model": candidate.model,
                "logical_calls": candidate.usage.logical_calls,
                "total_tokens": candidate.usage.total_tokens,
                "executed": False,
            }
        )
        return 0

    if args.command == "adopt-afl-uav-audited-source":
        from .afl_uav.artifact import adopt_rejected_source_after_human_review

        candidate_run_id = args.run_id or "afl-uav-human-reviewed-source"
        candidate_dir = (
            Path(args.run_dir)
            if args.run_dir is not None
            else config.output.results_dir
            / "afl_uav_candidates"
            / candidate_run_id
        )
        candidate, manifest_path = adopt_rejected_source_after_human_review(
            args.base_candidate,
            args.failed_audit,
            candidate_dir,
            approved_source_hash=args.approve_source_hash,
        )
        _print(
            {
                "candidate_id": candidate.candidate_id,
                "candidate": str(manifest_path.resolve()),
                "candidate_source": str(
                    (manifest_path.parent / candidate.solver_filename).resolve()
                ),
                "source_hash_to_approve": candidate.solver_hash,
                "logical_calls": candidate.usage.logical_calls,
                "total_tokens": candidate.usage.total_tokens,
                "human_review": candidate.human_review,
                "executed": False,
            }
        )
        return 0

    if args.command == "build-afl-uav":
        from .afl_uav.artifact import build_solver_artifact
        from .experiments.afl_uav import load_afl_uav_environment

        datasets = ensure_dataset(config)
        environment = load_afl_uav_environment(
            args.map_path,
            datasets["train"][0],
        )
        if all(
            environment.geometry_hash != candidate.geometry_hash
            for candidate in datasets["train"]
        ):
            raise ValueError("AFL-UAV solver artifacts may only be generated from Train maps")
        artifact_run_id = args.run_id or (
            f"afl-uav-{args.provider}-{environment.map_id}"
        )
        artifact_dir = (
            Path(args.run_dir)
            if args.run_dir is not None
            else config.output.results_dir
            / "afl_uav_artifacts"
            / artifact_run_id
        )
        artifact, manifest_path = build_solver_artifact(
            config,
            environment,
            artifact_dir,
            provider=args.provider,
            model=args.model,
            execute_untrusted_code=bool(args.execute_untrusted_code),
        )
        _print(
            {
                "artifact_id": artifact.artifact_id,
                "artifact": str(manifest_path.resolve()),
                "solver_hash": artifact.solver_hash,
                "provider": artifact.provider,
                "model": artifact.model,
                "generated_from_map_id": artifact.generated_from_map_id,
                "role_events": len(artifact.role_events),
                "provider_calls": len(artifact.provider_calls),
                "research_claim_eligible": artifact.research_claim_eligible,
            }
        )
        return 0

    if args.command == "benchmark-planners":
        from .planning_benchmarks.runner import run_planner_benchmark

        report = run_planner_benchmark(
            config,
            split=args.split,
            planners=args.planners,
            maps_per_class=args.maps_per_class,
            time_limit_seconds=args.time_limit,
            max_objective_evaluations=args.max_evaluations,
            repetitions=args.repetitions,
            afl_artifacts=_parse_afl_artifacts(args.afl_artifact),
            evolutionary_afl_artifacts=_parse_afl_artifacts(
                args.evolutionary_afl_artifact
            ),
            run_id=args.run_id,
            run_dir=args.run_dir,
        )
        _print(report)
        return 0

    paths = _new_paths(config, args)
    configure_logging(paths.log_file, verbose=bool(args.verbose))
    datasets = ensure_dataset(config)
    if args.command == "run-search":
        summary = run_search_workflow(config, paths, datasets["train"])
        update_latest(config, paths.run_id, paths.result_dir)
        _print(summary)
        return 0
    if args.command in {"evolve", "demo"}:
        result, figures = run_evolution_workflow(config, paths, datasets)
        diagnosis = run_diagnosis_workflow(config, paths.result_dir, figure_dir=paths.figure_dir)
        update_latest(config, paths.run_id, paths.result_dir)
        _print(
            {
                "run_id": paths.run_id,
                "run_dir": str(paths.result_dir.resolve()),
                "database": str(paths.database.resolve()),
                "generations": len(result.generations),
                "trace_count": result.trace_count,
                "operators_profiled": len(diagnosis["operator_profiles"]),
                "candidates_generated": sum(len(item.proposals) for item in result.generations),
                "retained_candidates": result.retained_candidates,
                "test_pairs": len(result.test_outcomes),
                "figures": [str(Path(path).resolve()) for path in figures],
            }
        )
        return 0
    if args.command == "run-baselines":
        report = run_baselines_workflow(config, paths, datasets)
        update_latest(config, paths.run_id, paths.result_dir)
        _print(report)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
