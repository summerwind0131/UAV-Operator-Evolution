"""Isolated repeated median benchmark for the Phase-1 manager critical path."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def _worker(source_root: Path, output_root: Path, index: int) -> dict[str, float | int]:
    sys.path.insert(0, str(source_root / "src"))

    from uav_operator_evolution.config import load_config
    from uav_operator_evolution.environment.generator import MapGenerator
    from uav_operator_evolution.evolution.manager import OperatorEvolutionManager

    config = load_config(source_root / "configs" / "smoke.yaml").model_copy(deep=True)
    config.search.train_iterations = 16
    config.search.validation_iterations = 8
    config.search.test_iterations = 8
    config.evolution.generations = 1
    config.evolution.candidates_per_generation = 1
    config.evolution.min_runtime_reduction = 1.0
    config.evolution.min_runtime_effective_call_rate = 1.0
    generator = MapGenerator(
        config.seed,
        grid_resolution=5.0,
        generation_attempts=20,
    )
    datasets = {
        "train": [
            generator.generate_map(
                "train", 0, "sparse", width=50, height=50, safety_distance=1
            )
        ],
        "validation": [
            generator.generate_map(
                "validation",
                0,
                "medium",
                width=50,
                height=50,
                safety_distance=1,
            )
        ],
        "test": [
            generator.generate_map(
                "test", 0, "dense", width=55, height=55, safety_distance=1
            )
        ],
    }
    database = output_root / f"manager-benchmark-{index}.sqlite"
    started = time.perf_counter()
    result = OperatorEvolutionManager(config, database).run(
        datasets,
        f"manager-benchmark-{index}",
    )
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    return {
        "elapsed_ms": elapsed_ms,
        "trace_count": result.trace_count,
        "final_population_size": len(result.final_population),
    }


def main() -> int:
    args = _parser().parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if args.repetitions < 3:
        raise ValueError("repetitions must be at least 3 for a median benchmark")
    output_root.mkdir(parents=True, exist_ok=True)

    if args.worker:
        print(json.dumps(_worker(source_root, output_root, args.worker_index)))
        return 0

    samples: list[float] = []
    trace_counts: set[int] = set()
    population_sizes: set[int] = set()
    script = Path(__file__).resolve()
    for index in range(args.repetitions):
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--source-root",
                str(source_root),
                "--output-root",
                str(output_root),
                "--repetitions",
                str(args.repetitions),
                "--worker",
                "--worker-index",
                str(index),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        samples.append(float(payload["elapsed_ms"]))
        trace_counts.add(int(payload["trace_count"]))
        population_sizes.add(int(payload["final_population_size"]))
    if len(trace_counts) != 1 or len(population_sizes) != 1:
        raise RuntimeError("benchmark repetitions produced inconsistent semantics")
    print(
        json.dumps(
            {
                "schema": "phase1-manager-benchmark-v1",
                "source_root": str(source_root),
                "repetitions": args.repetitions,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "trace_count": next(iter(trace_counts)),
                "final_population_size": next(iter(population_sizes)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
