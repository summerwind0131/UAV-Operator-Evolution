"""Run-directory and logging helpers used by CLI workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Filesystem locations belonging to one immutable experiment run."""

    run_id: str
    result_dir: Path
    figure_dir: Path
    database: Path
    log_file: Path

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        command: str,
        run_id: str | None = None,
        run_dir: str | Path | None = None,
    ) -> "RunPaths":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_command = command.replace("_", "-")
        identifier = run_id or f"{safe_command}-{stamp}-{config.config_hash[:8]}"
        result_dir = Path(run_dir) if run_dir else config.output.results_dir / identifier
        figure_dir = config.output.figures_dir / identifier
        result_dir.mkdir(parents=True, exist_ok=False if run_dir is None else True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            run_id=identifier,
            result_dir=result_dir,
            figure_dir=figure_dir,
            database=result_dir / "experiment.sqlite",
            log_file=result_dir / "run.log",
        )


def configure_logging(log_file: Path | None = None, verbose: bool = False) -> None:
    """Configure concise console logging and an optional UTF-8 run log."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
