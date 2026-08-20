"""Bounded subprocess execution for generated AFL-UAV solver artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from .models import ExecutionReport
from .validation import GeneratedCodePolicy


class GeneratedSolverRunner:
    """Run a generated solver with time, output, environment, and AST limits.

    The runner reduces accidental exposure but is not a system-level sandbox.
    Real-model-generated code therefore requires an explicit opt-in at the
    workflow boundary.
    """

    def __init__(
        self,
        *,
        policy: GeneratedCodePolicy | None = None,
        max_output_chars: int = 20_000,
    ) -> None:
        self.policy = policy or GeneratedCodePolicy()
        self.max_output_chars = max_output_chars

    def execute(
        self,
        *,
        solver_path: str | Path,
        source: str,
        instance_path: str | Path,
        output_path: str | Path,
        iterations: int,
        timeout_seconds: float,
        max_source_chars: int,
        seed: int | None = None,
        max_evaluations: int | None = None,
    ) -> ExecutionReport:
        issues = self.policy.validate(source, max_source_chars=max_source_chars)
        if issues:
            return ExecutionReport(
                status="policy_rejected",
                error="; ".join(issues),
            )
        solver = Path(solver_path).resolve()
        instance = Path(instance_path).resolve()
        output = Path(output_path).resolve()
        if not solver.is_file():
            return ExecutionReport(status="output_error", error=f"solver not found: {solver}")
        if not instance.is_file():
            return ExecutionReport(status="output_error", error=f"instance not found: {instance}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        command = [
            sys.executable,
            "-I",
            str(solver),
            "--path",
            str(instance),
            "--iteration",
            str(max(0, int(iterations))),
            "--output",
            str(output),
        ]
        if seed is not None:
            command.extend(["--seed", str(int(seed))])
        if max_evaluations is not None:
            command.extend(["--max-evaluations", str(max(1, int(max_evaluations)))])
        safe_environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH"}
        }
        safe_environment["PYTHONHASHSEED"] = "0"
        safe_environment["PYTHONIOENCODING"] = "utf-8"
        started = perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(solver.parent),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                env=safe_environment,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = (perf_counter() - started) * 1_000.0
            return ExecutionReport(
                status="timeout",
                duration_ms=duration,
                stdout=self._clip(exc.stdout),
                stderr=self._clip(exc.stderr),
                error=f"solver exceeded {timeout_seconds:.3f}s timeout",
            )
        except OSError as exc:
            duration = (perf_counter() - started) * 1_000.0
            return ExecutionReport(
                status="runtime_error",
                duration_ms=duration,
                error=f"failed to start solver: {type(exc).__name__}: {exc}",
            )
        duration = (perf_counter() - started) * 1_000.0
        stdout = self._clip(completed.stdout)
        stderr = self._clip(completed.stderr)
        if completed.returncode != 0:
            return ExecutionReport(
                status="runtime_error",
                return_code=completed.returncode,
                duration_ms=duration,
                stdout=stdout,
                stderr=stderr,
                error=f"solver exited with code {completed.returncode}",
            )
        if not output.is_file():
            return ExecutionReport(
                status="output_error",
                return_code=completed.returncode,
                duration_ms=duration,
                stdout=stdout,
                stderr=stderr,
                error="solver exited successfully but did not create the JSON output",
            )
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ExecutionReport(
                status="output_error",
                return_code=completed.returncode,
                duration_ms=duration,
                stdout=stdout,
                stderr=stderr,
                error=f"invalid solver output JSON: {type(exc).__name__}: {exc}",
            )
        if not isinstance(payload, dict):
            return ExecutionReport(
                status="output_error",
                return_code=completed.returncode,
                duration_ms=duration,
                stdout=stdout,
                stderr=stderr,
                error="solver output JSON must be an object",
            )
        return ExecutionReport(
            status="success",
            return_code=completed.returncode,
            duration_ms=duration,
            stdout=stdout,
            stderr=stderr,
            output_payload=payload,
        )

    def _clip(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        else:
            text = str(value)
        return text[: self.max_output_chars]


__all__ = ["GeneratedSolverRunner"]
