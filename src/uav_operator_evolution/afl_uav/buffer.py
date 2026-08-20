"""Content-addressed solver buffer mirroring AFL's description-to-code reuse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..reproducibility import canonical_json, stable_hash
from .models import UAVProblemDescription, UAVSolverInstance
from .validation import STAGE_CONTRACTS


class SolverBuffer:
    """Persist only independently validated solver source for later map instances."""

    schema_version = "afl-uav-solver-buffer-v4"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @classmethod
    def key(
        cls,
        description: UAVProblemDescription,
        instance: UAVSolverInstance,
    ) -> str:
        contract = description.model_dump(mode="json")
        contract.pop("source_hash", None)
        payload = {
            "schema_version": cls.schema_version,
            "problem_contract": contract,
            "instance_schema": instance.schema_version,
            "objective_weights": instance.objective_weights.model_dump(mode="json"),
            "grid_resolution": instance.grid_resolution,
            "max_waypoints": instance.max_waypoints,
            "function_contracts": [
                {
                    "name": item.name,
                    "primary_function": item.primary_function,
                    "signature": item.signature,
                }
                for item in STAGE_CONTRACTS
            ],
        }
        return stable_hash(payload)

    def load(self, key: str) -> str | None:
        source_path = self.root / f"{key}.py"
        metadata_path = self.root / f"{key}.json"
        if not source_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source = source_path.read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            return None
        if metadata.get("schema_version") != self.schema_version:
            return None
        if metadata.get("key") != key:
            return None
        if metadata.get("source_hash") != stable_hash({"source": source}):
            return None
        if metadata.get("externally_validated") is not True:
            return None
        return source

    def store(
        self,
        key: str,
        source: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        self.root.mkdir(parents=True, exist_ok=True)
        source_path = self.root / f"{key}.py"
        metadata_path = self.root / f"{key}.json"
        source_path.write_text(source, encoding="utf-8")
        payload = {
            **dict(metadata or {}),
            "schema_version": self.schema_version,
            "key": key,
            "source_hash": stable_hash({"source": source}),
            "externally_validated": True,
        }
        metadata_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        return source_path, metadata_path


__all__ = ["SolverBuffer"]
