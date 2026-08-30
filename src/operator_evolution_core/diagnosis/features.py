"""Versioned mappings from domain feature groups to generic trace paths."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FeatureCatalog:
    """Resolve stable domain group names without teaching core their meaning."""

    domain_id: str
    version: str
    groups: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9._-]{0,99}", self.domain_id):
            raise ValueError("domain_id must be a lowercase stable identifier")
        if not self.version.strip():
            raise ValueError("feature catalog version must not be empty")
        for name, path in self.groups.items():
            if not name.strip() or not path.strip():
                raise ValueError("feature catalog names and paths must not be empty")

    def resolve(self, name: str) -> str:
        """Resolve a registered alias, preserving explicit legacy paths."""

        normalized = str(name)
        return self.groups.get(normalized, normalized)


__all__ = ["FeatureCatalog"]

