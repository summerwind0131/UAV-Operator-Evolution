"""Stable seed and content hashing utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    """Return a SHA-256 hash for a JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def derive_seed(master_seed: int, *parts: object) -> int:
    """Derive a stable NumPy/SQLite-compatible non-negative 63-bit seed."""

    payload = canonical_json([int(master_seed), *[str(part) for part in parts]])
    # SQLite INTEGER is signed 64-bit, so reserve the sign bit while retaining
    # vastly more independent streams than an experiment can consume.
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def make_rng(master_seed: int, *parts: object) -> np.random.Generator:
    """Create an independent deterministic random generator."""

    return np.random.default_rng(derive_seed(master_seed, *parts))
