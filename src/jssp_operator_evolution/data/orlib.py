"""Strict parser for the archived OR-Library ``jobshop1`` text format."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..models import JobShopInstance, Operation

ORLIB_JOBSHOP1_URL = (
    "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt"
)
ORLIB_JOBSHOP1_SHA256 = (
    "7f36d103332f94cfdeb76358e40927a57cfa7a426ebeaabd4de666350c69f028"
)
ORLIB_EXPECTED_INSTANCES = 82
_INSTANCE = re.compile(r"^\s*instance\s+(\S+)\s*$", re.IGNORECASE)
_FAMILY = re.compile(r"^[a-z]+", re.IGNORECASE)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _meaningful(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) != {"+"}


def parse_jobshop1(
    path: str | Path,
    *,
    verify_archive_hash: bool = True,
) -> tuple[JobShopInstance, ...]:
    source_path = Path(path)
    if verify_archive_hash:
        actual = sha256_file(source_path)
        if actual != ORLIB_JOBSHOP1_SHA256:
            raise ValueError(
                f"jobshop1 archive hash mismatch: expected "
                f"{ORLIB_JOBSHOP1_SHA256}, got {actual}"
            )
    lines = source_path.read_text(encoding="ascii").splitlines()
    instances: list[JobShopInstance] = []
    cursor = 0
    while cursor < len(lines):
        match = _INSTANCE.match(lines[cursor])
        if match is None:
            cursor += 1
            continue
        name = match.group(1).lower()
        cursor += 1
        meaningful: list[str] = []
        while cursor < len(lines) and len(meaningful) < 2:
            if _meaningful(lines[cursor]):
                meaningful.append(lines[cursor].strip())
            cursor += 1
        if len(meaningful) != 2:
            raise ValueError(f"instance {name} has incomplete description/header")
        description, shape_line = meaningful
        shape = [int(value) for value in shape_line.split()]
        if len(shape) != 2:
            raise ValueError(f"instance {name} has invalid jobs/machines header")
        job_count, machine_count = shape
        jobs: list[tuple[Operation, ...]] = []
        while cursor < len(lines) and len(jobs) < job_count:
            line = lines[cursor]
            cursor += 1
            if not _meaningful(line):
                continue
            values = [int(value) for value in line.split()]
            if len(values) != machine_count * 2:
                raise ValueError(
                    f"instance {name} job {len(jobs)} has {len(values)} values; "
                    f"expected {machine_count * 2}"
                )
            jobs.append(
                tuple(
                    Operation(machine=values[offset], duration=values[offset + 1])
                    for offset in range(0, len(values), 2)
                )
            )
        if len(jobs) != job_count:
            raise ValueError(f"instance {name} has incomplete job rows")
        family_match = _FAMILY.match(name)
        if family_match is None:
            raise ValueError(f"instance {name} has no source family prefix")
        instances.append(
            JobShopInstance.create(
                instance_id=name,
                jobs=tuple(jobs),
                machines=machine_count,
                source="or-library-jobshop1",
                source_family=family_match.group(0).lower(),
                description=description,
            )
        )
    if len(instances) != ORLIB_EXPECTED_INSTANCES:
        raise ValueError(
            f"jobshop1 must contain {ORLIB_EXPECTED_INSTANCES} instances; "
            f"parsed {len(instances)}"
        )
    identifiers = [instance.instance_id for instance in instances]
    hashes = [instance.content_hash for instance in instances]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("jobshop1 contains duplicate instance identifiers")
    if len(hashes) != len(set(hashes)):
        raise ValueError("jobshop1 contains duplicate normalized content")
    return tuple(instances)


__all__ = [
    "ORLIB_EXPECTED_INSTANCES",
    "ORLIB_JOBSHOP1_SHA256",
    "ORLIB_JOBSHOP1_URL",
    "parse_jobshop1",
    "sha256_file",
]
