from __future__ import annotations

import tarfile
from pathlib import Path

from scripts.package_hidden_test_v2_results import (
    sha256_file,
    write_deterministic_archive,
)


def test_deterministic_archive_normalizes_order_and_metadata(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    first = root / "z" / "last.txt"
    second = root / "a" / "first.txt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("last\n", encoding="utf-8")
    second.write_text("first\n", encoding="utf-8")
    output = tmp_path / "release.tar.gz"

    write_deterministic_archive(root, [first, second], output)
    initial_hash = sha256_file(output)
    write_deterministic_archive(root, [first, second], output)

    assert sha256_file(output) == initial_hash
    with tarfile.open(output, "r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == ["a/first.txt", "z/last.txt"]
        assert all(member.mtime == 0 for member in members)
        assert all(member.uid == member.gid == 0 for member in members)
        assert all(member.mode == 0o644 for member in members)
