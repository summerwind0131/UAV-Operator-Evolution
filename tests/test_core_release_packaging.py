from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_core_distribution_metadata_is_versioned_without_source_duplication() -> None:
    metadata_path = ROOT / "packaging" / "core" / "pyproject.toml"
    metadata = tomllib.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["project"]["name"] == "trajectory-operator-evolution"
    assert metadata["project"]["version"] == "0.1.0"
    assert not list((ROOT / "packaging" / "core").rglob("*.py"))
    assert (ROOT / "src" / "operator_evolution_core" / "__init__.py").is_file()
