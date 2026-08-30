from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def prepare_source_tree(destination: Path) -> None:
    """Create an isolated core distribution tree from the monorepo sources."""

    package_root = destination / "src" / "operator_evolution_core"
    shutil.copytree(ROOT / "src" / "operator_evolution_core", package_root)
    shutil.copy2(ROOT / "packaging" / "core" / "pyproject.toml", destination)
    shutil.copy2(ROOT / "packaging" / "core" / "README.md", destination)
    shutil.copy2(ROOT / "LICENSE", destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the standalone core wheel and sdist from monorepo sources."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dist/core"),
        help="artifact directory, relative to the repository root by default",
    )
    args = parser.parse_args()
    output = args.out_dir
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="trajectory-core-build-") as temp:
        source_tree = Path(temp)
        prepare_source_tree(source_tree)
        subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(output), str(source_tree)],
            check=True,
        )


if __name__ == "__main__":
    main()
