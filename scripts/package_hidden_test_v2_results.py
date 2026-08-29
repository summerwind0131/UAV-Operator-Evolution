"""Build a deterministic release archive for the consumed Hidden Test-v2.

The command is deliberately read-only with respect to frozen data and results.
It validates the existing lifecycle receipts, packages a sorted file set with
normalized tar/gzip metadata, and writes a content-addressed release receipt.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Iterable


ARCHIVE_NAME = "uav2d-hidden-test-v2-final-v1.tar.gz"
DATA_RELATIVE = Path("data/benchmarks/uav2d-hidden-test-v2")
RESULTS_RELATIVE = Path(
    "artifacts/planning_benchmarks/uav2d-hidden-test-v2-final"
)
REPORT_RELATIVES = (
    Path("docs/hidden_test_v2.md"),
    Path("docs/hidden_test_v2_final_evaluation.md"),
    Path("docs/hidden_test_v2_final_report.md"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_release_files(repo_root: Path) -> list[Path]:
    roots = (repo_root / DATA_RELATIVE, repo_root / RESULTS_RELATIVE)
    missing = [str(path) for path in roots if not path.is_dir()]
    missing.extend(
        str(repo_root / relative)
        for relative in REPORT_RELATIVES
        if not (repo_root / relative).is_file()
    )
    if missing:
        raise FileNotFoundError(f"missing release inputs: {missing}")

    files = [path for root in roots for path in root.rglob("*") if path.is_file()]
    files.extend(repo_root / relative for relative in REPORT_RELATIVES)
    return sorted(files, key=lambda path: path.relative_to(repo_root).as_posix())


def validate_closed_evaluation(repo_root: Path) -> dict[str, object]:
    data_root = repo_root / DATA_RELATIVE
    results_root = repo_root / RESULTS_RELATIVE
    opening = json.loads(
        (data_root / "opening_receipt.json").read_text(encoding="utf-8")
    )
    execution = json.loads(
        (results_root / "execution_receipt.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (results_root / "audit_receipt.json").read_text(encoding="utf-8")
    )

    if opening.get("status") != "authorized_open":
        raise ValueError("Hidden Test-v2 does not have an authorized opening receipt")
    if execution.get("status") != "complete":
        raise ValueError("Hidden Test-v2 execution is not complete")
    if audit.get("status") != "passed":
        raise ValueError("Hidden Test-v2 audit did not pass")
    for receipt in (execution, audit):
        if receipt.get("records") != 6960 or receipt.get("unique_records") != 6960:
            raise ValueError("Hidden Test-v2 receipt does not identify 6,960 unique rows")

    checked_hashes = {
        "benchmark_runs_sha256": sha256_file(results_root / "benchmark_runs.csv"),
        "benchmark_paths_sha256": sha256_file(results_root / "benchmark_paths.jsonl"),
        "normalized_runs_sha256": sha256_file(
            results_root / "normalized_benchmark_runs.csv"
        ),
        "audit_report_sha256": sha256_file(results_root / "audit_report.json"),
        "audit_report_markdown_sha256": sha256_file(
            results_root / "audit_report.md"
        ),
        "audit_summary_sha256": sha256_file(results_root / "audit_summary.csv"),
    }
    for field, actual in checked_hashes.items():
        if field == "audit_report_markdown_sha256":
            continue
        expected = audit.get(field)
        if expected != actual:
            raise ValueError(f"{field} mismatch: expected {expected}, got {actual}")

    return {
        "preregistration_id": opening["preregistration_id"],
        "opening_id": opening["opening_id"],
        "execution_receipt_id": execution["execution_receipt_id"],
        "audit_receipt_id": audit["audit_receipt_id"],
        **checked_hashes,
    }


def write_deterministic_archive(
    repo_root: Path,
    files: Iterable[Path],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    with temporary_path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            with tarfile.open(
                fileobj=gzip_handle,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                ordered_files = sorted(
                    files, key=lambda path: path.relative_to(repo_root).as_posix()
                )
                for path in ordered_files:
                    data = path.read_bytes()
                    name = path.relative_to(repo_root).as_posix()
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.pax_headers = {}
                    archive.addfile(info, io.BytesIO(data))
    temporary_path.replace(output_path)


def package(repo_root: Path, output_path: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    output_path = output_path.resolve()
    identities = validate_closed_evaluation(repo_root)
    files = collect_release_files(repo_root)
    write_deterministic_archive(repo_root, files, output_path)
    archive_sha256 = sha256_file(output_path)

    checksum_path = output_path.with_name(f"{output_path.name}.sha256")
    checksum_path.write_text(
        f"{archive_sha256}  {output_path.name}\n", encoding="utf-8", newline="\n"
    )
    receipt = {
        "schema_version": "uav2d-hidden-test-v2-release-receipt-v1",
        "status": "packaged",
        "release_id": "uav2d-hidden-test-v2-final-v1",
        "archive_name": output_path.name,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": output_path.stat().st_size,
        "included_files": len(files),
        "source_roots": [DATA_RELATIVE.as_posix(), RESULTS_RELATIVE.as_posix()],
        **identities,
    }
    receipt_path = output_path.with_name(
        "uav2d-hidden-test-v2-final-v1.receipt.json"
    )
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        args.repo_root / "artifacts" / "releases" / ARCHIVE_NAME
    )
    receipt = package(args.repo_root, output)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
