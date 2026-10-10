"""Build a deterministic direction-balanced test and matching archived final outputs.

Run inside a Slurm CPU allocation. Existing output files are never silently replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pct4b.schema import evaluation_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-direction", type=int, default=200)
    parser.add_argument("--seed", default="hexabee-ablation-20261010")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("A Slurm allocation is required")
    if args.per_direction < 1:
        parser.error("--per-direction must be positive")
    test_files = sorted(args.test_dir.glob("*.parquet"))
    final_files = sorted(args.final_dir.glob("*.parquet"))
    if not test_files or not final_files:
        raise FileNotFoundError("Test or final-output Parquet files missing")
    if (args.output_root / "selection_manifest.json").exists():
        raise FileExistsError("Selection already exists; inspect it before rebuilding")

    rows: list[dict] = []
    schema = pq.read_schema(test_files[0])
    for path in test_files:
        for row in pq.read_table(path).to_pylist():
            if row["id"] != evaluation_id(row):
                raise ValueError(f"Invalid test identity in {path}")
            rows.append(row)
    if len(rows) != 36572 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected 36,572 unique archived test rows; found {len(rows)}")

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["source_language"], row["target_language"])].append(row)
    if len(groups) != 30:
        raise ValueError(f"Expected 30 directions, found {len(groups)}")

    def rank(row: dict) -> tuple[str, str]:
        value = hashlib.sha256(f"{args.seed}:{row['id']}".encode()).hexdigest()
        return value, row["id"]

    selected_ids: set[str] = set()
    for direction, direction_rows in groups.items():
        if len(direction_rows) < args.per_direction:
            raise ValueError(f"Only {len(direction_rows)} test rows in {direction}")
        selected_ids.update(row["id"] for row in sorted(direction_rows, key=rank)[: args.per_direction])
    selected = [row for row in rows if row["id"] in selected_ids]
    expected = args.per_direction * 30
    if len(selected) != expected or len(selected_ids) != expected:
        raise ValueError("Balanced selection produced the wrong row count")

    final_by_id: dict[str, dict] = {}
    for path in final_files:
        for row in pq.read_table(path).to_pylist():
            identifier = row["id"]
            if identifier not in selected_ids:
                continue
            if identifier in final_by_id:
                raise ValueError(f"Duplicate final prediction: {identifier}")
            final_by_id[identifier] = row
    if set(final_by_id) != selected_ids:
        raise ValueError(f"Final predictions missing {len(selected_ids - set(final_by_id))} IDs")
    for row in selected:
        final = final_by_id[row["id"]]
        for key in ("source_language", "target_language", "source", "reference"):
            if final[key] != row[key]:
                raise ValueError(f"Final prediction mismatch: {row['id']} {key}")
        if not isinstance(final["candidate"], str):
            raise ValueError(f"Invalid final candidate: {row['id']}")

    test_output = args.output_root / "test" / "part-00000.parquet"
    full_output = args.output_root / "full" / "predictions.jsonl"
    for output in (test_output, full_output):
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
    test_tmp = test_output.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(selected, schema=schema), test_tmp, compression="zstd")
    os.replace(test_tmp, test_output)
    full_tmp = full_output.with_suffix(".jsonl.tmp")
    with full_tmp.open("w", encoding="utf-8") as stream:
        for row in selected:
            final = final_by_id[row["id"]]
            record = {key: final[key] for key in (
                "id", "model", "source_language", "target_language", "source", "reference", "candidate"
            )}
            record["inference"] = {"provenance": "archived_full_test", "engine": "published_serving"}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(full_tmp, full_output)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "seed": args.seed,
        "per_direction": args.per_direction,
        "rows": expected,
        "directions": {f"{source}>{target}": args.per_direction for source, target in sorted(groups)},
        "selection_rule": "200 smallest SHA256(seed + ':' + archived test ID) values per direction",
        "selected_ids_sha256": hashlib.sha256("\n".join(sorted(selected_ids)).encode()).hexdigest(),
        "original_test_files_sha256": {path.name: sha256_file(path) for path in test_files},
        "original_final_files_sha256": {path.name: sha256_file(path) for path in final_files},
        "test_file": str(test_output),
        "test_file_sha256": sha256_file(test_output),
        "final_predictions": str(full_output),
        "final_predictions_sha256": sha256_file(full_output),
    }
    atomic_json(args.output_root / "selection_manifest.json", manifest)
    print(f"COMPLETE rows={expected} directions={len(groups)} output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
