"""Bounded-memory, atomic Parquet shards with source-aware resumability."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..io import atomic_json, digest, file_hash, read_records

SCHEMAS = {
    "text": pa.schema([("id", pa.string()), ("text", pa.string()), ("language", pa.string())]),
    "sft": pa.schema(
        [
            (k, pa.string())
            for k in (
                "id",
                "system",
                "instruct",
                "output",
                "component",
                "task",
                "source_language",
                "target_language",
            )
        ]
    ),
    "pairs": pa.schema(
        [
            (k, pa.string())
            for k in ("id", "source_language", "target_language", "source", "reference")
        ]
    ),
    "prompts": pa.schema(
        [(k, pa.string()) for k in ("id", "component", "language", "source", "target", "text")]
    ),
    "predictions": pa.schema(
        [
            (k, pa.string())
            for k in (
                "id",
                "model",
                "source_language",
                "target_language",
                "source",
                "reference",
                "candidate",
                "judge_reference_json",
                "judge_no_reference_json",
            )
        ]
    ),
}


def adapt(rows, kind: str, meta: dict):
    """Preserve source strings; attach provenance without inventing missing labels."""
    from ..schema import normalize_pair

    for index, raw in enumerate(rows):
        if kind == "text":
            if not isinstance(raw.get("text"), str):
                raise ValueError("PT input requires string text")
            yield {
                "id": digest([meta["component"], index, raw["text"]]),
                "text": raw["text"],
                "language": meta["language"],
            }
        elif kind == "sft":
            for key in ("system", "instruct", "output"):
                if not isinstance(raw.get(key), str):
                    raise ValueError(f"Historical SFT input contains nonstring {key}")
            yield {
                "id": digest([meta["component"], index, raw]),
                **{key: raw[key] for key in ("system", "instruct", "output")},
                **{
                    key: meta.get(key)
                    for key in ("component", "task", "source_language", "target_language")
                },
            }
        elif kind == "prompts":
            yield {
                "id": digest([index, raw]),
                **{
                    key: raw.get(key)
                    for key in ("component", "language", "source", "target", "text")
                },
            }
        else:
            row = normalize_pair(raw)
            output = {
                key: row[key]
                for key in ("id", "source_language", "target_language", "source", "reference")
            }
            if kind == "predictions":
                candidate = raw[meta["candidate_field"]]
                if candidate is not None and not isinstance(candidate, str):
                    raise ValueError("Candidate must be string or null")
                output.update(model=meta["model"], candidate=candidate or "")
                for view in ("reference", "no_reference"):
                    output["judge_" + view + "_json"] = (
                        json.dumps(raw["judge_" + view], ensure_ascii=False)
                        if "judge_" + view in raw
                        else None
                    )
            yield output


def prepare_shards(
    source: Path,
    target: Path,
    kind: str,
    meta: dict,
    shard_rows: int = 25000,
    expected_rows: int | None = None,
) -> dict:
    """Reuse a complete source partition only when its fingerprint and files agree."""
    if shard_rows < 1:
        raise ValueError("shard_rows must be positive")
    target = Path(target)
    stamp = {
        "bytes": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns,
        "kind": kind,
        "metadata": meta,
        "shard_rows": shard_rows,
        "schema_version": 1,
    }
    done = target / "partition.json"
    if done.exists():
        saved = json.loads(done.read_text())
        if saved["source_fingerprint"] != stamp:
            raise ValueError(
                f"Changed input/configuration for completed partition {target.name}; choose a new output directory"
            )
        if expected_rows is not None and saved["rows"] != expected_rows:
            raise ValueError("Completed partition row count differs from requested expectation")
        for entry in saved["files"]:
            path = target / entry["name"]
            if not path.is_file() or file_hash(path) != entry["sha256"]:
                raise ValueError(f"Corrupted completed shard: {entry['name']}")
        return saved
    work = target.with_name(target.name + ".partial")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    batch, inventory, total = [], [], 0

    def flush():
        nonlocal total
        if not batch:
            return
        name = f"part-{len(inventory):05d}.parquet"
        table = pa.Table.from_pylist(batch, schema=SCHEMAS[kind])
        pq.write_table(
            table,
            work / name,
            compression="zstd",
            compression_level=5,
            row_group_size=min(2000, shard_rows),
            use_dictionary=True,
        )
        total += len(batch)
        inventory.append(
            {
                "name": name,
                "rows": len(batch),
                "bytes": (work / name).stat().st_size,
                "sha256": file_hash(work / name),
            }
        )
        batch.clear()

    for row in adapt(read_records(source), kind, meta):
        batch.append(row)
        if len(batch) >= shard_rows:
            flush()
    flush()
    if expected_rows is not None and total != expected_rows:
        raise ValueError(
            f"Count mismatch for {meta['component']}: expected {expected_rows}, found {total}"
        )
    result = {"source_fingerprint": stamp, "rows": total, "files": inventory}
    atomic_json(work / "partition.json", result)
    if target.exists():
        raise ValueError(f"Incomplete existing destination: {target}; move it aside explicitly")
    os.replace(work, target)
    return result
