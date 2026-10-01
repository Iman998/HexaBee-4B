"""Losslessly pool prepared SFT partitions by public configuration."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..io import atomic_json, file_hash


def rewrite_shard(job):
    source, target, marker = map(Path, (job["source"], job["target"], job["marker"]))
    fingerprint = {"sha256": job["sha256"], "component": job["component"]}
    if marker.exists() and target.exists():
        saved = json.loads(marker.read_text())
        if saved["input"] == fingerprint and file_hash(target) == saved["sha256"]:
            return saved
        raise ValueError("Consolidated shard changed")
    if file_hash(source) != job["sha256"]:
        raise ValueError("Source differs from archived manifest")
    original = pq.read_table(source)
    index = original.schema.get_field_index("component")
    if index < 0:
        raise ValueError("Missing SFT component field")
    updated = original.set_column(index, "component", pa.array([job["component"]] * len(original)))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".parquet.partial")
    pq.write_table(
        updated,
        temporary,
        compression="zstd",
        compression_level=3,
        row_group_size=2000,
        use_dictionary=True,
    )
    restored = pq.read_table(temporary)
    if not restored.equals(updated) or not restored.drop(["component"]).equals(
        original.drop(["component"])
    ):
        raise ValueError("Consolidation changed training content or identity")
    os.replace(temporary, target)
    result = {
        "input": fingerprint,
        "name": target.name,
        "rows": len(restored),
        "bytes": target.stat().st_size,
        "sha256": file_hash(target),
        "content_preserved": True,
        "config": job["component"],
        "split": job["split"],
    }
    atomic_json(marker, result)
    return result


def stage_consolidated(entry, destination, scratch):
    source_root = Path(entry["consolidate_from"])
    manifest = json.loads((source_root / "release_manifest.json").read_text())
    if not manifest.get("complete") or manifest["repo_id"] != entry["repo_id"]:
        raise ValueError("Need matching complete source release")
    jobs, expected, indices, seen = [], defaultdict(int), defaultdict(int), set()
    for spec in entry["sources"]:
        config, split, old = spec["config"], spec.get("split", "train"), spec["source_partition"]
        if any("/" in x or ".." in x for x in (config, split, old)):
            raise ValueError("Unsafe partition name")
        key = config, split
        prefix = f"data/{old}/{split}/"
        found = [
            (n, v)
            for n, v in manifest["files"].items()
            if n.startswith(prefix) and n.endswith(".parquet")
        ]
        if not found:
            raise ValueError("Missing source partition")
        expected[key] += spec["expected_rows"]
        for name, info in sorted(found):
            if name in seen:
                raise ValueError("Repeated source shard")
            seen.add(name)
            filename = f"part-{indices[key]:05d}.parquet"
            indices[key] += 1
            jobs.append(
                {
                    "source": str(source_root / name),
                    "target": str(destination / "data" / config / split / filename),
                    "marker": str(
                        scratch / "consolidation" / config / split / (filename + ".json")
                    ),
                    "component": config,
                    "split": split,
                    "sha256": info["sha256"],
                }
            )
    if seen != {n for n in manifest["files"] if n.endswith(".parquet")}:
        raise ValueError("Source inventory would be lost")
    workers = int(entry.get("consolidation_workers", 6))
    if not 1 <= workers <= 16:
        raise ValueError("Consolidation workers must be 1–16")
    results, total = defaultdict(list), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(rewrite_shard, jobs), 1):
            results[result["config"], result["split"]].append(result)
            total += result["rows"]
            if i % 20 == 0 or i == len(jobs):
                status = {
                    "completed_shards": i,
                    "total_shards": len(jobs),
                    "verified_rows": total,
                    "content_preserved": True,
                }
                atomic_json(scratch / "consolidation_progress.json", status)
                print(json.dumps(status), flush=True)
    partitions, configs = [], defaultdict(list)
    for (config, split), files in results.items():
        rows = sum(f["rows"] for f in files)
        if rows != expected[config, split]:
            raise ValueError("Merged count differs from source count")
        atomic_json(
            destination / "data" / config / split / "partition.json",
            {
                "rows": rows,
                "component": config,
                "content_preserved": True,
                "files": [{k: f[k] for k in ("name", "rows", "bytes", "sha256")} for f in files],
            },
        )
        partitions.append({"config": config, "split": split, "rows": rows, "shards": len(files)})
        configs[config].append({"split": split, "path": f"data/{config}/{split}/*.parquet"})
    return {"rows": total, "partitions": partitions, "content_preserved": True}, [
        {"config_name": name, **({"default": True} if i == 0 else {}), "data_files": files}
        for i, (name, files) in enumerate(configs.items())
    ]
