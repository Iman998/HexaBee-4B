"""Prepare one independently publishable repository at a time."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

from ..io import atomic_json, file_hash, write_records
from .cards import build_card
from .evidence import aggregate, load_direction_scores
from .parquet import prepare_shards

MODEL_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "adapter_config.json",
    "adapter_model.safetensors",
}


def copy_file(source: Path, destination: Path) -> None:
    """Use a hard link for large immutable weights on one volume, otherwise copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size and file_hash(
            destination
        ) == file_hash(source):
            return
        raise ValueError(f"Destination already differs: {destination.name}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def overview_source(key: str) -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "hub-overviews" / f"{key}.png"


def sanitize_config(data):
    if isinstance(data, dict):
        return {
            k: sanitize_config(v)
            for k, v in data.items()
            if k not in {"_name_or_path", "name_or_path", "model_path"}
        }
    if isinstance(data, list):
        return [sanitize_config(v) for v in data]
    if isinstance(data, str) and data.startswith(("/data/", "/mnt/", "/home/")):
        return Path(data).name
    return data


def stage_model(entry: dict, destination: Path) -> dict:
    source = Path(entry["source"])
    if not source.is_dir():
        raise ValueError("Model source directory does not exist")
    names = [
        p
        for p in source.iterdir()
        if p.is_file()
        and (p.name in MODEL_FILES or p.name.startswith("model-") and p.suffix == ".safetensors")
    ]
    if not any(p.suffix == ".safetensors" for p in names):
        raise ValueError("No safetensors weights in model source")
    for path in names:
        target = destination / path.name
        if path.suffix == ".json":
            values = sanitize_config(json.loads(path.read_text()))
            if path.name == "adapter_config.json":
                values["base_model_name_or_path"] = entry["base_model"]
                values["revision"] = None
            atomic_json(target, values)
        else:
            copy_file(path, target)
    if (destination / "model.safetensors.index.json").exists():
        index = json.loads((destination / "model.safetensors.index.json").read_text())
        if any(not (destination / name).is_file() for name in set(index["weight_map"].values())):
            raise ValueError("Model index references a missing shard")
    stats = {
        "base_model": entry["base_model"],
        "weight_bytes": sum(p.stat().st_size for p in destination.glob("*.safetensors")),
    }
    if entry["kind"] == "adapter":
        saved = json.loads((destination / "adapter_config.json").read_text())
        stats["adapter"] = {
            k: saved.get(k)
            for k in ("r", "lora_alpha", "lora_dropout", "use_dora", "use_rslora", "target_modules")
        }
    for name in ("train_results.json", "eval_results.json"):
        if (source / name).exists():
            atomic_json(
                destination / "training" / name,
                sanitize_config(json.loads((source / name).read_text())),
            )
    return stats


def stage_data(entry: dict, destination: Path, scratch: Path) -> tuple[dict, list[dict]]:
    sources = list(entry.get("sources", []))
    if entry["kind"] == "prompts":
        from ..prompts import import_pools

        path = scratch / "normalized_prompt_pools.jsonl"
        if not path.exists():
            write_records(path, import_pools(entry["source"]))
        sources = [
            {
                "path": str(path),
                "config": "prompts",
                "split": "train",
                "kind": "prompts",
                "component": "original_prompt_pools",
                "expected_rows": 341142,
            }
        ]
    results, configurations = [], {}
    for spec in sources:
        config, split = spec["config"], spec["split"]
        target = destination / "data" / config / split
        stats = prepare_shards(
            Path(spec["path"]),
            target,
            spec["kind"],
            spec["metadata"]
            if "metadata" in spec
            else {
                k: v
                for k, v in spec.items()
                if k not in {"path", "expected_rows", "config", "split", "kind"}
            },
            shard_rows=entry.get("shard_rows", 25000),
            expected_rows=spec.get("expected_rows"),
        )
        configurations.setdefault(config, []).append(
            {"split": split, "path": f"data/{config}/{split}/*.parquet"}
        )
        results.append(
            {"config": config, "split": split, "rows": stats["rows"], "shards": len(stats["files"])}
        )
        print(json.dumps({"repository": entry["repo_id"], **results[-1]}), flush=True)
    return {"rows": sum(r["rows"] for r in results), "partitions": results}, [
        {"config_name": name, **({"default": True} if i == 0 else {}), "data_files": files}
        for i, (name, files) in enumerate(configurations.items())
    ]


def stage_metrics(entry: dict, destination: Path) -> dict:
    source = Path(entry["source"])
    for path in source.iterdir():
        if not path.is_file() or path.suffix not in {".csv", ".json"}:
            continue
        if path.suffix == ".csv":
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                for key in row:
                    row[key] = row[key].replace("PCT-4B", "HexaBee-4B")
            target = destination / "results" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        else:
            text = path.read_text().replace("PCT-4B", "HexaBee-4B")
            atomic_json(destination / "results" / path.name, json.loads(text))
    scores = load_direction_scores(source / "all_direction_metrics.csv")
    if entry.get("figures"):
        for figure in sorted(Path(entry["figures"]).glob("*")):
            if figure.is_file() and figure.suffix in {".pdf", ".svg", ".png"}:
                copy_file(figure, destination / "figures" / figure.name)
    return {
        "partitions": metric_partitions(destination),
        "benchmark": aggregate(scores),
        "direction_scores": scores,
        "systems": 3,
        "directions": 30,
        "test_items_per_model": 36572,
        "judge_items_per_model": 300,
        "renamed_system": {"historical": "PCT-4B", "public": "HexaBee-4B"},
    }


def prepare(config: dict, key: str) -> Path:
    entries = config["repositories"]
    entry = next(e for e in entries if e["key"] == key)
    destination = Path(config["staging_root"]) / key
    destination.mkdir(parents=True, exist_ok=True)
    scratch = Path(config["scratch_root"]) / key
    scratch.mkdir(parents=True, exist_ok=True)
    configs = []
    if entry["kind"] in {"model", "adapter"}:
        stats = stage_model(entry, destination)
    elif entry["kind"] == "metrics":
        stats = stage_metrics(entry, destination)
        configs = metric_configs()
    else:
        stats, configs = stage_data(entry, destination, scratch)
    if key == "model":
        add_model_benchmark(config, stats)
    assets = Path(config["brand_assets"])
    copy_file(assets / entry["banner"], destination / "assets/banner.png")
    copy_file(overview_source(key), destination / "assets/overview.png")
    license_text = (
        "# Model terms\n\nThese weights derive from Gemma 3 and remain subject to the "
        "[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and prohibited-use policy. "
        "This publication does not relicense the base model.\n"
        if entry["repo_type"] == "model"
        else "# Upstream terms and attribution\n\nThis repository releases historical research artifacts. "
        "Wikipedia-derived content retains its applicable source attribution and share-alike obligations. "
        "The archived exports do not recover per-row article revisions or complete upstream license metadata. "
        "Teacher-generated text and candidate outputs also remain subject to their applicable upstream terms. "
        "No blanket software license is asserted over these data. See the card for provenance.\n"
    )
    (destination / "LICENSE.md").write_text(license_text)
    atomic_json(destination / "release_stats.json", stats)
    (destination / "README.md").write_text(build_card(entry, entries, stats, configs))
    manifest = {
        "repo_id": entry["repo_id"],
        "repo_type": entry["repo_type"],
        "complete": True,
        "files": {
            p.relative_to(destination).as_posix(): {
                "bytes": p.stat().st_size,
                "sha256": file_hash(p),
            }
            for p in sorted(destination.rglob("*"))
            if p.is_file() and p.name != "release_manifest.json"
        },
    }
    atomic_json(destination / "release_manifest.json", manifest)
    print(json.dumps({"ready": entry["repo_id"], "files": len(manifest["files"])}), flush=True)
    return destination


def add_model_benchmark(config: dict, stats: dict) -> None:
    entry = next((e for e in config["repositories"] if e["kind"] == "metrics"), None)
    if entry:
        source = Path(entry["source"]) / "all_direction_metrics.csv"
        if source.is_file():
            stats["benchmark"] = aggregate(load_direction_scores(source))


def refresh_card(config: dict, key: str) -> Path:
    """Rebuild card metadata for a completed, idle release without reconverting data."""
    import yaml

    entry = next(e for e in config["repositories"] if e["key"] == key)
    destination = Path(config["staging_root"]) / key
    manifest_path = destination / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete") or manifest["repo_id"] != entry["repo_id"]:
        raise ValueError("Cannot refresh an incomplete or mismatched release")
    stats = json.loads((destination / "release_stats.json").read_text())
    previous_card = (destination / "README.md").read_text()
    metadata = yaml.safe_load(previous_card.split("---", 2)[1])
    if key == "model":
        add_model_benchmark(config, stats)
    if entry["kind"] == "metrics":
        metadata["configs"] = metric_configs()
        scores = load_direction_scores(destination / "results" / "all_direction_metrics.csv")
        stats.update(
            benchmark=aggregate(scores),
            direction_scores=scores,
            partitions=metric_partitions(destination),
        )
    atomic_json(destination / "release_stats.json", stats)
    (destination / "README.md").write_text(
        build_card(entry, config["repositories"], stats, metadata.get("configs", []))
    )
    overview = destination / "assets" / "overview.png"
    overview.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(overview_source(key), overview)
    for name in ("README.md", "release_stats.json", "assets/overview.png"):
        path = destination / name
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    atomic_json(manifest_path, manifest)
    return destination


def metric_configs() -> list[dict]:
    return [
        {
            "config_name": "direction_metrics",
            "default": True,
            "data_files": [{"split": "test", "path": "results/all_direction_metrics.csv"}],
        },
        {
            "config_name": "paired_judge_scores",
            "data_files": [{"split": "test", "path": "results/paired_scores.csv"}],
        },
    ]


def metric_partitions(destination: Path) -> list[dict]:
    output = []
    for name, filename in (
        ("direction_metrics", "all_direction_metrics.csv"),
        ("paired_judge_scores", "paired_scores.csv"),
    ):
        with (destination / "results" / filename).open() as handle:
            rows = sum(1 for _ in csv.DictReader(handle))
        output.append({"config": name, "split": "test", "rows": rows})
    return output
