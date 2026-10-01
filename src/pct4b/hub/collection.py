"""Group verified public repositories without changing their independent cards."""

from __future__ import annotations

import json
from pathlib import Path

from ..io import atomic_json
from .publish import authenticate


def publish_collection(config: dict, execute: bool = False, token_file: Path | None = None) -> dict:
    entries = config["repositories"]
    title = "HexaBee-4B · Models, Data & Evaluation"
    if not execute:
        return {
            "dry_run": True,
            "collection_title": title,
            "repositories": [e["repo_id"] for e in entries],
        }
    api = authenticate(config["owner"], token_file)
    for entry in entries:
        marker = Path(config["scratch_root"]) / entry["key"] / "published.json"
        if not marker.exists() or not json.loads(marker.read_text()).get("published"):
            raise ValueError(f"Publish and verify {entry['repo_id']} before adding the collection")
        info = api.repo_info(entry["repo_id"], repo_type=entry["repo_type"])
        if info.private:
            raise ValueError(f"Collection requires public repository: {entry['repo_id']}")
    collection = api.create_collection(
        title,
        namespace=config["owner"],
        private=True,
        exists_ok=True,
        description="Six-language translation with multilingual answering and bidirectional supervision. "
        "Separate merged models, LoRA stages, training corpora, prompts, predictions and evaluations.",
    )
    for entry in entries:
        api.add_collection_item(
            collection.slug,
            item_id=entry["repo_id"],
            item_type=entry["repo_type"],
            note=entry["scope"],
            exists_ok=True,
        )
    collection = api.update_collection_metadata(collection.slug, private=False, theme="yellow")
    result = {
        "published": True,
        "url": collection.url,
        "slug": collection.slug,
        "repositories": len(entries),
    }
    atomic_json(Path(config["scratch_root"]) / "collection.json", result)
    return result
