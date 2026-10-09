"""Remap archived HexaBee LoRA keys to the installed Gemma 3 module layout.

Run inside a Slurm CPU allocation. The original adapter is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


OLD_PREFIX = "base_model.model.model.language_model.layers."
NEW_PREFIX = "base_model.model.language_model.model.layers."
EXPECTED_KEYS = 476


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run in a Slurm allocation")
    source = args.source.resolve()
    destination = args.destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Destination must be separate from the source adapter")
    source_weights = source / "adapter_model.safetensors"
    with safe_open(source_weights, framework="pt", device="cpu") as archive:
        source_keys = list(archive.keys())
        if len(source_keys) != EXPECTED_KEYS or any(
            not key.startswith(OLD_PREFIX) for key in source_keys
        ):
            raise ValueError("Unexpected adapter key count or prefix")
        mapped = {key.replace(OLD_PREFIX, NEW_PREFIX, 1): archive.get_tensor(key) for key in source_keys}
    if len(mapped) != EXPECTED_KEYS:
        raise ValueError("Remapped adapter keys are not unique")
    modules = set()
    for key in mapped:
        if key.endswith(".lora_A.weight"):
            modules.add(key.removeprefix("base_model.model.").removesuffix(".lora_A.weight"))
        elif not key.endswith(".lora_B.weight"):
            raise ValueError(f"Unexpected tensor name {key}")
    if len(modules) * 2 != EXPECTED_KEYS:
        raise ValueError("Expected paired LoRA A and B tensors")
    config = json.loads((source / "adapter_config.json").read_text())
    config["target_modules"] = sorted(modules)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "adapter_model.safetensors").exists():
        raise FileExistsError(destination / "adapter_model.safetensors")
    processor_files = {
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        "special_tokens_map.json", "preprocessor_config.json", "processor_config.json",
        "chat_template.jinja", "added_tokens.json", "generation_config.json",
    }
    for item in source.iterdir():
        if item.is_file() and item.name in processor_files:
            if item.stat().st_size > 100_000_000:
                raise ValueError(f"Unexpectedly large processor file: {item}")
            shutil.copy2(item, destination / item.name)
    (destination / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    temporary = destination / "adapter_model.safetensors.tmp"
    save_file(mapped, temporary)
    os.replace(temporary, destination / "adapter_model.safetensors")
    manifest = {
        "source": str(source),
        "source_sha256": sha256(source_weights),
        "destination_sha256": sha256(destination / "adapter_model.safetensors"),
        "old_prefix": OLD_PREFIX,
        "new_prefix": NEW_PREFIX,
        "tensor_count": len(mapped),
        "module_count": len(modules),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "remap_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"REMAPPED tensors={len(mapped)} modules={len(modules)} destination={destination}")


if __name__ == "__main__":
    main()
