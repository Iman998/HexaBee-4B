"""Render a training YAML and register a dataset without launching training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pct4b.io import atomic_json, atomic_write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["multilingual", "translation"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--dataset-file", required=True, help="Filename relative to dataset-dir")
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--config-output", required=True, type=Path)
    args = parser.parse_args()
    if args.stage == "multilingual" and not args.adapter:
        parser.error("multilingual stage requires the previous Wikipedia adapter")
    filename = Path(args.dataset_file)
    if filename.is_absolute() or ".." in filename.parts:
        parser.error("dataset-file must stay inside dataset-dir")
    if not (args.dataset_dir / filename).is_file():
        parser.error("dataset file does not exist")
    root = Path(__file__).resolve().parents[1]
    template = (root / "configs" / f"train_{args.stage}.yaml").read_text()
    deepspeed = args.config_output.parent / "ds_z0_config.json"
    values = {
        "MODEL_PATH": args.model,
        "ADAPTER_PATH": args.adapter or "",
        "DATASET_DIRECTORY": str(args.dataset_dir.resolve()),
        "OUTPUT_DIRECTORY": args.training_output,
        "DEEPSPEED_CONFIG": str(deepspeed.resolve()),
    }
    for key, value in values.items():
        template = template.replace(": " + key, ": " + json.dumps(value))
    info_path = args.dataset_dir / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info["pct_" + args.stage] = {
        "file_name": args.dataset_file,
        "formatting": "alpaca",
        "columns": {"prompt": "instruct", "response": "output", "system": "system"},
    }
    atomic_json(info_path, info)
    atomic_json(deepspeed, json.loads((root / "configs/ds_z0_config.json").read_text()))
    atomic_write(args.config_output, lambda f: f.write(template))
    print("Wrote training YAML, DeepSpeed configuration, and dataset_info.json.")


if __name__ == "__main__":
    main()
