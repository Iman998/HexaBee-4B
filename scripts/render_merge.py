"""Render an adapter-merge configuration without loading model weights."""

import argparse
import json
from pathlib import Path

from pct4b.io import atomic_write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--merged-output", required=True)
    parser.add_argument("--config-output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    template = (root / "configs/merge_adapter.yaml").read_text()
    for key, value in {
        "MODEL_PATH": args.model,
        "ADAPTER_PATH": args.adapter,
        "MERGED_OUTPUT_PATH": args.merged_output,
    }.items():
        template = template.replace(": " + key, ": " + json.dumps(value))
    atomic_write(args.config_output, lambda f: f.write(template))
    print(f"Wrote {args.config_output}")


if __name__ == "__main__":
    main()
