"""Verify released scores and regenerate every direction table without an API or GPU."""

import argparse
from pathlib import Path

from pct4b.artifacts import rebuild_paper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path, default=root / "results/paper")
    parser.add_argument("--output", type=Path, default=root / "runs/paper")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    result = rebuild_paper(args.input, args.output, args.plots)
    print(
        f"Verified {result['direction_rows']} direction rows and {result['phase2_components']} data components."
    )
    print(f"Wrote tables, summaries, and verification to {args.output}")


if __name__ == "__main__":
    main()
