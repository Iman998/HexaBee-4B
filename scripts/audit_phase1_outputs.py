"""Record why Phase 1 was excluded from the balanced checkpoint comparison.

Run in a Slurm CPU allocation. This diagnostic uses every saved Phase 1 prediction
and the same identities from the other available intermediate checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import sacrebleu
from sacrebleu.metrics import CHRF


def read_progress(path: Path) -> dict[str, tuple[str, int, int]]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        return {
            identifier: (candidate, tokens, hit_limit)
            for identifier, candidate, tokens, hit_limit in connection.execute(
                "SELECT id, candidate, output_tokens, hit_limit FROM predictions"
            )
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("A Slurm allocation is required")
    ablation = args.paper_root / "ablation_eval"
    test = args.paper_root / "huggingface-release/repositories/data-translation/data/aligned_pairs/test"
    rows = {
        row["id"]: row
        for path in sorted(test.glob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    }
    progress = {
        variant: read_progress(ablation / variant / "progress.sqlite")
        for variant in ("phase1", "phase12", "phase3")
    }
    phase1_ids = list(progress["phase1"])
    if not phase1_ids:
        raise ValueError("No saved Phase 1 outputs")
    if any(identifier not in rows for identifier in phase1_ids):
        raise ValueError("Phase 1 IDs missing from test")
    if any(not set(phase1_ids) <= set(progress[variant]) for variant in ("phase12", "phase3")):
        raise ValueError("Comparison checkpoints lack saved predictions for Phase 1 IDs")

    scorer = CHRF(word_order=2)
    by_direction: dict[str, list[str]] = defaultdict(list)
    for identifier in phase1_ids:
        row = rows[identifier]
        by_direction[f"{row['source_language']}>{row['target_language']}"].append(identifier)
    score_summary = {}
    for variant, predictions in progress.items():
        direction_scores = {
            direction: scorer.corpus_score(
                [predictions[identifier][0] for identifier in ids],
                [[rows[identifier]["reference"] for identifier in ids]],
            ).score
            for direction, ids in sorted(by_direction.items())
        }
        pooled = scorer.corpus_score(
            [predictions[identifier][0] for identifier in phase1_ids],
            [[rows[identifier]["reference"] for identifier in phase1_ids]],
        ).score
        score_summary[variant] = {
            "pooled_chrfpp": pooled,
            "direction_macro_chrfpp": sum(direction_scores.values()) / len(direction_scores),
            "directions_below_10": sum(value < 10 for value in direction_scores.values()),
            "direction_scores": direction_scores,
        }
    failures = []
    for identifier in phase1_ids:
        if len(failures) >= 6:
            break
        row = rows[identifier]
        candidate = progress["phase1"][identifier][0]
        if len(candidate) < 50:
            continue
        failures.append({
            "id": identifier,
            "direction": f"{row['source_language']}>{row['target_language']}",
            "source_excerpt": row["source"][:180],
            "reference_excerpt": row["reference"][:180],
            "candidate_excerpt": candidate[:180],
        })
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "purpose": "Translation capability diagnostic before the balanced 6,000-example evaluation",
        "sample_note": "All saved Phase 1 predictions from the stopped full-test run; this is an incomplete, order-dependent diagnostic sample, not a representative full-test score.",
        "sample_size": len(phase1_ids),
        "directions": len(by_direction),
        "direction_min_n": min(map(len, by_direction.values())),
        "direction_max_n": max(map(len, by_direction.values())),
        "sample_ids_sha256": hashlib.sha256("\n".join(sorted(phase1_ids)).encode()).hexdigest(),
        "phase1_hit_5000_token_limit": sum(value[2] for value in progress["phase1"].values()),
        "phase1_empty": sum(not value[0].strip() for value in progress["phase1"].values()),
        "metric": "chrF++ (word_order=2)",
        "sacrebleu_version": sacrebleu.__version__,
        "scores_on_identical_ids": score_summary,
        "illustrative_examples": failures,
        "decision": "Exclude Phase 1 from the balanced 6,000-example automatic-metric run because these outputs predominantly fail to translate the requested source into the requested target language.",
        "raw_output": str(ablation / "phase1/progress.sqlite"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"COMPLETE sample={len(phase1_ids)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
