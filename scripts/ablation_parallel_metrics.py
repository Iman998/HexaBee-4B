"""Checkpointed, direction-parallel automatic metrics for one 30-direction model.

This reproduces pct4b.evaluation.score_metrics for BLEU, chrF, chrF++, and TER,
including corpus sufficient statistics and direction-macro averaging. Run only
inside a Slurm CPU allocation; never concurrently write the same output root.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER

from pct4b.evaluation import paired_rows


METRICS = ("bleu", "chrf", "chrfpp", "ter")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def direction_scores(task: tuple[str, str, list[dict]]) -> dict:
    model, pair, rows = task
    hypotheses = [row["candidate"] for row in rows]
    references = [[row["reference"] for row in rows]]
    scorers = {
        "bleu": BLEU(tokenize="zh" if pair.endswith("-zh") else "13a"),
        "chrf": CHRF(word_order=0),
        "chrfpp": CHRF(word_order=2),
        "ter": TER(normalized=True, asian_support=True),
    }
    record = {
        "model": model,
        "direction": pair,
        "n": len(rows),
        "scores": {},
        "signatures": {},
        "statistics": {},
    }
    for name in METRICS:
        scorer = scorers[name]
        stats = scorer._extract_corpus_statistics(hypotheses, references)
        total = np.asarray(stats).sum(axis=0).tolist()
        record["scores"][name] = scorer._compute_score_from_stats(total).score
        record["signatures"][name] = str(scorer.get_signature())
        record["statistics"][name] = total
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-direction", type=int, default=200)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("A Slurm allocation is required")
    if args.workers < 1 or args.per_direction < 1:
        parser.error("workers and per-direction must be positive")
    models = paired_rows([args.input])
    if len(models) != 1:
        raise ValueError("Expected exactly one model per metric job")
    model, items = next(iter(models.items()))
    if len(items) != args.per_direction * 30:
        raise ValueError(f"Expected {args.per_direction * 30} predictions, found {len(items)}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for identifier in sorted(items):
        row = items[identifier]
        groups[row["source_language"] + "-" + row["target_language"]].append(row)
    if len(groups) != 30 or any(len(rows) != args.per_direction for rows in groups.values()):
        raise ValueError("Expected 30 directions with the same per-direction count")

    progress_dir = args.output.with_suffix(".directions")
    progress_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "input_sha256": file_hash(args.input),
        "metrics": list(METRICS),
        "per_direction": args.per_direction,
        "sacrebleu_version": sacrebleu.__version__,
        "script_sha256": file_hash(Path(__file__)),
    }
    config_path = progress_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Existing direction progress belongs to a different input or scorer")
    else:
        save_json(config_path, config)

    completed: dict[str, dict] = {}
    pending: list[tuple[str, str, list[dict]]] = []
    for pair, rows in sorted(groups.items()):
        path = progress_dir / f"{pair}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["direction"] != pair or record["n"] != len(rows) or record["model"] != model:
                raise ValueError(f"Invalid saved direction result: {path}")
            completed[pair] = record
            print(f"DIRECTION_REUSED {pair}", flush=True)
        else:
            pending.append((model, pair, rows))
    print(
        f"START job={os.environ['SLURM_JOB_ID']} model={model} rows={len(items)} "
        f"directions_done={len(completed)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(direction_scores, task): task[1] for task in pending}
        for future in concurrent.futures.as_completed(futures):
            pair = futures[future]
            record = future.result()
            save_json(progress_dir / f"{pair}.json", record)
            completed[pair] = record
            print(f"DIRECTION_DONE {pair} n={record['n']} remaining={30-len(completed)}", flush=True)

    results = [completed[pair] for pair in sorted(groups)]
    report = {
        "schema_version": 1,
        "scope": "raw stored outputs; no wrapper stripping",
        "paired_id_sets_equal": True,
        "metrics": list(METRICS),
        "results": results,
        "macro": {model: {name: statistics.mean(row["scores"][name] for row in results) for name in METRICS}},
        "pooled": {model: {"n": len(items), "empty_candidates": sum(not row["candidate"] for row in items.values())}},
        "comet": None,
    }
    for name, scorer in (
        ("chrf", CHRF(word_order=0)),
        ("chrfpp", CHRF(word_order=2)),
        ("ter", TER(normalized=True, asian_support=True)),
    ):
        total = np.asarray([row["statistics"][name] for row in results]).sum(axis=0).tolist()
        report["pooled"][model][name] = scorer._compute_score_from_stats(total).score
    save_json(args.output, report)
    print(f"COMPLETE rows={len(items)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
