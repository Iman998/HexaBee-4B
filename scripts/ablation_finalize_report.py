"""Audit the matched HexaBee outputs and write the non-judge ablation report.

Run only inside a Slurm CPU allocation after all three metric jobs succeed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from pct4b.schema import evaluation_id


VARIANTS = ("phase12", "phase3", "full")
EXPECTED_MODELS = {
    "phase12": "HexaBee phases 1+2",
    "phase3": "HexaBee phase 3 only (v2)",
    "full": "HexaBee-4B",
}
METRICS = ("bleu", "chrf", "chrfpp", "ter")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_atomic(path: Path, value: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def fmt(value: float) -> str:
    return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("A Slurm allocation is required")
    root = args.root
    selection_path = root / "selection_manifest.json"
    selection = json.loads(selection_path.read_text())
    if selection["rows"] != 6000 or selection["per_direction"] != 200:
        raise ValueError("Selection manifest is not the approved 6,000-row subset")
    test_path = root / "test/part-00000.parquet"
    if digest_file(test_path) != selection["test_file_sha256"]:
        raise ValueError("Selected test Parquet checksum changed")
    test_rows = pq.read_table(test_path).to_pylist()
    if len(test_rows) != 6000:
        raise ValueError(f"Selected test has {len(test_rows)} rows")
    test_by_id = {}
    counts: Counter[str] = Counter()
    for row in test_rows:
        identifier = row["id"]
        if identifier in test_by_id or identifier != evaluation_id(row):
            raise ValueError(f"Duplicate or invalid test identity: {identifier}")
        test_by_id[identifier] = row
        counts[f"{row['source_language']}-{row['target_language']}"] += 1
    if len(counts) != 30 or set(counts.values()) != {200}:
        raise ValueError(f"Expected 200 rows in each of 30 directions: {counts}")
    selected_hash = hashlib.sha256("\n".join(sorted(test_by_id)).encode()).hexdigest()
    if selected_hash != selection["selected_ids_sha256"]:
        raise ValueError("Selected test identity hash changed")

    predictions: dict[str, dict[str, dict]] = {}
    reports: dict[str, dict] = {}
    metric_by_direction: dict[str, dict[str, dict]] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        pred_path = root / variant / "predictions.jsonl"
        metric_path = root / variant / "metrics.json"
        if variant == "full" and digest_file(pred_path) != selection["final_predictions_sha256"]:
            raise ValueError("Archived final subset predictions checksum changed")
        items: dict[str, dict] = {}
        with pred_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                identifier = row["id"]
                if identifier in items:
                    raise ValueError(f"Duplicate prediction ID in {variant}: {identifier}")
                if identifier not in test_by_id:
                    raise ValueError(f"Unknown prediction ID in {variant}: {identifier}")
                if row["model"] != EXPECTED_MODELS[variant]:
                    raise ValueError(f"Wrong model label in {variant}")
                if not isinstance(row.get("candidate"), str):
                    raise ValueError(f"Invalid candidate in {variant}: {identifier}")
                source = test_by_id[identifier]
                for key in ("source_language", "target_language", "source", "reference"):
                    if row[key] != source[key]:
                        raise ValueError(f"Identity field mismatch in {variant}: {identifier} {key}")
                items[identifier] = row
        if set(items) != set(test_by_id):
            raise ValueError(f"Prediction ID set mismatch in {variant}: {len(items)} rows")
        predictions[variant] = items
        report = json.loads(metric_path.read_text())
        if report.get("metrics") != list(METRICS) or not report.get("paired_id_sets_equal"):
            raise ValueError(f"Metric schema mismatch in {variant}")
        if len(report.get("results", [])) != 30:
            raise ValueError(f"Expected 30 metric directions in {variant}")
        model = EXPECTED_MODELS[variant]
        if set(report["macro"]) != {model} or set(report["pooled"]) != {model}:
            raise ValueError(f"Metric model mismatch in {variant}")
        if report["pooled"][model]["n"] != 6000:
            raise ValueError(f"Metric row count mismatch in {variant}")
        directions = {}
        for entry in report["results"]:
            direction = entry["direction"]
            if (
                direction in directions or entry["model"] != model
                or counts.get(direction) != entry["n"]
                or set(entry["scores"]) != set(METRICS)
            ):
                raise ValueError(f"Invalid metric direction in {variant}: {direction}")
            directions[direction] = entry
        if set(directions) != set(counts):
            raise ValueError(f"Missing metric directions in {variant}")
        for metric in METRICS:
            recalculated = statistics.mean(entry["scores"][metric] for entry in directions.values())
            if not math.isclose(recalculated, report["macro"][model][metric], abs_tol=1e-9):
                raise ValueError(f"Macro {metric} mismatch in {variant}")
        reports[variant] = report
        metric_by_direction[variant] = directions
        input_hashes[variant] = {
            "predictions_sha256": digest_file(pred_path),
            "metrics_sha256": digest_file(metric_path),
        }

    triage = json.loads((root / "phase1_triage.json").read_text())
    if triage["sample_size"] != 360 or triage["directions"] != 30:
        raise ValueError("Phase 1 diagnostic does not match the recorded sample")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "test_rows": 6000,
        "directions": 30,
        "per_direction": 200,
        "selected_ids_sha256": selected_hash,
        "model_labels": EXPECTED_MODELS,
        "direction_macro": {
            variant: reports[variant]["macro"][EXPECTED_MODELS[variant]] for variant in VARIANTS
        },
        "delta_vs_final": {
            variant: {
                metric: reports[variant]["macro"][EXPECTED_MODELS[variant]][metric]
                - reports["full"]["macro"][EXPECTED_MODELS["full"]][metric]
                for metric in METRICS
            }
            for variant in ("phase12", "phase3")
        },
        "empty_candidates": {
            variant: sum(not row["candidate"].strip() for row in predictions[variant].values())
            for variant in VARIANTS
        },
        "hit_5000_token_limit": {
            variant: sum(bool(row.get("hit_token_limit")) for row in predictions[variant].values())
            for variant in ("phase12", "phase3")
        },
        "phase1_triage": {
            "sample_size": triage["sample_size"],
            "direction_macro_chrfpp": triage["scores_on_identical_ids"]["phase1"]["direction_macro_chrfpp"],
            "hit_5000_token_limit": triage["phase1_hit_5000_token_limit"],
        },
        "input_hashes": input_hashes,
    }

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["direction", "n", "variant", *METRICS])
    for direction in sorted(counts):
        for variant in VARIANTS:
            entry = metric_by_direction[variant][direction]
            writer.writerow([direction, entry["n"], variant, *(entry["scores"][metric] for metric in METRICS)])

    lines = [
        "# HexaBee checkpoint comparison on 6,000 matched translations",
        "",
        "The archived synthetic test was sampled deterministically: 200 examples in each of the 30 translation directions. All three reported systems were checked against the same 6,000 IDs, source texts, and references. The test and output hashes are in `selection_manifest.json` and `ablation_summary.json`.",
        "",
        "## Direction-macro automatic metrics",
        "",
        "| System | BLEU ↑ | chrF ↑ | chrF++ ↑ | TER ↓ | Empty outputs | 5,000-token cap |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    names = {"phase12": "Phases 1+2", "phase3": "Phase 3-only v2", "full": "Final HexaBee"}
    for variant in VARIANTS:
        scores = summary["direction_macro"][variant]
        cap = str(summary["hit_5000_token_limit"][variant]) if variant != "full" else "n/a"
        lines.append(
            f"| {names[variant]} | {fmt(scores['bleu'])} | {fmt(scores['chrf'])} | "
            f"{fmt(scores['chrfpp'])} | {fmt(scores['ter'])} | "
            f"{summary['empty_candidates'][variant]} | {cap} |"
        )
    lines.extend([
        "",
        "Scores use raw stored candidates with no wrapper stripping. BLEU uses the Chinese tokenizer for Chinese targets and 13a otherwise; chrF++ uses word order 2. Each direction has weight 1/30. `per_direction_metrics.csv` contains all 90 system-direction rows, and each `metrics.json` preserves scorer signatures and sufficient statistics.",
        "",
        "## Phase 1 diagnostic and interpretation",
        "",
        f"The stopped Phase 1 inference produced {triage['sample_size']} saved outputs across all 30 directions. Its direction-macro chrF++ was {fmt(triage['scores_on_identical_ids']['phase1']['direction_macro_chrfpp'])}; {triage['phase1_hit_5000_token_limit']}/{triage['sample_size']} outputs hit the 5,000-token cap. Inspection found source copying, wrong target languages, and unrelated text. Phase 1 was therefore excluded from the 6,000-example automatic-metric run. This 360-output sample came from the original test order and is a translation-capability diagnostic, not a representative full-test estimate.",
        "",
        "The phase-3-only v2 adapter uses a different LoRA configuration from the published final stage, so it is a translation-only comparator rather than a strictly matched removal of phases 1 and 2. The archived final outputs also used a different serving engine from the new Transformers inference. These differences limit causal claims about the contribution of each stage.",
        "",
        "## chrF++ by translation direction",
        "",
        "| Direction | Phases 1+2 | Phase 3-only v2 | Final HexaBee |",
        "|:--|--:|--:|--:|",
    ])
    for direction in sorted(counts):
        lines.append(
            f"| {direction} | "
            f"{fmt(metric_by_direction['phase12'][direction]['scores']['chrfpp'])} | "
            f"{fmt(metric_by_direction['phase3'][direction]['scores']['chrfpp'])} | "
            f"{fmt(metric_by_direction['full'][direction]['scores']['chrfpp'])} |"
        )
    lines.extend([
        "",
        "This comparison reports BLEU, chrF, chrF++, and TER. It does not rerun the LLM judge or COMET. The paper's prior LLM-judge analysis used 300 matched examples per model, ten per direction.",
        "",
        "## Reproducibility files",
        "",
        "- `selection_manifest.json`: deterministic selection rule and hashes",
        "- `phase1_triage.json`: Phase 1 translation-capability check",
        "- `run_manifest.json`: Slurm jobs, resources, and source commit",
        "- `<variant>/predictions.jsonl`: all selected raw outputs",
        "- `<variant>/metrics.json`: automatic scores and sufficient statistics",
        "- `per_direction_metrics.csv` and `ablation_summary.json`: compact machine-readable comparison",
        "",
    ])

    save_atomic(root / "per_direction_metrics.csv", csv_buffer.getvalue())
    save_atomic(root / "ablation_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    save_atomic(root / "ABLATION_REPORT.md", "\n".join(lines))
    print(f"COMPLETE report={root / 'ABLATION_REPORT.md'} rows=6000 directions=30", flush=True)


if __name__ == "__main__":
    main()
