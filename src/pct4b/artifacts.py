"""Rebuild paper tables and verify released aggregates without redistributing texts."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from .io import atomic_json
from .report import build_report
from .schema import SECTIONS, language


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_paper(root: Path) -> dict:
    """Check per-direction scores, sufficient statistics, counts, and paired intervals."""
    from sacrebleu.metrics import BLEU, CHRF, TER

    rows = list(csv.DictReader((root / "all_direction_metrics.csv").open()))
    if len(rows) != 90 or len({(r["model"], r["direction"]) for r in rows}) != 90:
        raise ValueError("Expected three systems over 30 distinct directions")
    lookup = {(r["model"], r["direction"]): r for r in rows}
    checked = 0
    for filename in ("metrics_full_bleu_chrf.json", "metrics_full_ter_accelerated.json"):
        for record in read_json(root / filename)["results"]:
            row = lookup[record["model"], record["pair"]]
            scorers = {
                "bleu": BLEU(tokenize="zh" if record["pair"].endswith("-Chinese") else "13a"),
                "chrf": CHRF(),
                "ter": TER(normalized=True, asian_support=True),
            }
            for name, stats in record["statistics"].items():
                value = scorers[name]._compute_score_from_stats(stats).score
                if not math.isclose(value, float(row[name]), abs_tol=1e-9):
                    raise ValueError(f"Inconsistent {name}: {record['model']}, {record['pair']}")
                checked += 1
    chrf = read_json(root / "full_test_chrfpp.json")["models"]
    comet = read_json(root / "comet_full_summary.json")["models"]
    for row in rows:
        model, direction = row["model"], row["direction"]
        for name, value in [
            ("chrfpp", chrf[model]["per_direction"][direction]["chrfpp"]),
            ("comet", 100 * comet[model]["per_direction"][direction]["comet"]),
        ]:
            if not math.isclose(float(row[name]), value, abs_tol=1e-9):
                raise ValueError(f"Inconsistent {name}: {model}, {direction}")
    for model in chrf:
        if sum(int(r["n"]) for r in rows if r["model"] == model) != 36572:
            raise ValueError(f"Incorrect test count for {model}")
    phase2 = read_json(root / "phase2_summary.json")
    components = list(csv.DictReader((root / "phase2_by_direction.csv").open()))
    if len(components) != 329 or sum(int(r["n"]) for r in components) != phase2["final_total"]:
        raise ValueError("Phase-2 component counts do not match")
    for task, counts in phase2["by_task"].items():
        if sum(int(r["n"]) for r in components if r["task"] == task) != counts["total"]:
            raise ValueError(f"Incorrect count for {task}")
    paired = list(csv.DictReader((root / "paired_scores.csv").open()))
    summary = read_json(root / "evaluation_summary.json")
    models = {model: {r["id"]: r for r in paired if r["model"] == model} for model in chrf}
    keys = sorted(models["PCT-4B"])
    if len(keys) != 300 or any(set(items) != set(keys) for items in models.values()):
        raise ValueError("Paired judge identity sets differ")
    groups = [
        np.array([i for i, key in enumerate(keys) if models["PCT-4B"][key]["pair"] == direction])
        for direction in sorted({r["pair"] for r in paired})
    ]
    if len(groups) != 30 or any(len(group) != 10 for group in groups):
        raise ValueError("Expected 10 judged items per direction")
    rng = np.random.default_rng(summary["seed"])
    indices = np.concatenate(
        [
            rng.choice(g, size=(summary["bootstrap_replicates"], len(g)), replace=True)
            for g in groups
        ],
        axis=1,
    )
    differences = {}
    for baseline in models:
        if baseline == "PCT-4B":
            continue
        differences[baseline] = {}
        for view in ("reference", "no_reference"):
            delta = np.array(
                [
                    int(models["PCT-4B"][key][view]) - int(models[baseline][key][view])
                    for key in keys
                ]
            )
            differences[baseline][view] = {
                "mean": float(delta.mean()),
                "ci95": np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist(),
            }
    for baseline, views in differences.items():
        for view, computed in views.items():
            recorded = summary["paired_differences"]["PCT-4B minus " + baseline][view]
            if not np.allclose(
                [computed["mean"], *computed["ci95"]],
                [recorded["mean"], *recorded["ci95"]],
                rtol=0,
                atol=1e-9,
            ):
                raise ValueError(f"Paired interval mismatch: {baseline}, {view}")
    macros = {
        m: {
            k: statistics.mean(float(r[k]) for r in rows if r["model"] == m)
            for k in ("bleu", "chrf", "chrfpp", "ter", "comet")
        }
        for m in models
    }
    return {
        "status": "passed",
        "direction_rows": len(rows),
        "recomputed_statistic_scores": checked,
        "phase2_components": len(components),
        "phase2_records": phase2["final_total"],
        "judge_items_per_model": len(keys),
        "macro": macros,
        "recomputed_paired_differences": differences,
        "scope": "Aggregate consistency and score-only bootstrap; raw-text metrics require prediction files.",
    }


def rebuild_paper(root: Path, output: Path, plots: bool = False) -> dict:
    verification = verify_paper(root)
    csv_rows = list(csv.DictReader((root / "all_direction_metrics.csv").open()))
    models = list(verification["macro"])
    for scope, metrics in [
        ("full", ["bleu", "chrf", "chrfpp", "ter", "comet"]),
        ("judge", ["chrfpp", "ter", "comet"]),
    ]:
        rows = []
        for raw in csv_rows:
            s, t = map(language, raw["direction"].split("-"))
            row = {
                "model": raw["model"],
                "direction": s + "-" + t,
                "n": int(raw["n" if scope == "full" else "judge_n"]),
                "scores": {m: float(raw[m if scope == "full" else m + "_judge"]) for m in metrics},
            }
            if scope == "judge":
                for view in ("reference", "no_reference"):
                    row["judge_" + view] = {
                        "n": row["n"],
                        "score": float(raw["judge_" + view]),
                        "section_scores": {k: float(raw[view + "_" + k]) for k in SECTIONS},
                    }
            rows.append(row)
        report = {
            "metrics": metrics,
            "results": rows,
            "macro": {m: {} for m in models},
            "scope": scope,
        }
        build_report(report, output / scope, "PCT-4B", plots)
    atomic_json(output / "verification.json", verification)
    return verification
