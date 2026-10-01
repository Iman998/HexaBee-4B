"""Public comparison tables computed directly from archived direction scores."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

METRICS = ("bleu", "chrf", "chrfpp", "ter", "comet")
LABELS = ("BLEU ↑", "chrF ↑", "chrF++ ↑", "TER ↓", "COMET ×100 ↑")
MODELS = ("Gemma 3 4B IT", "Gemma 3 12B IT", "HexaBee-4B")
LANGUAGES = ("Persian", "English", "Arabic", "Chinese", "Hebrew", "Spanish")


def load_direction_scores(source: Path) -> list[dict]:
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["model"] = row["model"].replace("PCT-4B", "HexaBee-4B")
    identities = {(r["model"], r["direction"]) for r in rows}
    if len(rows) != 90 or len(identities) != 90:
        raise ValueError("Expected exactly three systems across 30 directions")
    if {r["model"] for r in rows} != set(MODELS):
        raise ValueError("Unexpected benchmark system labels")
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for model in MODELS:
        group = [r for r in rows if r["model"] == model]
        if not group:
            raise ValueError(f"No scores for {model}")
        output.append(
            {
                "model": model,
                "n": sum(int(r["n"]) for r in group),
                "directions": len(group),
                **{m: statistics.mean(float(r[m]) for r in group) for m in METRICS},
            }
        )
    return output


def ranked_table(rows: list[dict], metrics=METRICS, labels=LABELS) -> str:
    lines = ["| System | " + " | ".join(labels) + " |", "|:--|" + "--:|" * len(metrics)]
    for row in rows:
        values = []
        for metric in metrics:
            order = sorted({round(float(r[metric]), 2) for r in rows}, reverse=metric != "ter")
            value = round(float(row[metric]), 2)
            label = f"{value:.2f}"
            if value == order[0]:
                label = f"**{label}**"
            elif len(order) > 1 and value == order[1]:
                label = f"<u>{label}</u>"
            values.append(label)
        model = f"**{row['model']}**" if row["model"] == "HexaBee-4B" else row["model"]
        lines.append("| " + model + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def benchmark_markdown(rows: list[dict]) -> str:
    return (
        "## Translation benchmark\n\n"
        + ranked_table(rows)
        + "\n\nEqual-weight macro over **30 directions**, **36,572 test items per model**. "
        "Best displayed values are bold; the second distinct value is underlined. "
        "TER is lower-is-better. Synthetic references and differing archived inference "
        "templates limit comparisons. These are archived research results, not an official "
        "shared-task ranking.\n"
    )


def direction_appendix(rows: list[dict]) -> str:
    text = [
        "## Results by target language\n",
        "Automatic metrics below use the full test. "
        "The separate judge tables use ten matched items per direction and mode.\n",
    ]
    for target in LANGUAGES:
        group = [r for r in rows if r["direction"].endswith("-" + target)]
        text += [
            f"<details>\n<summary><strong>{target} · five incoming directions</strong></summary>\n",
            "### Target macro\n",
            ranked_table(aggregate(group)),
        ]
        for direction in sorted({r["direction"] for r in group}):
            subset = sorted(
                [r for r in group if r["direction"] == direction],
                key=lambda r: MODELS.index(r["model"]),
            )
            counts = {int(r["n"]) for r in subset}
            if len(counts) != 1:
                raise ValueError("Mismatched direction sample counts")
            text += [
                f"\n#### {direction} · n = {counts.pop():,} per model\n",
                ranked_table(subset),
                "\nJudge subset · n = 10 per model, each mode\n",
                ranked_table(
                    subset,
                    ("judge_reference", "judge_no_reference"),
                    ("Reference-aware ↑", "Reference-free ↑"),
                ),
            ]
        text += ["\n</details>\n"]
    text += [
        "\nAll eight judge dimensions in both modes are retained in "
        "[the direction-level CSV](results/all_direction_metrics.csv). "
        "[Paired intervals](results/evaluation_summary.json) use 10,000 "
        "direction-stratified bootstrap replicates.\n"
    ]
    return "\n".join(text)
