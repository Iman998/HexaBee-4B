"""Tables and summaries from one explicitly scoped evaluation report."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from .io import atomic_json, atomic_write
from .schema import SECTIONS

VIEWS = ("reference", "no_reference")


def latex(value: object) -> str:
    """Escape identifiers before including them in a LaTeX table."""
    escapes = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(escapes.get(c, c) for c in str(value))


def ranked(value: float, others: list[float], lower: bool = False, digits: int = 2) -> str:
    """Rank distinct displayed values, giving tied rounded scores equal emphasis."""
    order = sorted({round(v, digits) for v in others}, reverse=not lower)
    label = f"{value:.{digits}f}"
    position = order.index(round(value, digits))
    if position == 0:
        return r"\textbf{" + label + "}"
    if position == 1:
        return r"\underline{" + label + "}"
    return label


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))

    def writer(handle):
        output = csv.DictWriter(handle, fieldnames=fields)
        output.writeheader()
        output.writerows(rows)

    atomic_write(path, writer)


def _flatten(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        record = {k: row[k] for k in ("model", "direction", "n")}
        record.update(row["scores"])
        for view in VIEWS:
            key = "judge_" + view
            if key in row:
                record[key] = row[key]["score"]
                record[key + "_n"] = row[key]["n"]
                record.update({view + "_" + k: v for k, v in row[key]["section_scores"].items()})
        output.append(record)
    return output


def _table(
    rows: list[dict], columns: list[tuple[str, str]], caption: str, focus: str | None
) -> str:
    """Compare systems within each labeled group in one shared table."""
    lines = [
        r"\begin{table*}[t]\centering\scriptsize",
        r"\begin{tabular}{ll" + "r" * len(columns) + "}",
        r"\toprule",
        r"\rowcolor{PCTHeader} "
        + " & ".join(
            r"{\color{white}\bfseries " + latex(x) + "}"
            for x in ["Group", "System"] + [label for _, label in columns]
        )
        + r"\\",
        r"\midrule",
    ]
    for group in dict.fromkeys(row["group"] for row in rows):
        subset = [r for r in rows if r["group"] == group]
        for index, row in enumerate(subset):
            cells = [latex(group) if index == 0 else "", latex(row["model"])]
            cells += [
                ranked(row[key], [x[key] for x in subset], lower=key == "ter") for key, _ in columns
            ]
            lines.append(
                (r"\rowcolor{PCTRow} " if row["model"] == focus else "") + " & ".join(cells) + r"\\"
            )
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule\end{tabular}", r"\caption{" + latex(caption) + "}", r"\end{table*}"]
    return "\n".join(lines)


def build_report(
    report: dict, outdir: str | Path, focus: str | None = None, plots: bool = False
) -> None:
    """Write all directions, target macros, ranked tables, and optional vector plots."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    flat = _flatten(report["results"])
    if not flat:
        raise ValueError("Cannot report an empty evaluation")
    models = list(report["macro"])
    metrics = report["metrics"]
    _write_csv(outdir / "directions.csv", flat)
    directions = sorted({r["direction"] for r in flat})
    targets = sorted({d.split("-")[1] for d in directions})
    judge_columns = [
        key
        for view in VIEWS
        for key in ["judge_" + view] + [view + "_" + section for section in SECTIONS]
    ]
    available = [key for key in judge_columns if any(key in r for r in flat)]
    if any(any(key not in row for row in flat) for key in available):
        raise ValueError(
            "Report has incomplete judge coverage; use a separate judged-subset report"
        )
    summary_rows = []
    summaries = {}
    for target in ["all"] + targets:
        summaries[target] = {}
        for model in models:
            group = [
                r
                for r in flat
                if r["model"] == model
                and (target == "all" or r["direction"].endswith("-" + target))
            ]
            if not group:
                raise ValueError(f"No results for {model}, target {target}")
            values = {
                key: statistics.mean(row[key] for row in group) for key in metrics + available
            }
            summaries[target][model] = values
            summary_rows.append(
                {
                    "group": target,
                    "model": model,
                    "directions": len(group),
                    "n": sum(row["n"] for row in group),
                    **values,
                }
            )
    atomic_json(outdir / "summaries.json", summaries)
    _write_csv(outdir / "summaries.csv", summary_rows)
    auto_labels = {
        "bleu": "BLEU",
        "chrf": "chrF",
        "chrfpp": "chrF++",
        "ter": "TER",
        "comet": "COMET",
    }
    automatic = [(key, auto_labels[key]) for key in metrics]
    tex = [
        "% Requires booktabs and xcolor[table]. Best bold; second distinct score underlined.",
        r"\definecolor{PCTHeader}{HTML}{172B4D}",
        r"\definecolor{PCTRow}{HTML}{EAF1FF}",
        _table(
            summary_rows,
            automatic,
            "Direction-macro scores, overall and by target. "
            "TER is lower-is-better; COMET is multiplied by 100.",
            focus,
        ),
    ]
    # Five incoming directions x systems fit together; one table per target.
    for target in targets:
        subset = [
            {**r, "group": r["direction"]} for r in flat if r["direction"].endswith("-" + target)
        ]
        tex.append(
            _table(
                subset,
                automatic,
                f"All translation directions into {target}. "
                "Raw stored outputs; no wrapper stripping.",
                focus,
            )
        )
    section_labels = ["Sem.", "Term.", "Flu.", "Coh.", "Context", "Lang.", "Style", "Format"]
    for view in VIEWS:
        if "judge_" + view not in available:
            continue
        columns = [("judge_" + view, "Overall")] + list(
            zip([view + "_" + key for key in SECTIONS], section_labels, strict=True)
        )
        tex.append(
            _table(
                summary_rows,
                columns,
                f"Judge ({view.replace('_', ' ')}): "
                "overall and target-language direction macros. All scores are out of 100.",
                focus,
            )
        )
        for target in targets:
            subset = [
                {**r, "group": r["direction"]}
                for r in flat
                if r["direction"].endswith("-" + target)
            ]
            tex.append(
                _table(
                    subset,
                    columns,
                    f"Judge ({view.replace('_', ' ')}) for every "
                    f"direction into {target}. All eight rubric dimensions are reported.",
                    focus,
                )
            )
    atomic_write(outdir / "tables.tex", lambda f: f.write("\n\n".join(tex)))
    if plots:
        from .plotting import target_comparison

        target_comparison(summaries, models, metrics, outdir, focus)
