#!/usr/bin/env python3
"""Paper-native vector figures with a shared visual grammar and source-backed labels."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Rectangle,
)
from matplotlib.path import Path as MPath

R = Path(__file__).resolve().parents[1]
F = R / "figures"
INK = "#172B4D"
SLATE = "#64748B"
MUTED = "#718096"
GRID = "#E7ECF3"
BLUE = "#2563EB"
GOLD = "#C68A24"
LIGHT = "#EFF4FC"
COLORS = {"Gemma 3 4B IT": SLATE, "Gemma 3 12B IT": GOLD, "PCT-4B": BLUE}
MARKERS = {"Gemma 3 4B IT": "o", "Gemma 3 12B IT": "s", "PCT-4B": "D"}
MODELS = list(COLORS)
CODES = ["fa", "en", "ar", "zh", "he", "es"]
NAMES = {
    "fa": "Persian",
    "en": "English",
    "ar": "Arabic",
    "zh": "Chinese",
    "he": "Hebrew",
    "es": "Spanish",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.65,
        "mathtext.fontset": "dejavusans",
    }
)


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(F / (name + "." + ext), dpi=220, bbox_inches="tight", facecolor="white")
    svg = F / (name + ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def header(fig, title, subtitle, x=0.06):
    fig.text(x, 0.975, title, ha="left", va="top", size=11, weight="bold", color=INK)
    fig.text(x, 0.92, subtitle, ha="left", va="top", size=7.3, color=MUTED)


def clean(ax, xgrid=True):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#B9C4D3")
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", length=3)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.set_axisbelow(True)


def legend(fig, y=0.845, x=0.075):
    hs = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[m],
            color=COLORS[m],
            linestyle="",
            markerfacecolor="white" if i == 0 else COLORS[m],
            markersize=5,
            label=m.replace("PCT-4B", "HexaBee-4B"),
        )
        for i, m in enumerate(MODELS)
    ]
    fig.legend(
        handles=hs,
        loc="upper left",
        bbox_to_anchor=(x, y),
        ncol=3,
        frameon=False,
        fontsize=7.3,
        handletextpad=0.55,
        columnspacing=1.8,
        borderaxespad=0,
    )


def label(ax, x, y, s, **kw):
    return ax.text(x, y, s, **kw)


def rounded(ax, x, y, w, h, fc="white", ec=GRID, lw=0.8):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.10", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    return p


def arrow(ax, a, b, color=SLATE, style="arc3", lw=1.1):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, connectionstyle=style
        )
    )


def icon(ax, x, y, kind, color, scale=0.5):
    # Original line-art geometry, no borrowed branding.
    if kind == "doc":
        ax.add_patch(Rectangle((x, y), scale * 0.7, scale, fc="none", ec=color, lw=1))
        for k in [0.25, 0.48, 0.71]:
            ax.plot([x + 0.12 * scale, x + 0.56 * scale], [y + k * scale] * 2, color=color, lw=1)
    elif kind == "chat":
        rounded(ax, x, y, scale, scale * 0.68, fc="none", ec=color, lw=1)
        ax.plot(
            [x + 0.25 * scale, x + 0.18 * scale, x + 0.4 * scale],
            [y, y - 0.18 * scale, y],
            color=color,
            lw=1,
        )
        for a in [0.27, 0.50, 0.73]:
            ax.add_patch(Circle((x + a * scale, y + 0.35 * scale), 0.025 * scale, color=color))
    elif kind == "mt":
        ax.text(x, y, "A", color=color, size=14, weight="bold")
        ax.text(x + 0.35, y - 0.22, "文", color=color, size=11) if False else None
        ax.annotate(
            "",
            xy=(x + 0.62, y + 0.22),
            xytext=(x + 0.23, y + 0.22),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1},
        )
        ax.annotate(
            "",
            xy=(x + 0.23, y - 0.02),
            xytext=(x + 0.62, y - 0.02),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1},
        )
    elif kind == "model":
        for xx, ys in [
            (x, [y, y + 0.24, y + 0.48]),
            (x + 0.35, [y + 0.12, y + 0.36]),
            (x + 0.7, [y + 0.24]),
        ]:
            for yy in ys:
                ax.add_patch(Circle((xx, yy), 0.045, fc="white", ec=color, lw=1))
        for yy in [y, y + 0.24, y + 0.48]:
            for zz in [y + 0.12, y + 0.36]:
                ax.plot([x + 0.05, x + 0.3], [yy, zz], color=color, lw=0.65)
        for yy in [y + 0.12, y + 0.36]:
            ax.plot([x + 0.4, x + 0.65], [yy, y + 0.24], color=color, lw=0.65)


def method_overview():
    # Main method: two actual teachers and clear supervision branches, then adapted student.
    fig = plt.figure(figsize=(7.3, 3.65))
    ax = fig.add_axes([0.015, 0.015, 0.97, 0.97])
    ax.set(xlim=(0, 14), ylim=(0, 7))
    ax.axis("off")
    ax.text(0.2, 6.8, "TRANSLATOR ADAPTATION", weight="bold", size=11)
    ax.text(
        0.2,
        6.38,
        "Wikipedia-derived multilingual supervision  +  independent synthetic translation",
        size=7.8,
        color=MUTED,
    )
    # main upstream teacher and data card
    rounded(ax, 0.2, 2.15, 2.62, 3.7, fc="#F7F9FC")
    icon(ax, 0.42, 5.02, "doc", SLATE, 0.5)
    ax.text(1.0, 5.45, "PERSIAN", size=7.2, weight="bold")
    ax.text(1.0, 5.12, "WIKIPEDIA", size=7.2, weight="bold")
    ax.text(0.43, 4.67, "Question / answer records", size=6.5)
    rounded(ax, 0.42, 3.75, 2.18, 0.65, fc="white", ec="#CBD5E1")
    ax.text(1.51, 4.15, "Gemma 3 27B IT", ha="center", size=8, weight="bold")
    ax.text(1.51, 3.89, "translation teacher", ha="center", size=6.4, color=MUTED)
    ax.text(0.44, 3.29, "fa → en, ar", size=8.2, weight="bold", color=SLATE)
    ax.text(0.44, 2.94, "en → zh, he, es", size=8.2, weight="bold", color=SLATE)
    ax.text(0.43, 2.44, r"Variants: $q_s,\;a_s,\;a_t$", size=7.4)
    # branching outputs into same target generation loss
    for y, color, heading, eq, sys, n, kind in [
        (
            4.15,
            GOLD,
            "MULTILINGUAL ANSWERING",
            r"$q_s(c)\;\longrightarrow\;a_t^v(c)$",
            "Answer in t  ·  extent / style",
            "19.46M records  |  5.76M cross-language",
            "chat",
        ),
        (
            2.05,
            BLUE,
            "ANSWER-TEXT TRANSLATION",
            r"$a_s(c)\;\longrightarrow\;a_t(c)$",
            "Translate from s into t",
            "10.70M records  |  30 directions",
            "mt",
        ),
    ]:
        rounded(ax, 3.38, y, 5.18, 1.68, fc="white", ec="#D4DDEA")
        ax.add_patch(Rectangle((3.38, y), 0.055, 1.68, fc=color, lw=0))
        icon(ax, 3.66, y + 1.04, kind, color, 0.43)
        ax.text(4.36, y + 1.36, heading, size=7.3, weight="bold", color=color)
        ax.text(4.36, y + 0.88, eq, size=12, color=INK)
        ax.text(3.66, y + 0.5, "System: " + sys, size=7)
        ax.text(3.66, y + 0.2, n, size=6.7, color=MUTED)
    arrow(ax, (2.84, 4.38), (3.33, 4.99), GOLD)
    arrow(ax, (2.84, 3.25), (3.33, 2.92), BLUE)
    # receiving student stage2
    rounded(ax, 9.10, 3.05, 2.68, 2.4, fc="#F4F7FC", ec="#B8C9E5")
    icon(ax, 9.35, 4.55, "model", BLUE)
    ax.text(10.95, 4.93, "PHASE 2", ha="center", size=7, weight="bold", color=BLUE)
    ax.text(10.44, 4.14, "Gemma 3 4B student", ha="center", size=7.6, weight="bold")
    ax.text(10.44, 3.72, "Response-token loss", ha="center", size=7.2)
    ax.text(10.44, 3.35, "30.15M selected records", ha="center", size=7.0, color=MUTED)
    arrow(ax, (8.60, 4.95), (9.05, 4.60), GOLD)
    arrow(ax, (8.60, 2.92), (9.05, 3.69), BLUE)
    # final teacher stage3 then translator endpoint
    rounded(ax, 3.38, 0.38, 5.18, 1.18, fc="#F8FAFD", ec="#D4DDEA")
    ax.text(3.65, 1.22, "DeepSeek-V3.1", size=8.4, weight="bold", color=INK)
    ax.text(3.65, 0.84, "Native source texts + synthetic translations", size=7.1)
    ax.text(3.65, 0.54, "658,274 forward / reverse SFT examples", size=7.1, color=BLUE)
    rounded(ax, 9.10, 0.38, 2.68, 1.18, fc="#EDF3FF", ec="#B8C9E5")
    ax.text(10.44, 1.22, "PHASE 3", size=7, weight="bold", ha="center", color=BLUE)
    ax.text(10.44, 0.8, "Fresh translation LoRA", size=7.7, ha="center")
    ax.text(10.44, 0.53, "after merging phase 2", size=6.7, ha="center", color=MUTED)
    arrow(ax, (8.60, 0.98), (9.05, 0.98), BLUE)
    arrow(ax, (10.44, 3.0), (10.44, 1.61), BLUE)
    ax.text(10.6, 2.27, "merge", size=6.6, color=MUTED)
    rounded(ax, 12.15, 0.38, 1.65, 5.06, fc="#172B4D", ec="#172B4D")
    ax.text(12.98, 4.93, "HexaBee-4B", ha="center", size=11, weight="bold", color="white")
    ax.text(12.98, 4.48, "FINAL TRANSLATOR", ha="center", size=6, color="#C7D8FB")
    for i, code in enumerate(["FA  ·  EN", "AR  ·  ZH", "HE  ·  ES"]):
        ax.text(12.98, 3.63 - i * 0.5, code, ha="center", size=8, color="white")
    ax.text(12.98, 1.79, "30", ha="center", size=21, color="white", weight="bold")
    ax.text(12.98, 1.4, "directions", ha="center", size=7, color="#C7D8FB")
    ax.text(12.98, 0.75, "merged weights", ha="center", size=6.8, color="#C7D8FB")
    arrow(ax, (11.8, 0.98), (12.12, 0.98), BLUE)
    save(fig, "method_overview")


def target_results():
    # Target result comparison; two compatible metrics, explicit focused scales.
    ch = json.load(open(R / "results/paper/full_test_chrfpp.json"))["models"]
    co = json.load(open(R / "results/paper/comet_full_summary.json"))["models"]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), gridspec_kw={"wspace": 0.38})
    fig.subplots_adjust(left=0.12, right=0.94, top=0.72, bottom=0.18)
    header(
        fig,
        "Translation quality by target language",
        "Full test: 36,572 examples  ·  equal weight for each of five incoming directions",
    )
    legend(fig, 0.85, 0.12)
    for ax, key, title, lim in zip(
        axes, ["chrfpp", "comet"], ["chrF++", "COMET ×100"], [(30, 96), (79, 94)], strict=True
    ):
        for i, t in enumerate(CODES):
            vals = []
            for m in MODELS:
                ds = ch[m]["per_direction"] if key == "chrfpp" else co[m]["per_direction"]
                v = statistics.mean(d[key] for p, d in ds.items() if p.endswith("-" + NAMES[t]))
                vals.append(v if key == "chrfpp" else 100 * v)
            ax.plot(
                [min(vals), max(vals)],
                [i, i],
                color="#D0D8E5",
                lw=3,
                solid_capstyle="round",
                zorder=1,
            )
            for j, (m, v) in enumerate(zip(MODELS, vals, strict=True)):
                ax.scatter(
                    v,
                    i,
                    s=27 if j < 2 else 40,
                    marker=MARKERS[m],
                    facecolor="white" if j == 0 else COLORS[m],
                    edgecolor=COLORS[m],
                    linewidth=0.95,
                    zorder=4,
                )
            ax.text(
                vals[-1] + (1.4 if key == "chrfpp" else 0.35),
                i,
                f"{vals[-1]:.1f}",
                va="center",
                size=7,
                color=BLUE,
                weight="bold",
            )
        ax.set_yticks(range(6), [NAMES[t] for t in CODES])
        ax.invert_yaxis()
        ax.set_xlim(*lim)
        ax.set_ylim(5.5, -0.5)
        clean(ax)
        ax.set_title(title, loc="left", weight="bold", pad=13)
        ax.set_xlabel("Score  ·  focused scale", size=7, color=MUTED)
    save(fig, "target_comparison")


def direction_gains():
    ch = json.load(open(R / "results/paper/full_test_chrfpp.json"))["models"]
    # All direction gains in two aligned, annotated tile matrices.
    cmap = LinearSegmentedColormap.from_list(
        "paper_blue", ["#F1F5FE", "#CDDDFC", "#85AFF7", "#3A7BEA", "#174AAB"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.35))
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.12, top=0.74, wspace=0.32)
    header(
        fig,
        "Translation gains across all 30 directions",
        "HexaBee-4B minus each public comparator  ·  corpus chrF++ per direction  ·  higher is better",
    )
    for ax, b in zip(axes, MODELS[:2], strict=True):
        mat = np.full((6, 6), np.nan)
        for i, ss in enumerate(CODES):
            for j, tt in enumerate(CODES):
                if i != j:
                    pp = NAMES[ss] + "-" + NAMES[tt]
                    mat[i, j] = (
                        ch["PCT-4B"]["per_direction"][pp]["chrfpp"]
                        - ch[b]["per_direction"][pp]["chrfpp"]
                    )
        norm = Normalize(0, 30)
        for i in range(6):
            for j in range(6):
                rounded(
                    ax,
                    j - 0.465,
                    i - 0.465,
                    0.93,
                    0.93,
                    fc="#F2F4F7" if i == j else cmap(norm(mat[i, j])),
                    ec="white",
                    lw=0.4,
                )
                ax.text(
                    j,
                    i,
                    "—" if i == j else f"+{mat[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.3,
                    weight="bold" if i != j and mat[i, j] > 20 else "normal",
                    color="white" if i != j and mat[i, j] > 21 else INK,
                )
        ax.set(xlim=(-0.55, 5.55), ylim=(5.55, -0.55), aspect="equal")
        ax.set_xticks(range(6), [s.upper() for s in CODES])
        ax.set_yticks(range(6), [s.upper() for s in CODES])
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=6)
        ax.spines[:].set_visible(False)
        ax.set_title("vs " + b, loc="left", size=8.8, weight="bold", pad=24)
        ax.set_xlabel("TARGET", size=6.5, color=MUTED, labelpad=9)
        ax.set_ylabel("SOURCE", size=6.5, color=MUTED, labelpad=8)
    cax = fig.add_axes([0.945, 0.20, 0.012, 0.45])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 30), cmap=cmap), cax=cax, ticks=[0, 10, 20, 30]
    )
    cax.tick_params(labelsize=6, length=2)
    cax.spines[:].set_visible(False)
    save(fig, "direction_gains_both")


def corpus_composition():
    # Proportional source-to-task flow. Widths preserve absolute counts.
    s = json.load(open(R / "results/paper/phase2_summary.json"))
    families = list(s["retained_by_family"])
    labels = [
        "General QA A",
        "Standard QA A",
        "Title QA A",
        "General QA B",
        "Standard QA B",
        "Title QA B",
    ]
    total = s["final_total"]
    fig = plt.figure(figsize=(7.3, 3.5))
    ax = fig.add_axes([0.015, 0.04, 0.97, 0.76])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    header(
        fig,
        "Phase-2 data composition",
        "30,153,404 retained records  ·  ribbon widths encode counts  ·  QA includes all response styles",
    )
    scale = 0.77 / total
    gap = 0.027
    top = 0.98
    src = []
    for f, lab in zip(families, labels, strict=True):
        n = sum(s["retained_by_family"][f].values())
        h = n * scale
        src.append((f, lab, top - h, top))
        top -= h + gap
    qa = s["qa_total"]
    mt = s["by_task"]["translate"]["total"]
    qa0 = 1 - qa * scale
    mt0 = qa0 - 0.10 - mt * scale
    targets = {"qa": [qa0, qa0], "mt": [mt0, mt0]}
    # Stack source contributions in bottom-to-top order in targets to reduce crossing.
    for f, lab, lo, hi in reversed(src):
        nmt = s["retained_by_family"][f]["translate"]
        nqa = sum(s["retained_by_family"][f].values()) - nmt
        cur = lo
        for group, n, col in [("mt", nmt, BLUE), ("qa", nqa, GOLD)]:
            h = n * scale
            dst = targets[group][1]
            targets[group][1] += h
            verts = [
                (0.37, cur),
                (0.50, cur),
                (0.62, dst),
                (0.76, dst),
                (0.76, dst + h),
                (0.62, dst + h),
                (0.50, cur + h),
                (0.37, cur + h),
                (0.37, cur),
            ]
            patch = PathPatch(
                MPath(
                    verts,
                    [
                        MPath.MOVETO,
                        MPath.CURVE4,
                        MPath.CURVE4,
                        MPath.CURVE4,
                        MPath.LINETO,
                        MPath.CURVE4,
                        MPath.CURVE4,
                        MPath.CURVE4,
                        MPath.CLOSEPOLY,
                    ],
                ),
                fc=col,
                alpha=0.26,
                ec="white",
                lw=0.3,
            )
            ax.add_patch(patch)
            cur += h
        for _group, n, col, y in [("mt", nmt, BLUE, lo), ("qa", nqa, GOLD, lo + nmt * scale)]:
            ax.add_patch(Rectangle((0.354, y), 0.016, n * scale, fc=col, lw=0))
        ax.text(0.338, (lo + hi) / 2, lab, ha="right", va="center", size=7.2)
        ax.text(
            0.375,
            hi + 0.004,
            f"{(hi - lo) / scale / 1e6:.2f}M",
            ha="left",
            va="bottom",
            size=6,
            color=MUTED,
        )
    for lo, n, col, title in [(qa0, qa, GOLD, "ANSWERING"), (mt0, mt, BLUE, "TRANSLATION")]:
        ax.add_patch(Rectangle((0.76, lo), 0.018, n * scale, fc=col, lw=0))
        mid = lo + n * scale / 2
        ax.text(0.8, mid + 0.08, title, color=col, size=8, weight="bold")
        ax.text(0.8, mid - 0.015, f"{n / 1e6:.2f}M", size=16, weight="bold", color=INK)
        ax.text(0.8, mid - 0.09, f"{100 * n / total:.1f}% of phase 2", size=6.8, color=MUTED)
    save(fig, "corpus_composition")


def answering_coverage():
    cmap = LinearSegmentedColormap.from_list(
        "coverage_blue", ["#F1F5FE", "#CDDDFC", "#85AFF7", "#3A7BEA", "#174AAB"]
    )
    # Answering coverage tiles: annotate all cells, distinguish zero and known conflict.
    rr = list(csv.DictReader(open(R / "results/paper/phase2_by_direction.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.3))
    fig.subplots_adjust(left=0.1, right=0.91, bottom=0.13, top=0.74, wspace=0.32)
    header(
        fig,
        "Multilingual answering coverage",
        "Question language × response language  ·  counts use a shared logarithmic color scale",
    )
    norm = LogNorm(800, 5e6)
    for ax, allqa, title in zip(
        axes, [False, True], ["Complete-answer QA", "All answering variants"], strict=True
    ):
        mat = np.zeros((6, 6))
        for r in rr:
            if r["task"] == "translate" or (not allqa and r["task"] != "complete"):
                continue
            mat[CODES.index(r["source"]), CODES.index(r["target"])] += int(r["n"])
        for i in range(6):
            for j in range(6):
                n = mat[i, j]
                rounded(
                    ax,
                    j - 0.465,
                    i - 0.465,
                    0.93,
                    0.93,
                    fc="#F4F5F8" if not n else cmap(norm(n)),
                    ec="white",
                    lw=0.4,
                )
                tx = (
                    "0"
                    if n == 0
                    else f"{n / 1e6:.2f}M"
                    if n >= 1e6
                    else f"{n / 1e3:.0f}k"
                    if n >= 10000
                    else f"{n / 1e3:.1f}k"
                    if n >= 1000
                    else str(int(n))
                )
                ax.text(
                    j, i, tx, ha="center", va="center", size=7, color="white" if n > 500000 else INK
                )
                if j == 5 and i in [0, 3]:
                    ax.add_patch(
                        Rectangle((j - 0.44, i - 0.44), 0.88, 0.88, fill=False, ec=GOLD, lw=1.8)
                    )
                    ax.text(j + 0.27, i - 0.19, "!", size=7, weight="bold", color=GOLD)
        ax.set(xlim=(-0.55, 5.55), ylim=(5.55, -0.55), aspect="equal")
        ax.set_xticks(range(6), [s.upper() for s in CODES])
        ax.set_yticks(range(6), [s.upper() for s in CODES])
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=6)
        ax.spines[:].set_visible(False)
        ax.set_title(title, loc="left", size=8.8, weight="bold", pad=24)
        ax.set_xlabel("RESPONSE", size=6.5, color=MUTED, labelpad=8)
        ax.set_ylabel("QUESTION", size=6.5, color=MUTED, labelpad=8)
    cax = fig.add_axes([0.945, 0.22, 0.012, 0.43])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[1e3, 1e4, 1e5, 1e6])
    cax.tick_params(labelsize=6, length=2)
    cax.spines[:].set_visible(False)
    fig.text(
        0.1,
        0.035,
        "Gold outline: the 1,961 Spanish-target complete-answer records request English.",
        size=6.7,
        color=GOLD,
    )
    save(fig, "qa_direction_counts")


def judge_rubric():
    # Rubric comparison and bootstrap evidence.
    e = json.load(open(R / "results/paper/evaluation_summary.json"))
    sec = [
        "semantic_accuracy",
        "terminology",
        "fluency",
        "cohesion",
        "contextual_appropriateness",
        "target_language_adherence",
        "style",
        "formatting",
    ]
    labs = [
        "Semantics",
        "Terminology",
        "Fluency",
        "Cohesion",
        "Context",
        "Response language",
        "Style",
        "Formatting",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.55))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.72, bottom=0.12, wspace=0.35)
    header(
        fig,
        "Judge rubric comparison",
        "300 matched identities  ·  eight dimensions in each view  ·  scores are not accuracy percentages",
    )
    legend(fig, 0.85, 0.18)
    for ax, v, title in zip(
        axes, ["reference", "no_reference"], ["Reference-assisted", "Reference-free"], strict=True
    ):
        for i, k in enumerate(sec):
            vals = [e["models"][m][v]["sections"][k] for m in MODELS]
            ax.plot([min(vals), max(vals)], [i, i], color="#D9E0EB", lw=2.5, zorder=1)
            for j, (m, a) in enumerate(zip(MODELS, vals, strict=True)):
                ax.scatter(
                    a,
                    i,
                    s=23 if j < 2 else 34,
                    marker=MARKERS[m],
                    facecolor="white" if j == 0 else COLORS[m],
                    edgecolor=COLORS[m],
                    lw=0.8,
                    zorder=3,
                )
        ax.set_yticks(range(8), labs if ax == axes[0] else [""] * 8)
        ax.set_xlim(70, 100)
        ax.set_ylim(7.5, -0.5)
        ax.set_title(title, loc="left", weight="bold", pad=12)
        ax.set_xlabel("Judge score  ·  focused scale", size=7, color=MUTED)
        clean(ax)
    save(fig, "judge_rubric_profile")


def training_loss():
    # Training diagnostic: full raw signal and labeled 25-entry rolling median.
    st = json.load(open(R / "results/plot_inputs/training_loss.json"))
    hist = [r for r in st["log_history"] if "loss" in r]
    x = np.array([r["step"] for r in hist])
    y = np.array([r["loss"] for r in hist])
    med = np.array([np.median(y[max(0, i - 12) : min(len(y), i + 13)]) for i in range(len(y))])
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.23, top=0.72)
    header(fig, "Final-stage optimization", "Training loss only  ·  not a quality metric", x=0.16)
    ax.plot(x, y, color="#C6D5EF", lw=0.65, label="Logged loss")
    ax.plot(x, med, color=BLUE, lw=1.2, label="25-log rolling median")
    clean(ax, False)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Response-token NLL")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    save(fig, "training_loss")


def training_lineage():
    # Checkpoint lineage: adapter continuation and merge boundaries are explicit.
    fig = plt.figure(figsize=(7.3, 2.35))
    ax = fig.add_axes([0.015, 0.025, 0.97, 0.96])
    ax.set(xlim=(0, 14), ylim=(0, 4.2))
    ax.axis("off")
    ax.text(0.2, 4.02, "THREE-STAGE CHECKPOINT LINEAGE", size=11, weight="bold")
    ax.text(
        0.2,
        3.60,
        "One continued adapter across phases 1–2  ·  a fresh adapter after the first merge",
        size=7.2,
        color=MUTED,
    )
    cards = [
        (0.2, 2.0, 1.65, "BASE", "Gemma 3", "4B IT", None),
        (
            2.35,
            1.63,
            2.8,
            "PHASE 1",
            "Wikipedia",
            "continued pretraining",
            "3.40M tokenized examples",
        ),
        (
            5.65,
            1.63,
            2.8,
            "PHASE 2",
            "Answering +",
            "answer-text translation",
            "30.15M selected records",
        ),
        (
            8.95,
            1.63,
            2.8,
            "PHASE 3",
            "Bidirectional",
            "translation SFT",
            "658,274 selected records",
        ),
        (12.25, 2.0, 1.55, "FINAL", "HexaBee-4B", "merged", None),
    ]
    for i, (x, y, w, h, t1, t2, detail) in enumerate(cards):
        col = BLUE if i >= 3 else SLATE
        rounded(
            ax,
            x,
            y,
            w,
            1.35 if detail is None else 1.72,
            fc=INK if i == 4 else LIGHT,
            ec=INK if i == 4 else "#D4DDEA",
        )
        ax.text(
            x + 0.17,
            y + (1.04 if detail is None else 1.40),
            h,
            size=6.7,
            weight="bold",
            color="white" if i == 4 else col,
        )
        ax.text(
            x + 0.17,
            y + (0.65 if detail is None else 0.94),
            t1,
            size=8,
            weight="bold",
            color="white" if i == 4 else INK,
        )
        ax.text(
            x + 0.17,
            y + (0.30 if detail is None else 0.57),
            t2,
            size=7.7,
            color="#DCE7FF" if i == 4 else INK,
        )
        if detail:
            ax.text(x + 0.17, y + 0.22, detail, size=6.8, color=MUTED)
    for a, b in [(1.89, 2.3), (5.18, 5.6), (8.48, 8.9), (11.78, 12.2)]:
        arrow(ax, (a, 2.59), (b, 2.59), BLUE if a > 8 else SLATE)
    ax.plot([2.50, 8.29], [1.15, 1.15], color=GOLD, lw=2, solid_capstyle="round")
    ax.text(
        5.4,
        0.68,
        "Same LoRA adapter: initialize → continue",
        ha="center",
        size=7,
        color=GOLD,
        weight="bold",
    )
    ax.plot([9.12, 11.57], [1.15, 1.15], color=BLUE, lw=2, solid_capstyle="round")
    ax.text(10.35, 0.68, "Fresh LoRA adapter", ha="center", size=7, color=BLUE, weight="bold")
    ax.text(8.70, 3.43, "MERGE", ha="center", size=5.8, color=BLUE)
    ax.text(11.99, 3.43, "MERGE", ha="center", size=5.8, color=BLUE)
    save(fig, "training_lineage")


def main():
    F.mkdir(exist_ok=True)
    method_overview()
    target_results()
    direction_gains()
    corpus_composition()
    answering_coverage()
    judge_rubric()
    training_loss()
    training_lineage()
    print("Rebuilt 8 figure families as PDF, SVG, and PNG in figures/.")


if __name__ == "__main__":
    main()
