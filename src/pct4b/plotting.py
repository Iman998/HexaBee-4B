"""Publication-ready vector comparison plots; imported only with --plots."""

from pathlib import Path


def target_comparison(
    summaries: dict, models: list[str], metrics: list[str], outdir: Path, focus: str | None = None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    targets = [t for t in summaries if t != "all"]
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 9})
    palette = ["#64748B", "#C68A24", "#0F766E", "#9333EA"]
    colors = {
        m: "#2563EB" if m == focus else palette[i % len(palette)] for i, m in enumerate(models)
    }
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(max(7, 2.6 * len(metrics)), 3.5), squeeze=False
    )
    for ax, metric in zip(axes[0], metrics, strict=True):
        for j, target in enumerate(targets):
            values = [summaries[target][model][metric] for model in models]
            ax.plot([min(values), max(values)], [j, j], color="#D6DFEC", lw=3, zorder=1)
        for i, model in enumerate(models):
            values = [summaries[t][model][metric] for t in targets]
            ax.scatter(
                values,
                np.arange(len(targets)),
                label=model,
                s=48 if model == focus else 32,
                marker=["o", "s", "D", "^"][i % 4],
                color=colors[model],
                zorder=2,
            )
        ax.set_yticks(range(len(targets)), targets)
        ax.invert_yaxis()
        ax.set_title(metric + (" ↓" if metric == "ter" else " ↑"), fontweight="bold")
        ax.grid(axis="x", color="#E7ECF3")
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0, 0].legend(frameon=False, bbox_to_anchor=(0, 1.42), loc="upper left", ncol=len(models))
    fig.suptitle("Direction-macro comparison by target language", y=1.04, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(outdir / ("target_comparison." + ext), bbox_inches="tight", dpi=200)
    plt.close(fig)
