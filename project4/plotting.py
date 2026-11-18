"""Shared figure style so every member's plots look the same in the report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order: series k always gets COLORS[k]. Never cycle past 3
# on one axis; fold extra series into small multiples instead.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6")
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "lines.linewidth": 2.0, "legend.frameon": False,
    })


def save(fig, path_stem):
    """Save as PDF (for LaTeX) and PNG (for slides / notes)."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".pdf"))
    fig.savefig(path_stem.with_suffix(".png"))
    plt.close(fig)
