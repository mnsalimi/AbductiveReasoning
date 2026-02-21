"""
reporting/plots.py
------------------
Evolution line plots.
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Evolution line plots
# ---------------------------------------------------------------------------

def build_evolution_plots(combined: pd.DataFrame, base_dir: str) -> None:
    """Produce one PNG per (metric_col × status) pair."""
    metric_cols = [c for c in combined.columns if c.endswith("_count")]

    for col in metric_cols:
        if "Status" in combined.columns:
            for status in combined["Status"].unique():
                _evolution_plot(combined[combined["Status"] == status], col, base_dir, suffix=status.lower())
        else:
            _evolution_plot(combined, col, base_dir)


def _evolution_plot(df: pd.DataFrame, metric_col: str, base_dir: str, suffix: str = "") -> None:
    if metric_col not in df.columns:
        return
    try:
        pivot = df.pivot_table(index="Checkpoint", columns="Dataset", values=metric_col, aggfunc="mean")
    except Exception:
        return
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for ds in pivot.columns:
        ax.plot(pivot.index, pivot[ds], marker="o", label=ds, linewidth=2)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{metric_col} – evolution across checkpoints" + (f" ({suffix})" if suffix else ""))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = f"evolution_{metric_col}{'_' + suffix if suffix else ''}.png"
    plt.savefig(os.path.join(base_dir, fname), dpi=150)
    plt.close(fig)
