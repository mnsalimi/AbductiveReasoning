"""
reporting/plots.py
------------------
Evolution line plots.

Plotting behaviour is driven by ``config.SAMPLE_CORRECT_RATIO``:
  == 1.0  →  only "Correct" items were sampled → produce a single "correct" plot
  == 0.0  →  only "Incorrect" items were sampled → produce a single "incorrect" plot
  otherwise → produce three plots: "correct", "incorrect", and a "mix"
              (all statuses averaged together)
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import config

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evolution line plots
# ---------------------------------------------------------------------------

def build_evolution_plots(combined: pd.DataFrame, base_dir: str) -> None:
    """Produce evolution PNGs based on ``config.SAMPLE_CORRECT_RATIO``.

    * ratio == 1.0  →  only a "correct" plot per metric column
    * ratio == 0.0  →  only an "incorrect" plot per metric column
    * anything else →  three plots: "correct", "incorrect", and "mix"
                       where "mix" averages over all statuses
    """
    metric_cols = [c for c in combined.columns if c.endswith("_count")]
    ratio = config.SAMPLE_CORRECT_RATIO

    has_status = "Status" in combined.columns

    for col in metric_cols:
        if ratio == 1.0:
            # Only correct
            sub = combined[combined["Status"] == "Correct"] if has_status else combined
            _evolution_plot(sub, col, base_dir, suffix="correct")

        elif ratio == 0.0:
            # Only incorrect
            sub = combined[combined["Status"] == "Incorrect"] if has_status else combined
            _evolution_plot(sub, col, base_dir, suffix="incorrect")

        else:
            # Both individual statuses plus a mixed aggregate
            if has_status:
                for status in ("Correct", "Incorrect"):
                    sub = combined[combined["Status"] == status]
                    if not sub.empty:
                        _evolution_plot(sub, col, base_dir, suffix=status.lower())
                # Mix: aggregate over all statuses
                _evolution_plot(combined, col, base_dir, suffix="mix")
            else:
                _evolution_plot(combined, col, base_dir, suffix="mix")
