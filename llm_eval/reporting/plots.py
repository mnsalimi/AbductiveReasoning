"""
reporting/plots.py
------------------
Evolution line plots and reasoning-tier stacked bar charts.
"""

from __future__ import annotations

import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Tier constants (shared between functions)
# ---------------------------------------------------------------------------

_TIER_ORDER = [
    "1. Collapsed (no markers)",
    "2a. Simple Elimination (neg=1)",
    "2b. Rigorous Elimination (neg≥2)",
    "3a. Light Exploration (branch≥1)",
    "3b. Heavy Exploration (branch≥3)",
]

_TIER_COLORS = ["#d62728", "#ff7b00", "#fde802", "#74c476", "#006d2c"]


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


# ---------------------------------------------------------------------------
# Reasoning-tier stacked bar charts
# ---------------------------------------------------------------------------

def generate_tier_plots(base_dir: str) -> None:
    """
    Produce a stacked bar chart per dataset showing the distribution of
    reasoning complexity tiers across checkpoints.

    Requires both 'neg_constraint' and 'branchiness' metrics to be active.
    Tiers are derived from neg_constraint_count and branchiness_count columns.
    Only called when both metrics are enabled (guarded in main.py).
    """
    log_files = glob.glob(os.path.join(base_dir, "checkpoint-*", "detailed_metrics_log.csv"))
    if not log_files:
        return

    frames: list[pd.DataFrame] = []
    for path in log_files:
        try:
            ckpt_num = int(os.path.basename(os.path.dirname(path)).split("-")[-1])
            df = pd.read_csv(path)
            df["Checkpoint"] = ckpt_num
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return

    full = pd.concat(frames, ignore_index=True)

    branch_col = "branchiness_count" if "branchiness_count" in full.columns else None
    neg_col = "neg_constraint_count" if "neg_constraint_count" in full.columns else None

    def _tier(row: pd.Series) -> str:
        b = row.get(branch_col, 0) or 0 if branch_col else 0
        n = row.get(neg_col, 0) or 0 if neg_col else 0
        if b >= 3:
            return "3b. Heavy Exploration (branch≥3)"
        if b > 0:
            return "3a. Light Exploration (branch≥1)"
        if n >= 2:
            return "2b. Rigorous Elimination (neg≥2)"
        if n > 0:
            return "2a. Simple Elimination (neg=1)"
        return "1. Collapsed (no markers)"

    full["Tier"] = full.apply(_tier, axis=1)

    for ds in full["Dataset"].unique():
        ds_df = full[full["Dataset"] == ds]
        counts = ds_df.groupby(["Checkpoint", "Tier"]).size().unstack(fill_value=0)
        pct = counts.div(counts.sum(axis=1), axis=0) * 100
        for t in _TIER_ORDER:
            if t not in pct.columns:
                pct[t] = 0
        pct = pct[_TIER_ORDER]

        fig, ax = plt.subplots(figsize=(10, 6))
        pct.plot(kind="bar", stacked=True, ax=ax, color=_TIER_COLORS, width=0.75,
                 edgecolor="black", linewidth=0.4)
        ax.set_title(f"Reasoning Tier Distribution – {ds}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("% of items")
        ax.legend(title="Tier", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.xticks(rotation=0)
        for container in ax.containers:
            labels = [f"{v.get_height():.1f}%" if v.get_height() > 5 else "" for v in container]
            ax.bar_label(container, labels=labels, label_type="center",
                         color="white", fontsize=8, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(base_dir, f"tier_distribution_{ds}.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[OK] Tier plot → {path}")
