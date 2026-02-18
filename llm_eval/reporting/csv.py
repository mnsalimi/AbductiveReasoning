"""
reporting/csv.py
----------------
Per-checkpoint CSV writing and the end-of-run full debug log.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _ensure(*dirs: str) -> None:
    import os
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-checkpoint CSV saving
# ---------------------------------------------------------------------------

def save_checkpoint_csvs(
    checkpoint_num: int,
    checkpoint_dir_name: str,
    result_rows: list[dict],
    active_metric_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Write per-item and summary CSVs for one checkpoint.

    Returns (unnorm_summary_df, norm_summary_df) with a ``Checkpoint`` column
    already set, ready to be appended to the global summary list.
    """
    import os
    _ensure(config.UNNORM_DIR, config.NORM_DIR)

    unnorm_rows: list[dict] = []
    norm_rows: list[dict] = []

    for row in result_rows:
        wc = row.get("word_count", 0) or 0
        norm_factor = 100.0 / wc if wc > 0 else 0.0

        base = {
            "Dataset": row["dataset"],
            "Status": row["status"],
            "Problem ID": row["pid"],
            "Word Count": wc,
        }

        unnorm_item = dict(base)
        norm_item = dict(base)

        for mname in active_metric_names:
            mdata = row.get("metrics", {}).get(mname, {})
            count = mdata.get("example_count", 0) or 0
            detected = mdata.get("detected", False)
            analysis = mdata.get("reasoning", "")
            examples_str = "; ".join(
                e.get("excerpt", "") for e in mdata.get("examples", []) if isinstance(e, dict)
            )

            unnorm_item[f"{mname}_count"] = count
            unnorm_item[f"{mname}_detected"] = detected
            unnorm_item[f"{mname}_analysis"] = analysis
            unnorm_item[f"{mname}_examples"] = examples_str

            norm_item[f"{mname}_count"] = count * norm_factor
            norm_item[f"{mname}_detected"] = detected
            norm_item[f"{mname}_analysis"] = analysis
            norm_item[f"{mname}_examples"] = examples_str

        unnorm_rows.append(unnorm_item)
        norm_rows.append(norm_item)

    unnorm_df = pd.DataFrame(unnorm_rows)
    norm_df = pd.DataFrame(norm_rows)

    count_cols = [f"{m}_count" for m in active_metric_names if f"{m}_count" in unnorm_df.columns]
    detected_cols = [f"{m}_detected" for m in active_metric_names if f"{m}_detected" in unnorm_df.columns]

    def _summarise(df: pd.DataFrame) -> pd.DataFrame:
        agg: dict[str, str] = {c: "mean" for c in count_cols if c in df.columns}
        agg.update({c: "mean" for c in detected_cols if c in df.columns})
        if "Word Count" in df.columns:
            agg["Word Count"] = "mean"
        summary = df.groupby(["Dataset", "Status"]).agg(agg).reset_index()
        summary["Checkpoint"] = checkpoint_num
        return summary

    unnorm_summary = _summarise(unnorm_df)
    norm_summary = _summarise(norm_df)

    for data_df, base_dir, summary_df in [
        (unnorm_df, config.UNNORM_DIR, unnorm_summary),
        (norm_df, config.NORM_DIR, norm_summary),
    ]:
        ckpt_out = os.path.join(base_dir, checkpoint_dir_name)
        _ensure(ckpt_out)
        data_df.to_csv(os.path.join(ckpt_out, "detailed_metrics_log.csv"), index=False)
        summary_df.to_csv(os.path.join(ckpt_out, "summary_metrics.csv"), index=False)

    return unnorm_summary, norm_summary


# ---------------------------------------------------------------------------
# Full debug log (written once at the end across all checkpoints)
# ---------------------------------------------------------------------------

def write_debug_logs(all_results: list[dict]) -> None:
    """
    Write one consolidated CSV per dataset containing every evaluated item
    across all checkpoints.
    """
    import os
    _ensure(config.LOG_DIR)

    by_dataset: dict[str, list[dict]] = {}
    for r in all_results:
        ds = r.get("dataset", "unknown")
        by_dataset.setdefault(ds, []).append(r)

    for ds, records in by_dataset.items():
        rows = []
        for rec in records:
            row = {
                "timestamp": rec.get("timestamp"),
                "run_id": rec.get("run_id"),
                "checkpoint": rec.get("checkpoint"),
                "dataset": rec.get("dataset"),
                "problem_id": rec.get("pid"),
                "status": rec.get("status"),
                "true_label": rec.get("true_label"),
                "predicted_label": rec.get("pred_label"),
                "question": rec.get("question"),
                "word_count": rec.get("word_count"),
                "reasoning": rec.get("reasoning"),
            }
            for mname, mdata in (rec.get("metrics") or {}).items():
                row[f"{mname}_type"] = mdata.get("type")
                row[f"{mname}_detected"] = mdata.get("detected")
                row[f"{mname}_example_count"] = mdata.get("example_count")
                row[f"{mname}_analysis"] = mdata.get("reasoning")
                row[f"{mname}_examples"] = _safe_json(mdata.get("examples", []))
            rows.append(row)

        out_path = os.path.join(config.LOG_DIR, f"{ds}_full_debug_{config.RUN_ID}.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Debug log → {out_path}  ({len(rows)} rows)")
