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
            mtype = mdata.get("type", "")
            count = mdata.get("example_count", 0) or 0
            detected = mdata.get("detected", False)
            analysis = mdata.get("reasoning", "")
            score = mdata.get("score")  # float | None

            # For coverage metrics, detail_str lists each observation detail + addressed flag
            if mtype == "coverage":
                examples_str = "; ".join(
                    f"[{'Y' if e.get('addressed') else 'N'}] {e.get('detail', '')}"
                    for e in mdata.get("examples", []) if isinstance(e, dict)
                )
            else:
                examples_str = "; ".join(
                    e.get("excerpt", "") for e in mdata.get("examples", []) if isinstance(e, dict)
                )

            unnorm_item[f"{mname}_count"] = count
            unnorm_item[f"{mname}_detected"] = detected
            unnorm_item[f"{mname}_analysis"] = analysis
            unnorm_item[f"{mname}_examples"] = examples_str
            if score is not None:
                unnorm_item[f"{mname}_score"] = score

            norm_item[f"{mname}_count"] = count * norm_factor
            norm_item[f"{mname}_detected"] = detected
            norm_item[f"{mname}_analysis"] = analysis
            norm_item[f"{mname}_examples"] = examples_str
            if score is not None:
                norm_item[f"{mname}_score"] = score

        unnorm_rows.append(unnorm_item)
        norm_rows.append(norm_item)

    unnorm_df = pd.DataFrame(unnorm_rows)
    norm_df = pd.DataFrame(norm_rows)

    count_cols = [f"{m}_count" for m in active_metric_names if f"{m}_count" in unnorm_df.columns]
    detected_cols = [f"{m}_detected" for m in active_metric_names if f"{m}_detected" in unnorm_df.columns]
    score_cols = [f"{m}_score" for m in active_metric_names if f"{m}_score" in unnorm_df.columns]

    def _summarise(df: pd.DataFrame) -> pd.DataFrame:
        agg: dict[str, str] = {c: "mean" for c in count_cols if c in df.columns}
        agg.update({c: "mean" for c in detected_cols if c in df.columns})
        agg.update({c: "mean" for c in score_cols if c in df.columns})
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
    Write two consolidated debug files per dataset across all checkpoints:

    1. ``{dataset}_full_debug.jsonl`` – one JSON object per row combining the
       full source-item fields (observation_1/2, hypothesis_1/2, true_label,
       predicted_label, the model reasoning, correct) with the evaluated metric
       results.  ``timestamp`` and ``run_id`` are intentionally omitted so the
       file is stable across re-runs.

    2. ``{dataset}_full_debug_{RUN_ID}.csv`` – same content as a spreadsheet,
       without ``timestamp`` / ``run_id`` columns, with the source-item fields
       prepended.
    """
    import os
    _ensure(config.LOG_DIR)

    by_dataset: dict[str, list[dict]] = {}
    for r in all_results:
        ds = r.get("dataset", "unknown")
        by_dataset.setdefault(ds, []).append(r)

    for ds, records in by_dataset.items():
        rows = []
        jsonl_lines: list[str] = []

        for rec in records:
            status = rec.get("status", "")

            # All raw source-item fields, exactly as they appear in the JSON
            # file (works for ART, copa_guess_effect, or any other schema).
            raw_item: dict = rec.get("item") or {}

            # ── JSONL record ───────────────────────────────────────────────
            jsonl_obj: dict = {
                **raw_item,          # all source fields first
                "checkpoint":  rec.get("checkpoint"),
                "dataset":     rec.get("dataset"),
                "problem_id":  rec.get("pid"),
                "word_count":  rec.get("word_count"),
                "metrics": {
                    mname: {
                        "type":          mdata.get("type"),
                        "detected":      mdata.get("detected"),
                        "example_count": mdata.get("example_count"),
                        "analysis":      mdata.get("reasoning"),
                        "examples":      mdata.get("examples", []),
                        "tokens":        mdata.get("tokens") or {},
                        "error":         mdata.get("error") or "",
                    }
                    for mname, mdata in (rec.get("metrics") or {}).items()
                },
            }
            jsonl_lines.append(json.dumps(jsonl_obj, ensure_ascii=False))

            # ── CSV row ────────────────────────────────────────────────────
            row: dict = {
                "checkpoint": rec.get("checkpoint"),
                "dataset":    rec.get("dataset"),
                "problem_id": rec.get("pid"),
                "status":     status,
                **raw_item,          # all source fields
                "word_count": rec.get("word_count"),
            }
            for mname, mdata in (rec.get("metrics") or {}).items():
                tok = mdata.get("tokens") or {}
                row[f"{mname}_type"]             = mdata.get("type")
                row[f"{mname}_detected"]         = mdata.get("detected")
                row[f"{mname}_example_count"]    = mdata.get("example_count")
                row[f"{mname}_analysis"]         = mdata.get("reasoning")
                row[f"{mname}_examples"]         = _safe_json(mdata.get("examples", []))
                row[f"{mname}_tokens_input"]     = tok.get("input")
                row[f"{mname}_tokens_output"]    = tok.get("output")
                # Only written when the model separates reasoning tokens
                if "reasoning" in tok:
                    row[f"{mname}_tokens_reasoning"] = tok["reasoning"]
                if "cached_input" in tok:
                    row[f"{mname}_tokens_cached_input"] = tok["cached_input"]
                row[f"{mname}_error"]            = mdata.get("error") or ""
            rows.append(row)

        # Write JSONL
        jsonl_path = os.path.join(config.LOG_DIR, f"{ds}_full_debug.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""))
        print(f"[OK] Full-debug JSONL → {jsonl_path}  ({len(jsonl_lines)} records)")

        # Write CSV
        csv_path = os.path.join(config.LOG_DIR, f"{ds}_full_debug_{config.RUN_ID}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Full-debug CSV  → {csv_path}  ({len(rows)} rows)")
