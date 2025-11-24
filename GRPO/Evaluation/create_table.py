#!/usr/bin/env python3
"""
Aggregate metrics from checkpoint JSONs into a single table.

Directory structure (example):

checkpoints/
  ckpt-001/
    gsm8k/
      all_cases.json
    art/
      all_cases.json
  ckpt-002/
    gsm8k/
      all_cases.json

Each all_cases.json is either:
  {
      "accuracy": 0.87,
      "f1": 0.80,
      ...
  }
or:
  {
      "metrics": {
          "accuracy": 0.87,
          "f1": 0.80,
          ...
      },
      ...
  }

Output:
  - Prints a pretty table to stdout.
  - Writes an Excel file with columns:
        checkpoint, dataset_metric1, dataset_metric2, ...
    and a sheet name controlled by --run.
  - Cells better than the raw model are highlighted in yellow.
  - Two extra summary columns are added at the end:
        better_metrics_than_raw
        better_datasets_than_raw
"""

import os
import json
import argparse
import csv
import numpy as np
from typing import Dict, Any, List, Tuple, Set, Union
import pandas as pd
from openpyxl.styles import PatternFill, Font

from evaluate_aime_raw_vs_finetuned import find_best_checkpoint  

Scalar = Union[int, float, str]


def is_scalar(x: Any) -> bool:
    return isinstance(x, (int, float, str))


def load_metrics_from_json(json_path: str) -> Dict[str, Scalar]:
    """Load scalar metrics from all_cases.json.

    Supports:
      - top-level metrics (keys are metric names)
      - or a 'metrics' sub-dict
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If there's an explicit "metrics" dict, prefer that
    if isinstance(data, dict) and "metrics" in data and isinstance(data["metrics"], dict):
        metrics_dict = data["metrics"]
    else:
        metrics_dict = data

    out: Dict[str, Scalar] = {}
    if isinstance(metrics_dict, dict):
        for k, v in metrics_dict.items():
            if is_scalar(v):
                out[k] = v
    return out


def collect_all_rows(root_dir: str, run: str, best_checkpoint: str = None, model_name: str = "qwen2.5-3B") -> Tuple[List[Dict[str, Scalar]], List[str]]:
    """Walk the checkpoints directory and collect rows + column names.

    Returns:
      rows: list of dicts, each representing one checkpoint
      columns: ordered list of column names (including 'checkpoint')
    """
    rows: List[Dict[str, Scalar]] = []
    all_metric_cols: Set[str] = set()

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    # ---- RAW MODEL ROW ----
    ckpt_path = os.path.join(root_dir, "raw_model")  
    row: Dict[str, Scalar] = {"checkpoint": f"{model_name}"}
    for dataset_name in sorted(os.listdir(ckpt_path)):
        dataset_path = os.path.join(ckpt_path, dataset_name)
        dataset_name = dataset_name.lower()
        if not os.path.isdir(dataset_path):
            continue

        json_path = os.path.join(dataset_path, "raw_results_train_all.json")
        if not os.path.isfile(json_path):
            continue

        try:
            metrics = load_metrics_from_json(json_path)
        except Exception as e:
            print(f"[WARN] Failed to read {json_path}: {e}")
            continue

        for metric_name, metric_value in metrics.items():
            if "hamming_accuracy" in metric_name:
                col_name = f"{dataset_name}_hamming_acc"
            elif "accuracy" in metric_name:
                col_name = f"{dataset_name}_acc"
            elif "macro_f1" in metric_name or "f1_macro" in metric_name:
                col_name = f"{dataset_name}_f1"
            elif "f1" in metric_name:
                col_name = f"{dataset_name}_f1"
            elif "precision" in metric_name:
                col_name = f"{dataset_name}_precision"
            elif "recall" in metric_name:
                col_name = f"{dataset_name}_recall"
            elif "exact_match_accuracy" in metric_name:
                col_name = f"{dataset_name}_EM"
            else:
                continue

            # IMPORTANT CHANGE: store raw numeric values, not formatted strings
            if col_name not in row:
                row[col_name] = metric_value
                all_metric_cols.add(col_name)

    rows.append(row)

    # ---- CHECKPOINT ROWS ----
    root_dir = os.path.join(root_dir, run)
    # Each subdirectory in root_dir is treated as a checkpoint
    training_step = [int(dir.split("-")[-1]) for dir in os.listdir(root_dir)]
    for ckpt_name in [f"checkpoint-{str(dir)}" for dir in sorted(training_step)]:
        ckpt_path = os.path.join(root_dir, ckpt_name)
        if not os.path.isdir(ckpt_path) or "checkpoint" not in ckpt_name:
            continue
        
        if best_checkpoint and ckpt_name == best_checkpoint:
            row: Dict[str, Scalar] = {"checkpoint": ckpt_name+"(best)"}
        else:
            row: Dict[str, Scalar] = {"checkpoint": ckpt_name}

        # Each subdirectory here is treated as a dataset
        for dataset_name in sorted(os.listdir(ckpt_path)):
            dataset_path = os.path.join(ckpt_path, dataset_name)
            dataset_name = dataset_name.lower()
            if not os.path.isdir(dataset_path):
                continue

            json_path = os.path.join(dataset_path, "all_cases.json")
            if not os.path.isfile(json_path):
                json_path = os.path.join(dataset_path, "all_casses.json")
                if not os.path.isfile(json_path):
                    continue
            
            try:
                metrics = load_metrics_from_json(json_path)
            except Exception as e:
                print(f"[WARN] Failed to read {json_path}: {e}")
                continue

            for metric_name, metric_value in metrics.items():
                if "hamming_accuracy" in metric_name:
                    col_name = f"{dataset_name}_hamming_acc"
                elif "accuracy" in metric_name:
                    col_name = f"{dataset_name}_acc"
                elif "macro_f1" in metric_name or "f1_macro" in metric_name:
                    col_name = f"{dataset_name}_f1"
                elif "f1" in metric_name:
                    col_name = f"{dataset_name}_f1"
                elif "precision" in metric_name:
                    col_name = f"{dataset_name}_precision"
                elif "recall" in metric_name:
                    col_name = f"{dataset_name}_recall"
                elif "exact_match_accuracy" in metric_name:
                    col_name = f"{dataset_name}_EM"
                else:
                    continue

                # IMPORTANT CHANGE: store raw numeric values, not formatted strings
                if col_name not in row:
                    row[col_name] = metric_value
                    all_metric_cols.add(col_name)

        rows.append(row)

    # Order columns: checkpoint first, then all metrics sorted
    ordered_cols = ["checkpoint"] + sorted(all_metric_cols)
    return rows, ordered_cols


def format_value(v: Scalar) -> str:
    """Format a scalar value nicely for the ASCII table."""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_ascii_table(rows: List[Dict[str, Scalar]], columns: List[str]) -> None:
    """Print a simple ASCII table to stdout."""
    if not rows:
        print("No data found.")
        return

    # Compute column widths
    col_widths = {}
    for col in columns:
        max_len = len(col)
        for row in rows:
            val = format_value(row.get(col, ""))
            max_len = max(max_len, len(val))
        col_widths[col] = max_len

    # Build separator line
    sep = "+".join("-" * (col_widths[col] + 2) for col in columns)
    sep = f"+{sep}+"

    # Header
    header_cells = []
    for col in columns:
        header_cells.append(f" {col.ljust(col_widths[col])} ")
    header_line = "|" + "|".join(header_cells) + "|"

    print(sep)
    print(header_line)
    print(sep)

    # Rows
    for row in rows:
        cells = []
        for col in columns:
            val = format_value(row.get(col, ""))
            cells.append(f" {val.ljust(col_widths[col])} ")
        line = "|" + "|".join(cells) + "|"
        print(line)
    print(sep)


def write_csv(rows: List[Dict[str, Scalar]], columns: List[str], out_path: str) -> None:
    """Write data to a CSV file."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            # Ensure all keys exist
            full_row = {col: row.get(col, "") for col in columns}
            writer.writerow(full_row)
    print(f"CSV written to: {out_path}")


def clean_sheet_name(name):
    invalid = ['\\', '/', '*', '?', ':', '[', ']']
    for c in invalid:
        name = name.replace(c, '_')
    return name[:31]  # Excel also limits sheet names to 31 chars


def write_excel(
    rows: List[Dict[str, Scalar]],
    columns: List[str],
    out_path: str,
    sheet_name: str = "Sheet1",
    best_checkpoint: str = None,
    model_name: str = "qwen2.5-3B"
) -> None:
    """Write data to an Excel file (one sheet named by sheet_name).

    - Cells where checkpoint metric > raw_model metric are colored yellow.
    - Two extra columns are appended:
        better_metrics_than_raw
        better_datasets_than_raw
    """
    # clean sheet name:
    sheet_name = clean_sheet_name(sheet_name)

    # Build initial DataFrame from rows/columns
    table = [{col: row.get(col, "") for col in columns} for row in rows]
    df = pd.DataFrame(table, columns=columns)

    # All metric columns (everything except "checkpoint")
    metric_cols = [c for c in df.columns if c != "checkpoint"]

    # Convert metric columns to numeric where possible (NaNs if not numeric)
    if metric_cols:
        df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce")
        df[metric_cols] = np.round(df[metric_cols], 4)

    # --- Compute better-than-raw masks and counts ---
    has_baseline = False
    better_mask = None  # DataFrame[bool], shape (n_rows, n_metric_cols)

    if "checkpoint" in df.columns and (df["checkpoint"] == f"{model_name}").any() and metric_cols:
        has_baseline = True
        raw_idx = df.index[df["checkpoint"] == f"{model_name}"][0]
        raw_values = df.loc[raw_idx, metric_cols]

        # True where checkpoint metric > raw metric (broadcasted across rows)
        better_mask = df[metric_cols].gt(raw_values)
        better_mask = better_mask.fillna(False)

        # 1) Number of yellow cells (better metrics) per row
        df["better_metrics_than_raw"] = better_mask.sum(axis=1)

        # 2) Number of datasets with at least one improved metric per row
        dataset_to_cols = {}
        for col in metric_cols:
            dataset_name = col.split("_", 1)[0]
            dataset_to_cols.setdefault(dataset_name, []).append(col)

        better_datasets_counts: List[int] = []
        for idx, row_bool in better_mask.iterrows():
            count = 0
            for ds, ds_cols in dataset_to_cols.items():
                # If any metric for this dataset is better than raw
                if row_bool[ds_cols].any():
                    count += 1
            better_datasets_counts.append(count)
        df["better_datasets_than_raw"] = better_datasets_counts
    else:
        # Fallback if no raw_model row is found
        df["better_metrics_than_raw"] = 0
        df["better_datasets_than_raw"] = 0

    # Final column order: original columns + the two summary columns
    final_columns = columns + ["better_metrics_than_raw", "better_datasets_than_raw"]

    # --- Write to Excel and color cells ---
    mode = "a" if os.path.exists(out_path) else "w"

    if mode == "a":
        with pd.ExcelWriter(out_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name, columns=final_columns)

            # Apply yellow highlighting where metric > raw
            if has_baseline and better_mask is not None and metric_cols:
                ws = writer.sheets[sheet_name]
                green_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")

                # Map column name -> Excel column index (1-based)
                col_index_map = {col: idx + 1 for idx, col in enumerate(final_columns)}

                for r_idx, row_label in enumerate(df.index):
                    # Excel rows are 1-based, +1 for header row
                    excel_row = r_idx + 2
                    for col in metric_cols:
                        if bool(better_mask.loc[row_label, col]):
                            excel_col = col_index_map[col]
                            ws.cell(row=excel_row, column=excel_col).fill = green_fill
            
            # if best_checkpoint:
            #     ws = writer.sheets[sheet_name]
            #     checkpoint_col = final_columns.index("checkpoint") + 1
            #     for r_idx, row_label in enumerate(df.index):
            #         if df.loc[row_label, "checkpoint"] == best_checkpoint:
            #             excel_row = r_idx + 2  
            #             cell = ws.cell(row=excel_row, column=checkpoint_col)
            #             cell.font = Font(color="008000")  
    else:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name, columns=final_columns)

            if has_baseline and better_mask is not None and metric_cols:
                ws = writer.sheets[sheet_name]
                green_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
                col_index_map = {col: idx + 1 for idx, col in enumerate(final_columns)}

                for r_idx, row_label in enumerate(df.index):
                    excel_row = r_idx + 2
                    for col in metric_cols:
                        if bool(better_mask.loc[row_label, col]):
                            excel_col = col_index_map[col]
                            ws.cell(row=excel_row, column=excel_col).fill = green_fill
            
            if best_checkpoint:
                ws = writer.sheets[sheet_name]
                checkpoint_col = final_columns.index("checkpoint") + 1
                for r_idx, row_label in enumerate(df.index):
                    if df.loc[row_label, "checkpoint"] == best_checkpoint:
                        excel_row = r_idx + 2
                        cell = ws.cell(row=excel_row, column=checkpoint_col)
                        cell.font = Font(color="008000")

    print(f"Excel written to: {out_path} (sheet: {sheet_name})")


def main():
    parser = argparse.ArgumentParser(
        description="Create a metrics table from checkpoint JSON files."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./GRPO/Evaluation/",
        help="Root directory containing checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="./GRPO/Evaluation//metrics_summary.xlsx",
        help="Output Excel file path (default: metrics_summary.xlsx)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="dt11.18.17:40_e20_unsloth_Qwen2.5_3B_Instruct_unsloth_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b16",
        help="Name of the Excel sheet (subsheet) to write results into.",
    )
    parser.add_argument(
        "--best_checkpoint",
        type=str,
        default=None,
        help="Name of checkpoint whose name in the first column will be colored green.",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="qwen2.5-3B",
        help="Name of the base model we trained on",
    )

    args = parser.parse_args()

    if args.best_checkpoint is None:
        BASE_RESULTS_DIR="/home/moein_salimi/users/Nima/AbductiveReasoning/GRPO/results"
        TRAINING_DIR=f"{BASE_RESULTS_DIR}/Training_{args.run}"
        FINAL_DIR=f"{BASE_RESULTS_DIR}/{args.run}"
        if os.path.isdir(TRAINING_DIR):
            TRAINING_BASE = TRAINING_DIR
        elif os.path.isdir(FINAL_DIR):
            TRAINING_BASE = FINAL_DIR
        else:
            print(f"ERROR: Could not find checkpoint directory.")
            print(f"Tried:")
            print(f"  {TRAINING_DIR}")
            print(f"  {FINAL_DIR}")
            return 

        best_path, _ = find_best_checkpoint(TRAINING_BASE)
        args.best_checkpoint = os.path.basename(best_path) if best_path else None

    rows, columns = collect_all_rows(args.root, args.run, args.best_checkpoint, args.base_model_name)
    # print_ascii_table(rows, columns)
    # write_csv(rows, columns, args.out_csv)
    write_excel(rows, columns, args.out_csv, sheet_name=args.run, best_checkpoint=args.best_checkpoint, model_name=args.base_model_name) 


if __name__ == "__main__":
    main()
