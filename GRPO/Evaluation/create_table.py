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
  - Writes a CSV file with columns:
        checkpoint, dataset_metric1, dataset_metric2, ...
"""

import os
import json
import argparse
import csv
from typing import Dict, Any, List, Tuple, Set, Union
import pandas as pd


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


def collect_all_rows(root_dir: str) -> Tuple[List[Dict[str, Scalar]], List[str]]:
    """Walk the checkpoints directory and collect rows + column names.

    Returns:
      rows: list of dicts, each representing one checkpoint
      columns: ordered list of column names (including 'checkpoint')
    """
    rows: List[Dict[str, Scalar]] = []
    all_metric_cols: Set[str] = set()

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    ckpt_path = os.path.join(root_dir, "raw_model")  
    row: Dict[str, Scalar] = {"checkpoint": "raw_model"}
    for dataset_name in sorted(os.listdir(ckpt_path)):
        dataset_path = os.path.join(ckpt_path, dataset_name)
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
                col_name = f"{dataset_name}_hamming_accuracy"
            elif "accuracy" in metric_name :
                col_name = f"{dataset_name}_accuracy"
            elif "macro_f1" in metric_name or "f1_macro" in metric_name:
                col_name = f"{dataset_name}_f1"
            elif "f1" in metric_name :
                col_name = f"{dataset_name}_f1"
            elif "precision" in metric_name :
                col_name = f"{dataset_name}_precision"
            elif "recall" in metric_name :
                col_name = f"{dataset_name}_recall"
            elif "exact_match_accuracy" in metric_name :
                col_name = f"{dataset_name}_EM"
            else:
                continue
            if col_name not in row:    
                row[col_name] = format_value(metric_value)
                all_metric_cols.add(col_name)

    rows.append(row)

    # Each subdirectory in root_dir is treated as a checkpoint
    for ckpt_name in sorted(os.listdir(root_dir)):
        ckpt_path = os.path.join(root_dir, ckpt_name)
        if not os.path.isdir(ckpt_path) or "checkpoint" not in ckpt_name:
            continue

        row: Dict[str, Scalar] = {"checkpoint": ckpt_name}

        # Each subdirectory here is treated as a dataset
        for dataset_name in sorted(os.listdir(ckpt_path)):
            dataset_path = os.path.join(ckpt_path, dataset_name)
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
                    col_name = f"{dataset_name}_hamming_accuracy"
                elif "accuracy" in metric_name :
                    col_name = f"{dataset_name}_accuracy"
                elif "macro_f1" in metric_name or "f1_macro" in metric_name:
                    col_name = f"{dataset_name}_f1"
                elif "f1" in metric_name :
                    col_name = f"{dataset_name}_f1"
                elif "precision" in metric_name :
                    col_name = f"{dataset_name}_precision"
                elif "recall" in metric_name :
                    col_name = f"{dataset_name}_recall"
                elif "exact_match_accuracy" in metric_name :
                    col_name = f"{dataset_name}_EM"
                else:
                    continue
                if col_name not in row:  
                    row[col_name] = format_value(metric_value)
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

def write_excel(rows: List[Dict[str, Scalar]], columns: List[str], out_path: str) -> None:
    """Write data to an Excel file."""
    table = [{col: row.get(col, "") for col in columns} for row in rows]
    df = pd.DataFrame(table, columns=columns)
    df.to_excel(out_path, index=False)
    print(f"Excel written to: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a metrics table from checkpoint JSON files."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="checkpoints",
        help="Root directory containing checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="metrics_summary.csv",
        help="Output CSV file path (default: metrics_summary.csv)",
    )
    args = parser.parse_args()

    rows, columns = collect_all_rows(args.root)
    print_ascii_table(rows, columns)
    # write_csv(rows, columns, args.out_csv)
    write_excel(rows, columns, args.out_csv)


if __name__ == "__main__":
    main()
