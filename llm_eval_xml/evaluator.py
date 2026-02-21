"""
evaluator.py
------------
Orchestrates per-item evaluation across all active metrics.

``process_single_item`` is designed to be called from a ThreadPoolExecutor –
it is the only place that calls into the metrics and assembles results.
"""

from __future__ import annotations

import datetime
from typing import Any

import config
from data_loader import get_labels, extract_reasoning
from metrics.base import MetricResult
from metrics.registry import get_active_metrics


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_single_item(
    task: tuple,
) -> dict[str, Any]:
    """
    Evaluate all active metrics for a single reasoning item.

    Parameters
    ----------
    task : tuple
        ``(dataset_name, checkpoint_num, problem_id, reasoning_text, raw_item)``

    Returns
    -------
    dict
        A flat result dict ready to be assembled into DataFrames / logs.
    """
    dataset_name, checkpoint, pid, reasoning, item = task
    print(f"  [Processing] {dataset_name}  ckpt={checkpoint}  pid={pid}")

    run_id = config.RUN_ID

    # ── Item metadata ──────────────────────────────────────────────────────
    true_label, pred_label = get_labels(item) if isinstance(item, dict) else (None, None)
    is_correct = str(true_label) == str(pred_label) if (true_label is not None and pred_label is not None) else None
    status = ("Correct" if is_correct else "Incorrect") if is_correct is not None else "Unknown"

    # ART-specific fields
    obs1 = item.get("observation_1", "") if isinstance(item, dict) else ""
    obs2 = item.get("observation_2", "") if isinstance(item, dict) else ""
    hyp1 = item.get("hypothesis_1", "") if isinstance(item, dict) else ""
    hyp2 = item.get("hypothesis_2", "") if isinstance(item, dict) else ""
    question = item.get("question", "") if isinstance(item, dict) else ""

    # MedQA / fallback problem text
    problem_text = question or f"{obs1} | {obs2}".strip(" |")

    true_text = item.get(f"hypothesis_{true_label}", "") if true_label in (1, 2) and isinstance(item, dict) else ""
    pred_text = item.get(f"hypothesis_{pred_label}", "") if pred_label in (1, 2) and isinstance(item, dict) else ""

    # Word count (no regex – just whitespace split)
    word_count = len(reasoning.split()) if reasoning else 0

    # ── Metric evaluation ─────────────────────────────────────────────────
    active_metrics = get_active_metrics(config.DISABLED_METRICS)
    metric_results: dict[str, MetricResult] = {}

    for metric_name, metric in active_metrics.items():
        result = metric.evaluate(
            reasoning,
            dataset=dataset_name,
            problem_id=str(pid),
            checkpoint=str(checkpoint),
            run_id=run_id,
        )
        metric_results[metric_name] = result
        if result.error:
            print(f"    [FAIL] {metric_name}: {result.error[:100]}")
        else:
            print(f"    [OK]   {metric_name}")

    print(
        f"  [Done]       {dataset_name}  ckpt={checkpoint}  pid={pid}"
    )

    # ── Assemble output ───────────────────────────────────────────────────
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "run_id": run_id,
        "dataset": dataset_name,
        "checkpoint": checkpoint,
        "pid": pid,
        "status": status,
        "true_label": true_label,
        "pred_label": pred_label,
        "problem_text": problem_text,
        "question": question,
        "obs1": obs1,
        "obs2": obs2,
        "hyp1": hyp1,
        "hyp2": hyp2,
        "true_text": true_text,
        "pred_text": pred_text,
        "reasoning": reasoning,
        "word_count": word_count,
        # Per-metric structured results
        "metrics": {
            name: {
                "type": active_metrics[name].metric_type,
                "detected": r.detected,
                "reasoning": r.reasoning,
                "examples": r.examples,
                "example_count": len(r.examples),
                "error": r.error,
            }
            for name, r in metric_results.items()
        },
    }
