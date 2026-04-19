"""
Dataset-specific prompt note rendering.
"""

from __future__ import annotations

import importlib
import re

CANONICAL_DATASET_ALIASES: dict[str, str] = {
    "unknown": "unknown",
    "art": "art",
    "medqa": "medqa",
    "strategyqa": "strategyqa",
    "copa": "copa_guess_effect",
    "copa_guess_effect": "copa_guess_effect",
    "defeasible_nli": "defeasible_nli",
    "goemotion": "goemotion",
    "musr_murder": "musr",
    "musr_object": "musr",
    "musr_team": "musr",
    "musr": "musr",
    "neulr_abductive": "neulr_abductive",
}

DEFAULT_DATASET_NOTE = (
    "Apply this metric strictly to the current dataset using only evidence from the reasoning trace."
)
DEFAULT_FEW_SHOT_BLOCK = ""

METRIC_PROMPT_MODULES: dict[str, str] = {
    "branchiness": "prompts.counting.branchiness",
    "backtracking": "prompts.counting.backtracking",
    "uncertainty_markers": "prompts.counting.uncertainty_markers",
    "prior": "prompts.counting.prior",
    "differential_elimination": "prompts.counting.differential_elimination",
    "uncertainty_language": "prompts.binary.uncertainty_language",
    "detail_coverage": "prompts.binary.detail_coverage",
    "differential_evaluation": "prompts.binary.differential_evaluation",
    "evidence_explanation_directionality": "prompts.binary.evidence_explanation_directionality",
    "observation_coverage": "prompts.coverage.observation_coverage",
    "rationale_graph": "prompts.graph_structure.rationale_graph",
}


def _normalize_dataset_name(dataset: str | None) -> str:
    if dataset is None:
        return "unknown"
    key = re.sub(r"[^a-z0-9_]+", "_", dataset.strip().lower()).strip("_")
    if not key:
        return "unknown"
    return CANONICAL_DATASET_ALIASES.get(key, key)


def _load_metric_prompt_maps(metric_name: str) -> tuple[dict[str, str], dict[str, str]]:
    module_path = METRIC_PROMPT_MODULES.get(metric_name)
    if not module_path:
        return {}, {}

    module = importlib.import_module(module_path)
    notes = getattr(module, "DATASET_SPECIFIC_NOTES", {})
    few_shots = getattr(module, "DATASET_FEW_SHOT_EXAMPLES", {})

    notes_map = notes if isinstance(notes, dict) else {}
    few_shots_map = few_shots if isinstance(few_shots, dict) else {}
    return notes_map, few_shots_map


def _render_dataset_note(metric_name: str, dataset: str | None) -> str:
    notes_map, _ = _load_metric_prompt_maps(metric_name)
    ds_key = _normalize_dataset_name(dataset)
    return notes_map.get(ds_key, DEFAULT_DATASET_NOTE)


def _render_dataset_few_shot(metric_name: str, dataset: str | None) -> str:
    _, few_shots_map = _load_metric_prompt_maps(metric_name)
    ds_key = _normalize_dataset_name(dataset)
    return few_shots_map.get(ds_key, DEFAULT_FEW_SHOT_BLOCK)


def render_system_prompt(system_prompt: str, metric_name: str, dataset: str | None) -> str:
    prompt = system_prompt
    dataset_note = _render_dataset_note(metric_name, dataset)
    few_shot = _render_dataset_few_shot(metric_name, dataset)
    has_note_placeholder = "{dataset_specific_note}" in prompt
    has_few_shot_placeholder = "{dataset_few_shot_examples}" in prompt
    prompt = prompt.replace("{dataset_specific_note}", dataset_note)
    prompt = prompt.replace("{dataset_few_shot_examples}", few_shot)

    if not has_note_placeholder:
        prompt = (
            f"{prompt}\n\n## Dataset-specific note (current dataset only)\n\n{dataset_note}"
        )
        if few_shot:
            prompt = f"{prompt}\n\n## Few-shot demonstrations\n\n{few_shot}"
    elif not has_few_shot_placeholder and few_shot:
        prompt = f"{prompt}\n\n## Few-shot demonstrations\n\n{few_shot}"
    return prompt
