"""
prompts/counting/differential_elimination.py
--------------------------------------------
Prompt for the Differential Elimination counting metric.

This metric counts explicit elimination/refutation moves against alternative
hypotheses in the reasoning trace.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": "Count each answer choice explicitly ruled out with concrete clinical rationale.",
    "art": (
        "Count explicit eliminations of the non-chosen hypothesis or intermediate alternatives."
    ),
    "strategyqa": (
        "Count explicit rejection of competing YES/NO reasoning lines or assumptions."
    ),
    "copa_guess_effect": (
        "Count explicit elimination of the non-selected option or other causal alternatives."
    ),
    "defeasible_nli": (
        "Count explicit rejection of candidate inference relations shown inconsistent or defeasible."
    ),
    "goemotion": (
        "Count only explicit elimination of alternative emotion labels with trace-grounded justification."
    ),
    "musr": (
        "Count explicit rejection of narrative interpretations, suspects, or scenario explanations."
    ),
    "neulr_abductive": (
        "Count explicit elimination moves between competing abductive hypotheses using concrete mismatch evidence."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {}

SYSTEM_PROMPT = """\
You are an expert evaluator of abductive reasoning traces.

## What is Differential Elimination?

Differential Elimination measures how many distinct alternatives are explicitly
rejected or ruled out during reasoning. Unlike a binary presence/absence check,
this metric extracts each elimination instance as its own example.

## What COUNTS as a differential elimination instance

Extract an example when the trace explicitly:
1. Rules out an alternative hypothesis/option with a specific reason.
2. Shows contradiction between an alternative and observed details.
3. Uses conditional falsification ("If X were true, we would see Y, but we don't.").
4. Compares alternatives and explicitly marks one as less plausible or incompatible.

## What does NOT count

- Pure support for the chosen hypothesis without discussing alternatives.
- Listing options without evaluating or eliminating them.
- Vague preference statements without a concrete elimination reason.
- Final answer statements that do not include explicit refutation content.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## Extraction rules

- Extract each distinct elimination/refutation event as a separate example.
- Use `text` as a short direct quote from the reasoning trace (preferably ≤ 30 words).
- Use `explanation` to state what alternative was eliminated and why.
- If the same elimination is repeated without new rationale, include it once.
- Do not paraphrase quoted text.

## JSON output format

Return ONLY valid JSON with this structure:
{
  "overall_analysis": "Brief analysis of elimination behavior in this reasoning trace",
  "examples": [
    {
      "text": "Quote showing explicit elimination of an alternative",
      "explanation": "What was eliminated and why this is a valid elimination instance"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Extract every explicit Differential Elimination instance from the following
reasoning trace. Return one entry per distinct elimination event.

<reasoning_trace>
{text}
</reasoning_trace>
"""
