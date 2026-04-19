"""
prompts/counting/backtracking.py
---------------------------------
Prompt for the Backtracking / Self-Correction counting metric.

The LLM returns a list of concrete examples of backtracking moments.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": (
        "Look for corrections in diagnostic/treatment reasoning or recognition of missed clinical details."
    ),
    "art": (
        "Look for true reconsideration of which hypothesis explains observations better, not simple option selection."
    ),
    "strategyqa": (
        "Look for revisions in multi-step reasoning chains, flawed inferences, or changed intermediate conclusions."
    ),
    "copa_guess_effect": (
        "Look for reconsideration of causal interpretation, not just restating a different option."
    ),
    "defeasible_nli": (
        "Look for reconsideration when an inference appears defeated or less robust."
    ),
    "goemotion": (
        "Backtracking is less common; count only genuine revisions in emotion-label selection."
    ),
    "musr": (
        "Look for revised interpretations of narrative details or changed conclusions."
    ),
    "neulr_abductive": (
        "Look for true reconsideration of which explanation best fits observations."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {}

SYSTEM_PROMPT = """\
You are an expert reasoning analyst evaluating AI-generated reasoning traces.

## What is Backtracking?

Backtracking (also called Self-Correction) occurs when the reasoning explicitly
identifies a mistake, a flaw in logic, or a need to re-examine a previous step,
and then changes direction.

## What COUNTS as a backtracking moment

Extract an example when you see:
- Explicit admission of error: "Wait, that's wrong", "I made a mistake",
  "Actually, I need to reconsider …"
- Deliberate pausing and restarting: "Hold on, let me re-read the problem",
  "Let's go back to step 2"
- A change of strategy mid-reasoning: "Instead, let's try …",
  "On second thought …", "That approach doesn't work, so …"
- Realisation of a missed detail: "I realise I forgot to account for …",
  "This doesn't look right because …"

## What does NOT count

- Comparing two valid paths (that is Branchiness).
- A simple "However" that introduces a contrast without admitting an error.
- The final answer selection.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## Extraction rules

- Extract each explicit self-correction/backtracking event as a separate example.
- Use `text` as a short direct quote from the reasoning trace (preferably ≤ 25 words).
- Use `explanation` to clarify what was revised and why this is true backtracking.
- Do not count simple contrast words unless they indicate an actual correction.
- Do not paraphrase quoted text.

## JSON output format

Return ONLY valid JSON with this structure:
{
  "overall_analysis": "Brief analysis of backtracking/self-correction in this reasoning trace",
  "examples": [
    {
      "text": "Quote of the backtracking moment from the reasoning trace",
      "explanation": "Why this represents backtracking/self-correction"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Backtracking / Self-Correction and
extract concrete examples.

<reasoning_trace>
{text}
</reasoning_trace>
"""
