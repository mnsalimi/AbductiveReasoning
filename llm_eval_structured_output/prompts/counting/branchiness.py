"""
prompts/counting/branchiness.py
--------------------------------
Prompt for the Branchiness counting metric.

The LLM returns a list of concrete examples of branching moments in the
reasoning trace – NOT a count.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": (
        "Answer options are part of the question. Do not count restating options as branching; "
        "count only genuine differential exploration (e.g., diagnoses or treatment paths)."
    ),
    "art": (
        "Hypothesis 1 and Hypothesis 2 are given explicitly. Do not count simple option selection; "
        "count only internal exploration within reasoning about hypotheses."
    ),
    "strategyqa": (
        "Look for exploration of multiple reasoning paths or alternative inference chains to reach the answer."
    ),
    "copa_guess_effect": (
        "Do not count simple choice selection; count exploration within cause-effect reasoning."
    ),
    "defeasible_nli": (
        "Look for exploration of stronger vs weaker inferences and reasoning about potentially defeasible conclusions."
    ),
    "goemotion": (
        "This is primarily classification; count only if the model meaningfully explores multiple emotion labels."
    ),
    "musr": (
        "Look for exploration of different narrative interpretations or scenario solutions."
    ),
    "neulr_abductive": (
        "Do not count superficial option comparison; count genuine exploration of why one explanation is better."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {}

SYSTEM_PROMPT = """\
You are an expert reasoning analyst evaluating AI-generated reasoning traces.

## What is Branchiness?

Branchiness measures whether the reasoning **genuinely explores multiple distinct
possibilities** (alternative hypotheses, approaches, or cases) rather than
following a single linear path.

## What COUNTS as a branching moment

Extract an example when you see:
1. Building and comparing multiple hypotheses with their implications/evidence
   ("If diagnosis X we'd expect F… If diagnosis Y we'd expect G…").
2. Trying two or more different solution methods and comparing outcomes.
3. Conditional planning that develops multiple flows
   ("If result is positive, then … If negative, then …") beyond a trivial mention.

## What does NOT count

- The final answer selection or conclusion.
- A brief mention of an alternative followed by immediate rejection with no exploration.
- Simple step-by-step narration (First / Next / Then).
- Listing the given answer options without exploring them.
- Purely negative reasoning ("Option A is wrong because …") — that is captured by
  the neg_constraint metric.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## Extraction rules

- Extract each distinct branching moment as a separate example.
- Use `text` as a short direct quote from the reasoning trace (preferably ≤ 25 words).
- Use `explanation` to state why that quote is a genuine branch and not linear narration.
- If the same branch is repeated with no new reasoning content, extract it once.
- Do not paraphrase quoted text.

## JSON output format

Return ONLY valid JSON with this structure:
{
  "overall_analysis": "Brief analysis of branchiness in this reasoning trace",
  "examples": [
    {
      "text": "Quote of the branching moment from the reasoning trace",
      "explanation": "Why this represents branching"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Branchiness and extract concrete examples.

<reasoning_trace>
{text}
</reasoning_trace>
"""
