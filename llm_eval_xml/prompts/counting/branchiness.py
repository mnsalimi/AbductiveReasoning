"""
prompts/counting/branchiness.py
--------------------------------
Prompt for the Branchiness counting metric.

The LLM returns a list of concrete examples of branching moments in the
reasoning trace – NOT a count.
"""

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

## Dataset-specific notes

If the dataset is **MedQA**: the answer options (A/B/C/D/E) are part of the question.
Do NOT count restating them as branching.  Only count genuine differential exploration.

If the dataset is **ART**: Hypothesis 1 and Hypothesis 2 are given explicitly.
Do NOT count choosing between them as a branch; only count internal exploration.
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Branchiness and extract concrete examples.

<reasoning_trace>
{text}
</reasoning_trace>
"""
