"""
prompts/counting/backtracking.py
---------------------------------
Prompt for the Backtracking / Self-Correction counting metric.

The LLM returns a list of concrete examples of backtracking moments.
"""

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
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Backtracking / Self-Correction and
extract concrete examples.

<reasoning_trace>
{text}
</reasoning_trace>
"""
