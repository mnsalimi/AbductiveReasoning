"""
prompts/counting/self_verification.py
--------------------------------------
Prompt for the Self-Verification counting metric.

The LLM returns a list of concrete examples of verification moments.
"""

SYSTEM_PROMPT = """\
You are an expert reasoning analyst evaluating AI-generated reasoning traces.

## What is Self-Verification?

Self-Verification occurs when the reasoning explicitly checks, confirms, or
validates a previous step, calculation, or conclusion.

## What COUNTS as a self-verification moment

Extract an example when you see:
- Explicit verification verbs: "Verify", "Confirm", "Double-check",
  "Cross-check", "Validate", "Inspect", "Test"
- Cautionary intent-to-verify phrases: "To be sure …", "Let me make sure …",
  "Ensure that …", "Let's see if …"
- Logical consistency checks: "Does this match …?", "Is this consistent with …?",
  "This agrees with …", "This satisfies …"
- Calculation re-checks: "Recalculate", "Plug in", "Substitute back",
  "Sanity check"

## What does NOT count

- Simple forward reasoning steps that happen to use "check" as a casual verb.
- Mentioning a requirement without actually verifying it.
- The final answer selection.
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Self-Verification and extract
concrete examples.

<reasoning_trace>
{text}
</reasoning_trace>
"""
