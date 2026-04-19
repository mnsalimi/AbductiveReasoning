"""
prompts/binary/differential_evaluation.py
---------------------------------------
Prompt for the Differential Evaluation (IBE) binary metric.

Differential Evaluation: The active refutation of alternative hypotheses.
The model explains why other plausible answers are incorrect or less likely
given the specific context (Inference to the Best Explanation).

The LLM answers yes/no and explains why.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": (
        "Look for explicit comparison showing why the selected option outperforms alternatives."
    ),
    "art": (
        "Check whether reasoning explains why chosen hypothesis is better than rejected hypothesis."
    ),
    "strategyqa": (
        "Look for explicit rejection of one YES/NO line with reasons, not only support for the final choice."
    ),
    "copa_guess_effect": (
        "Check whether chosen option is explicitly favored over alternative with concrete reasoning."
    ),
    "defeasible_nli": (
        "Look for comparison among possible premise-hypothesis relations, with explicit refutation where applicable."
    ),
    "goemotion": (
        "If present, look for reasoning that rejects alternative emotion labels with justification."
    ),
    "musr": (
        "Look for reasoning that compares plausible narrative explanations and rejects weaker alternatives."
    ),
    "neulr_abductive": (
        "Check explicit comparative reasoning between the two hypotheses."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {}

SYSTEM_PROMPT = """\
You are an expert evaluator of abductive reasoning traces.

## What is Differential Evaluation (IBE - Inference to the Best Explanation)?

Differential Evaluation measures whether the reasoning trace actively compares
and refutes alternative hypotheses, not just describes why the chosen hypothesis
is good. The model should explicitly explain why the OTHER options are wrong or
less likely, rather than just presenting the chosen option in isolation.

## Examples of differential evaluation (detected = true)

- The reasoning explicitly compares options and states why alternatives fail:
  "Option A is incorrect because it assumes X which contradicts the observation
   that Y, whereas Option B explains this."
- The model eliminates alternatives through logical deduction:
  "We can rule out Option C because..."
- The response contrasts the chosen hypothesis with alternatives:
  "Unlike Option 1 which only explains the main event, Option 2 also accounts
   for the timing detail."
- The model shows active hypothesis elimination:
  "If Option A were true, we would expect to see X, but we don't."

## Examples that do NOT show differential evaluation (detected = false)

- Only describes why the chosen hypothesis is good without explaining why others fail:
  "Option B is correct because it explains the observation."
- Ignores or doesn't mention alternatives at all.
- Merely states the answer without comparative analysis.
- Treats all options equally without showing active elimination of alternatives.

## Important

Look for active, explicit refutation or elimination of alternatives. Simply
considering multiple options is not enough - there must be active reasoning
about why alternatives are less suitable or incorrect.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## Extraction rules

- Set `detected` to true only when alternatives are explicitly refuted, ruled out, or shown less plausible.
- Set `detected` to false if the trace only supports the chosen option without comparative elimination.
- Keep `reasoning` specific to comparative/refutational evidence in the trace.
- Set `evidence` to a direct quote that best demonstrates elimination/refutation.
- Use empty `evidence` only when `detected` is false and no quote supports detection.

## JSON output format

Return ONLY valid JSON with this structure:
{
  "detected": true,
  "reasoning": "Step-by-step explanation of why differential evaluation is present/absent",
  "evidence": "Direct quote from the text supporting the decision (leave empty if detected is false)"
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Differential Evaluation (active
refutation of alternative hypotheses).

<reasoning_trace>
{text}
</reasoning_trace>
"""
