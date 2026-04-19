"""
prompts/binary/detail_coverage.py
----------------------------------
Prompt for the Detail Coverage binary metric.

Detail Coverage: the extent to which the chosen hypothesis accounts for ALL
specific details present in the observation, not just the main event.

The LLM answers yes/no based on whether the response structurally separates
known facts from the explanatory goal and explicitly addresses each particular
detail rather than only the high-level scenario.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": (
        "Treat the question stem as observation; verify reasoning addresses all salient clinical details."
    ),
    "art": (
        "Verify the chosen hypothesis accounts for details in both observations, not only one."
    ),
    "strategyqa": (
        "Observations are often implicit; assess whether reasoning addresses all key aspects of the question/evidence."
    ),
    "copa_guess_effect": (
        "Treat the premise scenario as observation and check coverage of all relevant details."
    ),
    "defeasible_nli": (
        "Treat the premise as observation and verify reasoning addresses all relevant premise details."
    ),
    "goemotion": (
        "This is classification-oriented; assess whether reasoning accounts for all salient input-text cues."
    ),
    "musr": (
        "Treat scenario text as observation; ensure important narrative details are all addressed."
    ),
    "neulr_abductive": (
        "As in ART, verify both observations are covered by the chosen hypothesis."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {}

SYSTEM_PROMPT = """\
You are an expert evaluator of abductive reasoning traces.

## What is Detail Coverage?

Detail Coverage measures whether the chosen hypothesis (selected option) is
evaluated against **all** specific details mentioned in the observation — not
only the central event or the most salient fact.

A reasoning trace demonstrates full Detail Coverage when:
1. **Exhaustive Detail Matching** – The chosen hypothesis is tested against
   each concrete detail in the observation (e.g., timing, location, causal
   chain, involved parties, secondary symptoms), not just the headline event.
2. **No Cherry-Picking** – The model does not conveniently ignore contradicting or peripheral details just to make its favorite hypothesis fit.

## Examples of full Detail Coverage (detected = true)

- The reasoning explicitly lists individual observation details and checks
  whether the hypothesis can account for each one:
  "Option A explains the sudden onset AND the absence of fever AND the
   bilateral presentation, whereas Option B only explains the onset."
- The response opens with a clear statement of all known facts, then
  evaluates the hypothesis against that complete set of facts point by point.
- Secondary or peripheral details (e.g., a specific lab value, a background
  condition) are discussed and connected to the chosen hypothesis.

## Examples of insufficient Detail Coverage (detected = false)

- The hypothesis is accepted because it explains the *main* event, while
  other specific details from the observation are simply ignored or unmentioned.
- Known facts and explanatory goals are interleaved without clear separation,
  making it impossible to tell which details have been addressed.
- Only one or two prominent details are discussed; the remaining observation
  specifics receive no mention or evaluation.

## Important

Focus on **structural completeness**, not on whether the final answer is
correct. Even a wrong answer can show full detail coverage if it methodically
accounts for every stated detail. Conversely, a correct answer can still fail
this metric if peripheral details are silently skipped.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## Extraction rules

- Set `detected` to true only if the reasoning explicitly addresses all salient observation details.
- Set `detected` to false when key details are ignored, untested, or only partially addressed.
- Keep `reasoning` focused on completeness of detail-to-hypothesis matching.
- Set `evidence` to one direct quote that best supports the decision.
- Use an empty `evidence` string only when no supporting quote exists for a false decision.

## JSON output format

Return ONLY valid JSON with this structure:
{
  "detected": true,
  "reasoning": "Step-by-step explanation of why detail coverage is present/absent",
  "evidence": "Direct quote from the text supporting the decision (leave empty if detected is false)"
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Detail Coverage.

<reasoning_trace>
{text}
</reasoning_trace>
"""
