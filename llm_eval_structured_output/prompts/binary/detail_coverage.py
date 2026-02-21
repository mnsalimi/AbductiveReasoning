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
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Detail Coverage.

<reasoning_trace>
{text}
</reasoning_trace>
"""
