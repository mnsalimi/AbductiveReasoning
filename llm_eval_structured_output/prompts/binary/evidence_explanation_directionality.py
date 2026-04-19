"""
prompts/binary/evidence_explanation_directionality.py
---------------------------------------------------
Prompt for the Evidence-Explanation Directionality Awareness binary metric.

Evidence-Explanation Directionality: The model demonstrates awareness that
abduction runs from evidence to explanation (not explanation to evidence,
as in prediction/deduction). The model distinguishes what is observed from
what needs to be explained.

The LLM answers yes/no and explains why.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": (
        "Check that reasoning starts from patient evidence and seeks explanation, rather than assuming an answer first."
    ),
    "art": (
        "Check that reasoning starts from observations and seeks explanatory hypotheses, not the reverse."
    ),
    "strategyqa": (
        "Check that reasoning starts from given evidence/question context and supports the conclusion from it."
    ),
    "copa_guess_effect": (
        "Check that reasoning starts from the premise observation and seeks best explanation/cause-effect account."
    ),
    "defeasible_nli": (
        "Check that reasoning starts from premise evidence and evaluates hypothesis relation from that evidence."
    ),
    "goemotion": (
        "Look for reasoning that starts from input text cues before selecting emotion labels."
    ),
    "musr": (
        "Check that reasoning starts from scenario details and builds explanations, not back-fitted support."
    ),
    "neulr_abductive": (
        "As in ART, ensure observation-to-explanation direction is explicit."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {}

SYSTEM_PROMPT = """\
You are an expert evaluator of abductive reasoning traces.

## What is Evidence-Explanation Directionality Awareness?

Abduction (inference to the best explanation) moves FROM the evidence/observations
TO the explanatory hypothesis. This is different from deduction (hypothesis → prediction)
or induction (examples → generalization).

A reasoning trace demonstrates this awareness when:
1. It explicitly separates what is OBSERVED from what needs to be EXPLAINED
2. It treats the observations as GIVEN and tries to find their CAUSE
3. It doesn't treat the hypothesis as a given and work backward to find "evidence"
4. It shows awareness that we're finding the best explanation for given facts

## Examples of correct directionality (detected = true)

- Explicitly separates known facts from explanatory goals:
  "Given these observations (X, Y), we need to explain why they occurred."
- Shows awareness of moving from observation to explanation:
  "The observation is that A happened. What could explain this?"
- Distinguishes between what's given vs what's inferred:
  "We know observation X is true. To explain it, hypothesis Y would need to be true."
- Doesn't confuse prediction with explanation:
  "If we assume Y caused X, then we would predict..." (vs assuming Y and saying "we have evidence")

## Examples of incorrect or missing directionality (detected = false)

- Confuses abduction with deduction:
  "We know hypothesis Y is true, therefore evidence X must be true."
- Treats the hypothesis as given and finds "supporting evidence":
  "Since Option A is correct, we can see evidence for it in the observation."
- Works from hypothesis to observation instead of observation to hypothesis:
  "If Option B were true, then X would be true - and X is true, so B is correct"
  (This is deduction, not abduction)
- Doesn't distinguish between what is observed vs what is inferred.

## Important

The key is whether the model treats observations as fixed and seeks explanations
for them, rather than treating hypotheses as fixed and finding "evidence" for them.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## Extraction rules

- Set `detected` to true only when the trace clearly reasons from given evidence toward explanation.
- Set `detected` to false when it assumes a hypothesis first and then back-fits evidence.
- Keep `reasoning` focused on directionality cues in the trace.
- Set `evidence` to a direct quote that best demonstrates directionality (or its absence).
- Use empty `evidence` only when `detected` is false and no positive directionality quote applies.

## JSON output format

Return ONLY valid JSON with this structure:
{
  "detected": true,
  "reasoning": "Step-by-step explanation of why evidence-explanation directionality is present/absent",
  "evidence": "Direct quote from the text supporting the decision (leave empty if detected is false)"
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Evidence-Explanation Directionality
Awareness (does the model move from evidence to explanation, not the reverse?).

<reasoning_trace>
{text}
</reasoning_trace>
"""
