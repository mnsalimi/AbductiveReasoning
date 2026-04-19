"""
prompts/binary/uncertainty_language.py
----------------------------------------
Prompt for the Uncertainty Language binary metric.

Uncertainty Language: the use of probabilistic or hedging language rather than
expressing absolute certainty in conclusions or intermediate steps.

The LLM answers yes/no and explains why.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": (
        "Focus on hedging in diagnostic/treatment reasoning and interpretation, not restatement of fixed medical facts."
    ),
    "art": (
        "Look for uncertainty while comparing hypotheses and explaining observations."
    ),
    "strategyqa": (
        "Look for uncertainty in multi-step inference and in YES/NO conclusion confidence."
    ),
    "copa_guess_effect": (
        "Look for uncertainty in causal reasoning and option comparison."
    ),
    "defeasible_nli": (
        "Look for uncertainty in entailment judgments, especially under defeasible interpretations."
    ),
    "goemotion": (
        "Look for uncertainty in choosing among emotion labels."
    ),
    "musr": (
        "Look for uncertainty in interpreting narrative details, motives, or outcomes."
    ),
    "neulr_abductive": (
        "Look for uncertainty while weighing abductive explanations."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {}

SYSTEM_PROMPT = """\
You are an expert linguistic analyst evaluating AI-generated reasoning traces.

## What is Uncertainty Language?

Uncertainty Language refers to the use of probabilistic, hedging, or tentative
language rather than absolute certainty when the model draws conclusions or
makes claims during reasoning.

## Examples of uncertainty language (detected = true)

- Probabilistic qualifiers: "probably", "likely", "possibly", "might",
  "may", "could", "seems to", "appears to"
- Hedging phrases: "I believe", "I think", "I'm not sure but",
  "this suggests", "this is consistent with"
- Degree qualifiers: "most likely", "least likely", "more probable than"
- Epistemic markers: "it is uncertain whether", "we cannot be sure",
  "the evidence is inconclusive"

## Examples that do NOT indicate uncertainty language (detected = false)

- Confident logical deductions stated without hedges:
  "The answer is X because Y and Z."
- Purely factual recall without hedging: "The boiling point of water is 100 °C."
- Disclaimer boiler-plate that is not part of the original reasoning.

## Important

Focus on the *reasoning process* itself, not the final answer line.
A single strong hedging phrase that is central to the reasoning is enough for
detected = true.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## Extraction rules

- Set `detected` to true only when the reasoning trace contains genuine hedging/probabilistic language.
- Set `detected` to false for fully certain reasoning without meaningful uncertainty markers.
- Keep `reasoning` concise and specific to the trace.
- Set `evidence` to a direct quote when `detected` is true; set `evidence` to an empty string when `detected` is false.
- Do not paraphrase the evidence quote.

## JSON output format

Return ONLY valid JSON with this structure:
{
  "detected": true,
  "reasoning": "Step-by-step explanation of why uncertainty language is present/absent",
  "evidence": "Direct quote from the text supporting the decision (leave empty if detected is false)"
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Uncertainty Language.

<reasoning_trace>
{text}
</reasoning_trace>
"""
