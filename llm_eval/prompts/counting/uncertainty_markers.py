"""
prompts/counting/uncertainty_markers.py
----------------------------------------
Prompt for the Uncertainty Markers counting metric.

Unlike the binary `uncertainty_language` metric (which answers yes/no on whether
the reasoning hedges at all), this metric extracts *every individual probabilistic
word or phrase* as a separate example.  The count of examples is then used as a
quantitative measure of how densely hedged the reasoning trace is.
"""

SYSTEM_PROMPT = """\
You are an expert linguistic analyst evaluating AI-generated reasoning traces.

## What is an Uncertainty Marker?

An uncertainty marker is a **specific word or phrase** that signals the model is
expressing a degree of belief, possibility, or probability rather than stating
something as an established fact.  Your job is to locate every individual marker
that appears in the reasoning trace.

## Categories of uncertainty markers to extract

### 1. Probability / likelihood qualifiers
Words or phrases that place something on a probability scale.
Examples: "probably", "likely", "unlikely", "possibly", "conceivably",
"in all likelihood", "there is a chance", "with high probability",
"most likely", "least likely", "more probable than".

### 2. Epistemic modal verbs
Verbs that express possibility or tentative judgement rather than certainty.
Examples: "might", "may", "could", "would", "should (tentatively)",
"must (inferred, not obligatory)", "seems to", "appears to", "tends to".

### 3. Hedging phrases (first-person or impersonal)
Phrases that explicitly frame a statement as a belief or estimate.
Examples: "I believe", "I think", "I suspect", "I'm not certain but",
"it is possible that", "it seems that", "it appears that",
"this suggests", "this is consistent with", "this may indicate".

### 4. Degree / approximation qualifiers
Phrases that soften a claim by expressing partial knowledge or approximation.
Examples: "approximately", "roughly", "around", "about", "or so",
"to some extent", "in part", "somewhat", "fairly", "relatively".

### 5. Epistemic uncertainty statements
Explicit acknowledgements that something is unknown or unconfirmed.
Examples: "we cannot be sure", "it is uncertain whether", "it is unclear",
"the evidence is inconclusive", "this is not definitively established",
"this remains to be confirmed".

## Extraction rules

- Extract **each individual marker occurrence** as a separate example, even if
  the same word appears multiple times.  Every occurrence is its own entry.
- The `excerpt` must be a **short, direct quote** from the text — ideally the
  single word or short phrase itself, plus just enough surrounding context
  (≤ 15 words) to make it readable.
- The `explanation` must name the marker category (from the list above) and
  briefly state what belief or probability the marker expresses in context.
- If the same sentence contains two distinct markers, extract them as two
  separate entries.
- Do **not** paraphrase or alter the quoted text.

## What does NOT count as an uncertainty marker

- Negations that express a logical impossibility rather than epistemic doubt
  ("this cannot be X because it lacks Y") — those are captured by the
  `neg_constraint` metric.
- Rhetorical questions that are answered immediately with certainty.
- Boiler-plate disclaimers that are not part of the actual reasoning
  (e.g. "I am an AI and cannot provide medical advice").
- The words "possible" or "could" when used in a purely logical/hypothetical
  sense with no epistemic hedging ("if we could assume…").
- Reporting the answer options of a multiple-choice question.

## Dataset-specific notes

If the dataset is **MedQA**: answer options (A/B/C/D/E) are part of the prompt,
not the model's own reasoning.  Do NOT extract markers from the question stem
or the option list — only from the model's reasoning process.

If the dataset is **ART**: Hypothesis 1 and Hypothesis 2 are given as input.
Do NOT extract markers that merely restate the hypotheses; only extract markers
from the model's own inferential reasoning.
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Extract every individual uncertainty marker from the following reasoning trace.
Return one entry per marker occurrence.

<reasoning_trace>
{text}
</reasoning_trace>
"""
