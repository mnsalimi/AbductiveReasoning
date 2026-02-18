"""
prompts/counting/neg_constraint.py
------------------------------------
Prompt for the Negative Constraint Satisfaction counting metric.

The LLM returns a list of concrete examples where an option or hypothesis is
explicitly ruled out because of a contradiction with known facts or constraints.
"""

SYSTEM_PROMPT = """\
You are an expert reasoning analyst evaluating AI-generated reasoning traces.

## What is Negative Constraint Satisfaction?

Negative Constraint Satisfaction measures when the reasoning explicitly
*rules out* an option, hypothesis, or case by citing a contradiction with
given facts or constraints.

## What COUNTS as a ruling-out moment

Extract an example when the text clearly states:
  (Option X) cannot be true / is ruled out / is excluded BECAUSE of (reason).

This includes:
- Direct exclusion language: "rule out", "exclude", "eliminate", "cannot be",
  "is impossible", "is inconsistent with", "contradicts"
- Contraindications: "Drug X is contraindicated because …", "Option Y is
  incompatible because …"
- Missing required findings: "Diagnosis Z requires symptom W, which is absent"
- Temporal or demographic impossibilities:
  "This condition presents in children; the patient is 60"

## What does NOT count

- The final answer selection alone.
- Positive evidence for the chosen option.
- Vague preferences ("A is less likely than B") without a concrete reason.
- Restating the given options without ruling any out.
- Comparative hypothesis-exploration (Branchiness):
  "If A then X … If B then Y …" — that is Branchiness, not NegConstraint.

## Repetition rule

If the reasoning uses ONE blanket statement to eliminate multiple options
("none of A, B, or C fit because …"), extract it as ONE example.

## Dataset-specific notes

MedQA: Ruling out an answer choice due to a contraindication, an inconsistent
lab value, or a missing required finding counts.

ART: Ruling out a hypothesis because it contradicts an observation counts.
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Negative Constraint Satisfaction and
extract concrete examples of explicit ruling-out events.

<reasoning_trace>
{text}
</reasoning_trace>
"""
