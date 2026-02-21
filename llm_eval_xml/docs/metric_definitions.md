# Metric Definitions

This document defines each metric type and every metric currently implemented in the pipeline.  For instructions on adding new metrics see [adding_a_metric.md](adding_a_metric.md).

---

## Metric Types

### Binary Metric

A binary metric asks the judge LLM a single yes/no question about the reasoning trace.

- **Output:** `detected` (true/false), a `reasoning` explanation, and an optional `evidence` quote.
- **Count used in reports:** 1 if detected, 0 otherwise (one value per trace).
- **Use when:** you want to know *whether* a phenomenon exists in the trace, not *how many times*.

### Counting Metric

A counting metric asks the judge LLM to extract every individual occurrence of a phenomenon as a concrete example (an excerpt + explanation pair).

- **Output:** A list of `{excerpt, explanation}` items.
- **Count used in reports:** `len(examples)` — the raw number of extracted occurrences.  This can be normalized per 100 words in the `normalized/` results.
- **Use when:** you want to measure the *density* or *frequency* of a phenomenon, not just its presence.

---

## Implemented Metrics

### `uncertainty_language` — Binary

> **Does the reasoning trace use probabilistic or hedging language rather than absolute certainty?**

Captures whether the model expresses any degree of epistemic humility during its reasoning process.  A single strong hedging phrase that is central to the argument is enough for `detected = true`.

**Positive examples:** "probably", "likely", "I believe", "this suggests", "most likely", "we cannot be sure"  
**Does not count:** confident logical deductions stated without hedges, purely factual recall, boiler-plate disclaimers

---

### `uncertainty_markers` — Counting

> **How many individual probabilistic or hedging words and phrases appear in the reasoning trace?**

Where `uncertainty_language` gives a yes/no answer, `uncertainty_markers` extracts *every single occurrence* of a hedging word or phrase as its own entry.  The count gives a quantitative density measure of epistemic hedging.

Markers are grouped into five categories:
1. **Probability / likelihood qualifiers** — "probably", "likely", "possibly", "in all likelihood"
2. **Epistemic modal verbs** — "might", "may", "could", "seems to", "appears to"
3. **Hedging phrases** — "I believe", "I think", "this suggests", "this is consistent with"
4. **Degree / approximation qualifiers** — "approximately", "roughly", "to some extent", "somewhat"
5. **Epistemic uncertainty statements** — "we cannot be sure", "it is unclear", "the evidence is inconclusive"

**Does not count:** logical negations, boiler-plate disclaimers, purely logical hypotheticals ("if we could assume…")

---

### `branchiness` — Counting

> **How many times does the reasoning genuinely explore multiple distinct possibilities, hypotheses, or solution paths?**

Measures whether the model thinks divergently rather than following a single linear chain.  Each distinct branching moment (building two or more hypotheses in parallel, trying multiple methods, or developing multiple conditional flows) is extracted as one example.

**Positive examples:** "If diagnosis X we'd expect F… If diagnosis Y we'd expect G…", trying two solution methods and comparing them  
**Does not count:** the final answer selection, brief mention of an alternative followed by immediate rejection, simple step-by-step narration, restating the given answer options

---

### `backtracking` — Counting

> **How many times does the reasoning explicitly identify an error or flaw and change direction?**

Captures deliberate self-correction: the model realises something it said or computed is wrong and reverses course.  This is distinct from `branchiness` (exploring valid alternatives).

**Positive examples:** "Wait, that's wrong", "On second thought…", "Let me re-read the problem", "I realise I forgot to account for…"  
**Does not count:** comparing two valid paths (Branchiness), a simple "however" contrast without admitting an error, the final answer selection

## Relationship Between Metrics

```
Reasoning trace phenomenon
│
├── Is a phenomenon present at all?                → uncertainty_language  (binary)
│
├── How densely does it hedge?                     → uncertainty_markers   (counting)
│
├── Does it explore multiple paths in parallel?    → branchiness           (counting)
│
└── Does it catch and fix its own mistakes?        → backtracking          (counting)
```

Note that `uncertainty_language` and `uncertainty_markers` measure the **same underlying phenomenon** at different granularities — binary presence vs. raw occurrence count.  They are designed to complement rather than replace each other.
