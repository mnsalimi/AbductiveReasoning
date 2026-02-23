# Metric Definitions

This document defines each metric type and every metric currently implemented in the pipeline.  For instructions on adding new metrics see [adding_a_metric.md](adding_a_metric.md).

> **Controlling which metrics run:** edit `ACTIVE_METRICS` in `config.py`.  List the metric names you want; leave the list empty (`[]`) to run all registered metrics.

---

## Metric Types

### Binary Metric

A binary metric asks the judge LLM a single yes/no question about the reasoning trace.

- **Output:** `detected` (true/false), a `reasoning` explanation, and an `evidence` quote (empty when `detected` is false).
- **Count used in reports:** 1 if detected, 0 otherwise (one value per trace).
- **Use when:** you want to know *whether* a phenomenon exists in the trace, not *how many times*.

### Counting Metric

A counting metric asks the judge LLM to extract every individual occurrence of a phenomenon as a concrete example (an excerpt + explanation pair).

- **Output:** An `overall_analysis` string and a list of `{excerpt, explanation}` items.
- **Count used in reports:** `len(examples)` — the raw number of extracted occurrences.  This can be normalized per 100 words in the `normalized/` results.
- **Use when:** you want to measure the *density* or *frequency* of a phenomenon, not just its presence.

### Coverage Metric

A coverage metric asks the judge LLM to enumerate **all specific details** present in the observation and assess whether the reasoning trace explicitly connects each detail to the chosen hypothesis.

- **Output:** `observation_details` as a list of `{detail, addressed, evidence}` and an `overall_analysis` synthesis.
- **Score used in reports:** `addressed_count / total_details` (a float in 0.0–1.0). Aggregations typically use the mean score per dataset.
- **Use when:** you want to measure *how completely* a hypothesis accounts for the full set of observation details, not just whether it mentions them.

---

## Implemented Metrics

### `uncertainty_language` — Binary

> **Does the reasoning trace use probabilistic or hedging language rather than absolute certainty?**

Captures whether the model expresses any degree of epistemic humility during its reasoning process.  A single strong hedging phrase that is central to the argument is enough for `detected = true`.

**Positive examples:** "probably", "likely", "I believe", "this suggests", "most likely", "we cannot be sure"  
**Does not count:** confident logical deductions stated without hedges, purely factual recall, boiler-plate disclaimers

---

### `detail_coverage` — Binary

> **Does the reasoning trace account for all specific details of the observation, rather than focusing only on the main event?**

Captures whether the model exhaustively matches its hypothesis against every concrete detail in the observation (symptoms, timeline, lab values, contextual facts), rather than explaining only the most salient finding.

**Positive examples:** addressing each listed symptom individually, checking that the proposed answer is consistent with all given lab values, explicitly noting the absence of findings that would contradict the conclusion  
**Does not count:** a brief summary of the main finding only, restating the question, applying general medical or factual knowledge without cross-checking specific details

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

**Does not count:** boiler-plate disclaimers, purely logical hypotheticals ("if we could assume…")

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

---

### `observation_coverage` — Coverage

> **What fraction of specific observation details are explicitly accounted for by the chosen hypothesis?**

Extracts an exhaustive set of observation details and marks each one as addressed or not addressed.

- **Per-detail fields:**
	- `detail` — one concrete observation fact
	- `addressed` — whether the trace explicitly links that fact to the hypothesis
	- `evidence` — a direct quote showing the link (empty if `addressed = false`)

The metric score is:

$$
	ext{score} = \frac{\#\text{addressed details}}{\#\text{total details}}
$$

`detected` is set to true only when the score is 1.0 (all details addressed).

---

## Relationship Between Metrics

```
Reasoning trace phenomenon
│
├── Is a phenomenon present at all?                → uncertainty_language  (binary)
│
├── Does it cover all observation details?         → detail_coverage       (binary)
│
├── What fraction of details are addressed?        → observation_coverage  (coverage)
│
├── How densely does it hedge?                     → uncertainty_markers   (counting)
│
├── Does it explore multiple paths in parallel?    → branchiness           (counting)
│
└── Does it catch and fix its own mistakes?        → backtracking          (counting)
```

Note that `uncertainty_language` and `uncertainty_markers` measure the **same underlying phenomenon** at different granularities — binary presence vs. raw occurrence count.  They are designed to complement rather than replace each other.

---

## Dataset & Checkpoint Selection

Datasets and checkpoints to include in a run are controlled via `config.py`:

| Setting | Effect when empty (`[]`) | Effect when non-empty |
|---|---|---|
| `ACTIVE_DATASETS` | All dataset folders found in each checkpoint are evaluated | Only the listed dataset names are evaluated |
| `ACTIVE_METRICS` | All registered metrics are run | Only the listed metric names are run |

### `raw_model` as a checkpoint alias

A checkpoint directory named `raw_model` is automatically treated as **checkpoint-0** (the untrained baseline).  Both `checkpoint-0/` and `raw_model/` are discovered and reported identically — no manual renaming is needed.
