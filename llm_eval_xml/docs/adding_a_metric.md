# How to Add a New Metric

There are two paths. Follow **Path A** if your metric fits the yes/no (binary) or example-extraction (counting) model that already exists in the pipeline. Follow **Path B** if you need fundamentally different LLM output structure or evaluation logic.

---

## Path A — Adding a Binary or Counting Metric

This is the normal path. You only touch two things: a new prompt file and one registration entry.

### What is the difference?

| Type | What the LLM outputs | `detected` field | `examples` field |
|---|---|---|---|
| **Binary** | `true`/`false` + reasoning + one evidence quote | Direct from LLM | Empty (or one item if evidence exists) |
| **Counting** | A list of (excerpt, explanation) pairs | `true` when at least one example found | All extracted examples |

---

### Step 1 — Create the prompt file

**Binary** → create `prompts/binary/your_metric_name.py`  
**Counting** → create `prompts/counting/your_metric_name.py`

Both files must export exactly these two names:

```python
SYSTEM_PROMPT: str         # full instructions for the judge LLM
USER_PROMPT_TEMPLATE: str  # uses {text} and optionally {dataset}
```

#### Binary prompt template

```python
"""
prompts/binary/your_metric_name.py
"""

SYSTEM_PROMPT = """\
You are an expert reasoning analyst.

## What is <Your Metric>?
<Define the phenomenon clearly.>

## What COUNTS (detected = true)
<Bullet list of clear positive cases.>

## What does NOT count (detected = false)
<Bullet list of common false-positive traps.>

## Dataset-specific notes (optional)
If the dataset is MedQA: ...
If the dataset is ART: ...
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for <Your Metric>.

<reasoning_trace>
{text}
</reasoning_trace>
"""
```

See [`prompts/binary/uncertainty_language.py`](../prompts/binary/uncertainty_language.py) for a full real example.

#### Counting prompt template

```python
"""
prompts/counting/your_metric_name.py
"""

SYSTEM_PROMPT = """\
You are an expert reasoning analyst.

## What is <Your Metric>?
<Define the phenomenon.>

## What COUNTS as an example
<Numbered list of concrete extraction criteria.>

## What does NOT count
<Common false positives.>

## Dataset-specific notes (optional)
If the dataset is MedQA: ...
If the dataset is ART: ...
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for <Your Metric> and extract concrete examples.

<reasoning_trace>
{text}
</reasoning_trace>
"""
```

See [`prompts/counting/branchiness.py`](../prompts/counting/branchiness.py) for a full real example.

---

### Step 2 — Register the metric in `metrics/registry.py`

Open [`metrics/registry.py`](../metrics/registry.py) and make **three edits**:

**1. Import the prompts** (add to the relevant import section at the top):

```python
# Binary:
from prompts.binary.your_metric_name import (
    SYSTEM_PROMPT as YM_SYS,
    USER_PROMPT_TEMPLATE as YM_USR,
)

# OR Counting:
from prompts.counting.your_metric_name import (
    SYSTEM_PROMPT as YM_SYS,
    USER_PROMPT_TEMPLATE as YM_USR,
)
```

**2. Add an entry to the `METRICS` dict**:

```python
# Binary:
"your_metric_name": BinaryMetric(
    name="your_metric_name",
    description="One-line description shown in reports.",
    system_prompt=YM_SYS,
    user_prompt_template=YM_USR,
),

# OR Counting:
"your_metric_name": CountingMetric(
    name="your_metric_name",
    description="One-line description shown in reports.",
    system_prompt=YM_SYS,
    user_prompt_template=YM_USR,
),
```

**3. Make sure it is not disabled** in [`config.py`](../config.py):

```python
DISABLED_METRICS = ["backtracking", ...]  # remove your_metric_name if it appears here
```

That's it. The pipeline picks it up automatically — no changes needed to `evaluator.py`, `main.py`, or the reporting layer.

---

## Path B — Adding a Completely New Metric Type

Use this when `BinaryMetric` and `CountingMetric` are not enough — for example, you need a numeric score, a multi-label output, a multi-turn LLM call, or non-LLM computation.

You must touch **four** things.

---

### Step 1 — Define the Pydantic response schema

This is the structured contract between your code and the LLM. Create it in a new file under `metrics/`:

```python
# metrics/your_type.py
from pydantic import BaseModel, Field

class YourResponse(BaseModel):
    score: float = Field(..., description="Numeric score between 0 and 1.")
    label: str   = Field(..., description="Dominant category detected.")
    reasoning: str = Field(..., description="Step-by-step justification.")
    # Add whatever fields your metric needs.
```

The field descriptions are important — they become part of the LLM's context when it is asked to produce structured output.

---

### Step 2 — Write the metric class

Your class must inherit from `BaseMetric` ([`metrics/base.py`](../metrics/base.py)) and implement everything listed below.

```python
# metrics/your_type.py (continued)
from metrics.base import BaseMetric, MetricResult
import llm_client

class YourTypeMetric(BaseMetric):

    # ── Required class-level attribute ──────────────────────────────────────
    metric_type = "your_type"   # new string identifier for your category

    # ── Constructor ─────────────────────────────────────────────────────────
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        user_prompt_template: str,
        # add any extra params your type needs, e.g. score_threshold: float = 0.5
    ) -> None:
        self.name = name
        self.description = description
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        # store extra params as self attributes

    # ── Required: schema property ────────────────────────────────────────────
    @property
    def schema(self) -> type[YourResponse]:
        """Returned schema is passed to llm_client.ask_llm as response_schema."""
        return YourResponse

    # ── Required: evaluate() ────────────────────────────────────────────────
    def evaluate(
        self,
        text: str,
        *,
        dataset: str = "unknown",
        problem_id: str = "N/A",
        checkpoint: str = "N/A",
        run_id: str | None = None,
    ) -> MetricResult:
        """
        Must always return a MetricResult, even on error.

        MetricResult fields:
            metric_name  – always set to self.name
            detected     – bool: was the phenomenon present? (required)
            reasoning    – str: model's justification (required)
            examples     – list[dict{"excerpt", "explanation"}]: evidence items
            error        – str: non-empty only on failure
            raw          – dict: the full LLM payload for logging/debugging
        """
        # 1. Guard against empty input — always return MetricResult, never raise
        if not isinstance(text, str) or not text.strip():
            return MetricResult(metric_name=self.name, error="Empty input.")

        # 2. Truncate to stay within context limits (same threshold as other metrics)
        trimmed = text[:15_000] + "\n…(truncated)" if len(text) > 15_000 else text

        # 3. Build prompts
        user_prompt = self._user_prompt_template.format(text=trimmed, dataset=dataset)
        system_prompt = self._system_prompt + "\n\n" + _YOUR_FORMAT_INSTRUCTIONS

        # 4. Call the LLM — pass all identifiers for caching and logging
        payload = llm_client.ask_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=YourResponse,
            dataset=dataset,
            problem_id=problem_id,
            metric_type=self.name,
            checkpoint=checkpoint,
            run_id=run_id,
        )

        # 5. Extract fields from the returned dict
        score: float   = float(payload.get("score", 0.0))
        label: str     = payload.get("label", "")
        reasoning: str = payload.get("reasoning", "")

        # 6. Decide what "detected" means for your metric
        detected = score > 0.5  # replace with your own logic

        # 7. Build and return MetricResult
        return MetricResult(
            metric_name=self.name,
            detected=detected,
            reasoning=reasoning,
            examples=[],   # populate with {"excerpt": ..., "explanation": ...} dicts if relevant
            raw=payload,
        )


# ── XML format instructions appended to the system prompt ───────────────────
# The llm_client XML parser maps these tag names to your Pydantic field names.
_YOUR_FORMAT_INSTRUCTIONS = """\
## Output format

Return your analysis using the following XML tags:

<score>[float between 0.0 and 1.0]</score>
<label>[dominant category]</label>
<reasoning>[step-by-step justification]</reasoning>

CRITICAL: You MUST include all three XML tags in your response.
"""
```

> **Why XML tags?**  
> `llm_client` parses XML tags from the LLM's response and maps tag names back to Pydantic field names. The tag name must exactly match the Pydantic field name. Study the existing format instruction blocks in [`metrics/binary.py`](../metrics/binary.py) and [`metrics/counting.py`](../metrics/counting.py) for reference.

---

### Step 3 — Create the prompt file

Same as Path A. Create a new subfolder for your type:

```
prompts/
└── your_type/
    ├── __init__.py              ← empty file, required for Python imports
    └── your_metric_name.py     ← SYSTEM_PROMPT + USER_PROMPT_TEMPLATE
```

Use `{text}` and optionally `{dataset}` as the only placeholders in `USER_PROMPT_TEMPLATE`.

---

### Step 4 — Register in `metrics/registry.py`

```python
from metrics.your_type import YourTypeMetric
from prompts.your_type.your_metric_name import (
    SYSTEM_PROMPT as YM_SYS,
    USER_PROMPT_TEMPLATE as YM_USR,
)

METRICS: dict[str, BaseMetric] = {
    # ... existing metrics ...
    "your_metric_name": YourTypeMetric(
        name="your_metric_name",
        description="One-line description.",
        system_prompt=YM_SYS,
        user_prompt_template=YM_USR,
    ),
}
```

Then make sure it is active in [`config.py`](../config.py) (not listed in `DISABLED_METRICS`).

---

## Summary Checklist

| | Path A — Binary | Path A — Counting | Path B — New Type |
|---|---|---|---|
| New prompt file | `prompts/binary/name.py` | `prompts/counting/name.py` | `prompts/your_type/name.py` + `__init__.py` |
| New `__init__.py` in prompt folder | Not needed | Not needed | **Required** |
| New metric class file | Not needed | Not needed | `metrics/your_type.py` |
| Pydantic response schema | `BinaryResponse` (already exists) | `CountingResponse` (already exists) | **You define `YourResponse`** |
| Inherit `BaseMetric` | Done by `BinaryMetric` | Done by `CountingMetric` | **You must inherit it** |
| Declare `metric_type` | Already `"binary"` | Already `"counting"` | **You must set a new string** |
| Implement `schema` property | Already done | Already done | **You must implement it** |
| Implement `evaluate()` | Already done | Already done | **You must implement it, always return `MetricResult`** |
| XML output instructions | Already appended | Already appended | **You must write and append your own** |
| Import in `registry.py` | Prompts only | Prompts only | Class + prompts |
| Add to `METRICS` dict | `BinaryMetric(...)` | `CountingMetric(...)` | `YourTypeMetric(...)` |
| Enable in `config.py` | Remove from `DISABLED_METRICS` | Remove from `DISABLED_METRICS` | Remove from `DISABLED_METRICS` |
