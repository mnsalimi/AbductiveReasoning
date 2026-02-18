# LLM Evaluation Pipeline

A clean, extensible pipeline for evaluating reasoning metrics on model checkpoints using an LLM-as-judge approach.

## Project structure

```
llm_eval/
├── main.py                     ← entry point
├── config.py                   ← all settings (model, sampling, API, paths)
├── llm_client.py               ← OpenAI wrapper, caching, XML parsing
├── data_loader.py              ← checkpoint discovery, item loading, sampling
├── evaluator.py                ← per-item orchestration
├── results.py                  ← backward-compatible facade over reporting/
├── pyproject.toml              ← project metadata, dependencies, linter config
├── requirements.txt
├── .env.example                ← copy to .env and fill in your API key
│
├── metrics/
│   ├── base.py                 ← MetricResult dataclass + abstract BaseMetric
│   ├── binary.py               ← BinaryMetric class (yes/no + reasoning)
│   ├── counting.py             ← CountingMetric class (list of examples)
│   └── registry.py             ← METRICS dict – add new metrics here
│
├── prompts/
│   ├── binary/
│   │   └── uncertainty_language.py   ← example binary metric prompt
│   └── counting/
│       ├── branchiness.py
│       ├── backtracking.py
│       ├── self_verification.py
│       └── neg_constraint.py
│
└── reporting/                  ← output-generation package
    ├── csv.py                  ← per-checkpoint CSV writing + debug logs
    ├── excel.py                ← colour-coded Excel workbook builder
    └── plots.py                ← evolution line plots + tier bar charts
```

## Metric types

### Binary metrics
The LLM reasons about whether a phenomenon is present (`detected: true/false`) and explains why.  It also quotes the strongest piece of supporting evidence.

**Example:** `uncertainty_language` – Does the model use probabilistic language rather than expressing absolute certainty?

### Counting metrics
The LLM does **not** produce a number.  Instead it returns a list of concrete **examples** (excerpt + explanation) of the phenomenon.  The pipeline derives a count as `len(examples)` for plotting.

**Examples:** `branchiness`, `backtracking`, `self_verification`, `neg_constraint`

## Quick start

```bash
pip install -r requirements.txt
# or, using pyproject.toml:
pip install -e .

# Set up credentials
cp .env.example .env          # then fill in OPENAI_API_KEY (and optionally OPENAI_BASE_URL)

# Edit config.py to set your model, sampling, and paths
python main.py
```

## LLM output format

The judge LLM responds in **XML** (plain text mode, no JSON mode).

**Binary metrics** expect:
```xml
<detected>true</detected>
<reasoning>…</reasoning>
<evidence>…</evidence>
```

**Counting metrics** expect:
```xml
<analysis>…</analysis>
<matches>
  <match>
    <excerpt>…</excerpt>
    <explanation>…</explanation>
  </match>
</matches>
```

The `_parse_xml_response()` function in `llm_client.py` handles extraction and maps the tags back to the same Pydantic schemas used throughout the pipeline.

## How to add a new metric

### New binary metric

1. Create `prompts/binary/my_metric.py` with:
   ```python
   SYSTEM_PROMPT = "..."
   USER_PROMPT_TEMPLATE = "...\n\n<reasoning_trace>\n{text}\n</reasoning_trace>"
   ```
2. In `metrics/registry.py`:
   ```python
   from prompts.binary.my_metric import SYSTEM_PROMPT as MM_SYS, USER_PROMPT_TEMPLATE as MM_USR
   # ...inside METRICS dict:
   "my_metric": BinaryMetric(
       name="my_metric",
       description="One-line description.",
       system_prompt=MM_SYS,
       user_prompt_template=MM_USR,
   ),
   ```
That's it.

### New counting metric

Same steps but use `prompts/counting/` and `CountingMetric`.

## Outputs

```
results/
├── unnormalized/
│   ├── checkpoint-<N>/
│   │   ├── detailed_metrics_log.csv   ← per-item raw counts
│   │   └── summary_metrics.csv        ← per-dataset averages
│   ├── all_checkpoints_summary.csv
│   ├── checkpoint_comparison.xlsx     ← colour-coded comparison table
│   ├── evolution_<metric>_*.png       ← line plots
│   └── tier_distribution_<dataset>.png
├── normalized/                        ← same files but counts per 100 words
└── llm_logs/
    ├── <dataset>_llm_responses.jsonl  ← raw LLM call log (for debugging)
    └── <dataset>_full_debug_<run>.csv ← all items × all checkpoints
```

## Configuration reference (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `JUDGE_MODEL` | `deepseek-ai/DeepSeek-V3` | LLM used for judging |
| `N_SAMPLES` | `2` | Items per dataset per checkpoint |
| `MAX_WORKERS` | `5` | Parallel threads |
| `SAMPLE_CORRECT_RATIO` | `1.0` | Fraction of correct items in sample |
| `RANDOM_SEED` | `42` | Reproducibility seed |
| `DISABLED_METRICS` | `[]` | Names of metrics to skip |
| `CLEAR_PREVIOUS_OUTPUTS` | `False` | Delete existing JSONL logs on start |
