# LLM Evaluation Pipeline

A clean, extensible pipeline for evaluating reasoning metrics on model checkpoints using an LLM-as-judge approach.

## Project structure

```
llm_eval/
├── main.py                     ← entry point
├── config.py                   ← all settings (model, sampling, API, paths)
├── llm_client.py               ← OpenAI wrapper, structured-output parsing, caching
├── data_loader.py              ← checkpoint discovery, item loading, sampling
├── evaluator.py                ← per-item orchestration
├── results.py                  ← backward-compatible facade over reporting/
├── pyproject.toml              ← project metadata, dependencies, linter config
├── requirements.txt
├── .env                        ← API credentials (never commit – in .gitignore)
├── .gitignore
│
├── docs/
│   ├── adding_a_metric.md      ← step-by-step guide for adding new metrics
│   └── metric_definitions.md  ← definitions of every metric and metric type
│
├── metrics/
│   ├── base.py                 ← MetricResult dataclass + abstract BaseMetric
│   ├── binary.py               ← BinaryMetric class (yes/no + reasoning)
│   ├── counting.py             ← CountingMetric class (list of examples)
│   └── registry.py             ← METRICS dict – add new metrics here
│
├── prompts/
│   ├── binary/
│   │   ├── uncertainty_language.py      ← binary: presence of hedging language
│   │   └── detail_coverage.py           ← binary: hypothesis covers all observation details
│   └── counting/
│       ├── branchiness.py               ← counting: parallel hypothesis exploration
│       ├── backtracking.py              ← counting: explicit self-correction moments
│       └── uncertainty_markers.py       ← counting: individual hedging word occurrences
│
├── checkpoints/                ← input data (model checkpoint outputs)
│   ├── checkpoint-0/
│   └── checkpoint-4096/
│
├── reporting/                  ← output-generation package
│   ├── csv.py                  ← per-checkpoint CSV writing + debug logs
│   ├── excel.py                ← colour-coded Excel workbook builder
│   └── plots.py                ← evolution line plots
│
└── results/                    ← generated outputs (see Outputs section)
```

## Metric types

See **[docs/metric_definitions.md](docs/metric_definitions.md)** for full definitions of every metric and both metric types.

### Binary metrics
The LLM reasons about whether a phenomenon is present (`detected: true/false`) and explains why.  It also quotes the strongest piece of supporting evidence.

**Metrics:** `uncertainty_language`, `detail_coverage`

### Counting metrics
The LLM does **not** produce a number.  Instead it returns a list of concrete **examples** (excerpt + explanation) of the phenomenon.  The pipeline derives a count as `len(examples)` for plotting.

**Metrics:** `branchiness`, `backtracking`, `uncertainty_markers`

## Quick start

```bash
pip install -r requirements.txt
# or, using pyproject.toml:
pip install -e .

# Set up credentials – add your API key to .env
# OPENAI_API_KEY=your_key_here
# OPENAI_BASE_URL=https://api.hyperbolic.xyz/v1   (optional)

# Edit config.py to set your model, sampling, and paths
python main.py
```

## LLM output format

The judge LLM is called via the **OpenAI Structured Outputs** API (`client.chat.completions.parse`).  Responses are validated and deserialized directly into Pydantic models — no regex or XML parsing.

**Binary metrics** use `BinaryResponse`:
```python
class BinaryResponse(BaseModel):
    detected: bool        # True if the phenomenon is present
    reasoning: str        # Step-by-step justification
    evidence: str         # Direct supporting quote (empty if detected=False)
```

**Counting metrics** use `CountingResponse`:
```python
class ExampleItem(BaseModel):
    excerpt: str          # Exact short quote from the reasoning text
    explanation: str      # Why this excerpt is an instance of the phenomenon

class CountingResponse(BaseModel):
    overall_analysis: str          # Brief summary of findings
    examples: list[ExampleItem]    # All extracted occurrences (empty list if none)
```

The `ask_llm()` function in `llm_client.py` handles the API call, structured-output parsing, JSONL logging, and in-memory caching.

## How to add a new metric

See **[docs/adding_a_metric.md](docs/adding_a_metric.md)** for the full step-by-step guide.

Two paths are covered:
- **Path A** — add a binary (yes/no) or counting (example-extraction) metric using the existing classes. Requires only a new prompt file and one line in `metrics/registry.py`.
- **Path B** — add a completely new metric type with custom LLM output structure. Covers writing the Pydantic schema, the metric class, the prompt file, and registration.

## Outputs

```
results/
├── unnormalized/
│   ├── checkpoint-<N>/
│   │   ├── detailed_metrics_log.csv   ← per-item raw counts
│   │   └── summary_metrics.csv        ← per-dataset averages
│   ├── all_checkpoints_summary.csv
│   ├── checkpoint_comparison.xlsx     ← colour-coded comparison table
│   └── evolution_<metric>_*.png       ← line plots
├── normalized/                        ← same files but counts per 100 words
└── llm_logs/
    ├── <dataset>_llm_responses.jsonl  ← raw LLM call log (for debugging)
    └── <dataset>_full_debug_<run>.csv ← all items × all metrics × all checkpoints
                                          (includes <metric>_error column per metric)
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
