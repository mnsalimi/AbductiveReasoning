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
│   ├── coverage.py             ← CoverageMetric class (per-detail coverage + score)
│   └── registry.py             ← METRICS dict – add new metrics here
│
├── prompts/
│   ├── binary/
│   │   ├── uncertainty_language.py      ← binary: presence of hedging language
│   │   └── detail_coverage.py           ← binary: hypothesis covers all observation details
│   ├── coverage/
│   │   └── observation_coverage.py      ← coverage: per-detail observation coverage + score
│   └── counting/
│       ├── branchiness.py               ← counting: parallel hypothesis exploration
│       ├── backtracking.py              ← counting: explicit self-correction moments
│       └── uncertainty_markers.py       ← counting: individual hedging word occurrences
│
├── checkpoints/                ← input data (model checkpoint outputs)
│   ├── checkpoint-0/           ← baseline; ``raw_model/`` is treated as an alias
│   ├── raw_model/              ← optional alias for checkpoint-0
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

See **[docs/metric_definitions.md](docs/metric_definitions.md)** for full definitions of every metric and metric type.

### Binary metrics
The LLM reasons about whether a phenomenon is present (`detected: true/false`) and explains why.  It also quotes the strongest piece of supporting evidence.

**Metrics:** `uncertainty_language`, `detail_coverage`

### Counting metrics
The LLM does **not** produce a number.  Instead it returns a list of concrete **examples** (excerpt + explanation) of the phenomenon.  The pipeline derives a count as `len(examples)` for plotting.

**Metrics:** `branchiness`, `backtracking`, `uncertainty_markers`

### Coverage metrics
The LLM extracts an exhaustive list of observation details, marks whether each one is explicitly addressed by the chosen hypothesis, and the pipeline computes a coverage score.

**Metrics:** `observation_coverage`

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

**Coverage metrics** use `ObservationCoverageResponse`:
```python
class ObservationDetail(BaseModel):
    detail: str           # One specific observation fact
    addressed: bool       # Was it connected to the hypothesis?
    evidence: str         # Quote from the trace (empty if addressed=False)

class ObservationCoverageResponse(BaseModel):
    observation_details: list[ObservationDetail]
    overall_analysis: str
```

Token usage (input/output, and optionally reasoning/cached input) is recorded per LLM call and propagated into the full-debug outputs.

The `ask_llm()` function in `llm_client.py` handles the API call, structured-output parsing, JSONL logging, and in-memory caching.

## How to add a new metric

See **[docs/adding_a_metric.md](docs/adding_a_metric.md)** for the full step-by-step guide.

Two paths are covered:
- **Path A** — add a binary (yes/no), counting (example-extraction), or coverage (per-detail) metric using the existing classes. Requires only a new prompt file and one line in `metrics/registry.py`.
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
    ├── <dataset>_llm_responses.jsonl  ← raw LLM call log (includes token usage per call)
    └── <dataset>_full_debug_<run>.csv ← all items × all metrics × all checkpoints
                                          (includes <metric>_error column per metric)
```

## Configuration reference (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `JUDGE_MODEL` | `gpt-5-nano` | LLM used for judging |
| `REASONING_EFFORT` | `"low"` | Reasoning token budget for GPT-5+ models (`"low"` / `"medium"` / `"high"`). Ignored for older models. |
| `N_SAMPLES` | `1` | Items per dataset per checkpoint |
| `MAX_WORKERS` | `5` | Parallel threads |
| `SAMPLE_CORRECT_RATIO` | `1.0` | Fraction of correct items in sample |
| `RANDOM_SEED` | `42` | Reproducibility seed |
| `ACTIVE_METRICS` | `["uncertainty_markers"]` | Names of metrics to run. Empty list activates **all** registered metrics. |
| `ACTIVE_DATASETS` | `["art", "copa_guess_effect"]` | Dataset folder names to evaluate. Empty list evaluates **all** datasets found in each checkpoint. |
| `CLEAR_PREVIOUS_OUTPUTS` | `False` | Delete existing JSONL logs on start |

## Changelog

### 2026-02-22
- **`data_loader.py` — auto-assign `problem_id` for datasets without IDs**  
  Some dataset JSON files (e.g. `art`, `copa_guess_effect`) do not include a `problem_id` field on each result item.  `load_items()` now assigns a sequential integer ID (0-based, matching the order items appear in the file) to any item that is missing this field.  This ensures every item passes through `build_pid_map()` and the sampling / cross-checkpoint comparison logic works correctly for these datasets.

---

## Model compatibility

The pipeline auto-detects the judge model family and adjusts the API call accordingly:

| Model family | Role used | Token limit param | Extras |
|---|---|---|---|
| GPT-5 and newer (`gpt-5*`) | `developer` | `max_completion_tokens` | `reasoning_effort` |
| Older / OSS models | `system` | `max_tokens` | `temperature=0.0` |

Detection is name-based: any model matching `gpt-N` where N ≥ 5 uses the modern path. Everything else uses the legacy path. To change the reasoning budget for GPT-5+ models, adjust `REASONING_EFFORT` in `config.py`.
