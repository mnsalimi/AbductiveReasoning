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
│       ├── uncertainty_markers.py       ← counting: individual hedging word occurrences
│       └── prior.py                     ← counting: prior probability / base-rate reasoning
│
├── scripts/
│   ├── generate_latex_slides.py ← generate Beamer .tex comparing one item across 2 checkpoints
│   └── gen_slides.sh            ← edit variables here and run to generate slides
│
├── checkpoints/                ← input data (model checkpoint outputs)
│   ├── raw_model/              ← treated as checkpoint-0
│   └── checkpoint-<N>/
│
├── reporting/                  ← output-generation package
│   ├── csv.py                  ← per-checkpoint CSV writing, debug logs, config snapshot
│   ├── excel.py                ← colour-coded Excel workbook builder
│   ├── plots.py                ← evolution line plots (respects SAMPLE_CORRECT_RATIO)
│   ├── comparison_logs.py      ← pairwise CSV diff logs (2-checkpoint runs)
│   └── detailed_logs.py        ← backward-compat alias for comparison_logs.py
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

**Metrics:** `branchiness`, `backtracking`, `uncertainty_markers`, `prior`

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

## LaTeX slide generation

`scripts/generate_latex_slides.py` produces a **Beamer .tex presentation** comparing a single evaluated item across two checkpoints for one metric.

### Quick way

Edit the variables at the top of `scripts/gen_slides.sh`, then run from the project root:

```bash
bash scripts/gen_slides.sh
```

### Full CLI

```bash
python scripts/generate_latex_slides.py \
    --dataset         copa_guess_effect \
    --problem_id      70 \
    --checkpoint_a    0 \
    --checkpoint_b    2560 \
    --metric          uncertainty_markers \
    --output          results/latex_slides \
    --log_dir         results/llm_logs \
    --checkpoints_dir checkpoints
```

Compile the output:
```bash
pdflatex results/latex_slides/copa_guess_effect_pid70_ckpt0vs2560_uncertainty_markers.tex
```

### Slide structure

| Slide | Content |
|---|---|
| `Problem N [DATASET]: Question` | Full question, options/hypotheses, true vs predicted answer with ✓/✗ |
| `Problem N: Reasoning (ckpt A)` | Cleaned reasoning trace (strips `<reasoning>` / `<answer>` wrappers) |
| `Problem N: {Metric} (ckpt A)` | Formatted metric result (type-aware, see below) |
| `Problem N: Reasoning (ckpt B)` | Same for checkpoint B |
| `Problem N: {Metric} (ckpt B)` | Same for checkpoint B |

Metric results are rendered according to type:
- **Binary** — large ✓/✗, detected status, reasoning text, evidence quote block
- **Counting** — prominent count, overall analysis, numbered list of (excerpt, why) pairs
- **Coverage** — `X/N = Y%` score, overall analysis, `tabularx` table with ✓/✗ per detail

Data is loaded from `results/llm_logs/{dataset}_full_debug.jsonl` (post-evaluation, includes metrics). If that file doesn't exist yet the script falls back to the raw checkpoint JSON (question + reasoning only, no metric data).

---

## How to add a new metric

See **[docs/adding_a_metric.md](docs/adding_a_metric.md)** for the full step-by-step guide.

Two paths are covered:
- **Path A** — add a binary (yes/no), counting (example-extraction), or coverage (per-detail) metric using the existing classes. Requires only a new prompt file and one line in `metrics/registry.py`.
- **Path B** — add a completely new metric type with custom LLM output structure. Covers writing the Pydantic schema, the metric class, the prompt file, and registration.

## Outputs

```
results/
├── run_config_<RUN_ID>.json           ← snapshot of every config setting for this run
├── unnormalized/
│   ├── checkpoint-<N>/
│   │   ├── detailed_metrics_log.csv   ← per-item raw counts
│   │   └── summary_metrics.csv        ← per-dataset averages
│   ├── all_checkpoints_summary.csv
│   ├── checkpoint_comparison.xlsx     ← colour-coded comparison table
│   └── evolution_<metric>_*.png       ← line plots (correct / incorrect / mix)
├── normalized/                        ← same files but counts per 100 words
├── llm_logs/
│   ├── <dataset>_llm_responses.jsonl  ← raw LLM call log (token usage per call)
│   └── <dataset>_full_debug_<run>.csv ← all items × all metrics × all checkpoints
├── comparison_logs/                   ← only when exactly 2 checkpoints are run
│   └── <dataset>/
│       └── <metric>/
│           ├── match.csv / mismatch.csv           (binary metrics)
│           └── A_gt_B.csv / A_eq_B.csv / A_lt_B.csv (counting & coverage)
└── latex_slides/
    └── <dataset>_pid<N>_ckpt<A>vs<B>_<metric>.tex
```

### Evolution plots and `SAMPLE_CORRECT_RATIO`

| `SAMPLE_CORRECT_RATIO` | Plots generated |
|---|---|
| `1.0` | `evolution_<metric>_correct.png` only |
| `0.0` | `evolution_<metric>_incorrect.png` only |
| any other value | `_correct.png`, `_incorrect.png`, and `_mix.png` (all statuses averaged) |

## Configuration reference (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `JUDGE_MODEL` | `gpt-5-nano` | LLM used for judging |
| `REASONING_EFFORT` | `"low"` | Reasoning token budget for GPT-5+ models (`"low"` / `"medium"` / `"high"`). Ignored for older models. |
| `N_SAMPLES` | `3` | Items per dataset per checkpoint |
| `MAX_WORKERS` | `5` | Parallel threads |
| `SAMPLE_CORRECT_RATIO` | `None` | Fraction of correct items in sample. `1.0` = all correct, `0.0` = all incorrect, `None` = pure random. |
| `RANDOM_SEED` | `42` | Reproducibility seed |
| `ACTIVE_METRICS` | `["prior", "uncertainty_markers", "observation_coverage"]` | Names of metrics to run. Empty list activates **all** registered metrics. |
| `ACTIVE_DATASETS` | `["medqa", "copa_guess_effect"]` | Dataset folder names to evaluate. Empty list evaluates **all** datasets found in each checkpoint. |
| `EXCLUDED_CHECKPOINTS` | `[]` | Checkpoint directory basenames to skip entirely (e.g. `["raw_model", "checkpoint-500"]`). |
| `CLEAR_PREVIOUS_OUTPUTS` | `False` | Delete existing JSONL logs on start |

## Changelog

### 2026-02-25
- **Excluded checkpoints** — new `EXCLUDED_CHECKPOINTS` list in `config.py`. Any checkpoint whose directory basename appears in this list is silently skipped by `find_checkpoint_dirs()`.
- **Run config snapshot** — every run writes `results/run_config_<RUN_ID>.json` capturing the full setup: model, effort, sampling params, active metrics/datasets, excluded/evaluated checkpoints, and all output paths.
- **Plot behaviour by `SAMPLE_CORRECT_RATIO`** — `reporting/plots.py` now generates only the relevant plots: correct-only (`== 1.0`), incorrect-only (`== 0.0`), or correct + incorrect + mix (any other value).
- **Pairwise comparison logs** — `reporting/comparison_logs.py` (replaces `detailed_logs.py`) produces per-metric CSV diffs when exactly two checkpoints are run. Binary metrics → match/mismatch buckets; counting/coverage → A > B / A = B / A < B buckets, each split by dataset.
- **LaTeX slide generator** — `scripts/generate_latex_slides.py` renders a Beamer `.tex` for any (dataset, problem_id, ckpt_A, ckpt_B, metric) tuple. Handles all dataset schemas (ART, COPA, MedQA, GoEmotion, etc.) and all three metric types cleanly. Shell wrapper at `scripts/gen_slides.sh`.

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
