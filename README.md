# Improving Abductive Reasoning in LLMs

> **CEDAR-GRPO** — NeurIPS submission

## Overview

We investigate whether post-training with **Group Relative Policy Optimization (GRPO)** can strengthen abductive reasoning as a *transferable* competence in large language models. We train on a deliberately domain-neutral mixture of abductive tasks (hypothesis generation and hypothesis selection) and evaluate on nine held-out benchmarks—none seen during training—spanning classic abductive selection, abstract rule induction, long-context multi-step inference, defeasible reasoning, and non-abductive controls.

Beyond end-task accuracy, we introduce **process-level metrics** (hypothesis branching, explicit backtracking, epistemic uncertainty marking, and observation coverage) to verify that improved benchmark performance co-occurs with measurably more exploratory and evidence-grounded reasoning traces.

## Repository layout

```
AbductiveReasoning/
├── GRPO/                          # Main GRPO training (notebooks) and all per-dataset evaluation scripts
│   ├── train_abductive_new.ipynb          # GRPO training — abductive mixture
│   ├── train_abductive_general_reasoning.ipynb  # GRPO training — general reasoning ablation
│   ├── prompts.py                         # Shared prompt builders for all datasets
│   └── Evaluation/                        # 25+ evaluation scripts + orchestrator
│       ├── evaluate_all.py                # Parallel evaluation orchestrator
│       └── evaluate_<dataset>_raw_vs_finetuned.py  (one per dataset)
├── SFT/                           # Supervised fine-tuning baseline
│   └── train_abductive_sft.py
├── llm_eval_structured_output/    # LLM-as-judge pipeline for process-level metrics
│   └── main.py
├── datasets/                      # Pre-loaded held-out evaluation datasets (JSONL)
├── direct_evaluation/             # Alternative evaluation pipeline (subset of datasets)
├── evaluate_sml/                  # Small/medium baseline model evaluation
├── correlation-analysis/          # Correlation analysis notebooks
└── requirements.txt
```

## Installation

Python ≥ 3.10 is required.

```bash
git clone https://github.com/<your-org>/AbductiveReasoning.git
cd AbductiveReasoning
pip install -r requirements.txt
```

For the process-level metrics pipeline only:

```bash
pip install -r llm_eval_structured_output/requirements.txt
# or
pip install -e llm_eval_structured_output/
```

## Models

We apply GRPO post-training to four base models:

| Model | HuggingFace ID |
|---|---|
| Qwen3-4B | `Qwen/Qwen3-4B` |
| Qwen3-8B | `Qwen/Qwen3-8B` |
| DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` |

All models are fine-tuned with the same PEFT configuration:

| Setting | Value |
|---|---|
| Quantization | 4-bit NF4 via `bitsandbytes` |
| LoRA target modules | `q, k, v, o, up, down, gate` projections |
| LoRA rank / alpha | 64 / 64 |
| Training method | GRPO (`trl.GRPOTrainer`) |

Fine-tuned weights are fully reproducible from the training scripts below.

## Training

### GRPO (main method)

Open the training notebook and run all cells:

```bash
jupyter notebook GRPO/train_abductive_new.ipynb
```

Key constants to set at the top of the notebook:

| Variable | Description |
|---|---|
| `MODEL_NAME` | HuggingFace model path |
| `OUTPUT_DIR` | Where checkpoints are saved |
| `NUM_TRAIN_EPOCHS` | Number of GRPO epochs (default: 20) |

Checkpoints are saved under `GRPO/results/<run_name>/`.

### SFT baseline

```bash
cd SFT
python train_abductive_sft.py
```

Outputs are saved under `SFT/results/<run_name>/`.

### Training data

| Dataset | Stage | Instances | Train | Val |
|---|---|---|---|---|
| UniADILR-HGc (abductive) | Stage II — hypothesis selection | 400 | 320 | 80 |
| Balanced COPA (cause) | Stage II — hypothesis selection | 400 | 320 | 80 |
| CauseLogics (Levels 3–4) | Stage II-adjacent | 400 | 320 | 80 |
| CLIMATE-FEVER | Stage II-adjacent | 400 | 320 | 80 |
| AbductionRules | Stage I — missing premise generation | 400 | 320 | 80 |
| Crypto (caesar + atbash) | Stage I — rule induction | 200 | 160 | 40 |
| List Function | Stage I — rule induction | 200 | 160 | 40 |
| **Total** | | **2,400** | **1,920** | **480** |

## Evaluation

### End-task accuracy

Run all held-out evaluations in parallel (configure paths at the top of `evaluate_all.py`):

```bash
python GRPO/Evaluation/evaluate_all.py \
    --parallel 2 \
    --checkpoint_path /path/to/checkpoint-<N> \
    --training_dir /path/to/GRPO/results/<run_name>
```

To evaluate a single dataset:

```bash
python GRPO/Evaluation/evaluate_art_raw_vs_finetuned.py \
    --checkpoint_path /path/to/checkpoint-<N> \
    --max_samples 200
```

Available dataset scripts: `art`, `copa` (cause/effect), `defeasible_nli`, `goEmotion`, `musr_murder_mystery`, `musr_object_placements`, `musr_team_allocation`, `neulr_abductive`, `strategyqa`, and 15+ more.

### Process-level metrics

Requires an OpenAI or Gemini API key. Copy `.env.example` and fill in credentials:

```bash
cp llm_eval_structured_output/.env.example llm_eval_structured_output/.env
# Edit .env: set OPENAI_API_KEY or GEMINI_BASE_URL
```

Place model checkpoint outputs under `llm_eval_structured_output/checkpoints/` then run:

```bash
cd llm_eval_structured_output
python main.py
```

Metrics evaluated: `branchiness`, `backtracking`, `uncertainty_markers`, `observation_coverage`, `evidence_explanation_directionality`. See [`llm_eval_structured_output/docs/metric_definitions.md`](llm_eval_structured_output/docs/metric_definitions.md) for full definitions.

## Results

Accuracy across all held-out evaluation datasets. **Bold** denotes best result per model. BASE = unmodified base model; ACC-GRPO = accuracy-only GRPO; CEDAR-GRPO = our method.

To reproduce these numbers:

```bash
# 1. Train
jupyter nbconvert --to notebook --execute GRPO/train_abductive_new.ipynb

# 2. Evaluate all held-out datasets
python GRPO/Evaluation/evaluate_all.py \
    --parallel 2 \
    --training_dir GRPO/results/<run_name>

# 3. Aggregate results into a table
python GRPO/Evaluation/create_table.py \
    --results_dir GRPO/Evaluation/multi_evaluation_results/<timestamp>
```

| Model | Method | ART | B-COPA | DefNLI | GoEmo. | MuSR-M | MuSR-O | MuSR-T | NeuLR | StratQA | MedQA | ML-Debug |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | Base | 65.25% | 84.40% | 80.75% | 27.75% | 20.40% | 19.53% | 51.60% | 35.00% | 38.25% | 35.50% | 25.25% |
| | Acc-GRPO | 71.75% | 86.80% | 86.75% | 32.25% | 41.60% | 32.42% | 48.80% | 38.75% | 43.25% | 37.50% | 28.50% |
| | CEDAR-GRPO | **72.25%** | **88.80%** | **88.50%** | **34.25%** | **48.40%** | **33.59%** | **55.60%** | **40.50%** | **44.75%** | **37.75%** | **29.00%** |
| Qwen3-8B | Base | 72.25% | 84.40% | 82.50% | 39.50% | 23.20% | 26.52% | 55.20% | 36.25% | 40.00% | 42.25% | 28.75% |
| | Acc-GRPO | 74.00% | 86.00% | 87.75% | 45.50% | 43.60% | 34.13% | 58.80% | 39.50% | 44.50% | 46.25% | 29.75% |
| | CEDAR-GRPO | **75.50%** | **89.00%** | **90.50%** | **47.00%** | **46.80%** | **37.19%** | **60.80%** | **43.75%** | **50.50%** | **49.25%** | **31.75%** |
| DeepSeek-R1-Distill-Qwen-7B | Base | 70.25% | 85.20% | 81.50% | 30.00% | 26.40% | 38.67% | 49.60% | 31.50% | 42.50% | 35.75% | 23.75% |
| | Acc-GRPO | 73.50% | 88.25% | 82.25% | 31.50% | 56.40% | 40.47% | 50.00% | 33.00% | 42.25% | 36.25% | 22.75% |
| | CEDAR-GRPO | **78.50%** | **89.25%** | **87.75%** | **35.50%** | **57.20%** | **49.83%** | **51.60%** | **35.50%** | **45.75%** | **39.50%** | **24.75%** |
| Llama-3.1-8B-Instruct | Base | 73.50% | 86.40% | 86.00% | 33.75% | 24.80% | 35.24% | 48.80% | 29.00% | 42.50% | 33.50% | 19.00% |
| | Acc-GRPO | 78.75% | 89.20% | 87.75% | 35.50% | 52.00% | 36.17% | 49.20% | 30.25% | 42.25% | 34.25% | 19.75% |
| | CEDAR-GRPO | **80.00%** | **90.40%** | **91.50%** | **39.25%** | **53.20%** | **36.91%** | **50.80%** | **32.25%** | **45.50%** | **35.75%** | **21.25%** |
