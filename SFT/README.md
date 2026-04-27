# SFT Training Pipeline (Abductive Reasoning)

This folder shares the GRPO pipeline assets (**model family, datasets, and evaluation stack**) but replaces RL/GRPO training with **Supervised Fine-Tuning (SFT)**.

## What is included

- `train_abductive_sft.py`
  - New SFT training pipeline.
  - Keeps the same default model setup used in GRPO (`unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit`).
  - Uses the same prompt-construction logic per dataset.
  - Reads data directly from `GRPO/dataset/` (no local copy).
  - Imports evaluation helpers directly from `GRPO/Evaluation/` (no local copy).

## Folder structure

```text
SFT/
├── train_abductive_sft.py   # SFT training script
├── README.md
└── visualize.ipynb          # Training curve / log visualisation

GRPO/                        # shared assets (read by SFT, not copied)
├── dataset/
│   ├── train_split.json
│   ├── val_split.json
│   └── ...
└── Evaluation/
    ├── evaluate_all.py
    ├── evaluate_*_raw_vs_finetuned.py
    └── ...
```

## Training method change

- **Before (GRPO):** policy optimization with rewards (`GRPOTrainer`).
- **Now (SFT):** supervised next-token training (`SFTTrainer`) on the same transformed prompt data.

In SFT mode, each sample is converted into a chat transcript:

1. system prompt
2. user prompt
3. assistant target structured as:

```
<think>
{rationale or placeholder sentence}
</think>
<answer>{ground_truth}</answer>
```

If the dataset provides a `rationale`, `explanation`, or `proof_text` field it is used verbatim inside `<think>`; otherwise a minimal placeholder is generated so the model learns the tag structure while still being trained on the correct answer.

This keeps output formatting fully aligned with your evaluation scripts.

## Default configuration (same core settings as GRPO)

- Model: `unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit`
- Quantization: 4-bit
- LoRA rank/alpha: 64 / 64
- Max sequence length: 4096
- Learning rate: `1e-5`
- Batch size: `4`
- Gradient accumulation: `1`
- Epochs: `1` (test default, same spirit as current GRPO notebook defaults)

All key constants are at the top of `train_abductive_sft.py`.

## How to run training

From repository root:

```bash
cd /home/runner/work/SFT-abduction-training/SFT-abduction-training/SFT
python train_abductive_sft.py
```

Outputs are saved under:

- `SFT/results/<run_name>/checkpoint/`
- final model: `SFT/results/<run_name>/checkpoint/final_model`

## How to run `visualize.ipynb` for SFT results

From repository root:

```bash
cd /home/runner/work/SFT-abduction-training/SFT-abduction-training/SFT
jupyter notebook visualize.ipynb
```

Then in the notebook:

1. Run all cells in order.
2. Keep the default call `generate_all_runs_in_results()` so it scans `SFT/results/`.
3. Confirm your run folders contain `training_log.json` (and optionally `validation_log.json`).

Generated plots are saved to:

- `SFT/results/<run_name>/plots/`

## How to run evaluation

The script uses `GRPO/Evaluation/` directly. You can run the same evaluation workflow as before, pointing to your SFT checkpoint directory.

Typical flow:

1. Train with `train_abductive_sft.py`
2. Use your existing `Evaluation` scripts/orchestrator in `GRPO/Evaluation/` with:
   - raw model path
   - SFT training/checkpoint path

## Notes

- `SFT/` is intentionally lightweight — it contains only the training script and notebook.
- Dataset splits and evaluation scripts are shared with GRPO; no duplication.
- Any changes to `GRPO/dataset/` or `GRPO/Evaluation/` are automatically picked up by SFT training.
