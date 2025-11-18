#!/bin/bash

scripts=(
    "evaluate_gsm8k_raw_vs_finetuned.py"
    "evaluate_aime_raw_vs_finetuned.py"
    "evaluate_aimo_raw_vs_finetuned.py"
    "evaluate_art_raw_vs_finetuned.py"
    "evaluate_copa_raw_vs_finetuned_guess_cause.py"
    "evaluate_copa_raw_vs_finetuned_guess_effect.py"
    "evaluate_copa_raw_vs_finetuned.py"
    "evaluate_goEmotion_raw_vs_finetuned.py"
)

CHECKPOINT_DIR="/home/moein_salimi/users/Nima/AbductiveReasoning/GRPO/results/dt11.10.16:42_e20_unsloth_Qwen2.5_3B_Instruct_unsloth_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b16/checkpoint"

COMMON_ARGS="--batch_size 256 --cuda_device 1 --evaluate_checkpoints 1"

for ckpt in "$CHECKPOINT_DIR"/checkpoint-*; do
    [ -d "$ckpt" ] || continue

    echo "====================================="
    echo "Using checkpoint: $ckpt"
    echo "====================================="

    for script in "${scripts[@]}"; do
        echo "Running $script with checkpoint $ckpt ..."
        python3 GRPO/Evaluation/"$script" $COMMON_ARGS --checkpoint_path "$ckpt"
        echo "Finished $script"
        echo "-------------------------------------"
    done
done
