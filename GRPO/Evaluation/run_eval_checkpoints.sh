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

BASE_RESULTS_DIR="/home/moein_salimi/users/Nima/AbductiveReasoning/GRPO/results"

RUN_NAME="dt11.15.23:13_e20_unsloth_Qwen2.5_3B_Instruct_unsloth_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b16"

TRAINING_DIR="$BASE_RESULTS_DIR/Training_${RUN_NAME}"
FINAL_DIR="$BASE_RESULTS_DIR/${RUN_NAME}"

if [ -d "$TRAINING_DIR/checkpoint" ]; then
    CHECKPOINT_DIR="$TRAINING_DIR/checkpoint"
elif [ -d "$FINAL_DIR/checkpoint" ]; then
    CHECKPOINT_DIR="$FINAL_DIR/checkpoint"
else
    echo "ERROR: Could not find checkpoint directory."
    echo "Tried:"
    echo "  $TRAINING_DIR/checkpoint"
    echo "  $FINAL_DIR/checkpoint"
    exit 1
fi

echo "Using checkpoint directory: $CHECKPOINT_DIR"
echo

COMMON_ARGS="--cuda_device 1 --evaluate_checkpoints 1"

declare -A BATCH_SIZES=(
    ["evaluate_gsm8k_raw_vs_finetuned.py"]=256
    ["evaluate_aime_raw_vs_finetuned.py"]=256
    ["evaluate_aimo_raw_vs_finetuned.py"]=64
    ["evaluate_art_raw_vs_finetuned.py"]=256
    ["evaluate_copa_raw_vs_finetuned_guess_cause.py"]=256
    ["evaluate_copa_raw_vs_finetuned_guess_effect.py"]=256
    ["evaluate_copa_raw_vs_finetuned.py"]=256
    ["evaluate_goEmotion_raw_vs_finetuned.py"]=128
)

for ckpt in "$CHECKPOINT_DIR"/checkpoint-*; do
    [ -d "$ckpt" ] || continue

    echo "====================================="
    echo "Using checkpoint: $ckpt"
    echo "====================================="

    for script in "${scripts[@]}"; do
        batch_size="${BATCH_SIZES[$script]:-256}"

        echo "Running $script with checkpoint $ckpt (batch_size=$batch_size) ..."
        python3 GRPO/Evaluation/"$script" \
            $COMMON_ARGS \
            --batch_size "$batch_size" \
            --checkpoint_path "$ckpt"

        echo "Finished $script"
        echo "-------------------------------------"
    done
    python3 GRPO/Evaluation/create_table.py \
        --root "./GRPO/Evaluation/" \
        --out_csv "./GRPO/Evaluation//metrics_summary.xlsx"
done
