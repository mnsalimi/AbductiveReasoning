#!/bin/bash

scripts=(
    "evaluate_medqa_raw_vs_finetuned.py"
    "evaluate_gsm8k_raw_vs_finetuned.py"
    "evaluate_aime_raw_vs_finetuned.py"
    "evaluate_aimo_raw_vs_finetuned.py"
    "evaluate_art_raw_vs_finetuned.py"
    "evaluate_copa_raw_vs_finetuned_guess_effect.py"
    "evaluate_goEmotion_raw_vs_finetuned.py"
)

BASE_RESULTS_DIR="/home/msalimi/users/Nima/AbductiveReasoning/GRPO/results"

RAW_MODEL_PATH="/home/msalimi/PLLMS/unsloth-Qwen2.5-14B-Instruct-unsloth-bnb-4bit"
# RAW_MODEL_PATH="/home/moein_salimi/PLLMS/unsloth-Qwen2.5-14B-Instruct-bnb-4bit"

RUN_NAME="dt11.26.15:08_e20_unsloth_Qwen2.5_14B_Instruct_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b4"
# RUN_NAME="dt11.23.10:54_e20_unsloth_Qwen2.5_14B_Instruct_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b8"

TRAINING_DIR="$BASE_RESULTS_DIR/Training_${RUN_NAME}"
FINAL_DIR="$BASE_RESULTS_DIR/${RUN_NAME}"

if [ -d "$TRAINING_DIR/checkpoint" ]; then
    CHECKPOINT_DIR="$TRAINING_DIR/checkpoint"
    TRAINING_BASE="$TRAINING_DIR"
elif [ -d "$FINAL_DIR/checkpoint" ]; then
    CHECKPOINT_DIR="$FINAL_DIR/checkpoint"
    TRAINING_BASE="$FINAL_DIR"
else
    echo "ERROR: Could not find checkpoint directory."
    echo "Tried:"
    echo "  $TRAINING_DIR/checkpoint"
    echo "  $FINAL_DIR/checkpoint"
    exit 1
fi

echo "Using checkpoint directory: $CHECKPOINT_DIR"
echo

COMMON_ARGS="--cuda_device 0 --evaluate_checkpoints 1"

declare -A BATCH_SIZES=(
    ["evaluate_medqa_raw_vs_finetuned.py"]=64
    ["evaluate_gsm8k_raw_vs_finetuned.py"]=128
    ["evaluate_aime_raw_vs_finetuned.py"]=256
    ["evaluate_aimo_raw_vs_finetuned.py"]=64
    ["evaluate_art_raw_vs_finetuned.py"]=256
    ["evaluate_copa_raw_vs_finetuned_guess_effect.py"]=256
    ["evaluate_goEmotion_raw_vs_finetuned.py"]=128
)

export TRAINING_BASE

for ckpt_name in $(ls -1 "$CHECKPOINT_DIR" | grep '^checkpoint-' | sort -t- -k2,2n); do
    ckpt="$CHECKPOINT_DIR/$ckpt_name"
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
            --checkpoint_path "$ckpt" \
            --run "$RUN_NAME"
            --raw_path "$RAW_MODEL_PATH"

        echo "Finished $script"
        echo "-------------------------------------"
        python3 GRPO/Evaluation/create_table.py \
            --root "./GRPO/Evaluation/" \
            --out_csv "./GRPO/Evaluation//metrics_summary.xlsx" \
            --run "$RUN_NAME" \
            --base_model_name "qwen2.5-14B"
    done
done
