#!/bin/bash

scripts=(
    # "evaluate_medqa_raw_vs_finetuned.py"
    # "evaluate_gsm8k_raw_vs_finetuned.py"
    # "evaluate_aime_raw_vs_finetuned.py"
    # "evaluate_aimo_raw_vs_finetuned.py"
    # "evaluate_art_raw_vs_finetuned.py"
    "evaluate_copa_raw_vs_finetuned_guess_cause.py"
    "evaluate_uniadilr_raw_vs_finetuned.py"
    # "evaluate_goEmotion_raw_vs_finetuned.py"
)

raw_paths=(
    "/home/msalimi/PLLMS/unsloth-Qwen2.5-14B-Instruct-unsloth-bnb-4bit"
    "/home/msalimi/PLLMS/unsloth-Qwen2.5-3B-Instruct-unsloth-bnb-4bit"
)

RUN_PREFIX="Dec1"

OUTPUT_PATH="/home/msalimi/sahand/AbductiveReasoning/Evaluation/raw_model_evaluation"

COMMON_ARGS="--cuda_device 0 --skip_finetuned"

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

script_name_color='\033[0;32m' # Green
model_info_color='\033[0;33m'  # Yellow
batch_info_color='\033[0;34m'  # Blue
no_color='\033[0m'             # Reset color

for script in "${scripts[@]}"; do
    for raw_path in "${raw_paths[@]}"; do
        batch_size="${BATCH_SIZES[$script]:-256}"
        model_name="${raw_path##*/}"   # Extract the model name
        RUN_NAME="${RUN_PREFIX}_${model_name}"
        echo -e "Running ${script_name_color}$script${no_color} with on model ${model_info_color}$model_name ${batch_info_color}(batch_size=$batch_size)${no_color}..."
        echo "python3 GRPO/Evaluation/\"$script\" \\
            $COMMON_ARGS \\
            --batch_size \"$batch_size\" \\
            --run \"$RUN_NAME\" \\
            --output_path \"$OUTPUT_PATH\" \\
            --raw_path \"$raw_path\""
    done
done