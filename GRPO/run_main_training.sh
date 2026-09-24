#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOK="$SCRIPT_DIR/train_abductive_new.ipynb"
EVAL_SCRIPT="$SCRIPT_DIR/Evaluation/run_eval_checkpoints_midtrain.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PAPERMILL_BIN="${PAPERMILL_BIN:-papermill}"
VLLM_BIN="${VLLM_BIN:-vllm}"

MODEL_KEY="qwen3-4b"
METHOD="cedar-grpo"
SEED=42
GPU_SPEC="auto"
SCREEN_NAME=""
INSIDE_SCREEN=0
ORIGINAL_ARGS=("$@")

usage() {
    cat <<'EOF'
Run one complete paper experiment: training, best-checkpoint evaluation, and table creation.

Usage:
  bash run_main_training.sh [options]

Options:
  --model NAME       qwen3-4b (default), qwen3-8b,
                     deepseek-r1-distill-qwen-7b, or llama-3.1-8b-instruct
  --method NAME      cedar-grpo (default) or cor-grpo
  --seed INTEGER     PyTorch/NumPy seed (default: 42)
  --gpu ID|auto      Physical GPU index, or GPU with most free memory (default: auto)
  --screen NAME      Start this run in a detached GNU screen session
  -h, --help         Show this help

Examples:
  bash run_main_training.sh --model qwen3-4b --method cedar-grpo --seed 42
  bash run_main_training.sh --model qwen3-8b --method cor-grpo --gpu 1 --screen cor_q8_s42

Environment overrides for the local judge:
  JUDGE_MODEL, JUDGE_GPU_MEMORY_UTILIZATION, JUDGE_MAX_MODEL_LEN,
  JUDGE_MAX_NUM_SEQS, JUDGE_MAX_BATCHED_TOKENS, JUDGE_PORT
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --model) [[ $# -ge 2 ]] || die "--model requires a value"; MODEL_KEY="$2"; shift 2 ;;
        --method) [[ $# -ge 2 ]] || die "--method requires a value"; METHOD="$2"; shift 2 ;;
        --seed) [[ $# -ge 2 ]] || die "--seed requires a value"; SEED="$2"; shift 2 ;;
        --gpu) [[ $# -ge 2 ]] || die "--gpu requires a value"; GPU_SPEC="$2"; shift 2 ;;
        --screen) [[ $# -ge 2 ]] || die "--screen requires a value"; SCREEN_NAME="$2"; shift 2 ;;
        --inside-screen) INSIDE_SCREEN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (use --help)" ;;
    esac
done

[[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"
[[ "$METHOD" == "cor-grpo" || "$METHOD" == "cedar-grpo" ]] || \
    die "--method must be cor-grpo or cedar-grpo"

case "$MODEL_KEY" in
    qwen3-4b)
        MODEL_ID="Qwen/Qwen3-4B"
        MODEL_LABEL="Qwen3-4B"
        ;;
    qwen3-8b)
        MODEL_ID="Qwen/Qwen3-8B"
        MODEL_LABEL="Qwen3-8B"
        ;;
    deepseek-r1-distill-qwen-7b)
        MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        MODEL_LABEL="DeepSeek-R1-Distill-Qwen-7B"
        ;;
    llama-3.1-8b-instruct)
        MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
        MODEL_LABEL="Llama-3.1-8B-Instruct"
        ;;
    *) die "Unsupported model '$MODEL_KEY' (use --help)" ;;
esac

if [[ -n "$SCREEN_NAME" && "$INSIDE_SCREEN" -eq 0 ]]; then
    [[ "$SCREEN_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || \
        die "--screen may contain only letters, digits, dot, underscore, and hyphen"
    command -v screen >/dev/null 2>&1 || die "GNU screen is not installed"
    SCREEN_LOG="$SCRIPT_DIR/screen_${SCREEN_NAME}.log"
    screen -L -Logfile "$SCREEN_LOG" -dmS "$SCREEN_NAME" \
        bash "$SCRIPT_DIR/run_main_training.sh" "${ORIGINAL_ARGS[@]}" --inside-screen
    echo "Started detached screen session: $SCREEN_NAME"
    echo "Attach with: screen -r $SCREEN_NAME"
    echo "Screen log: $SCREEN_LOG"
    exit 0
fi

for required in nvidia-smi curl bash "$PYTHON_BIN" "$PAPERMILL_BIN"; do
    command -v "$required" >/dev/null 2>&1 || die "Required command not found: $required"
done
[[ -f "$NOTEBOOK" ]] || die "Notebook not found: $NOTEBOOK"
[[ -f "$EVAL_SCRIPT" ]] || die "Evaluation script not found: $EVAL_SCRIPT"

if [[ "$GPU_SPEC" == "auto" ]]; then
    GPU_SPEC="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -n1 | cut -d, -f1 | tr -d ' ')"
fi
[[ "$GPU_SPEC" =~ ^[0-9]+$ ]] || die "Could not resolve a valid GPU index"
export CUDA_VISIBLE_DEVICES="$GPU_SPEC"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_COUNT="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
[[ "$GPU_COUNT" == "1" ]] || die "Expected exactly one visible GPU, found $GPU_COUNT"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${METHOD}_${MODEL_KEY}_seed${SEED}_${RUN_STAMP}"
RUN_DIR="$SCRIPT_DIR/results/$RUN_NAME"
EVAL_DIR="$SCRIPT_DIR/Evaluation/$RUN_NAME"
mkdir -p "$RUN_DIR" "$EVAL_DIR"

JUDGE_PID=""
JUDGE_PORT="${JUDGE_PORT:-8000}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
JUDGE_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1"

stop_judge() {
    if [[ -n "$JUDGE_PID" ]] && kill -0 "$JUDGE_PID" 2>/dev/null; then
        echo "Stopping local judge (PID $JUDGE_PID)..."
        kill "$JUDGE_PID" 2>/dev/null || true
        wait "$JUDGE_PID" 2>/dev/null || true
    fi
    JUDGE_PID=""
}
trap stop_judge EXIT INT TERM

start_judge() {
    command -v "$VLLM_BIN" >/dev/null 2>&1 || die "Required command not found: $VLLM_BIN"
    local judge_log="$RUN_DIR/judge_server.log"
    local memory_util="${JUDGE_GPU_MEMORY_UTILIZATION:-0.60}"
    local max_model_len="${JUDGE_MAX_MODEL_LEN:-8192}"
    local max_num_seqs="${JUDGE_MAX_NUM_SEQS:-8}"
    local max_batched_tokens="${JUDGE_MAX_BATCHED_TOKENS:-8192}"

    echo "Starting local $JUDGE_MODEL judge on logical GPU 0..."
    "$VLLM_BIN" serve "$JUDGE_MODEL" \
        --host 127.0.0.1 \
        --port "$JUDGE_PORT" \
        --served-model-name "$JUDGE_MODEL" \
        --tensor-parallel-size 1 \
        --dtype auto \
        --gpu-memory-utilization "$memory_util" \
        --max-model-len "$max_model_len" \
        --max-num-seqs "$max_num_seqs" \
        --max-num-batched-tokens "$max_batched_tokens" \
        --enforce-eager >"$judge_log" 2>&1 &
    JUDGE_PID=$!

    local waited=0
    local timeout="${JUDGE_STARTUP_TIMEOUT:-1800}"
    until curl --silent --fail "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null; do
        kill -0 "$JUDGE_PID" 2>/dev/null || {
            tail -n 100 "$judge_log" >&2 || true
            die "Local judge exited during startup"
        }
        ((waited >= timeout)) && die "Judge was not healthy after ${timeout}s; see $judge_log"
        sleep 5
        waited=$((waited + 5))
    done
    echo "Local judge is ready (startup ${waited}s)."

    "$PYTHON_BIN" - "$JUDGE_API_BASE" "$JUDGE_MODEL" <<'PY'
import sys
from openai import OpenAI

client = OpenAI(api_key="local", base_url=sys.argv[1])
response = client.chat.completions.create(
    model=sys.argv[2],
    messages=[{"role": "user", "content": "Reply with only the word READY."}],
    temperature=0.0,
    max_tokens=512,
    timeout=180,
)
content = response.choices[0].message.content
if not content or "READY" not in content.upper():
    raise SystemExit(f"Judge smoke test returned no usable final content: {content!r}")
print("Local judge chat-completions smoke test passed.")
PY
}

echo "============================================================"
echo "Run:        $RUN_NAME"
echo "Model:      $MODEL_ID"
echo "Method:     $METHOD"
echo "Seed:       $SEED (PyTorch and NumPy; pipeline/LoRA seed remains 3407)"
echo "GPU:        physical $GPU_SPEC (exposed to the run as logical GPU 0)"
echo "Epochs:     5"
echo "Results:    $RUN_DIR"
echo "============================================================"

if [[ "$METHOD" == "cedar-grpo" ]]; then
    start_judge
else
    echo "Cor-GRPO selected; no LLM judge server is needed."
fi

cd "$SCRIPT_DIR"
echo "Starting notebook training..."
"$PAPERMILL_BIN" "$NOTEBOOK" "$RUN_DIR/executed_notebook.ipynb" \
    --cwd "$SCRIPT_DIR" \
    -p MODEL_NAME "$MODEL_ID" \
    -p TRAINING_VARIANT "$METHOD" \
    -p RUN_NAME_OVERRIDE "$RUN_NAME" \
    -p NUM_TRAIN_EPOCHS 5 \
    -p TORCH_SEED "$SEED" \
    -p NUMPY_SEED "$SEED" \
    -p CUDA_VISIBLE_DEVICES "0" \
    -p LLM_JUDGE_API_KEY "local" \
    -p LLM_JUDGE_API_BASE "$JUDGE_API_BASE" \
    -p LLM_JUDGE_MODEL "$JUDGE_MODEL" \
    2>&1 | tee "$RUN_DIR/training_console.log"

# Evaluation needs the GPU memory held by gpt-oss-120b, so release it first.
stop_judge

BEST_CHECKPOINT="$($PYTHON_BIN - "$RUN_DIR" <<'PY'
import json
import pathlib
import re
import sys

run_dir = pathlib.Path(sys.argv[1])
checkpoint_dir = run_dir / "checkpoint"
saved = {
    int(match.group(1)): path.name
    for path in checkpoint_dir.iterdir()
    if path.is_dir() and (match := re.fullmatch(r"checkpoint-(\d+)", path.name))
}
if not saved:
    raise SystemExit("No checkpoint-* directory was produced")

metrics_path = run_dir / "val_metrics.json"
candidates = []
if metrics_path.exists():
    validation = json.loads(metrics_path.read_text(encoding="utf-8"))
    for step, name in saved.items():
        entry = validation.get(str(step), {})
        if isinstance(entry, dict) and "avg_reward" in entry:
            candidates.append((float(entry["avg_reward"]), step, name))

if candidates:
    print(max(candidates)[2])
else:
    print(saved[max(saved)])
PY
)"
[[ "$BEST_CHECKPOINT" =~ ^checkpoint-[0-9]+$ ]] || \
    die "Could not select a best checkpoint (got '$BEST_CHECKPOINT')"
echo "Selected best validation checkpoint: $BEST_CHECKPOINT"

echo "Starting held-out evaluation of the base model and best checkpoint..."
export OUTPUT_DIR="$EVAL_DIR"
export ROOT_DIR="$SCRIPT_DIR/Evaluation"
export BASE_RESULTS_DIR="$SCRIPT_DIR/results"
export RAW_MODEL_PATH="$MODEL_ID"
export RUN_NAME
export CHKPT_NAME="$BEST_CHECKPOINT"
export BASE_MODEL_NAME="$MODEL_LABEL"
export TRAIN_DATA="mixed"
export CUDA_DEVICE=0
export EVALUATE_CHECKPOINTS=1
export PYTHON_BIN

bash "$EVAL_SCRIPT" 2>&1 | tee "$RUN_DIR/evaluation_console.log"

TABLE_PATH="$EVAL_DIR/metrics_summary.xlsx"
[[ -f "$TABLE_PATH" ]] || die "Evaluation finished but the table was not created: $TABLE_PATH"

echo "============================================================"
echo "COMPLETE"
echo "Run directory:       $RUN_DIR"
echo "Best checkpoint:     $RUN_DIR/checkpoint/$BEST_CHECKPOINT"
echo "Evaluation results:  $EVAL_DIR"
echo "Two-row table:       $TABLE_PATH"
echo "============================================================"
