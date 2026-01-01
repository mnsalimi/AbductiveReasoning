#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime

# Model Configuration
# Choose ONE of the following options:

# Option 1: 3B model from HuggingFace (RECOMMENDED - no compatibility issues)
# MODEL_NAME = 'unsloth/Qwen2.5-3B-Instruct'

# Option 2: 7B model from HuggingFace (larger, slower, needs more VRAM)
# MODEL_NAME = 'unsloth/Qwen2.5-7B-Instruct'

# Option 3: Local 3B model (if you have it downloaded)
# MODEL_NAME = '/home/moein_salimi/PLLMS/Qwen3-4B-unsloth-bnb-4bit'
# MODEL_NAME = '/home/moein_salimi/users/Nima/AbductiveReasoning/GRPO/results/dt11.15.23:13_e20_unsloth_Qwen2.5_3B_Instruct_unsloth_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b16/checkpoint-1792'
# MODEL_NAME = '/PLLMShome/moein_salimi//unsloth-Qwen2.5-3B-Instrurct'
MODEL_NAME = '/home/moein_salimi/PLLMS/unsloth-Qwen2.5-14B-Instruct-bnb-4bit'
# MODEL_NAME = 'Qwen/Qwen3-4B-Thinking-2507'

# Option 4: Local 7B model (currently causing error)
# MODEL_NAME = '/home/moein_salimi/PLLMS/unsloth-Qwen2.5-7B-Instruct-bnb-4bit'
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
USE_VLLM = False
LORA_RANK = 64
LORA_ALPHA = 64
GPU_MEMORY_UTILIZATION = 1.0
MAX_SEQ_LENGTH = 4096
MAX_PROMPT_LENGTH = 2048
MAX_COMPLETION_LENGTH = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH

RESUME_FROM_CHECKPOINT = False
PREVIOUS_RUN_DIR = 'dt11.15.23:13_e20_unsloth_Qwen2.5_3B_Instruct_unsloth_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b16'

RUN_DESC = "test_run"
CUDA_VISIBLE_DEVICES = "0"

# Training Configuration
LEARNING_RATE = 1e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 2
LR_SCHEDULER_TYPE = "cosine"
OPTIM = "adamw_torch"
EPSILON = 0.2
BETA = 0.01

# Validation Configuration
EVAL_STEPS = 1  # Evaluate on validation set every N steps
SAVE_STEPS = 10  # Save checkpoint every N steps
LOG_VALIDATION = True  # Whether to log validation metrics
LOG_TRAIN_EVERY = 1  # Save training log every N completions (not every step)

# Training Loop Settings
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
NUM_GENERATIONS = 4 # Reduced for testing
MAX_GRAD_NORM = 0.1
TEMPERATURE = 0.7
NUM_TRAIN_EPOCHS = 1 # Reduced for testing

# Data Configuration
NUM_SAMPLES = 20  # Reduced for testing None for full dataset
TRAIN_SPLIT = 0.8
DATA_PATH = "dataset/abduction.jsonl"
ERROR_LOG_PATH = "error_log.log"
TRAINING_LOG_PATH = "training_log.json"
VALIDATION_LOG_PATH = "validation_log.json"
VALIDATION_METRICS_PATH = "val_metrics.json"

# List of datasets to exclude from training/validation
# Options: 'UniADILR', 'balanced_copa_cause_only', 'causelogics_level3&4', 'list_function', 'miniarc', 'climate_fever'
# EXCLUDED_DATASETS = ['UniADILR', 'balanced_copa_cause_only', 'causelogics_level3&4', 'climate_fever']
EXCLUDED_DATASETS = []

TRAIN_DATA_VAL = 'v1'

# System Prompt for Abductive Reasoning
SYSTEM_PROMPT_UniADILR = """
You are an expert in logical reasoning and abductive inference. Your task is to identify which sentences from a given context provide the necessary evidence to support or explain a hypothesis.

You will be provided with:
1. A Context containing multiple numbered sentences (sent1, sent2, sent3, etc.)
2. A Hypothesis that needs to be supported or explained

Your goal is to identify which sentence(s) from the context, when combined, provide the logical foundation for the hypothesis through abductive reasoning.

## Instructions:
1. Carefully read all sentences in the context
2. Analyze the hypothesis
3. Identify which sentences, when combined, best explain or support the hypothesis
4. Consider both direct evidence and logical connections

## Output Format:
You MUST provide your answer in the following format:

<think>
[Explain your thought process: why you selected these particular sentences and how they support the hypothesis]
</think>

<answer>
[Sentence numbers only, comma-separated. For example: 5, 13 or 2, 7, 9]
</answer>

CRITICAL: The answer section must contain ONLY the sentence numbers separated by commas. Do not include the word "sent" or any other text.
""".strip()


SYSTEM_PROMPT_balanced_copa_cause_only = """
You are an expert in logical reasoning and abductive inference. Your task is to determine which of two given choices represents the most plausible cause for a given premise.

You will be provided with:
1. A Premise describing a situation or event
2. Two Choices (Choice 1 and Choice 2)

Your goal is to select the choice that best explains WHY the premise happened - identifying the root cause that led to the described situation.

## Instructions:
1. Carefully read the premise
2. Evaluate both choices as potential causes
3. Consider common sense, real-world knowledge, and typical causal relationships when making your decision
4. Select the choice that represents the most plausible and direct cause

## Output Format:
You MUST provide your answer in the following format:

<think>
[Explain your thought process: why we should select one choice over the other or analyzing the cause or their relationships]
</think>

<answer>
[Either "1" or "2" - just the number, nothing else]
</answer>

CRITICAL: The answer section must contain ONLY the number 1 or 2. Do not include any other text, explanation, or punctuation.
""".strip()

# Random State Configuration
RANDOM_STATE = 3407
TORCH_SEED = 42
NUMPY_SEED = 42

# Environment Configuration
WANDB_DISABLED = "true"

#=======================================================================

# Output Configuration
def get_run_name():
    """Generate run name based on configuration"""
    model_name = MODEL_NAME.split("/")[-1].replace("-", "_")
    if LOAD_IN_8BIT:
        model_name += "_8bit"
    elif LOAD_IN_4BIT:
        model_name += "_bnb_4bit"
    now = datetime.now()
    name = f"dt{now.strftime('%m.%d.%H:%M')}_e{NUM_TRAIN_EPOCHS}_{model_name}_lr{LEARNING_RATE}_t{TEMPERATURE}_ε{EPSILON}_r{LORA_RANK}_b{PER_DEVICE_TRAIN_BATCH_SIZE}"
    if RUN_DESC:
        name += f"_{RUN_DESC}"
    return name

def get_results_dir(run_name=None):
    """Get results directory path"""
    if run_name is None:
        run_name = get_run_name()
    if RESUME_FROM_CHECKPOINT:
        run_name = PREVIOUS_RUN_DIR
    return f"results/{run_name}"


# In[ ]:


# Environment setup and configuration
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import torch

# Add current directory to path for imports
sys.path.append('.')

# Set random seeds for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(NUMPY_SEED)
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)

print(f"🎲 Random seeds set:")
print(f"   Python: {RANDOM_STATE}")
print(f"   NumPy: {NUMPY_SEED}")
print(f"   PyTorch: {TORCH_SEED}")

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["WANDB_DISABLED"] = WANDB_DISABLED

print("\n🔧 Abductive Reasoning Training Pipeline")
print("=" * 50)
print(f"Configuration loaded:")
print(f"  📦 Model: {MODEL_NAME}")
print(f"  🎯 Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
print(f"  📄 Samples: {NUM_SAMPLES}")
print(f"  🏃 Epochs: {NUM_TRAIN_EPOCHS}")
print(f"  📈 Learning rate: {LEARNING_RATE}")
print(f"  🌡️  Temperature: {TEMPERATURE}")
print(f"  🎮 GPU: {CUDA_VISIBLE_DEVICES}")


# In[ ]:


# Import required libraries
import torch
import json
import re
import time
from datasets import Dataset
from unsloth import FastLanguageModel
# import vllm
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
import matplotlib.pyplot as plt

print("🔍 System Check:")
print("=" * 30)

# Check GPU setup
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"Number of visible GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"✅ GPU Available")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ No GPU available!")
    
print(f"✅ PyTorch version: {torch.__version__}")


# In[ ]:


import json
from datasets import Dataset
from Evaluation.evaluate_uniadilr_raw_vs_finetuned import create_uniadilr_prompt
from Evaluation.evaluate_copa_raw_vs_finetuned_guess_cause import create_copa_prompt
from Evaluation.evaluate_climate_fever_raw_vs_finetuned import create_climate_fever_prompt
from Evaluation.evaluate_causelogics_raw_vs_finetuned import create_causelogics_prompt
from Evaluation.evaluate_list_function_raw_vs_finetuned import create_list_functions_prompt
from Evaluation.evaluate_miniarc_raw_vs_finetuned import create_acr_prompt

print("\n📂 Loading Pre-Split Data and Transforming")
print("=" * 40)

# Load the raw splits from JSON files
print("Loading train split...")
with open('./dataset/train_split.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print("Loading validation split...")
with open('./dataset/val_split.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)

# print("Loading test split...")
# with open('./dataset/test_split.json', 'r', encoding='utf-8') as f:
#     test_data = json.load(f)



def transform_to_prompt_format(example, record_id):
    """
    Transform the original JSONL format to the required prompt format.
    Handles both UniADILR and balanced_copa_cause_only datasets.
    """
    dataset_name = example.get('datasetName', '')
    rule_test_input = None
    rule_train_examples = None
    
    if dataset_name == 'UniADILR':
        # Create the system prompt and user prompt for UniADILR
        system_prompt, user_prompt = create_uniadilr_prompt(example)
        
        # Create the prompt structure
        prompt = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        ground_truth = json.dumps(example['proof'])
        
    elif dataset_name == 'balanced_copa_cause_only':
        # Create the system prompt and user prompt for COPA
        system_prompt, user_prompt = create_copa_prompt(example)
        
        # Create the prompt structure
        prompt = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        # For COPA, the ground truth is the label (1 or 2  [it is originally 0 or 1 but since I told the model in system prompt to give either 1 or 2 I made it 1 or 2])
        ground_truth = str(example['label'] + 1)

    elif dataset_name == 'causelogics_level3&4':
        # Create the system prompt and user prompt for CauseLogics
        system_prompt, user_prompt = create_causelogics_prompt(example)

        # Create the prompt structure
        prompt = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # Ground truth: normalize "True"/"False" → "TRUE"/"FALSE"
        ground_truth = example["Label"].upper()

    elif dataset_name == 'list_function':
        # Create the system prompt and user prompt for list function
        system_prompt, user_prompt = create_list_functions_prompt(example)

        # Create the prompt structure
        prompt = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # Ground truth / inputs: store as JSON strings to keep Arrow column types consistent
        raw_output = example['test'][0]['output']
        output_obj = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        ground_truth = json.dumps(output_obj, ensure_ascii=False)
        
        raw_input = example['test'][0]['input']
        input_obj = json.loads(raw_input) if isinstance(raw_input, str) else raw_input
        rule_test_input = json.dumps(input_obj, ensure_ascii=False)
        
        # Parse training examples as well (store as JSON string)
        rule_train_examples = []
        for ex in example['train']:
            ex_in = json.loads(ex['input']) if isinstance(ex['input'], str) else ex['input']
            ex_out = json.loads(ex['output']) if isinstance(ex['output'], str) else ex['output']
            rule_train_examples.append({'input': ex_in, 'output': ex_out})
        rule_train_examples = json.dumps(rule_train_examples, ensure_ascii=False)

    elif dataset_name == 'miniarc':
        # Create the system prompt and user prompt for miniarc (acr)
        system_prompt, user_prompt = create_acr_prompt(example)

        # Create the prompt structure
        prompt = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        # Ground truth / inputs: store as JSON strings to keep Arrow column types consistent
        ground_truth = json.dumps(example['test'][0]['output'], ensure_ascii=False)
        rule_test_input = json.dumps(example['test'][0]['input'], ensure_ascii=False)
        rule_train_examples = json.dumps(example['train'], ensure_ascii=False)

    elif dataset_name == 'climate_fever':
        # Create the system prompt and user prompt for climate fever
        system_prompt, user_prompt = create_climate_fever_prompt(example)

        # Create the prompt structure
        prompt = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        # ground truth: map the label 0-3 to the proper word
        LABEL_MAP = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO", 3: "DISPUTED"}
        ground_truth = LABEL_MAP[example["claim_label"]]

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    # Return the transformed example
    return {
        "prompt": prompt,
        "record_id": record_id,
        "ground_truth": ground_truth,
        "reasoning_type": example.get('reasoning_type', 'abduction'),
        "dataset_name": dataset_name,
        "rule_test_input": rule_test_input,
        "rule_train_examples": rule_train_examples
    }

# Transform each split
print(f"\nTransforming train data to prompt format... (Excluding: {EXCLUDED_DATASETS})")
train_transformed = []
for idx, example in enumerate(train_data):
    if example.get('datasetName') in EXCLUDED_DATASETS:
        continue
    train_transformed.append(transform_to_prompt_format(example, record_id=idx))

print(f"Transforming validation data to prompt format... (Excluding: {EXCLUDED_DATASETS})")
val_transformed = []
for idx, example in enumerate(val_data):
    if example.get('datasetName') in EXCLUDED_DATASETS:
        continue
    val_transformed.append(transform_to_prompt_format(example, record_id=idx))

# Limit dataset size for testing
if NUM_SAMPLES is not None and NUM_SAMPLES > 0:
    print(f"\n⚠️ Limiting dataset size for testing to {NUM_SAMPLES} samples")
    train_transformed = train_transformed[:NUM_SAMPLES]
    val_transformed = val_transformed[:max(1, int(NUM_SAMPLES * 0.2))]
else:
    total = len(train_transformed) + len(val_transformed)
    print(f"\n⚠️ Using full dataset: {total} samples")
    val_transformed = val_transformed[:max(1, int(total * 0.2))]


# print("Transforming test data to prompt format...")
# test_transformed = []
# for idx, example in enumerate(test_data):
#     test_transformed.append(transform_to_prompt_format(example, record_id=idx))

print(f"✅ Transformed all splits")

# Convert to HuggingFace datasets
print("\nConverting to HuggingFace datasets...")
train_ds = Dataset.from_list(train_transformed)
val_ds = Dataset.from_list(val_transformed)
# test_ds = Dataset.from_list(test_transformed)

# Display the first training example to verify format
print("\n" + "="*80)
print("🔍 FIRST TRAINING EXAMPLE (to verify system prompt)")
print("="*80)
first_example = train_ds[0]
print(f"\n📋 Example keys: {list(first_example.keys())}")
print(f"\n🆔 Record ID: {first_example.get('record_id', 'N/A')}")
print("\n💬 PROMPT STRUCTURE:")
print("-" * 80)
for i, msg in enumerate(first_example['prompt']):
    role = msg.get('role', 'unknown')
    content = msg.get('content', '')
    print(f"\n[Message {i+1}] Role: {role.upper()}")
    print("-" * 40)
    # Show first 500 characters of content to avoid overwhelming output
    if len(content) > 500:
        print(f"{content[:500]}...")
        print(f"\n... (Content truncated - total length: {len(content)} characters)")
    else:
        print(content)
    print("-" * 40)

# Log the prompt structure to a file
log_file = './prompt_structure_log.txt'
with open(log_file, 'w', encoding='utf-8') as f:
    for i, msg in enumerate(first_example['prompt']):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        f.write(f"\n[Message {i+1}] Role: {role.upper()}\n")
        f.write("-" * 40 + "\n")
        f.write(content + "\n")
        f.write("-" * 40 + "\n")

print(f"✅ Prompt structure logged to: {log_file}")


print("\n" + "="*80)



total = len(train_ds) + len(val_ds)
print(f"\n✅ Datasets loaded, transformed, and ready!")
print(f"\n📈 Dataset Statistics:")
print(f"   Total samples: {total:,}")
print(f"   Training samples: {len(train_ds):,} ({len(train_ds)/total*100:.1f}%)")
print(f"   Validation samples: {len(val_ds):,} ({len(val_ds)/total*100:.1f}%)")
# print(f"   Test samples: {len(test_ds):,} ({len(test_ds)/total*100:.0f}%)")


# In[ ]:


# Verify loaded datasets
print("\n🛠️  Verifying Loaded Datasets")
print("=" * 35)

# Calculate prompt statistics from loaded datasets
prompt_lengths = []
for ds in [train_ds, val_ds]:
    for example in ds:
        # Extract user prompt length from the prompt field
        for msg in example['prompt']:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                prompt_lengths.append(len(msg.get('content', '')))
                break

print(f"✅ Datasets ready for training!")
print(f"   Total prompts: {len(prompt_lengths):,}")
print(f"   Max prompt length: {max(prompt_lengths)} characters")
print(f"   Average prompt length: {sum(prompt_lengths)/len(prompt_lengths):.0f} characters")
print(f"\n   Sample keys in training data: {list(train_ds[0].keys())}")

# Show example of answer field
print(f"\n📋 Example answer from first training sample:")
print(f"   answer: {train_ds[0]['ground_truth']}")
print(f"   answer type: {type(train_ds[0]['ground_truth'])}")

# Show a snippet of the user prompt for context
print(f"\n📝 Example user prompt (first 200 chars):")
for msg in train_ds[0]['prompt']:
    if isinstance(msg, dict) and msg.get('role') == 'user':
        user_content = msg.get('content', '')
        print(f"   {user_content[:200]}...")
        break


# In[ ]:


from unsloth import FastLanguageModel, is_bfloat16_supported
from huggingface_hub import HfApi
import os
from tqdm.auto import tqdm
import time

start_time = time.time()

def format_bytes(bytes_value):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def get_model_size(model_name):
    """Try to get model size from HuggingFace Hub"""
    try:
        api = HfApi()
        model_info = api.model_info(model_name)
        # Sum up all file sizes
        total_size = sum(file.size for file in model_info.siblings if file.size)
        return total_size
    except:
        return None

# Configure download settings
print("🔧 Configuring Hugging Face Hub download settings...")
print("=" * 60)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "240"
print("✓ Download timeout: 240 seconds per chunk")
print("✓ Using default retry settings")
print()

# Get model size info
print("📊 Fetching model information...")
model_size = get_model_size(MODEL_NAME)
if model_size:
    print(f"✓ Model size: {format_bytes(model_size)}")
    print(f"✓ Estimated download time: ~{model_size / (10 * 1024 * 1024):.0f} seconds (at 10 MB/s)")
else:
    print("⚠ Could not determine model size")
print()

# Load model with progress tracking
print("🤖 Model Setup")
print("=" * 60)
print(f"📦 Model: {MODEL_NAME}")
print(f"🔢 Max sequence length: {MAX_SEQ_LENGTH}")
print(f"⚙️  Quantization: {'4-bit' if LOAD_IN_4BIT else '8-bit' if LOAD_IN_8BIT else 'None'}")
print(f"🚀 Fast inference (vLLM): {USE_VLLM}")
print(f"💾 GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
print()

print("⏳ Downloading and loading model...")
print("   (This may take several minutes depending on your connection)")
print()

download_start = time.time()

# Create a simple progress indicator
class ProgressCallback:
    def __init__(self):
        self.last_print = time.time()
        self.dots = 0
    
    def update(self):
        current = time.time()
        if current - self.last_print > 2:  # Print every 2 seconds
            self.dots = (self.dots + 1) % 4
            elapsed = current - download_start
            print(f"\r   Downloading{'.' * (self.dots + 1)}{' ' * (3 - self.dots)} " +
                  f"[{elapsed:.0f}s elapsed]", end='', flush=True)
            self.last_print = current

progress = ProgressCallback()

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        max_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
        fast_inference=USE_VLLM,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    print("\r" + " " * 80 + "\r", end='')  # Clear progress line
    
    download_time = time.time() - download_start
    print(f"✅ Model downloaded and loaded successfully!")
    print(f"⏱️  Total time: {download_time:.1f}s ({download_time/60:.1f} minutes)")
    
    if model_size:
        avg_speed = model_size / download_time
        print(f"📈 Average speed: {format_bytes(avg_speed)}/s")
    print()
    
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    raise

# Configure tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Configured pad token")
    print()

# Apply LoRA
print("🔧 Applying LoRA configuration...")
print("-" * 60)

lora_start = time.time()

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    use_gradient_checkpointing="unsloth",
    random_state=RANDOM_STATE
)

lora_time = time.time() - lora_start

print(f"✅ LoRA configured successfully! ({lora_time:.1f}s)")
print()

# Model statistics
print("📊 Model Statistics")
print("=" * 60)
print(f"🎯 LoRA Configuration:")
print(f"   • Rank (r): {LORA_RANK}")
print(f"   • Alpha: {LORA_ALPHA}")
print(f"   • Target modules: 7 (q, k, v, o, gate, up, down projections)")
print()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
frozen_params = total_params - trainable_params

print(f"🔢 Parameters:")
print(f"   • Total: {total_params:,}")
print(f"   • Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"   • Frozen: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
print()

total_setup_time = time.time() - start_time
print(f"⏱️  Total Setup Time: {total_setup_time:.1f}s ({total_setup_time/60:.1f} minutes)")
print(f"   • Model download/load: {download_time:.1f}s")
print(f"   • LoRA configuration: {lora_time:.1f}s")
print("=" * 60)
print("✨ Ready to train!")


# In[ ]:


import multiprocessing
import re
import json

def extract_code(response):
    """
    Extracts the Python code block from the model's response.
    Looks for code between <answer> tags and then cleans up markdown code blocks.
    """
    # Handle list input (e.g., from GRPOTrainer if it passes messages)
    if isinstance(response, list):
        if len(response) > 0 and isinstance(response[-1], dict) and 'content' in response[-1]:
            response = response[-1]['content']
        elif len(response) > 0 and isinstance(response[0], str):
            response = response[0]
        else:
            response = str(response)

    tag_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.IGNORECASE | re.DOTALL)
    if not tag_match:
        return None
    
    code = tag_match.group(1).strip()
    # Remove markdown code blocks if present
    code = re.sub(r'^```python\s*', '', code)
    code = re.sub(r'^```\s*', '', code)
    code = re.sub(r'\s*```$', '', code)
    
    return code

def string_to_list(s):
    """
    Safely converts a string representation of a 1D list of integers
    (e.g., "[1, 2, 3]") into a Python list ([1, 2, 3]).
    Returns None on failure.
    """
    if not s:
        return None
    try:
        # json.loads is safer than eval()
        parsed_list = json.loads(s)

        # Basic validation: ensure it's a list of ints
        if isinstance(parsed_list, list) and all(isinstance(val, int) for val in parsed_list):
            return parsed_list
        else:
            return None
    except:
        return None

def _execute_worker(code_str, input_data, queue, dataset_name):
    """Worker function for multiprocessing execution to handle timeouts."""
    local_namespace = {}
    # Add numpy if needed (common for grid tasks)
    try:
        import numpy as np
        local_namespace['np'] = np
    except ImportError:
        pass

    try:
        # Execute the code to define the 'transform' function
        # Use a clean globals dict but allow builtins
        exec(code_str, {"__builtins__": __builtins__, "np": local_namespace.get('np')}, local_namespace)

        if 'transform' not in local_namespace:
            queue.put((None, "Error: Function 'transform' not found in generated code."))
            return

        result = local_namespace['transform'](input_data)

        # Basic validation based on dataset
        if dataset_name == 'miniarc':
            if not isinstance(result, list):
                queue.put((None, f"Error: MiniARC expected list, got {type(result).__name__}"))
                return
            # Check if it's a 2D list of ints
            if not all(isinstance(row, list) and all(isinstance(val, int) for val in row) for row in result):
                queue.put((None, "Error: MiniARC expected 2D list of integers"))
                return
        elif dataset_name == 'list_function':
            if not isinstance(result, list):
                queue.put((None, f"Error: List Function expected list, got {type(result).__name__}"))
                return
            if not all(isinstance(x, int) for x in result):
                queue.put((None, "Error: List Function expected list of integers"))
                return

        queue.put((result, None))

    except Exception as e:
        queue.put((None, f"Execution Error: {str(e)}"))

def execute_transform(code_str, input_data, dataset_name, timeout=5):
    """
    Executes the model-generated code and runs the 'transform' function with a timeout.
    """
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_execute_worker, args=(code_str, input_data, queue, dataset_name))
    p.start()

    # Wait for the process to finish or timeout
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None, f"Timeout Error: Execution took longer than {timeout} seconds."

    if not queue.empty():
        return queue.get()

    return None, "Execution Error: Unknown error during subprocess execution."


# In[ ]:


import logging

from Evaluation.evaluate_miniarc_raw_vs_finetuned import grids_match
from Evaluation.evaluate_list_function_raw_vs_finetuned import lists_match

# Setup reward function and output directories
print("\n🏯 Reward Function Setup")
print("=" * 30)

# Create run name and directories
run_name = get_run_name()
results_dir = get_results_dir(run_name)

os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "checkpoint"), exist_ok=True)

# # Add 'Training_' prefix to results directory
# sep_idx = results_dir.find('/') + 1
# results_dir = results_dir[:sep_idx] + 'Training_' + results_dir[sep_idx:]
# os.rename(get_results_dir(), results_dir)

logging.basicConfig(
    filename=os.path.join(results_dir, ERROR_LOG_PATH),
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
 )

print(f"📁 Results directory: {results_dir}")
print(f"🏷️  Run name: {run_name}")

def extract_prediction(text: str, datasetName: str):
    """
    Extract the model's answer from <answer>...</answer> and return it in the right type
    for each dataset (set/int/str/list).
    """
    # Handle list input (e.g., from GRPOTrainer if it passes messages)
    if isinstance(text, list):
        if len(text) > 0 and isinstance(text[-1], dict) and 'content' in text[-1]:
            text = text[-1]['content']
        elif len(text) > 0 and isinstance(text[0], str):
            text = text[0]
        else:
            text = str(text)

    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    answer = (m.group(1).strip() if m else "").strip()

    if datasetName == "UniADILR":
        # sentence indices as a set of ints
        nums = re.findall(r"\b(\d+)\b", answer)
        return set(int(n) for n in nums)

    if datasetName == "balanced_copa_cause_only":
        # "1" or "2"
        return int(answer) if answer else -123

    if datasetName == "causelogics_level3&4":
        # "TRUE"/"FALSE"
        return answer.upper()

    if datasetName == "climate_fever":
        # expected one of: SUPPORTS / REFUTES / NOT ENOUGH INFO / DISPUTED
        return answer.upper()

    if datasetName in ("list_function", "miniarc"):
        # you stored GT as string; compare as stripped string
        return answer

    # fallback
    return answer


def parse_ground_truth(gt, datasetName: str):
    """
    Convert stored ground_truth into the same type as extract_prediction returns.
    """
    if datasetName == "UniADILR":
        # gt is json.dumps(example['proof']) in your transform
        proof_str = json.loads(gt) if isinstance(gt, str) else gt
        # proof_str might be like: 'sent5 & sent13 -> hypothesis'
        if isinstance(proof_str, str):
            left = proof_str.split("->")[0] if "->" in proof_str else proof_str
            nums = re.findall(r"sent(\d+)", left)
            return set(int(n) for n in nums)
        # if proof is already structured, adapt here
        raise ValueError(f"Unexpected UniADILR proof type: {type(proof_str)}")

    if datasetName == "balanced_copa_cause_only":
        # stored as "1" or "2"
        return int(gt)

    if datasetName == "causelogics_level3&4":
        # stored as "TRUE"/"FALSE"
        return str(gt).upper()

    if datasetName == "climate_fever":
        # stored as "SUPPORTS"/"REFUTES"/...
        return str(gt).upper()

    if datasetName == "list_function":
        # stored as example['test'][0]['output']  (already a string)
        return str(gt).strip()

    if datasetName == "miniarc":
        # stored as str(example['test'][0]['output'])  (string representation of 2D list)
        return str(gt).strip()

    return gt


class AbductiveRewardFunction:
    """Deterministic reward function for abductive reasoning task."""
    
    def __init__(self, dataset, tokenizer, output_path, log_every=50):
        self.dataset = dataset  # Keep for validation only
        self.tokenizer = tokenizer
        self.output_path = output_path
        self.current_epoch = 1
        self.training_log = []
        self.step_losses = []
        self.log_every = log_every
        
        print("🛠️ Building prompt-to-[ground_truth, datasetName, rule_test_input, rule_train_examples] lookup table for reward function...")
        self.lookup_table = {}
        missing_ground_truths = 0
        flag = False
        for record in self.dataset:
            # Apply chat template to get the exact string the trainer will pass
            full_prompt = self.tokenizer.apply_chat_template(record['prompt'], tokenize=False, add_generation_prompt=True)
            
            if not flag:
                print(f"Example formatted prompt: {full_prompt[:200]}...")
                flag = True

            datasetName = record['dataset_name']
            ground_truth = record.get('ground_truth', '')
            rule_test_input = record.get('rule_test_input')
            rule_train_examples = record.get('rule_train_examples')
            
            if ground_truth is not None:
                # If multiple records have the exact same prompt, this will overwrite.
                # This is usually fine if the ground_truth is also the same.
                self.lookup_table[full_prompt] = {
                    'ground_truth': ground_truth,
                    'dataset_name': datasetName,
                    'rule_test_input': rule_test_input,
                    'rule_train_examples': rule_train_examples
                }
            else:
                missing_ground_truths += 1

                
        
        print(f"✅ Lookup table built. Contains {len(self.lookup_table)} entries.")
        if missing_ground_truths > 0:
            print(f"   ⚠️ Warning: {missing_ground_truths} records in the dataset were missing a 'ground_truths' field.")

    def _calculate_code_reward(self, code, datasetName, test_input, ground_truth_raw, train_examples_raw):
        """
        Helper to calculate reward for code-based tasks by verifying against ALL examples.
        """
        if not code:
            return 0.0, "No code found", "No code found"

        # 1. Prepare all examples (test + train)
        all_examples = []
        
        # Add test example
        try:
            if isinstance(test_input, str):
                test_input_obj = json.loads(test_input)
            else:
                test_input_obj = test_input
        except:
            test_input_obj = string_to_list(test_input) if datasetName == 'list_function' else test_input
            
        # Ensure ground_truth_raw is also decoded if it's a string
        try:
            if isinstance(ground_truth_raw, str):
                gt_obj = json.loads(ground_truth_raw)
            else:
                gt_obj = ground_truth_raw
        except:
            gt_obj = string_to_list(ground_truth_raw) if datasetName == 'list_function' else ground_truth_raw

        all_examples.append({'input': test_input_obj, 'output': gt_obj, 'is_test': True})
        
        # Add training examples
        if isinstance(train_examples_raw, str):
            try:
                train_examples = json.loads(train_examples_raw)
            except:
                train_examples = None
        else:
            train_examples = train_examples_raw
            
        if train_examples:
            for ex in train_examples:
                ex_in = ex['input']
                if isinstance(ex_in, str):
                    try: ex_in = json.loads(ex_in)
                    except: 
                        if datasetName == 'list_function': ex_in = string_to_list(ex_in)
                ex_out_gt = ex['output']
                if isinstance(ex_out_gt, str):
                    try: ex_out_gt = json.loads(ex_out_gt)
                    except:
                        if datasetName == 'list_function': ex_out_gt = string_to_list(ex_out_gt)
                all_examples.append({'input': ex_in, 'output': ex_out_gt, 'is_test': False})
        
        # 2. Verify all examples
        passed_count = 0
        test_prediction = None
        test_error = None
        
        for ex in all_examples:
            ex_out, ex_err = execute_transform(code, ex['input'], datasetName)
            
            match = False
            if not ex_err:
                if datasetName == 'miniarc':
                    match = grids_match(ex_out, ex['output'])
                else:
                    match = lists_match(ex_out, ex['output'])
            
            if match:
                passed_count += 1
                if ex.get('is_test'):
                    test_prediction = ex_out
                    test_error = None
            elif ex.get('is_test'):
                test_prediction = f"Error: {ex_err}" if ex_err else ex_out
                test_error = ex_err
        
        # Reward is 1.0 only if ALL examples pass
        reward = 1.0 if passed_count == len(all_examples) else 0.0
        return reward, test_prediction, test_error
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def record_loss(self, step, loss):
        self.step_losses.append({"step": step, "loss": loss})
    
    def __call__(self, completions, prompts, **kwargs):
        """
        Calculate rewards using the pre-computed lookup table.
        
        Args:
            completions: List of generated text strings for each prompt in the batch.
                         Shape: (batch_size * num_generations)
            prompts: List of the formatted input text strings.
                     Shape: (batch_size * num_generations)
        """
        rewards = []
        
        # The `prompts` and `completions` are flattened lists of shape (batch_size * num_generations)
        for i, (prompt_text, completion_text) in enumerate(zip(prompts, completions)):
            try:
                # If prompt_text is a list (e.g. messages), convert it to a string using the chat template
                if isinstance(prompt_text, list):
                    prompt_text = self.tokenizer.apply_chat_template(prompt_text, tokenize=False, add_generation_prompt=True)
                
                # prompt_text and completion_text are strings
                entry = self.lookup_table.get(prompt_text)
                
                if entry is None:
                    logging.warning(f"Prompt not found in lookup table. Cannot calculate reward. Prompt: {prompt_text[:100]}...")
                    rewards.append(0.0) # Assign a neutral reward
                    continue
                
                datasetName = entry['dataset_name']
                ground_truth_raw = entry['ground_truth']
                # Decode JSON-string payloads for code-based datasets
                if datasetName in ("list_function", "miniarc") and isinstance(ground_truth_raw, str):
                    try:
                        ground_truth_raw = json.loads(ground_truth_raw)
                    except Exception:
                        # Fallback for list_function if something isn't strict JSON
                        if datasetName == 'list_function':
                            try:
                                ground_truth_raw = string_to_list(ground_truth_raw)
                            except Exception:
                                pass
                
                # Branching logic based on dataset type
                execution_error = None
                extracted_code = None
                if datasetName in ("list_function", "miniarc"):
                    # Code-based tasks
                    extracted_code = extract_code(completion_text)
                    reward, predicted, execution_error = self._calculate_code_reward(
                        extracted_code, 
                        datasetName, 
                        entry['rule_test_input'], 
                        ground_truth_raw, 
                        entry.get('rule_train_examples')
                    )
                else:
                    # Text-based tasks (UniADILR, COPA, etc.)
                    if "list" in datasetName.lower() or "mini" in datasetName.lower() or "arc" in datasetName.lower():
                         print(f"   ⚠️ Warning: Dataset '{datasetName}' fell into text-based processing path!")

                    ground_truth = parse_ground_truth(ground_truth_raw, datasetName)
                    predicted = extract_prediction(completion_text, datasetName)
                    reward = 1.0 if predicted == ground_truth else 0.0

                rewards.append(reward)
                
                # Log entry
                log_entry = {
                    'epoch': self.current_epoch,
                    'batch_idx': i,
                    'dataset_name': datasetName,
                    'input': prompt_text,
                    'ground_truth': ground_truth_raw if datasetName in ("list_function", "miniarc") else (sorted(list(ground_truth)) if datasetName == 'UniADILR' else ground_truth),
                    'predicted': predicted if datasetName in ("list_function", "miniarc") else (sorted(list(predicted)) if datasetName == 'UniADILR' else predicted),
                    'reward': reward,
                    'completion': completion_text,
                }

                if datasetName in ("list_function", "miniarc"):
                    log_entry['extracted_code'] = extracted_code
                    log_entry['execution_error'] = execution_error
                
                # VERIFICATION LOGGING
                if datasetName in ("list_function", "miniarc"):
                    print(f"   📝 Logged {datasetName}: Reward={reward:.1f} | Pred={str(predicted)[:30]}...")

                self.training_log.append(log_entry)
                
            except Exception as e:
                logging.exception(f"Error calculating reward for item {i}: {e}")
                rewards.append(0.0)
    
        # Save training log periodically
        if len(self.training_log) > 0 and len(self.training_log) % self.log_every == 0:
            try:
                with open(self.output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.training_log, f, ensure_ascii=False, indent=2)
                
                # Calculate full training average
                all_rewards = [r['reward'] for r in self.training_log]
                full_avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
                
                # Calculate step average (from the current batch of rewards)
                step_avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
                
                print(f"   💾 Saved {len(self.training_log)} completions log | Full avg: {full_avg_reward:.3f} | Step avg: {step_avg_reward:.3f}")
            except Exception as e:
                logging.warning(f"Failed to save training log: {e}")
        
        return rewards




    
    def evaluate_batch(self, completions, record_ids, validation_dataset=None):
        """Evaluate a batch of completions against ground truth.
        
        Args:
            completions: List of model outputs
            record_ids: List of indices into the dataset
            validation_dataset: Optional validation dataset
        
        Returns:
            List of dicts with reward, predicted, ground_truth, etc.
        """
        results = []
        
        # --- FIX STARTS HERE ---
        
        # 1. Determine which dataset to use for evaluation.
        #    If a validation_dataset is passed, use it. Otherwise, fall back to the
        #    dataset stored in the instance (likely the training set).
        dataset_to_use = validation_dataset if validation_dataset is not None else self.dataset
        
        # 2. Fetch the specific records from the dataset using the provided record_ids.
        #    This creates the 'records' variable that was missing.
        try:
            records = [dataset_to_use[i] for i in record_ids]
        except (IndexError, TypeError) as e:
            # Add error handling in case the IDs are out of bounds or dataset is not indexable
            logging.error(f"Failed to fetch records for evaluation using record_ids. Error: {e}")
            # Depending on desired behavior, you might want to return an empty list or raise the exception
            return []
            
        # --- FIX ENDS HERE ---
        
        # Now, the 'records' variable exists and the loop will work as intended.
        for idx, (completion, record) in enumerate(zip(completions, records)):
            try:
                datasetName = record.get('dataset_name', '')
                ground_truth_raw = record.get('ground_truth', '')
                if datasetName in ("list_function", "miniarc") and isinstance(ground_truth_raw, str):
                    try:
                        ground_truth_raw = json.loads(ground_truth_raw)
                    except Exception:
                        if datasetName == 'list_function':
                            try:
                                ground_truth_raw = string_to_list(ground_truth_raw)
                            except Exception:
                                pass
                
                # Branching logic based on dataset type
                execution_error = None
                extracted_code = None
                if datasetName in ("list_function", "miniarc"):
                    # Code-based tasks
                    extracted_code = extract_code(completion)
                    reward, predicted, execution_error = self._calculate_code_reward(
                        extracted_code, 
                        datasetName, 
                        record.get('rule_test_input'), 
                        ground_truth_raw, 
                        record.get('rule_train_examples')
                    )
                else:
                    # Text-based tasks
                    ground_truth = parse_ground_truth(ground_truth_raw, datasetName)
                    predicted = extract_prediction(completion, datasetName)
                    reward = 1.0 if predicted == ground_truth else 0.0
                
                # Extract input for logging
                input_prompt = record.get('prompt', [])
                user_content = ""
                for msg in input_prompt:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        user_content = msg.get('content', '')
                        break
                
                # Log entry
                log_entry = {
                    'epoch': self.current_epoch,
                    'record_id': record.get('record_id', idx),
                    'dataset_name': datasetName,
                    'input': user_content,
                    'ground_truth': ground_truth_raw if datasetName in ("list_function", "miniarc") else (sorted(list(ground_truth)) if datasetName == 'UniADILR' else ground_truth),
                    'predicted': predicted if datasetName in ("list_function", "miniarc") else (sorted(list(predicted)) if datasetName == 'UniADILR' else predicted),
                    'reward': reward,
                    'completion': completion,
                }

                if datasetName in ("list_function", "miniarc"):
                    log_entry['extracted_code'] = extracted_code
                    log_entry['execution_error'] = execution_error
                
                results.append({
                    'reward': reward,
                    'predicted': log_entry['predicted'],
                    'ground_truth': log_entry['ground_truth'],
                    'completion': completion,
                    'input': user_content,
                    'dataset_name': datasetName,
                    'extracted_code': extracted_code,
                    'execution_error': execution_error
                })
                
                self.training_log.append(log_entry)
                
            except Exception as e:
                logging.exception(f"Error evaluating completion {idx}: {e}")
                results.append({
                    'reward': 0.0,
                    'predicted': [],
                    'ground_truth': [],
                    'completion': completion,
                    'input': '',
                    'dataset_name': datasetName,
                })
        
        return results

# Create reward function

reward_fn = AbductiveRewardFunction(
    dataset=train_ds,
    tokenizer=tokenizer,
    output_path=os.path.join(results_dir, TRAINING_LOG_PATH),
    log_every=LOG_TRAIN_EVERY
)
reward_fn.__name__ = "AbductiveRewardFunction"

print(f"✅ Deterministic reward function configured")
print(f"   Type: Exact match (order-independent)")
print(f"   Output file: {TRAINING_LOG_PATH}")
print(f"   Log frequency: Every {LOG_TRAIN_EVERY} completions")


# In[ ]:


# Training configuration
print("\n⚙️ Training Configuration")
print("=" * 30)

training_args = GRPOConfig(
    learning_rate=LEARNING_RATE,
    adam_beta1=ADAM_BETA1,
    adam_beta2=ADAM_BETA2,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    optim=OPTIM,
    logging_steps=1,
    save_total_limit=20, #TODO: maybe more would be better
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    save_steps=SAVE_STEPS,
    max_grad_norm=MAX_GRAD_NORM,
    report_to=["tensorboard"],
    run_name=None,
    output_dir=os.path.join(results_dir, "checkpoint"),
    temperature=TEMPERATURE,
    epsilon=EPSILON,
    beta=BETA,
    )

print(f"Training Parameters:")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
print(f"   Epochs: {NUM_TRAIN_EPOCHS:,}")
print(f"   Save every: {SAVE_STEPS} steps")
print(f"   Max grad norm: {MAX_GRAD_NORM}")
print(f"   Temperature: {TEMPERATURE}")
print(f"   Warmup steps: {WARMUP_STEPS}")
print(f"   Weight decay: {WEIGHT_DECAY}")


# In[ ]:


import os
import subprocess
from typing import Dict, Any
import time
import threading
import queue

# ----------------------------------------------------------------------
# Job Queue and Worker Thread for Serialized Async Execution
# ----------------------------------------------------------------------
_evaluation_job_queue = queue.Queue()
_evaluation_worker_thread = None
_evaluation_worker_lock = threading.Lock()


def _evaluation_worker():
    """Background worker that processes evaluation jobs one at a time (serialized)."""
    while True:
        job_args = _evaluation_job_queue.get()
        if job_args is None:  # Sentinel value to stop the worker
            _evaluation_job_queue.task_done()
            break
        try:
            _execute_evaluation_job(*job_args)
        except Exception as e:
            print(f"[QUEUE ERROR] Evaluation job failed with exception: {e}")
        finally:
            _evaluation_job_queue.task_done()


def _ensure_evaluation_worker_running():
    """Ensure the background evaluation worker thread is running."""
    global _evaluation_worker_thread
    with _evaluation_worker_lock:
        if _evaluation_worker_thread is None or not _evaluation_worker_thread.is_alive():
            _evaluation_worker_thread = threading.Thread(
                target=_evaluation_worker, 
                daemon=True,
                name="EvaluationJobWorker"
            )
            _evaluation_worker_thread.start()


def _execute_evaluation_job(
    output_dir: str,
    root_dir: str,
    base_results_dir: str,
    raw_model_path: str,
    run_name: str,
    chkpt_name: str,
    base_model_name: str,
    train_data: str,
    cuda_device: int,
    evaluate_checkpoints: int,
):
    """
    The actual synchronous execution of the evaluation job.
    This runs inside the worker thread.
    """
    
    # ----------------------------------------------------------------------
    # 1. Prepare Environment Variables for the Bash Script
    # ----------------------------------------------------------------------
    
    # Start with a copy of the current environment (needed for PATH, etc.)
    bash_env: Dict[str, str] = os.environ.copy()
    
    # Add all function arguments to the environment dictionary as strings
    bash_env.update({
        "OUTPUT_DIR": str(output_dir),
        "ROOT_DIR": str(root_dir),
        "BASE_RESULTS_DIR": str(base_results_dir),
        "RAW_MODEL_PATH": str(raw_model_path),
        "RUN_NAME": str(run_name),
        "CHKPT_NAME": str(chkpt_name),
        "BASE_MODEL_NAME": str(base_model_name),
        "TRAIN_DATA": str(train_data),
        "CUDA_DEVICE": str(cuda_device),
        "EVALUATE_CHECKPOINTS": str(evaluate_checkpoints),
    })
    
    # Print environment configuration for debugging
    print("***********************************************")
    for key, value in bash_env.items():
        print(f"{key}: {value}")
    print("***********************************************")

    # ----------------------------------------------------------------------
    # 2. Execute the Bash Script
    # ----------------------------------------------------------------------
    
    bash_script_path = "Evaluation/run_eval_checkpoints_midtrain.sh"
    
    print("--- Executing Bash Script ---")
    print(f"Target Script: {bash_script_path}")
    print(f"RUN_NAME: {run_name}")
    print(f"CUDA_DEVICE: {cuda_device}")
    print("-----------------------------")

    try:
        # FIXED CODE (Calls bash directly)
        subprocess.run(
            ["bash", bash_script_path], 
            check=True, 
            text=True,
            env=bash_env
        )
        print("\nBash script executed successfully.")
        time.sleep(60)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Bash script failed with exit code {e.returncode}")
    except FileNotFoundError:
        print(f"\nERROR: The Bash script '{bash_script_path}' was not found.")


def run_evaluation_job(
    output_dir: str,
    root_dir: str,
    base_results_dir: str,
    raw_model_path: str,
    run_name: str,
    chkpt_name: str,
    base_model_name: str,
    train_data: str,
    cuda_device: int,
    evaluate_checkpoints: int,
):
    """
    Queues an evaluation job for async execution.
    
    - Returns immediately (non-blocking to main thread)
    - Jobs are processed one at a time in order (serialized in the background)
    """
    _ensure_evaluation_worker_running()
    
    print(f"[QUEUE] Adding job to queue: {run_name}")
    
    _evaluation_job_queue.put((
        output_dir,
        root_dir,
        base_results_dir,
        raw_model_path,
        run_name,
        chkpt_name,
        base_model_name,
        train_data,
        cuda_device,
        evaluate_checkpoints,
    ))


# ----------------------------------------------------------------------
# Optional: Utility functions for queue management
# ----------------------------------------------------------------------

def wait_for_all_evaluation_jobs():
    """Block until all queued evaluation jobs are complete."""
    _evaluation_job_queue.join()
    print("[QUEUE] All evaluation jobs completed.")


def shutdown_evaluation_worker():
    """Gracefully shutdown the worker thread after finishing current jobs."""
    _evaluation_job_queue.put(None)  # Sentinel to stop
    if _evaluation_worker_thread is not None:
        _evaluation_worker_thread.join()
    print("[QUEUE] Worker thread shut down.")


# In[ ]:


from transformers import DataCollatorWithPadding
try:
    from vllm import SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Define a dummy SamplingParams if vllm is not available
    class SamplingParams:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

print("\n🔄 Setting up Training Callbacks with Validation")
print("=" * 45)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=0.95, #TODO: Consider changing this
    max_tokens=MAX_COMPLETION_LENGTH,
 )

class EnhancedEpochCallback(TrainerCallback):
    """
    Custom callback to log epoch progress, manage rewards, and handle validation.
    - Logs start and end of each epoch.
    - Records step losses for the reward function.
    - Triggers validation at the end of each epoch and after every EVAL_STEPS steps.
    """
    def __init__(self, reward_fn, val_dataset, results_dir, use_vllm=False, eval_interval=EVAL_STEPS):
        self.reward_fn = reward_fn
        self.val_dataset = val_dataset
        self.step_count = 0
        self.last_eval_step = -1
        self.start_time = None
        self.validation_metrics = {}
        self.results_dir = results_dir
        self.trainer = None
        self.formatted_inputs = None
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self.eval_interval = eval_interval

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"🚀 Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Format each prompt individually to avoid truncation/concatenation issues
        self.formatted_inputs = [
            self.trainer.processing_class.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            ) for prompt in self.val_dataset['prompt']
        ]

        # Trigger validation for step 0 (raw model)
        if LOG_VALIDATION and self.trainer:
            print("\n📊 Evaluating raw model (step 0) before training...")
            self.evaluate_validation(
                self.trainer.model,
                self.trainer.processing_class,
                state.global_step,
            )

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch_idx = int(state.epoch) + 1  # Convert to 1-indexed
        self.reward_fn.set_epoch(epoch_idx)
        print(f"\n📍 Starting epoch {epoch_idx}")

    def on_step_end(self, args, state, control, **kwargs):
        current_loss = 'N/A'
        if state.log_history:
            current_loss = state.log_history[-1].get("loss", 'N/A')
            if current_loss != 'N/A':
                self.reward_fn.record_loss(state.log_history[-1]['step'], current_loss)
        
        self.step_count += 1
        if self.step_count % 50 == 0:
            elapsed = time.time() - self.start_time
            steps_per_sec = self.step_count / elapsed
            print(f"   Step {self.step_count} | Loss: {current_loss} | Speed: {steps_per_sec:.2f} steps/s")

        if (
            LOG_VALIDATION
            and self.eval_interval
            and (self.step_count % self.eval_interval == 0)
            and self.trainer
):
            self.evaluate_validation(
                self.trainer.model,
                self.trainer.processing_class,
                state.global_step,
            )

    def evaluate_validation(self, model, tokenizer, step):
        if step == self.last_eval_step:
            return
        self.last_eval_step = step
        
        print(f"\n🔍 Validation at step {step}:")

        try:
            val_rewards = []
            validation_log = []
            batch_size = PER_DEVICE_EVAL_BATCH_SIZE

            with torch.no_grad():
                for batch_num in range(0, len(self.val_dataset), batch_size):
                    FastLanguageModel.for_inference(model)
                    batch = self.formatted_inputs[batch_num:batch_num + batch_size]
                    
                    if self.use_vllm:
                        outputs = model.fast_generate(
                            batch,
                            lora_request=None,
                            sampling_params=sampling_params,
                        )
                        completions = [o.outputs[0].text.strip() for o in outputs]
                    else:
                        # Set padding side to left for generation
                        old_padding_side = tokenizer.padding_side
                        tokenizer.padding_side = "left"
                        
                        batch_encodings = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
                        outputs = model.generate(
                            **batch_encodings,
                            temperature=sampling_params.temperature,
                            top_p=sampling_params.top_p,
                            max_new_tokens=sampling_params.max_tokens,
                        )
                        
                        # Restore padding side
                        tokenizer.padding_side = old_padding_side
                        
                        prompt_lengths = batch_encodings["input_ids"].shape[1]
                        generated_tokens = outputs[:, prompt_lengths:]
                        completions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    
                        batch_indices = list(range(batch_num, batch_num + len(completions)))
                        results = self.reward_fn.evaluate_batch(completions, batch_indices, validation_dataset=self.val_dataset)

                    
                    for batch_idx, result in enumerate(results):
                        val_rewards.append(result["reward"])
                        log_entry = {
                            "record_id": self.val_dataset['record_id'][batch_num + batch_idx],
                            "input": batch[batch_idx],
                            "ground_truth": result["ground_truth"],
                            "predicted": result["predicted"],
                            "reward": result["reward"],
                            "completion": result["completion"],
                        }
                        if "extracted_code" in result:
                            log_entry["extracted_code"] = result["extracted_code"]
                        if "execution_error" in result:
                            log_entry["execution_error"] = result["execution_error"]
                        
                        validation_log.append(log_entry)
                        
            FastLanguageModel.for_training(model)
            
            if val_rewards:
                avg_val_reward = sum(val_rewards) / len(val_rewards) if val_rewards else 0.0
                print(f"   📊 Validation reward: {avg_val_reward:.4f} (n={len(val_rewards)})")

                # When called from on_epoch_end, state.epoch is N for the just-completed epoch N.
                # Use float epoch to avoid overwriting mid-epoch evaluations
                epoch_key = f"{self.trainer.state.epoch:.4f}"

                self.validation_metrics[epoch_key] = {
                    'avg_reward': avg_val_reward,
                    'num_samples': len(val_rewards)
                }

                # Save validation log
                val_log_path = os.path.join(self.results_dir, VALIDATION_LOG_PATH)
                existing_data = {}
                if os.path.exists(val_log_path):
                    with open(val_log_path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)

                existing_data[epoch_key] = validation_log
                with open(val_log_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)

                # Save validation metrics
                val_metrics_path = os.path.join(self.results_dir, VALIDATION_METRICS_PATH)
                all_metrics = {}
                if os.path.exists(val_metrics_path):
                    with open(val_metrics_path, "r", encoding="utf-8") as f:
                        all_metrics = json.load(f)

                all_metrics[epoch_key] = {
                    "avg_reward": avg_val_reward,
                    "num_samples": len(val_rewards)
                }
                with open(val_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(all_metrics, f, ensure_ascii=False, indent=2)
                
                try:
                    with open(self.reward_fn.output_path, 'w', encoding='utf-8') as f:
                        json.dump(self.reward_fn.training_log, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logging.warning(f"Failed to save training log after validation: {e}")
            else:
                logging.warning(f"⚠️  No validation rewards computed. Step: {step}")

        except Exception as e:
            logging.exception(f"❌ Validation error: {e}")

    def on_epoch_end(self, args, state, control, **kwargs):
        completed_epoch_idx = int(state.epoch)
        print(f"✅ Completed epoch {completed_epoch_idx}")

        # Trigger validation at the end of the epoch
        if LOG_VALIDATION:
            if self.trainer:
                # We use state.global_step to be consistent with Hugging Face's tracking
                self.evaluate_validation(self.trainer.model, self.trainer.processing_class, state.global_step)
            else:
                logging.warning("⚠️  No trainer assigned; cannot evaluate validation.")


    def on_save(self, args, state, control, **kwargs):
        print(f"💾 Checkpoint saved at step {state.global_step}")
        output_dir = os.path.join('Evaluation', f'{run_name}')
        run_evaluation_job(
            output_dir=output_dir,
            root_dir=os.path.dirname(output_dir),
            base_results_dir="results",
            raw_model_path=MODEL_NAME,
            run_name=run_name,
            chkpt_name=f'checkpoint-{state.global_step}',
            base_model_name=MODEL_NAME.split('/')[-1],
            train_data=TRAIN_DATA_VAL,
            cuda_device=str(int(CUDA_VISIBLE_DEVICES)+1),
            evaluate_checkpoints=1,
        )

# Initialize callback
enhanced_callback = EnhancedEpochCallback(
    reward_fn=reward_fn,
    val_dataset=val_ds,
    results_dir=results_dir,
    use_vllm=USE_VLLM,
    # eval_interval=EVAL_STEPS, #if we want to evaluate every EVAL_STEPS steps
)   

print("✅ Enhanced callbacks configured:")
print("   - Epoch management")
print("   - Progress tracking with loss")
print("   - Validation evaluation")
print("   - Validation JSON logging")
print("   - Checkpoint notifications")
print(f"   - Validation every {EVAL_STEPS} steps")


# In[ ]:


# Create trainer with enhanced validation
print("\n🏗️  Creating Trainer with Validation")
print("=" * 35)

try:
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_ds,
    )
    trainer.image_token_id = None
    trainer.vision_start_token_id = None
    trainer.vision_end_token_id = None
    
    enhanced_callback.trainer = trainer
    trainer.add_callback(enhanced_callback)
    
    print("✅ Trainer created successfully!")
    print(f"   Model: {type(model).__name__}")
    print(f"   Training samples: {len(train_ds):,}")
    print(f"   Validation samples: {len(val_ds):,}")
    print(f"   Reward functions: 1")
    print(f"   Callbacks: {len(trainer.callback_handler.callbacks)}")
    
except Exception as e:
    logging.exception(f"❌ Failed to create trainer: {e}")
    raise

print(f"\n📋 Training Summary:")
print(f"   Total training epochs: {NUM_TRAIN_EPOCHS}")
print(f"   Effective batch size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"   Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"   Generations per step: {NUM_GENERATIONS}")
print(f"   Output directory: {results_dir}")


# In[ ]:


import sys
from datetime import datetime
import signal

# Set up proper logging at the start of your notebook
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add a custom callback for better progress tracking
from transformers import TrainerCallback
import math

class DetailedProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()
        self.step_times = []
        self.last_log_time = time.time()
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step"""
        current_time = time.time()
        # Log every 10 steps or every 30 seconds, whichever comes first
        if state.global_step % 10 == 0 or (current_time - self.last_log_time) > 30:
            elapsed = current_time - self.start_time
            steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
            
            # Calculate ETA
            remaining_steps = state.max_steps - state.global_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            
            progress_pct = (state.global_step / state.max_steps) * 100
            
            print(f"\r⏳ Step {state.global_step}/{state.max_steps} ({progress_pct:.1f}%) | "
                  f"Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_str} | "
                  f"Epoch: {state.epoch:.1f}", end='', flush=True)
            
            self.last_log_time = current_time
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs:
            print()  # New line after progress bar
            log_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                  for k, v in logs.items() if k != 'epoch'])
            print(f"📊 {log_str}")
            logging.info(log_str)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        print()  # New line
        elapsed = time.time() - self.start_time
        print(f"\n✅ Epoch {int(state.epoch)} completed | "
              f"Total time: {elapsed/60:.1f}m | "
              f"Steps: {state.global_step}/{state.max_steps}")
        logging.info(f"Epoch {int(state.epoch)} completed")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the start of training"""
        print(f"\n🎯 Training will run for {state.max_steps} steps")
        print(f"📝 Logging every {args.logging_steps} steps")
        print(f"💾 Saving checkpoints every {args.save_steps} steps")
        print("-" * 70)
        logging.info("Training started")

# Add progress callback to trainer
progress_callback = DetailedProgressCallback()
trainer.add_callback(progress_callback)

# Handle keyboard interrupts gracefully
def signal_handler(sig, frame):
    print("\n⚠️  Interrupt signal received. Saving progress...")
    logging.warning("Training interrupted by user")
    trainer.save_model(os.path.join(results_dir, "checkpoint", "interrupted"))
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Start training with enhanced logging
print("\n🚀 Starting Training")
print("=" * 70)
print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🏷️  Run name: {run_name}")
print(f"📁 Output directory: {results_dir}")
print(f"🔍 Logs will be saved to: training_log.log")
print("-" * 70)

# Verify logging is working
logging.info(f"Starting training run: {run_name}")
logging.info(f"Output directory: {results_dir}")
logging.info(f"Training config: epochs={NUM_TRAIN_EPOCHS}, batch_size={PER_DEVICE_TRAIN_BATCH_SIZE}")

training_start_time = time.time()
last_checkpoint_time = training_start_time

try:
    # Verify trainer is set up correctly
    print("🔍 Verifying trainer configuration...")
    print(f"   • Total training steps: {trainer.args.max_steps}")
    print(f"   • Steps per epoch: {len(trainer.get_train_dataloader())}")
    print(f"   • Logging interval: {trainer.args.logging_steps} steps")
    print(f"   • Save interval: {trainer.args.save_steps} steps")
    print()
    
    # Force immediate logging
    sys.stdout.flush()
    logging.info("Calling trainer.train()...")
    
    # Start the training process
    print("🎬 Initiating training loop...\n")
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"⏱️  Duration: {training_duration/3600:.2f} hours ({training_duration/60:.1f} minutes)")
    print(f"📈 Average time per epoch: {training_duration/NUM_TRAIN_EPOCHS/60:.2f} minutes")
    print(f"🏁 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Training completed successfully in {training_duration/3600:.2f} hours")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    logging.warning("Training interrupted by user (KeyboardInterrupt)")
    print("💾 Saving current progress...")
    
except Exception as e:
    print(f"\n\n❌ Training failed with error!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\n📋 Full traceback:")
    logging.exception(f"Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    raise
    
finally:
    training_end_time = time.time()
    actual_duration = training_end_time - training_start_time
    
    print("\n" + "="*70)
    print("🔄 Cleanup and saving...")
    print("="*70)
    
    # Always try to save the current state
    try:
        # Save final training log
        if reward_fn and hasattr(reward_fn, 'training_log') and reward_fn.training_log:
            try:
                log_path = os.path.join(results_dir, "training_rewards.json")
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(reward_fn.training_log, f, ensure_ascii=False, indent=2)
                print(f"✅ Training log saved: {len(reward_fn.training_log)} entries")
                logging.info(f"Saved training log with {len(reward_fn.training_log)} entries")
            except Exception as e:
                print(f"⚠️  Failed to save training log: {e}")
                logging.warning(f"Failed to save training log: {e}")
        
        # # Rename results directory
        # if 'Training_' in results_dir:
        #     new_results_dir = results_dir.replace('Training_', '')
        #     os.rename(results_dir, new_results_dir)
        #     results_dir = new_results_dir
        #     print(f"✅ Results directory renamed")
        
        # Save final model
        final_model_path = os.path.join(results_dir, "checkpoint", "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        trainer.save_model(final_model_path)
        print(f"✅ Model saved to: {final_model_path}")
        logging.info(f"Final model saved to: {final_model_path}")
        
        
        # Trigger evaluation for the final model
        output_dir = os.path.join('Evaluation', f'{run_name}')
        run_evaluation_job(
            output_dir=output_dir,
            root_dir=os.path.dirname(output_dir),
            base_results_dir="results",
            raw_model_path=MODEL_NAME,
            run_name=run_name,
            chkpt_name='checkpoint/final_model',
            base_model_name=MODEL_NAME.split('/')[-1],
            train_data=TRAIN_DATA_VAL,
            cuda_device=str(int(CUDA_VISIBLE_DEVICES)+1),
            evaluate_checkpoints=1,
        )
        
        print(f"\n⏱️  Total elapsed time: {actual_duration/60:.1f} minutes")
        print("="*70)
        
    except Exception as e:
        print(f"⚠️  Error during cleanup: {e}")
        logging.exception("Error during cleanup")


# In[ ]:


# Optional: Visualize training progress
print("\n📊 Training Visualization")
print("=" * 30)

# Load validation metrics
val_metrics_path = os.path.join(results_dir, VALIDATION_METRICS_PATH)
if os.path.exists(val_metrics_path):
    with open(val_metrics_path, 'r') as f:
        val_metrics = json.load(f)
    
    epochs = [float(k) for k in val_metrics.keys()]
    rewards = [v['avg_reward'] for v in val_metrics.values()]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rewards, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Validation Reward', fontsize=12)
    plt.title('Validation Performance Over Training', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, 'validation_progress.png')
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Validation progress plot saved to: {plot_path}")
    plt.show()
else:
    print("⚠️  No validation metrics found")


# In[ ]:


wait_for_all_evaluation_jobs()
shutdown_evaluation_worker()

