#!/usr/bin/env python3
"""
AIMO Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the AIMO (AI Mathematical Olympiad) dataset.

Usage:
    python evaluate_aimo_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import os
import json
import argparse
import re
from datetime import datetime
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
import time
import warnings
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Allow path injection from orchestrator
RAW_MODEL_PATH = os.environ.get('EVAL_RAW_MODEL_PATH', 
    "/home/moein_salimi/PLLMS/unsloth-Qwen2.5-3B-Instruct-unsloth-bnb-4bit")
TRAINING_DIR = os.environ.get('EVAL_TRAINING_DIR',
    "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/results/dt11.10.16:42_e20_unsloth_Qwen2.5_3B_Instruct_unsloth_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b16")
CHECKPOINT_DIR = os.path.join(TRAINING_DIR, "checkpoint")
OUTPUT_DIR = os.environ.get('EVAL_OUTPUT_DIR',
    "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/aimo_evaluation_results")  # Change default per script
MODEL_TYPE = "hf"
CURRENT_LORA_REQUEST = None
CURRENT_LORA_INT_ID = 0

# ============================================================================
# Helper Functions
# ============================================================================

def find_best_checkpoint(training_dir):
    """Find the best checkpoint based on validation metrics."""
    print("\n📁 Finding best checkpoint...")
    
    val_metrics_path = os.path.join(training_dir, "val_metrics.json")
    checkpoint_dir = os.path.join(training_dir, "checkpoint")
    
    if not os.path.exists(val_metrics_path):
        print(f"⚠️  No val_metrics.json found, using latest checkpoint")
        checkpoints = [d for d in os.listdir(checkpoint_dir) 
                      if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            return os.path.join(checkpoint_dir, latest), 0.0
        return None, 0.0
    
    with open(val_metrics_path, 'r') as f:
        val_metrics = json.load(f)
    
    # Find epoch with highest avg_reward
    best_epoch = None
    best_score = 0.0
    
    for epoch_str, metrics in val_metrics.items():
        if metrics['avg_reward'] > best_score:
            best_score = metrics['avg_reward']
            best_epoch = float(epoch_str)
    
    if best_epoch is None:
        print("⚠️  No valid metrics found, using latest checkpoint")
        checkpoints = [d for d in os.listdir(checkpoint_dir) 
                      if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            return os.path.join(checkpoint_dir, latest), 0.0
        return None, 0.0
    
    # Find closest checkpoint
    checkpoints = [d for d in os.listdir(checkpoint_dir) 
                  if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    if not checkpoints:
        return None, 0.0
    
    checkpoint_steps = [(int(cp.split('-')[1]), cp) for cp in checkpoints]
    checkpoint_steps.sort()
    
    max_checkpoint_step = max(checkpoint_steps)[0]
    estimated_steps_per_epoch = max_checkpoint_step / 20.0
    target_step = int(best_epoch * estimated_steps_per_epoch)
    
    best_checkpoint = min(checkpoint_steps, key=lambda x: abs(x[0] - target_step))
    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint[1])
    
    print(f"✅ Best checkpoint: {best_checkpoint[1]}")
    print(f"   Validation score: {best_score:.4f} at epoch {best_epoch:.2f}")
    
    return checkpoint_path, best_score

def load_raw_model(device):
    """Load the raw/base model."""
    print(f"\n🤖 Loading raw model from: {RAW_MODEL_PATH}")
    print(f"- Model type: {MODEL_TYPE}")
    
    tokenizer = AutoTokenizer.from_pretrained(RAW_MODEL_PATH, trust_remote_code=True)
    
    if MODEL_TYPE == "hf":
        
        model = AutoModelForCausalLM.from_pretrained(
            RAW_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map={"": f"cuda:0"},
            trust_remote_code=True,
            load_in_4bit=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
    elif MODEL_TYPE == "vllm":
        model = LLM(
            model=RAW_MODEL_PATH,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            quantization="bitsandbytes",
            max_lora_rank=64
      )
    else:
        raise ValueError(f"Invalid model type: {MODEL_TYPE}")
    
    print("✅ Raw model loaded successfully")
    
    return model, tokenizer

# Global variable to hold the current LoRA request
CURRENT_LORA_REQUEST = None

def sanitize_name(name: str) -> str:
    """Convert a string into a safe identifier-style name."""
    return re.sub(r'\W|^(?=\d)', '_', name).upper()

def load_finetuned_model(checkpoint_path, device):
    """
    Load the fine-tuned model with LoRA adapter.
    For vLLM, creates and saves a global LoRARequest with a meaningful name.
    """
    global CURRENT_LORA_REQUEST, CURRENT_LORA_INT_ID

    print(f"\nLoading fine-tuned model (MODEL_TYPE={MODEL_TYPE})")
    print(f"Checkpoint path: {checkpoint_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(RAW_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create meaningful lora_name
    raw_model_name = sanitize_name(Path(RAW_MODEL_PATH).stem)
    checkpoint_name = sanitize_name(Path(checkpoint_path).stem)
    meaningful_lora_name = f"LORA_{raw_model_name}_{checkpoint_name}"

    if MODEL_TYPE.lower() == "hf":
        base_model = AutoModelForCausalLM.from_pretrained(
            RAW_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map={"": f'cuda:0'},
            trust_remote_code=True,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.eval()
        print("✅ HF LoRA model loaded successfully")
        return model, tokenizer

    elif MODEL_TYPE.lower() == "vllm":
        llm = LLM(
            model=RAW_MODEL_PATH,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            quantization="bitsandbytes",
            enable_lora=True,
            max_lora_rank=64
        )

        CURRENT_LORA_INT_ID += 1

        # Save LoRARequest to global variable
        CURRENT_LORA_REQUEST = LoRARequest(
            lora_name=meaningful_lora_name,
            lora_int_id=CURRENT_LORA_INT_ID,
            lora_path=checkpoint_path,
        )
        
        print(f"✅ vLLM base engine loaded. LoRA request saved as CURRENT_LORA_REQUEST with name {meaningful_lora_name}")
        return llm, tokenizer

    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


def create_aimo_prompt(problem):
    """Create a prompt for AIMO problem - handles LaTeX properly."""
    system_prompt = """You are an expert mathematician specializing in competition mathematics (AMC, AIME, etc.).

First, read the problem carefully, including all LaTeX mathematical notation, and solve it step by step. Then give the final answer as a single number.

Your entire output MUST use exactly the following format and nothing else (no text before, between, or after these tags):

<reasoning>
[here you write your chain-of-thought reasoning and intermediate steps]
</reasoning>
<answer>
[here you output ONLY the final answer as a number, decimal, or fraction a/b, with no extra words]
</answer>"""
    
    user_prompt = f"""Problem:
{problem}

Solve this problem and provide your final numerical answer."""
    
    return system_prompt, user_prompt

def extract_reasoning(response):
    """Extract chain-of-thought reasoning from <reasoning>...</reasoning> tags, if present."""
    match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_answer(response):
    """Extract the numerical answer from the <answer>...</answer> block.

    Supports integers, decimals, and simple fractions (a/b).
    Returns the answer as a cleaned string, or None if not found.
    """
    if not response:
        return None

    # Find the content inside <answer>...</answer>
    tag_match = re.search(
        r'<answer>\s*(.*?)\s*</answer>',
        response,
        re.IGNORECASE | re.DOTALL
    )
    if not tag_match:
        return None

    answer_content = tag_match.group(1).strip()

    # Handle possible \boxed{...} inside the answer block (extra robustness)
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer_content)
    if boxed_match:
        answer_content = boxed_match.group(1).strip()

    # Clean common wrappers/symbols
    answer_content = answer_content.replace('$', '').replace(',', '').strip()

    # Look for a number / decimal / fraction inside the answer content
    num_match = re.search(r'[+-]?\d+(?:\.\d+)?(?:/\d+)?', answer_content)
    if num_match:
        return num_match.group(0).strip()

    # Fallback: if nothing matched, return None
    return None


def normalize_answer(ans):
    """Normalize answers for comparison - handles integers, decimals, and fractions."""
    if ans is None:
        return None
    
    ans = str(ans).strip().lower()
    
    # Remove dollar signs, spaces, commas
    ans = ans.replace('$', '').replace(' ', '').replace(',', '')
    
    # Handle LaTeX fractions: \frac{a}{b} -> a/b
    ans = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', ans)
    
    # Remove other LaTeX commands
    ans = ans.replace('\\', '')
    
    # Try to evaluate fractions to compare as floats
    if '/' in ans:
        try:
            parts = ans.split('/')
            if len(parts) == 2:
                numerator = float(parts[0])
                denominator = float(parts[1])
                if denominator != 0:
                    return str(numerator / denominator)
        except:
            pass
    
    # Try to convert to float for comparison
    try:
        return str(float(ans))
    except:
        return ans


def evaluate_on_aimo(model, tokenizer, max_samples=None, model_name="Model", batch_size=1, split="test"):
    """Evaluate model on AIMO dataset with support for HF and vLLM models (LG1)."""
    print(f"\n🔍 Evaluating {model_name} on AIMO dataset...")

    # Load dataset
    try:
        dataset = load_dataset("AI-MO/aimo-validation-amc", split="train")
        print(f"✅ Loaded {len(dataset)} samples from AIMO validation (AMC) dataset")
        
        # Debug: Print sample
        print(f"\n📋 Sample from dataset:")
        print(f"   Problem: {dataset[0]['problem'][:150]}...")
        print(f"   Answer: {dataset[0]['answer']}")
        print(f"   Answer type: {type(dataset[0]['answer'])}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"📊 Evaluating on {len(dataset)} samples (limited)")
    
    results = []
    correct = 0
    total = 0
    failed_extractions = 0
    
    btime = time.time()
    # Process in batches
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {model_name}"):
        batch_end = min(batch_start + batch_size, len(dataset))

        # Prepare batch prompts
        formatted_prompts = []
        true_answers = []
        batch_data = []

        for idx in range(batch_start, batch_end):
            sample = dataset[idx]
            system_prompt, user_prompt = create_aimo_prompt(sample['problem'])
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except:
                formatted_prompt = f"{system_prompt}\n\n{user_prompt}"

            formatted_prompts.append(formatted_prompt)
            true_answers.append(str(sample['answer']))
            batch_data.append({'problem': sample['problem'], 'id': idx})

        # --- LG1: Generation ---
        if MODEL_TYPE.lower() == "hf":
            inputs = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=2048
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )

            for i in range(len(formatted_prompts)):
                input_len = inputs['input_ids'][i].shape[0]
                response = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
                predicted_answer = extract_answer(response) or "FAILED_EXTRACTION"
                reasoning = response
                norm_true = normalize_answer(true_answers[i])
                norm_pred = normalize_answer(predicted_answer)
                # Compare answers with tolerance
                is_correct = False
                if norm_true and norm_pred != "failed_extraction":
                    try:
                        is_correct = abs(float(norm_true) - float(norm_pred)) < 0.01
                    except:
                        is_correct = norm_pred == norm_true
                if is_correct:
                    correct += 1
                total += 1
                if predicted_answer == "FAILED_EXTRACTION":
                    failed_extractions += 1

                results.append({
                    'problem_id': batch_data[i]['id'],
                    'problem': batch_data[i]['problem'],
                    'true_answer': true_answers[i],
                    'predicted_answer': predicted_answer,
                    'reasoning': reasoning,
                    'correct': is_correct
                })

        elif MODEL_TYPE.lower() == "vllm":
            from vllm import SamplingParams
            sampling = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0)

            vllm_outputs = model.generate(
                formatted_prompts,
                sampling_params=sampling,
                lora_request=CURRENT_LORA_REQUEST
            )

            for i, out in enumerate(vllm_outputs):
                response = out.outputs[0].text
                predicted_answer = extract_answer(response) or "FAILED_EXTRACTION"
                reasoning = response
                norm_true = normalize_answer(true_answers[i])
                norm_pred = normalize_answer(predicted_answer)
                is_correct = False
                if norm_true and norm_pred != "failed_extraction":
                    try:
                        is_correct = abs(float(norm_true) - float(norm_pred)) < 0.01
                    except:
                        is_correct = norm_pred == norm_true
                if is_correct:
                    correct += 1
                total += 1
                if predicted_answer == "FAILED_EXTRACTION":
                    failed_extractions += 1

                results.append({
                    'problem_id': batch_data[i]['id'],
                    'problem': batch_data[i]['problem'],
                    'true_answer': true_answers[i],
                    'predicted_answer': predicted_answer,
                    'reasoning': reasoning,
                    'correct': is_correct
                })

        else:
            raise ValueError(f"Unsupported MODEL_TYPE={MODEL_TYPE}")

    etime = time.time()
    print(f"Batch processing time: {etime - btime:.2f} seconds")
    accuracy = correct / total if total > 0 else 0.0
    extraction_rate = (total - failed_extractions) / total if total > 0 else 0.0
    
    print(f"\n📊 {model_name} Results:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) - {correct}/{total} correct")
    print(f"   Extraction Rate: {extraction_rate:.4f} ({extraction_rate*100:.2f}%)")
    print(f"   Failed extractions: {failed_extractions}/{total} ({failed_extractions/total*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'failed_extractions': failed_extractions,
        'extraction_rate': extraction_rate,
        'time': etime - btime,
        'results': results
    }



def evaluate_model_with_dynamic_batch(model, tokenizer, args, model_name):
    """Evaluate a model with automatic batch-size backoff to avoid CUDA OOM."""
    results = None
    batch_size = args.batch_size
    
    while batch_size >= 1 and results is None:
        try:
            print(f"\n🧪 Evaluating {model_name} with batch_size={batch_size}")
            results = evaluate_on_aimo(
                model,
                tokenizer,
                args.max_samples,
                model_name,
                batch_size,
                args.split
            )
            print(f"✅ {model_name} evaluation succeeded with batch_size={batch_size}")
        except torch.cuda.OutOfMemoryError:
            print(f"⚠️ CUDA OutOfMemoryError at batch_size={batch_size}, halving batch size...")
            results = None
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ RuntimeError OOM at batch_size={batch_size}, halving batch size...")
                results = None
            else:
                raise
        
        if results is None:
            torch.cuda.empty_cache()
            batch_size = batch_size // 2
    
    if results is None:
        print(f"❌ {model_name}: still out of memory even with batch_size < 1, giving up.")
    
    return results

def ensure_raw_results_cached(args):
    """
    Ensure raw aimo results are cached on disk for the current configuration.
    Returns the loaded or newly computed raw_results dict.
    """
    dataset_name = "aimo"
    split = args.split
    sample_tag = f"max{args.max_samples}" if args.max_samples else "all"
    
    raw_results_dir = os.path.join(OUTPUT_DIR, "raw_model", dataset_name)
    os.makedirs(raw_results_dir, exist_ok=True)
    
    raw_results_file = os.path.join(
        raw_results_dir,
        f"raw_results_train_all.json"
    )
    
    if os.path.exists(raw_results_file):
        print(f"\n📂 Found cached raw model results: {raw_results_file}")
        with open(raw_results_file, "r") as f:
            raw_results = json.load(f)
        return raw_results
    
    print("\n🔁 No cached raw model results found for this configuration.")
    print("   Running raw model once and caching per-sample results...")
    
    raw_model, raw_tokenizer = load_raw_model(args.cuda_device)
    raw_results = evaluate_model_with_dynamic_batch(
        raw_model, raw_tokenizer, args, "Raw Model (cached)"
    )
    del raw_model
    torch.cuda.empty_cache()
    
    if raw_results is None:
        print("❌ Failed to compute raw model results; cannot cache.")
        return None
    
    raw_results_with_meta = {
        "model_path": RAW_MODEL_PATH,
        "dataset": dataset_name,
        "split": split,
        "max_samples": args.max_samples,
        **raw_results
    }
    
    with open(raw_results_file, "w") as f:
        json.dump(raw_results_with_meta, f, indent=2)
    print(f"💾 Cached raw model results saved to: {raw_results_file}")
    
    return raw_results_with_meta

def ensure_finetuned_results_cached(args, ckpt_name):
    """
    Ensure fine-tuned model results are cached on disk for the current configuration.
    Returns the loaded or newly computed fine-tuned results dict.
    """
    dataset_name = "aimo"
    ckpt_output_dir = os.path.join("/".join(OUTPUT_DIR.split("/")[:]), args.run, ckpt_name, dataset_name)
    if os.path.exists(ckpt_output_dir) and os.path.exists(os.path.join(ckpt_output_dir, "disagreement_cases.json")) and os.path.exists(os.path.join(ckpt_output_dir, "all_cases.json")):
        print(f"\n📂 Found cached fine-tuned model results: {ckpt_output_dir}")
        return True
    
    print("\n🔁 No cached fine-tuned model results found for this configuration.")
    return False


def evaluate_checkpoint_cases(args, checkpoint_path):
    """
    Given a single checkpoint, evaluate it vs cached raw results and save:
      - all_cases.json
      - disagreement_cases.json
    under: OUTPUT_DIR/<checkpoint_name>/aimo/
    """
    print(f"\n📁 Checkpoint path argument received: {checkpoint_path}")
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
        print(f"   Converted to absolute path: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint path does not exist: {checkpoint_path}")
        print(f"   Please check the path and try again.")
        return
    
    ckpt_name = os.path.basename(checkpoint_path.rstrip("/"))
    print(f"✅ Using checkpoint for per-case evaluation: {ckpt_name}")
    
    # Get cached (or newly computed) raw results
    raw_results = ensure_raw_results_cached(args)
    if raw_results is None:
        print("❌ Cannot evaluate checkpoint without raw model results.")
        return

    # Get cached (or newly computed) fine-tuned results
    if ensure_finetuned_results_cached(args, ckpt_name):
        print(f"✅ Using cached fine-tuned model results for per-case evaluation: {ckpt_name}")
        return
    
    # Evaluate fine-tuned checkpoint
    finetuned_model, finetuned_tokenizer = load_finetuned_model(checkpoint_path, args.cuda_device)
    finetuned_results = evaluate_model_with_dynamic_batch(
        finetuned_model,
        finetuned_tokenizer,
        args,
        f"Fine-tuned Model ({ckpt_name})"
    )
    del finetuned_model
    torch.cuda.empty_cache()
    
    if finetuned_results is None:
        print("❌ Fine-tuned model evaluation failed; aborting.")
        return
    
    # Build per-case comparison
    dataset_name = "aimo"
    ckpt_output_dir = os.path.join("/".join(OUTPUT_DIR.split("/")[:]), args.run ,ckpt_name, dataset_name)
    os.makedirs(ckpt_output_dir, exist_ok=True)
    
    raw_by_id = {idx + 1: r for idx, r in enumerate(raw_results["results"])}
    ft_by_id = {idx + 1: r for idx, r in enumerate(finetuned_results["results"])}
    
    disagreement_cases = []
    
    for pid, raw_r in raw_by_id.items():
        if pid not in ft_by_id:
            continue
        ft_r = ft_by_id[pid]
        
        case_entry = {
            "problem_id": pid,
            "problem": raw_r["problem"],          
            "true_answer": raw_r["true_answer"],  
            "raw": {
                "predicted_answer": raw_r["predicted_answer"],
                "reasoning": raw_r["reasoning"],
                "correct": raw_r["correct"]
            },
            "finetuned": {
                "predicted_answer": ft_r["predicted_answer"],
                "reasoning": ft_r["reasoning"],
                "correct": ft_r["correct"]
            }
        }
        
        if raw_r["correct"] == ft_r["correct"]:
            continue
        
        if raw_r["correct"] and not ft_r["correct"]:
            disagreement_type = "raw_correct_finetuned_wrong"
        else:
            disagreement_type = "finetuned_correct_raw_wrong"
        
        disagreement_cases.append({
            **case_entry,
            "disagreement_type": disagreement_type
        })
    
    disagreement_file = os.path.join(ckpt_output_dir, "disagreement_cases.json")
    with open(disagreement_file, "w") as f:
        json.dump(disagreement_cases, f, indent=2)
    print(f"💾 Disagreement cases saved to: {disagreement_file}")
    
    finetune_results_with_meta = {
        "dataset": dataset_name,
        "max_samples": args.max_samples,
        **finetuned_results
    }
    
    finetune_results_file = os.path.join(ckpt_output_dir, "all_cases.json")
    with open(finetune_results_file, "w") as f:
        json.dump(finetune_results_with_meta, f, indent=2)
    print(f"💾 finetune model results saved to: {finetune_results_file}")

    return {
        "raw_results": raw_results,
        "finetuned_results": finetuned_results,
        "all_cases_file": finetune_results_file,
        "disagreement_file": disagreement_file
    }



def save_results(raw_results, finetuned_results, best_checkpoint_info, output_dir):
    """Save evaluation results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save raw model results
    raw_output = {
        'model': RAW_MODEL_PATH,
        'evaluation_time': timestamp,
        'dataset': 'AIMO',
        'metrics': {
            'accuracy': raw_results['accuracy'],
            'extraction_rate': raw_results['extraction_rate']
        },
        'correct': raw_results['correct'],
        'total': raw_results['total'],
        'failed_extractions': raw_results['failed_extractions'],
        'detailed_results': raw_results['results']
    }
    
    raw_file = os.path.join(output_dir, f"raw_model_aimo_results_{timestamp}.json")
    with open(raw_file, 'w') as f:
        json.dump(raw_output, f, indent=2)
    print(f"\n💾 Raw model results saved to: {raw_file}")
    
    # Save fine-tuned model results
    finetuned_output = {
        'base_model': RAW_MODEL_PATH,
        'checkpoint': best_checkpoint_info['path'],
        'validation_score': best_checkpoint_info['score'],
        'evaluation_time': timestamp,
        'dataset': 'AIMO',
        'metrics': {
            'accuracy': finetuned_results['accuracy'],
            'extraction_rate': finetuned_results['extraction_rate']
        },
        'correct': finetuned_results['correct'],
        'total': finetuned_results['total'],
        'failed_extractions': finetuned_results['failed_extractions'],
        'detailed_results': finetuned_results['results']
    }
    
    finetuned_file = os.path.join(output_dir, f"finetuned_model_aimo_results_{timestamp}.json")
    with open(finetuned_file, 'w') as f:
        json.dump(finetuned_output, f, indent=2)
    print(f"💾 Fine-tuned model results saved to: {finetuned_file}")
    
    # Save comparison summary
    improvement = finetuned_results['accuracy'] - raw_results['accuracy']
    relative_improvement = (improvement / raw_results['accuracy'] * 100) if raw_results['accuracy'] > 0 else 0
    
    extraction_improvement = finetuned_results['extraction_rate'] - raw_results['extraction_rate']
    
    summary = {
        'evaluation_time': timestamp,
        'dataset': 'AIMO',
        'split': 'test',
        'num_samples': raw_results['total'],
        'raw_model': {
            'path': RAW_MODEL_PATH,
            'metrics': {
                'accuracy': raw_results['accuracy'],
                'extraction_rate': raw_results['extraction_rate']
            },
            'correct': raw_results['correct'],
            'total': raw_results['total'],
            'failed_extractions': raw_results['failed_extractions']
        },
        'finetuned_model': {
            'base_model': RAW_MODEL_PATH,
            'checkpoint': best_checkpoint_info['path'],
            'validation_score': best_checkpoint_info['score'],
            'metrics': {
                'accuracy': finetuned_results['accuracy'],
                'extraction_rate': finetuned_results['extraction_rate']
            },
            'correct': finetuned_results['correct'],
            'total': finetuned_results['total'],
            'failed_extractions': finetuned_results['failed_extractions']
        },
        'comparison': {
            'accuracy_improvement': improvement,
            'accuracy_relative_improvement_percent': relative_improvement,
            'extraction_improvement': extraction_improvement,
            'overall_improved': improvement > 0
        }
    }
    
    summary_file = os.path.join(output_dir, f"aimo_comparison_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Comparison summary saved to: {summary_file}")
    
    # Save disagreement and all cases summary
    raw_by_id = {idx+1: r for idx, r in enumerate(raw_results['results'])}
    ft_by_id = {idx+1: r for idx, r in enumerate(finetuned_results['results'])}
    
    disagreement_cases, all_cases = [], []
    
    for pid, raw_r in raw_by_id.items():
        if pid not in ft_by_id:
            continue
        ft_r = ft_by_id[pid]
        
        all_cases.append({
            "problem_id": pid,
            "problem": raw_r["problem"],          
            "true_answer": raw_r["true_answer"],  
            "raw": {
                "predicted_answer": raw_r["predicted_answer"],
                "reasoning": raw_r["reasoning"],
                "correct": raw_r["correct"]
            },
            "finetuned": {
                "predicted_answer": ft_r["predicted_answer"],
                "reasoning": ft_r["reasoning"],
                "correct": ft_r["correct"]
            }
        })
        
        if raw_r['correct'] == ft_r['correct']:
            continue
        
        if raw_r['correct'] and not ft_r['correct']:
            disagreement_type = "raw_correct_finetuned_wrong"
        else:
            disagreement_type = "finetuned_correct_raw_wrong"
        
        disagreement_cases.append({
            "problem_id": pid,
            "problem": raw_r["problem"],          
            "true_answer": raw_r["true_answer"],
            "raw": {
                "predicted_answer": raw_r["predicted_answer"],
                "reasoning": raw_r["reasoning"],
                "correct": raw_r["correct"]
            },
            "finetuned": {
                "predicted_answer": ft_r["predicted_answer"],
                "reasoning": ft_r["reasoning"],
                "correct": ft_r["correct"]
            },
            "disagreement_type": disagreement_type
        })
    
    disagreement_file = os.path.join(output_dir, f"disagreement_cases_{timestamp}.json")
    with open(disagreement_file, "w") as f:
        json.dump(disagreement_cases, f, indent=2)
    print(f"💾 Disagreement cases saved to: {disagreement_file}")
    
    all_cases_file = os.path.join(output_dir, f"all_cases_{timestamp}.json")
    with open(all_cases_file, "w") as f:
        json.dump(all_cases, f, indent=2)
    print(f"💾 All cases saved to: {all_cases_file}")
    
    return summary

def evaluate_all_checkpoints(args):
    """Evaluate all checkpoints in a directory."""
    checkpoint_dir = args.checkpoint_dir
    
    # Handle relative vs absolute paths
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Error: Checkpoint directory does not exist: {checkpoint_dir}")
        return
    
    print("="*80)
    print("🚀 AIMO EVALUATION: ALL CHECKPOINTS")
    print("="*80)
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Split: {args.split}")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    print("="*80)
    
    # Find all checkpoint directories
    all_items = os.listdir(checkpoint_dir)
    checkpoint_dirs = [
        d for d in all_items 
        if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    
    if not checkpoint_dirs:
        print(f"❌ No checkpoint directories found in: {checkpoint_dir}")
        print(f"   Looking for directories named 'checkpoint-*'")
        return
    
    # Sort checkpoints by number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
    
    print(f"\n📁 Found {len(checkpoint_dirs)} checkpoints:")
    for ckpt in checkpoint_dirs:
        print(f"   - {ckpt}")
    print()
    
    # Optionally evaluate raw model once
    raw_results = None
    if not args.skip_raw:
        print("\n" + "="*80)
        print("🤖 EVALUATING RAW MODEL (once)")
        print("="*80)
        raw_model, raw_tokenizer = load_raw_model(args.cuda_device)
        raw_results = evaluate_on_aimo(raw_model, raw_tokenizer, args.max_samples, "Raw Model", args.batch_size, args.split)
        if raw_results is None:
            print("❌ Failed to evaluate raw model")
            return
        del raw_model
        torch.cuda.empty_cache()
        print(f"\n✅ Raw model evaluation complete")
        print(f"   Accuracy: {raw_results['accuracy']:.4f} ({raw_results['accuracy']*100:.2f}%)")
    
    # Save detailed results to JSON
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    summary_data = {
        'evaluation_time': timestamp,
        'dataset': 'AIMO',
        'split': args.split,
        'checkpoint_directory': checkpoint_dir,
        'num_checkpoints_evaluated': len(checkpoint_dirs),
        'raw_model': {
            'path': RAW_MODEL_PATH,
            'results': raw_results if raw_results else 'not_evaluated'
        },
        'checkpoints': []
    }
    
    summary_file = os.path.join(OUTPUT_DIR, f"aimo_all_checkpoints_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Evaluate each checkpoint
    all_checkpoint_results = []
    
    for i, ckpt_name in enumerate(checkpoint_dirs, 1):
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)
        
        print("\n" + "="*80)
        print(f"🎯 EVALUATING CHECKPOINT {i}/{len(checkpoint_dirs)}: {ckpt_name}")
        print("="*80)
        
        try:
            # Load and evaluate checkpoint
            finetuned_model, finetuned_tokenizer = load_finetuned_model(checkpoint_path, args.cuda_device)
            finetuned_results = evaluate_on_aimo(
                finetuned_model, finetuned_tokenizer, args.max_samples, 
                f"{ckpt_name}", args.batch_size, args.split
            )
            
            if finetuned_results is None:
                print(f"❌ Failed to evaluate {ckpt_name}")
                continue
                
            del finetuned_model
            torch.cuda.empty_cache()
            
            # Store results
            checkpoint_info = {
                'checkpoint_name': ckpt_name,
                'checkpoint_path': checkpoint_path,
                'results': finetuned_results
            }
            
            summary_data["checkpoints"].append({
                'name': checkpoint_info['checkpoint_name'],
                'path': checkpoint_info['checkpoint_path'],
                'metrics': {
                    'accuracy': checkpoint_info['results']['accuracy'],
                    'extraction_rate': checkpoint_info['results']['extraction_rate']
                },
                'improvements_vs_raw': {
                    'accuracy_delta': checkpoint_info['results']['accuracy'] - raw_results['accuracy'] if raw_results else None,
                    'extraction_delta': checkpoint_info['results']['extraction_rate'] - raw_results['extraction_rate'] if raw_results else None
                } if raw_results else None
            })
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            all_checkpoint_results.append(checkpoint_info)
            
            print(f"\n✅ {ckpt_name} evaluation complete")
            print(f"   Accuracy: {finetuned_results['accuracy']:.4f} ({finetuned_results['accuracy']*100:.2f}%) - {finetuned_results['correct']}/{finetuned_results['total']} correct")
            print(f"   Extraction Rate: {finetuned_results['extraction_rate']:.4f} ({finetuned_results['extraction_rate']*100:.2f}%)")
            
            # Show improvement vs raw model if available
            if raw_results:
                acc_improvement = finetuned_results['accuracy'] - raw_results['accuracy']
                ext_improvement = finetuned_results['extraction_rate'] - raw_results['extraction_rate']
                print(f"   📈 Improvement vs Raw: Accuracy {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%), Extraction {ext_improvement:+.4f} ({ext_improvement*100:+.2f}%)")
            
        except Exception as e:
            print(f"❌ Error evaluating {ckpt_name}: {e}")
            continue
    
    # Save all results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create summary comparison
    print("\n" + "="*80)
    print("📊 SUMMARY: ALL CHECKPOINTS COMPARISON (AIMO)")
    print("="*80)
    
    if raw_results:
        print(f"\n🤖 RAW MODEL:")
        print(f"   Accuracy:        {raw_results['accuracy']:.4f} ({raw_results['accuracy']*100:.2f}%)")
        print(f"   Extraction Rate: {raw_results['extraction_rate']:.4f} ({raw_results['extraction_rate']*100:.2f}%)")
    
    print(f"\n🎯 FINE-TUNED CHECKPOINTS:")
    if raw_results:
        print(f"   {'Checkpoint':<20} {'Accuracy':<15} {'Extraction':<15} {'Acc Δ':<12} {'Ext Δ':<12}")
        print(f"   {'-'*80}")
        
        for checkpoint_info in all_checkpoint_results:
            res = checkpoint_info['results']
            acc_delta = res['accuracy'] - raw_results['accuracy']
            ext_delta = res['extraction_rate'] - raw_results['extraction_rate']
            
            print(f"   {checkpoint_info['checkpoint_name']:<20} "
                  f"{res['accuracy']:.4f} ({res['accuracy']*100:5.2f}%) "
                  f"{res['extraction_rate']:.4f}         "
                  f"{acc_delta:+.4f}      "
                  f"{ext_delta:+.4f}")
    else:
        print(f"   {'Checkpoint':<20} {'Accuracy':<15} {'Extraction Rate':<15}")
        print(f"   {'-'*60}")
        
        for checkpoint_info in all_checkpoint_results:
            res = checkpoint_info['results']
            print(f"   {checkpoint_info['checkpoint_name']:<20} "
                  f"{res['accuracy']:.4f} ({res['accuracy']*100:5.2f}%) "
                  f"{res['extraction_rate']:.4f} ({res['extraction_rate']*100:5.2f}%)")
    
    # Find best checkpoint
    if all_checkpoint_results:
        best_ckpt = max(all_checkpoint_results, key=lambda x: x['results']['accuracy'])
        print(f"\n🏆 BEST CHECKPOINT: {best_ckpt['checkpoint_name']}")
        print(f"   Accuracy: {best_ckpt['results']['accuracy']:.4f} ({best_ckpt['results']['accuracy']*100:.2f}%)")
        print(f"   Extraction Rate: {best_ckpt['results']['extraction_rate']:.4f} ({best_ckpt['results']['extraction_rate']*100:.2f}%)")
        
        if raw_results:
            best_acc_imp = best_ckpt['results']['accuracy'] - raw_results['accuracy']
            best_rel_imp = (best_acc_imp / raw_results['accuracy'] * 100) if raw_results['accuracy'] > 0 else 0
            print(f"   📈 Improvement vs Raw: Accuracy {best_acc_imp:+.4f} ({best_acc_imp*100:+.2f}%), Relative {best_rel_imp:+.2f}%")
    
    print(f"\n💾 All results saved to: {summary_file}")
    print("="*80 + "\n")

def print_comparison(summary):
    """Print formatted comparison results."""
    print("\n" + "="*80)
    print("📊 AIMO EVALUATION: RAW vs FINE-TUNED MODEL")
    print("="*80)
    
    raw_metrics = summary['raw_model']['metrics']
    ft_metrics = summary['finetuned_model']['metrics']
    
    print("\n🤖 RAW MODEL:")
    print(f"   Accuracy:  {raw_metrics['accuracy']:.4f} ({raw_metrics['accuracy']*100:.2f}%) - {summary['raw_model']['correct']}/{summary['raw_model']['total']} correct")
    print(f"   Extraction Rate: {raw_metrics['extraction_rate']:.4f} ({raw_metrics['extraction_rate']*100:.2f}%)")
    
    print("\n🎯 FINE-TUNED MODEL:")
    print(f"   Checkpoint: {os.path.basename(summary['finetuned_model']['checkpoint'])}")
    val_score = summary['finetuned_model']['validation_score']
    val_score_str = f"{val_score:.4f}" if isinstance(val_score, (int, float)) else str(val_score)
    print(f"   Validation Score: {val_score_str}")
    print(f"   Accuracy:  {ft_metrics['accuracy']:.4f} ({ft_metrics['accuracy']*100:.2f}%) - {summary['finetuned_model']['correct']}/{summary['finetuned_model']['total']} correct")
    print(f"   Extraction Rate: {ft_metrics['extraction_rate']:.4f} ({ft_metrics['extraction_rate']*100:.2f}%)")
    
    print("\n📈 IMPROVEMENTS:")
    comp = summary['comparison']
    acc_imp = comp['accuracy_improvement']
    acc_rel = comp['accuracy_relative_improvement_percent']
    ext_imp = comp['extraction_improvement']
    
    print(f"   Accuracy:  {acc_imp:+.4f} ({acc_imp*100:+.2f}%) | Relative: {acc_rel:+.2f}%")
    print(f"   Extraction: {ext_imp:+.4f} ({ext_imp*100:+.2f}%)")
    
    print("\n" + "-"*80)
    
    if comp['overall_improved']:
        print("✅ RESULT: Fine-tuning on your dataset IMPROVED performance on AIMO!")
        print(f"   • Accuracy improved by {acc_rel:.2f}% (relative)")
        print(f"   The model shows better olympiad-level math problem solving ability.")
    elif acc_imp < 0:
        print("⚠️  RESULT: Fine-tuning on your dataset DECREASED performance on AIMO.")
        print(f"   • Accuracy decreased by {acc_rel:.2f}% (relative)")
        print(f"   • This suggests potential overfitting to your training data.")
    else:
        print("➖ RESULT: Fine-tuning had NO SIGNIFICANT IMPACT on AIMO performance.")
        print(f"   The model maintained baseline olympiad math problem solving ability.")
    
    print("="*80 + "\n")

def main():
    global RAW_MODEL_PATH, OUTPUT_DIR, MODEL_TYPE
    parser = argparse.ArgumentParser(description='Evaluate raw vs fine-tuned model on AIMO dataset')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum number of samples to evaluate (default: all samples)')
    parser.add_argument('--cuda_device', type=str, default='0',
                       help='CUDA device to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation. Higher values (4-8) are faster but use more GPU memory (default: 1)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'validation'],
                       help='Dataset split to use (default: test)')
    parser.add_argument('--skip_raw', action='store_true',
                       help='Skip raw model evaluation (evaluate only fine-tuned model)')
    parser.add_argument('--skip_finetuned', action='store_true',
                       help='Skip fine-tuned model evaluation (evaluate only raw model)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to specific checkpoint to evaluate (e.g., /path/to/checkpoint-640). '
                            'If not provided, automatically selects the best checkpoint based on validation metrics.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Path to directory containing multiple checkpoints (e.g., /path/to/checkpoint/). '
                            'Will evaluate ALL checkpoint-* directories found. Cannot be used with --checkpoint_path.')
    parser.add_argument('--evaluate_checkpoints', type=int, default=0,
                       help='If set to 1, run per-checkpoint mode: '
                            'evaluate the given --checkpoint_path vs cached raw results and '
                            'save all_cases/disagreement_cases under OUTPUT_DIR/checkpoint/dataset_name.')
    parser.add_argument('--run', type=str, default="run",
                       help='Which training run to use for the output directory.')
    parser.add_argument('--raw_path', type=str, default=None,
                       help='The raw model path')
    parser.add_argument('--output_path', type=str, default=OUTPUT_DIR,
                       help='Model output path, defaults to env variable.')
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE,
                       help='Use VLLM or HuggingFace Transformers model, defaults to HF.')
        
    args = parser.parse_args()

    OUTPUT_DIR = args.output_path
    MODEL_TYPE = args.model_type.lower()
    
    
    
    # Validate arguments
    if args.checkpoint_path and args.checkpoint_dir:
        print("❌ Error: Cannot use both --checkpoint_path and --checkpoint_dir")
        print("   Use --checkpoint_path for a single checkpoint")
        print("   Use --checkpoint_dir to evaluate all checkpoints in a directory")
        return
    
    if args.evaluate_checkpoints == 1 and args.checkpoint_dir:
        print("❌ Error: --evaluate_checkpoints 1 is only supported with --checkpoint_path (single checkpoint).")
        print("   Please pass a single --checkpoint_path, or omit --evaluate_checkpoints to use --checkpoint_dir.")
        return
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    if args.raw_path:
        RAW_MODEL_PATH = args.raw_path
    
    # Special mode: per-checkpoint evaluation with cached raw results
    if args.evaluate_checkpoints == 1:
        if not args.checkpoint_path:
            print("❌ Error: --evaluate_checkpoints 1 requires --checkpoint_path to be set.")
            return
        
        print("="*80)
        print("🚀 AIMO PER-CHECKPOINT EVALUATION MODE")
        print("="*80)
        print(f"Raw Model:     {RAW_MODEL_PATH}")
        print(f"Output Dir:    {OUTPUT_DIR}")
        print(f"CUDA Device:   {args.cuda_device}")
        print(f"Split:         {args.split}")
        if args.max_samples:
            print(f"Max Samples:   {args.max_samples}")
        print(f"Checkpoint:    {args.checkpoint_path}")
        print("="*80)
        
        evaluate_checkpoint_cases(args, args.checkpoint_path)
        print(f"\n✅ Per-checkpoint evaluation finished for: {args.checkpoint_path}")
        print(f"   Results root directory: {OUTPUT_DIR}")
        return
    
    # If checkpoint_dir is provided, evaluate all checkpoints
    if args.checkpoint_dir:
        evaluate_all_checkpoints(args)
        return
    
    print("="*70)
    print("🚀 AIMO EVALUATION: RAW vs FINE-TUNED")
    print("="*70)
    print(f"Raw Model: {RAW_MODEL_PATH}")
    print(f"Training Dir: {TRAINING_DIR}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Split: {args.split}")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    if args.skip_raw:
        print(f"Mode: Fine-tuned model only")
    elif args.skip_finetuned:
        print(f"Mode: Raw model only")
    else:
        print(f"Mode: Both models (comparison)")
    print("="*70)
    
    # Determine which checkpoint to use
    if not args.skip_finetuned:
        if args.checkpoint_path:
            # Use user-provided checkpoint
            checkpoint_path = args.checkpoint_path
            
            # Debug: show what we received
            print(f"\n📁 Checkpoint path argument received: {checkpoint_path}")
            
            # Handle relative vs absolute paths
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.abspath(checkpoint_path)
                print(f"   Converted to absolute path: {checkpoint_path}")
            
            if not os.path.exists(checkpoint_path):
                print(f"❌ Error: Checkpoint path does not exist: {checkpoint_path}")
                print(f"   Please check the path and try again.")
                return
            
            print(f"✅ Using user-specified checkpoint: {os.path.basename(checkpoint_path)}")
            best_checkpoint_info = {
                'path': checkpoint_path,
                'score': 'N/A (manually specified)'
            }
        else:
            # Auto-select best checkpoint
            print("\n📁 No checkpoint path provided, auto-selecting best checkpoint...")
            best_checkpoint_path, best_score = find_best_checkpoint(TRAINING_DIR)
            if best_checkpoint_path is None:
                print("❌ No valid checkpoint found!")
                return
            best_checkpoint_info = {
                'path': best_checkpoint_path,
                'score': best_score
            }
    else:
        best_checkpoint_info = None
    
    # Evaluate raw model
    if not args.skip_raw:
        raw_model, raw_tokenizer = load_raw_model(args.cuda_device)
        raw_results = evaluate_on_aimo(raw_model, raw_tokenizer, args.max_samples, "Raw Model", args.batch_size, args.split)
        if raw_results is None:
            print("❌ Failed to evaluate raw model")
            return
        del raw_model  # Free memory
        torch.cuda.empty_cache()
    else:
        raw_results = None
        print("\n⏭️  Skipping raw model evaluation")
    
    # Evaluate fine-tuned model
    if not args.skip_finetuned:
        finetuned_model, finetuned_tokenizer = load_finetuned_model(best_checkpoint_info['path'], args.cuda_device)
        finetuned_results = evaluate_on_aimo(finetuned_model, finetuned_tokenizer, args.max_samples, "Fine-tuned Model", args.batch_size, args.split)
        if finetuned_results is None:
            print("❌ Failed to evaluate fine-tuned model")
            return
        del finetuned_model  # Free memory
        torch.cuda.empty_cache()
    else:
        finetuned_results = None
        print("\n⏭️  Skipping fine-tuned model evaluation")
    
    # Save and display results
    if raw_results and finetuned_results:
        summary = save_results(raw_results, finetuned_results, best_checkpoint_info, OUTPUT_DIR)
        print_comparison(summary)
    elif raw_results:
        print("\n✅ Raw model evaluation completed")
    elif finetuned_results:
        print("\n✅ Fine-tuned model evaluation completed")
    
    print(f"\n✅ All results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
