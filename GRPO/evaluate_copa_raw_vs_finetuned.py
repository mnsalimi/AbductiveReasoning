#!/usr/bin/env python3
"""
COPA Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the COPA (Choice of Plausible Alternatives) dataset.
Focuses on tasks where the effect is given and we need to identify the cause.

Usage:
    python evaluate_copa_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Allow path injection from orchestrator
RAW_MODEL_PATH = os.environ.get('EVAL_RAW_MODEL_PATH', 
    "/home/moein_salimi/PLLMS/unsloth-Qwen2.5-3B-Instruct-unsloth-bnb-4bit")
TRAINING_DIR = os.environ.get('EVAL_TRAINING_DIR',
    "/home/moein_salimi/users/Nima/abductive_reasoning_finetuning/results/abductive_dt10.25.17:43_e20_unsloth_Qwen2.5_3B_Instruct_unsloth_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b16_abductive-reasoning")
CHECKPOINT_DIR = os.path.join(TRAINING_DIR, "checkpoint")
OUTPUT_DIR = os.environ.get('EVAL_OUTPUT_DIR',
    "/home/moein_salimi/users/Nima/abductive_reasoning_finetuning/aimo_evaluation_results")  # Change default per script

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
    
    tokenizer = AutoTokenizer.from_pretrained(RAW_MODEL_PATH, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        RAW_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map={"": f"cuda:{device}"},
        trust_remote_code=True,
        load_in_4bit=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print("✅ Raw model loaded successfully")
    
    return model, tokenizer

def load_finetuned_model(checkpoint_path, device):
    """Load the fine-tuned model with LoRA adapter."""
    print(f"\n🎯 Loading fine-tuned model from: {checkpoint_path}")
    
    # Load base model
    base_tokenizer = AutoTokenizer.from_pretrained(RAW_MODEL_PATH, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        RAW_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map={"": f"cuda:{device}"},
        trust_remote_code=True,
        load_in_4bit=True,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    model.eval()
    print("✅ Fine-tuned model loaded successfully")
    
    return model, base_tokenizer

def create_copa_prompt(premise, choice1, choice2):
    """Create a prompt for COPA causal reasoning task.
    
    Args:
        premise: The effect that occurred
        choice1: First possible cause
        choice2: Second possible cause
    
    Returns:
        system_prompt, user_prompt
    """
    system_prompt = """You are an expert in causal reasoning. Given an effect, you need to identify which of two possible options is the most plausible cause.

IMPORTANT:
- Think step by step about the causal relationship
- Consider which option is the most direct and likely cause
- Provide your final answer as either "1" or "2" wrapped in <answer></answer> tags
- Format: <answer>1</answer> or <answer>2</answer>"""
    
    user_prompt = f"""Effect: {premise}

Which of the following is the most plausible CAUSE of this effect?

Option 1: {choice1}
Option 2: {choice2}

Think step by step about which option is the most likely cause, then provide your answer in <answer></answer> tags."""
    
    return system_prompt, user_prompt

def extract_answer(response):
    """Extract answer from model response.
    
    Returns:
        int: 0 or 1 representing the choice (0-indexed), or None if extraction fails
    """
    # Try to find <answer>X</answer> pattern
    answer_pattern = r'<answer>\s*(\d+)\s*</answer>'
    matches = re.findall(answer_pattern, response, re.IGNORECASE)
    
    if matches:
        answer = matches[-1].strip()  # Take the last match
        if answer in ['1', '2']:
            return int(answer) - 1  # Convert to 0-indexed
    
    # Fallback: Look for "answer: 1" or "answer: 2" patterns
    fallback_patterns = [
        r'(?:answer|choice)[\s:]+(\d+)',
        r'option\s+(\d+)',
        r'(?:^|\s)([12])(?:\s|$|\.|,)'
    ]
    
    for pattern in fallback_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            answer = matches[-1].strip()
            if answer in ['1', '2']:
                return int(answer) - 1
    
    return None

def evaluate_on_copa(model, tokenizer, max_samples=None, model_name="Model", batch_size=1, split="validation"):
    """Evaluate model on COPA dataset (cause questions only)."""
    print(f"\n🔍 Evaluating {model_name} on COPA dataset (split: {split})...")
    print(f"   Task: Identify the CAUSE given an EFFECT")
    print(f"   Batch size: {batch_size}")
    
    # Load COPA dataset
    print("Loading COPA dataset...")
    
    try:
        dataset = load_dataset("pkavumba/balanced-copa", split="train")
        print(f"Loaded {len(dataset)} samples from COPA dataset")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\n💡 Make sure you have internet connection and the dataset is accessible.")
        return None
    
    # Filter for "cause" questions only (where we're given the effect)
    cause_dataset = dataset.filter(lambda x: x['question'] == 'cause')
    print(f"Filtered to {len(cause_dataset)} 'cause' questions (given effect, find cause)")
    
    if len(cause_dataset) == 0:
        print("❌ No 'cause' questions found in dataset!")
        return None
    
    if max_samples:
        cause_dataset = cause_dataset.select(range(min(max_samples, len(cause_dataset))))
        print(f"Evaluating on {len(cause_dataset)} samples (limited)")
    else:
        print(f"Evaluating on {len(cause_dataset)} samples (full filtered {split} set)")
    
    results = []
    all_true_labels = []
    all_pred_labels = []
    failed_extractions = 0
    
    # Process in batches
    num_batches = (len(cause_dataset) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {model_name}"):
        # Get batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(cause_dataset))
        batch = cause_dataset[start_idx:end_idx]
        
        # Handle both single sample and batch cases
        if not isinstance(batch['premise'], list):
            batch = {k: [v] for k, v in batch.items()}
        
        batch_size_actual = len(batch['premise'])
        
        # Prepare prompts for batch
        formatted_prompts = []
        true_labels_batch = []
        batch_data = []
        
        for i in range(batch_size_actual):
            premise = batch['premise'][i]
            choice1 = batch['choice1'][i]
            choice2 = batch['choice2'][i]
            true_label = batch['label'][i]  # 0 or 1
            
            # Create prompt
            system_prompt, user_prompt = create_copa_prompt(premise, choice1, choice2)
            
            # Format with chat template if available
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback if chat template not available
                formatted_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            formatted_prompts.append(formatted_prompt)
            true_labels_batch.append(true_label)
            batch_data.append({
                'premise': premise,
                'choice1': choice1,
                'choice2': choice2,
                'true_label': true_label,
                'id': start_idx + i
            })
        
        # Tokenize batch with padding
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate for batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
            )
        
        # Process each output in batch
        for i in range(len(formatted_prompts)):
            # Decode response (skip input tokens)
            input_length = inputs['input_ids'][i].shape[0]
            response = tokenizer.decode(outputs[i][input_length:], skip_special_tokens=True)
            
            # Extract answer from response
            predicted_label = extract_answer(response)
            
            if predicted_label is None:
                failed_extractions += 1
                predicted_label = 0  # Default to 0 if extraction fails
            
            true_label = true_labels_batch[i]
            
            all_true_labels.append(true_label)
            all_pred_labels.append(predicted_label)
            
            # Store result
            is_correct = (predicted_label == true_label)
            results.append({
                'sample_id': batch_data[i]['id'],
                'premise': batch_data[i]['premise'],
                'choice1': batch_data[i]['choice1'],
                'choice2': batch_data[i]['choice2'],
                'true_label': true_label,
                'predicted_label': predicted_label,
                'response': response,
                'correct': is_correct
            })
    
    # Calculate metrics
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    
    # Accuracy
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    
    # F1, Precision, Recall (binary classification)
    f1 = f1_score(all_true_labels, all_pred_labels, average='binary', zero_division=0)
    precision = precision_score(all_true_labels, all_pred_labels, average='binary', zero_division=0)
    recall = recall_score(all_true_labels, all_pred_labels, average='binary', zero_division=0)
    
    # Count correct predictions
    correct_count = sum(1 for r in results if r['correct'])
    
    # Extraction rate
    extraction_rate = (len(results) - failed_extractions) / len(results) if results else 0.0
    
    print(f"\n📊 {model_name} Results:")
    print(f"   Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%) - {correct_count}/{len(results)} correct")
    print(f"   F1 Score:         {f1:.4f}")
    print(f"   Precision:        {precision:.4f}")
    print(f"   Recall:           {recall:.4f}")
    print(f"   Extraction Rate:  {extraction_rate:.4f} ({extraction_rate*100:.2f}%)")
    print(f"   Failed extractions: {failed_extractions}/{len(results)} ({failed_extractions/len(results)*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'extraction_rate': extraction_rate,
        'correct_count': correct_count,
        'total': len(results),
        'failed_extractions': failed_extractions,
        'results': results
    }


def save_results(raw_results, finetuned_results, best_checkpoint_info, output_dir):
    """Save evaluation results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save raw model results
    raw_output = {
        'model': RAW_MODEL_PATH,
        'evaluation_time': timestamp,
        'dataset': 'COPA (cause questions only)',
        'metrics': {
            'accuracy': raw_results['accuracy'],
            'f1': raw_results['f1'],
            'precision': raw_results['precision'],
            'recall': raw_results['recall'],
            'extraction_rate': raw_results['extraction_rate']
        },
        'correct_count': raw_results['correct_count'],
        'total': raw_results['total'],
        'failed_extractions': raw_results['failed_extractions'],
        'detailed_results': raw_results['results'][:100]  # Save first 100 for space
    }
    
    raw_file = os.path.join(output_dir, f"raw_model_copa_results_{timestamp}.json")
    with open(raw_file, 'w') as f:
        json.dump(raw_output, f, indent=2)
    print(f"\n💾 Raw model results saved to: {raw_file}")
    
    # Save fine-tuned model results
    finetuned_output = {
        'base_model': RAW_MODEL_PATH,
        'checkpoint': best_checkpoint_info['path'],
        'validation_score': best_checkpoint_info['score'],
        'evaluation_time': timestamp,
        'dataset': 'COPA (cause questions only)',
        'metrics': {
            'accuracy': finetuned_results['accuracy'],
            'f1': finetuned_results['f1'],
            'precision': finetuned_results['precision'],
            'recall': finetuned_results['recall'],
            'extraction_rate': finetuned_results['extraction_rate']
        },
        'correct_count': finetuned_results['correct_count'],
        'total': finetuned_results['total'],
        'failed_extractions': finetuned_results['failed_extractions'],
        'detailed_results': finetuned_results['results'][:100]  # Save first 100 for space
    }
    
    finetuned_file = os.path.join(output_dir, f"finetuned_model_copa_results_{timestamp}.json")
    with open(finetuned_file, 'w') as f:
        json.dump(finetuned_output, f, indent=2)
    print(f"💾 Fine-tuned model results saved to: {finetuned_file}")
    
    # Save comparison summary
    improvement_acc = finetuned_results['accuracy'] - raw_results['accuracy']
    improvement_f1 = finetuned_results['f1'] - raw_results['f1']
    
    summary = {
        'evaluation_time': timestamp,
        'dataset': 'COPA (cause questions only)',
        'split': 'validation',
        'num_samples': raw_results['total'],
        'raw_model': {
            'path': RAW_MODEL_PATH,
            'metrics': {
                'accuracy': raw_results['accuracy'],
                'f1': raw_results['f1'],
                'precision': raw_results['precision'],
                'recall': raw_results['recall'],
                'extraction_rate': raw_results['extraction_rate']
            }
        },
        'finetuned_model': {
            'base_model': RAW_MODEL_PATH,
            'checkpoint': best_checkpoint_info['path'],
            'validation_score': best_checkpoint_info['score'],
            'metrics': {
                'accuracy': finetuned_results['accuracy'],
                'f1': finetuned_results['f1'],
                'precision': finetuned_results['precision'],
                'recall': finetuned_results['recall'],
                'extraction_rate': finetuned_results['extraction_rate']
            }
        },
        'comparison': {
            'accuracy_improvement': improvement_acc,
            'f1_improvement': improvement_f1,
            'overall_improved': improvement_acc > 0
        }
    }
    
    summary_file = os.path.join(output_dir, f"copa_comparison_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Comparison summary saved to: {summary_file}")
    
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
    print("🚀 COPA EVALUATION: ALL CHECKPOINTS")
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
        raw_results = evaluate_on_copa(raw_model, raw_tokenizer, args.max_samples, "Raw Model", args.batch_size, args.split)
        if raw_results is None:
            print("❌ Failed to evaluate raw model")
            return
        del raw_model
        torch.cuda.empty_cache()
        print(f"\n✅ Raw model evaluation complete")
        print(f"   Accuracy: {raw_results['accuracy']:.4f}")
        print(f"   F1 Score: {raw_results['f1']:.4f}")
    
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
            finetuned_results = evaluate_on_copa(
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
            all_checkpoint_results.append(checkpoint_info)
            
            print(f"\n✅ {ckpt_name} evaluation complete")
            print(f"   Accuracy: {finetuned_results['accuracy']:.4f} ({finetuned_results['accuracy']*100:.2f}%)")
            print(f"   F1 Score: {finetuned_results['f1']:.4f}")
            
            # Show improvement vs raw model if available
            if raw_results:
                acc_improvement = finetuned_results['accuracy'] - raw_results['accuracy']
                f1_improvement = finetuned_results['f1'] - raw_results['f1']
                print(f"   📈 Improvement vs Raw: Acc {acc_improvement:+.4f}, F1 {f1_improvement:+.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {ckpt_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("📊 SUMMARY: ALL CHECKPOINTS COMPARISON (COPA)")
    print("="*80)
    
    if raw_results:
        print(f"\n🤖 RAW MODEL:")
        print(f"   Accuracy: {raw_results['accuracy']:.4f}")
        print(f"   F1 Score: {raw_results['f1']:.4f}")
    
    print(f"\n🎯 FINE-TUNED CHECKPOINTS:")
    if raw_results:
        print(f"   {'Checkpoint':<20} {'Accuracy':<15} {'F1 Score':<15} {'Acc Δ':<12} {'F1 Δ':<12}")
        print(f"   {'-'*80}")
        
        for ckpt_info in all_checkpoint_results:
            res = ckpt_info['results']
            acc_delta = res['accuracy'] - raw_results['accuracy']
            f1_delta = res['f1'] - raw_results['f1']
            
            print(f"   {ckpt_info['checkpoint_name']:<20} "
                  f"{res['accuracy']:.4f}         "
                  f"{res['f1']:.4f}         "
                  f"{acc_delta:+.4f}      "
                  f"{f1_delta:+.4f}")
    
    # Find best checkpoint
    if all_checkpoint_results:
        best_ckpt = max(all_checkpoint_results, key=lambda x: x['results']['accuracy'])
        print(f"\n🏆 BEST CHECKPOINT: {best_ckpt['checkpoint_name']}")
        print(f"   Accuracy: {best_ckpt['results']['accuracy']:.4f}")
        print(f"   F1 Score: {best_ckpt['results']['f1']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    summary_data = {
        'evaluation_time': timestamp,
        'dataset': 'COPA (cause questions only)',
        'split': args.split,
        'checkpoint_directory': checkpoint_dir,
        'num_checkpoints_evaluated': len(all_checkpoint_results),
        'raw_model': {
            'path': RAW_MODEL_PATH,
            'results': {k: v for k, v in raw_results.items() if k != 'results'} if raw_results else 'not_evaluated'
        },
        'checkpoints': [
            {
                'name': ckpt_info['checkpoint_name'],
                'path': ckpt_info['checkpoint_path'],
                'metrics': {
                    'accuracy': ckpt_info['results']['accuracy'],
                    'f1': ckpt_info['results']['f1'],
                    'precision': ckpt_info['results']['precision'],
                    'recall': ckpt_info['results']['recall']
                }
            }
            for ckpt_info in all_checkpoint_results
        ]
    }
    
    summary_file = os.path.join(OUTPUT_DIR, f"copa_all_checkpoints_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n💾 All results saved to: {summary_file}")
    print("="*80 + "\n")

def print_comparison(summary):
    """Print formatted comparison results."""
    print("\n" + "="*80)
    print("📊 COPA EVALUATION: RAW vs FINE-TUNED MODEL")
    print("="*80)
    
    raw_metrics = summary['raw_model']['metrics']
    ft_metrics = summary['finetuned_model']['metrics']
    
    print("\n🤖 RAW MODEL:")
    print(f"   Accuracy:  {raw_metrics['accuracy']:.4f} ({raw_metrics['accuracy']*100:.2f}%)")
    print(f"   F1 Score:  {raw_metrics['f1']:.4f}")
    print(f"   Precision: {raw_metrics['precision']:.4f}")
    print(f"   Recall:    {raw_metrics['recall']:.4f}")
    
    print("\n🎯 FINE-TUNED MODEL:")
    print(f"   Checkpoint: {os.path.basename(summary['finetuned_model']['checkpoint'])}")
    print(f"   Accuracy:  {ft_metrics['accuracy']:.4f} ({ft_metrics['accuracy']*100:.2f}%)")
    print(f"   F1 Score:  {ft_metrics['f1']:.4f}")
    print(f"   Precision: {ft_metrics['precision']:.4f}")
    print(f"   Recall:    {ft_metrics['recall']:.4f}")
    
    print("\n📈 IMPROVEMENTS:")
    comp = summary['comparison']
    acc_imp = comp['accuracy_improvement']
    f1_imp = comp['f1_improvement']
    
    print(f"   Accuracy:  {acc_imp:+.4f} ({acc_imp*100:+.2f}%)")
    print(f"   F1 Score:  {f1_imp:+.4f} ({f1_imp*100:+.2f}%)")
    
    print("\n" + "-"*80)
    
    if comp['overall_improved']:
        print("✅ RESULT: Fine-tuning IMPROVED causal reasoning performance!")
        print(f"   • Accuracy improved by {acc_imp:.4f}")
    else:
        print("⚠️  RESULT: Fine-tuning did not improve causal reasoning.")
    
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Evaluate raw vs fine-tuned model on COPA dataset')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum number of samples to evaluate (default: all samples)')
    parser.add_argument('--cuda_device', type=str, default='0',
                       help='CUDA device to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation (default: 4)')
    parser.add_argument('--split', type=str, default='validation', choices=['train', 'test', 'validation'],
                       help='Dataset split to use (default: validation)')
    parser.add_argument('--skip_raw', action='store_true',
                       help='Skip raw model evaluation')
    parser.add_argument('--skip_finetuned', action='store_true',
                       help='Skip fine-tuned model evaluation')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to specific checkpoint to evaluate')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Path to directory containing multiple checkpoints')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.checkpoint_path and args.checkpoint_dir:
        print("❌ Error: Cannot use both --checkpoint_path and --checkpoint_dir")
        return
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    
    # If checkpoint_dir is provided, evaluate all checkpoints
    if args.checkpoint_dir:
        evaluate_all_checkpoints(args)
        return
    
    print("="*70)
    print("🚀 COPA EVALUATION: RAW vs FINE-TUNED")
    print("="*70)
    print(f"Raw Model: {RAW_MODEL_PATH}")
    print(f"Training Dir: {TRAINING_DIR}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Split: {args.split}")
    print(f"Task: Identify CAUSE given EFFECT")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    print("="*70)
    
    # Determine which checkpoint to use
    if not args.skip_finetuned:
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.abspath(checkpoint_path)
            
            if not os.path.exists(checkpoint_path):
                print(f"❌ Error: Checkpoint path does not exist: {checkpoint_path}")
                return
            
            print(f"✅ Using user-specified checkpoint: {os.path.basename(checkpoint_path)}")
            best_checkpoint_info = {
                'path': checkpoint_path,
                'score': 'N/A (manually specified)'
            }
        else:
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
        raw_results = evaluate_on_copa(raw_model, raw_tokenizer, args.max_samples, "Raw Model", args.batch_size, args.split)
        if raw_results is None:
            print("❌ Failed to evaluate raw model")
            return
        del raw_model
        torch.cuda.empty_cache()
    else:
        raw_results = None
        print("\n⏭️  Skipping raw model evaluation")
    
    # Evaluate fine-tuned model
    if not args.skip_finetuned:
        finetuned_model, finetuned_tokenizer = load_finetuned_model(best_checkpoint_info['path'], args.cuda_device)
        finetuned_results = evaluate_on_copa(finetuned_model, finetuned_tokenizer, args.max_samples, "Fine-tuned Model", args.batch_size, args.split)
        if finetuned_results is None:
            print("❌ Failed to evaluate fine-tuned model")
            return
        del finetuned_model
        torch.cuda.empty_cache()
    else:
        finetuned_results = None
        print("\n⏭️  Skipping fine-tuned model evaluation")
    
    # Save and display results
    if raw_results and finetuned_results:
        summary = save_results(raw_results, finetuned_results, best_checkpoint_info, OUTPUT_DIR)
        print_comparison(summary)
    
    print(f"\n✅ All results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
