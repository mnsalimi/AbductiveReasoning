#!/usr/bin/env python3
"""
ART Dataset Evaluation: Raw vs Fine-tuned Qwen2.5-3B
Compares performance between the raw model and fine-tuned checkpoint on the ART dataset.

Usage:
    python evaluate_art_raw_vs_finetuned.py [--max_samples N] [--cuda_device N]
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
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
    
    # Map epoch to checkpoint number
    # Looking at validation log structure, checkpoints appear to be every 128 steps
    # Find the checkpoint closest to the best epoch
    checkpoints = [d for d in os.listdir(checkpoint_dir) 
        if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    if not checkpoints:
        return None, 0.0
    
    # Sort checkpoints by step number
    checkpoint_steps = [(int(cp.split('-')[1]), cp) for cp in checkpoints]
    checkpoint_steps.sort()
    
    # Estimate steps per epoch (roughly)
    # If we have 20 epochs total and checkpoint-896 is the last one:
    # steps_per_epoch ≈ 896 / (20 epochs) ≈ 44.8 steps/epoch
    max_checkpoint_step = max(checkpoint_steps)[0]
    estimated_steps_per_epoch = max_checkpoint_step / 20.0  # Assuming 20 epochs
    
    target_step = int(best_epoch * estimated_steps_per_epoch)
    
    # Find closest checkpoint
    best_checkpoint = min(checkpoint_steps, key=lambda x: abs(x[0] - target_step))
    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint[1])
    
    print(f"✅ Best checkpoint found: {best_checkpoint[1]}")
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

def create_art_prompt(obs1, obs2, hyp1, hyp2):
    """Create prompt for ART task."""
    system_prompt = """You are an expert in abductive reasoning. Given two observations and two hypotheses, select which hypothesis (1 or 2) best explains what happened between the observations.

Answer with ONLY the number 1 or 2. No other text."""
    
    user_prompt = f"""Observation 1: {obs1}
Observation 2: {obs2}

Hypothesis 1: {hyp1}
Hypothesis 2: {hyp2}

Which hypothesis better explains the transition from Observation 1 to Observation 2? Answer with just the number 1 or 2."""
    
    return system_prompt, user_prompt

def extract_answer(response):
    """Extract the hypothesis number (1 or 2) from model response."""
    # Clean the response
    response = response.strip().lower()
    
    # Try various patterns
    patterns = [
        r'^\s*(\d)\s*$',  # Just a number
        r'(?:hypothesis\s*)?(\d)',  # "hypothesis 1" or just "1"
        r'(?:answer|select|choose)(?:\s+is)?\s*(\d)',  # "answer is 1"
        r'(?:^|\s)(\d)(?:\s|$|\.)',  # Number with spaces
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            num = match.group(1)
            if num in ['1', '2']:
                return int(num)
    
    # Check if response starts with 1 or 2
    if response.startswith('1'):
        return 1
    if response.startswith('2'):
        return 2
    
    # Look for "first" or "second"
    if 'first' in response[:20]:
        return 1
    if 'second' in response[:20]:
        return 2
    
    return None  # Unable to extract

def evaluate_on_art(model, tokenizer, max_samples=None, model_name="Model", batch_size=1,split='validation'):
    """Evaluate model on ART dataset with batch processing support."""
    print(f"\n🔍 Evaluating {model_name} on ART dataset...")
    print(f"   Batch size: {batch_size}")
    
    # Load ART dataset
    print("Loading ART dataset...")
    dataset = load_dataset("allenai/art", split="validation")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Evaluating on {len(dataset)} samples (limited)")
    else:
        print(f"Evaluating on {len(dataset)} samples (full validation set)")
    
    results = []
    correct = 0
    total = 0
    failed_extractions = 0
    
    # Process in batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {model_name}"):
        # Get batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]
        
        # Handle both single sample and batch cases
        if not isinstance(batch['observation_1'], list):
            batch = {k: [v] for k, v in batch.items()}
        
        batch_size_actual = len(batch['observation_1'])
        
        # Prepare prompts for batch
        formatted_prompts = []
        true_labels = []
        batch_data = []
        
        for i in range(batch_size_actual):
            obs1 = batch['observation_1'][i]
            obs2 = batch['observation_2'][i]
            hyp1 = batch['hypothesis_1'][i]
            hyp2 = batch['hypothesis_2'][i]
            true_label = int(batch['label'][i])
            
            # Create prompt
            system_prompt, user_prompt = create_art_prompt(obs1, obs2, hyp1, hyp2)
            
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
            true_labels.append(true_label)
            batch_data.append({
                'obs1': obs1,
                'obs2': obs2,
                'hyp1': hyp1,
                'hyp2': hyp2
            })
        
        # Tokenize batch with padding
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate for batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=1e-5,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
            )
        
        # Process each output in batch
        for i in range(batch_size_actual):
            # Decode response (skip input tokens)
            input_length = inputs['input_ids'][i].shape[0]
            response = tokenizer.decode(outputs[i][input_length:], skip_special_tokens=True)
            
            # Extract answer
            predicted_label = extract_answer(response)
            
            if predicted_label is None:
                failed_extractions += 1
                predicted_label = 1  # Default to 1 if extraction fails
            
            # Check correctness
            true_label = true_labels[i]
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct += 1
            total += 1
            
            # Store result
            results.append({
                'observation_1': batch_data[i]['obs1'],
                'observation_2': batch_data[i]['obs2'],
                'hypothesis_1': batch_data[i]['hyp1'],
                'hypothesis_2': batch_data[i]['hyp2'],
                'true_label': true_label,
                'predicted_label': predicted_label,
                'response': response,
                'correct': is_correct
            })
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate comprehensive metrics
    y_true = [r['true_label'] for r in results]
    y_pred = [r['predicted_label'] for r in results]
    
    # Calculate precision, recall, f1 (macro and weighted averages)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2])
    
    print(f"\n📊 {model_name} Results:")
    print(f"   Accuracy:  {accuracy:.4f} ({correct}/{total})")
    print(f"   Precision: {precision_macro:.4f} (macro), {precision_weighted:.4f} (weighted)")
    print(f"   Recall:    {recall_macro:.4f} (macro), {recall_weighted:.4f} (weighted)")
    print(f"   F1-Score:  {f1_macro:.4f} (macro), {f1_weighted:.4f} (weighted)")
    print(f"   Failed extractions: {failed_extractions}/{total} ({failed_extractions/total*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'failed_extractions': failed_extractions,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support_per_class.tolist(),
        'confusion_matrix': conf_matrix.tolist(),
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
        'metrics': {
            'accuracy': raw_results['accuracy'],
            'precision_macro': raw_results['precision_macro'],
            'precision_weighted': raw_results['precision_weighted'],
            'recall_macro': raw_results['recall_macro'],
            'recall_weighted': raw_results['recall_weighted'],
            'f1_macro': raw_results['f1_macro'],
            'f1_weighted': raw_results['f1_weighted']
        },
        'per_class_metrics': {
            'precision': raw_results['precision_per_class'],
            'recall': raw_results['recall_per_class'],
            'f1': raw_results['f1_per_class'],
            'support': raw_results['support_per_class']
        },
        'confusion_matrix': raw_results['confusion_matrix'],
        'correct': raw_results['correct'],
        'total': raw_results['total'],
        'failed_extractions': raw_results['failed_extractions'],
        'detailed_results': raw_results['results']
    }
    
    raw_file = os.path.join(output_dir, f"raw_model_results_{timestamp}.json")
    with open(raw_file, 'w') as f:
        json.dump(raw_output, f, indent=2)
    print(f"\n💾 Raw model results saved to: {raw_file}")
    
    # Save fine-tuned model results
    finetuned_output = {
        'base_model': RAW_MODEL_PATH,
        'checkpoint': best_checkpoint_info['path'],
        'validation_score': best_checkpoint_info['score'],
        'evaluation_time': timestamp,
        'metrics': {
            'accuracy': finetuned_results['accuracy'],
            'precision_macro': finetuned_results['precision_macro'],
            'precision_weighted': finetuned_results['precision_weighted'],
            'recall_macro': finetuned_results['recall_macro'],
            'recall_weighted': finetuned_results['recall_weighted'],
            'f1_macro': finetuned_results['f1_macro'],
            'f1_weighted': finetuned_results['f1_weighted']
        },
        'per_class_metrics': {
            'precision': finetuned_results['precision_per_class'],
            'recall': finetuned_results['recall_per_class'],
            'f1': finetuned_results['f1_per_class'],
            'support': finetuned_results['support_per_class']
        },
        'confusion_matrix': finetuned_results['confusion_matrix'],
        'correct': finetuned_results['correct'],
        'total': finetuned_results['total'],
        'failed_extractions': finetuned_results['failed_extractions'],
        'detailed_results': finetuned_results['results']
    }
    
    finetuned_file = os.path.join(output_dir, f"finetuned_model_results_{timestamp}.json")
    with open(finetuned_file, 'w') as f:
        json.dump(finetuned_output, f, indent=2)
    print(f"💾 Fine-tuned model results saved to: {finetuned_file}")
    
    # Save comparison summary
    improvement = finetuned_results['accuracy'] - raw_results['accuracy']
    relative_improvement = (improvement / raw_results['accuracy'] * 100) if raw_results['accuracy'] > 0 else 0
    
    f1_improvement = finetuned_results['f1_macro'] - raw_results['f1_macro']
    precision_improvement = finetuned_results['precision_macro'] - raw_results['precision_macro']
    recall_improvement = finetuned_results['recall_macro'] - raw_results['recall_macro']
    
    summary = {
        'evaluation_time': timestamp,
        'dataset': 'allenai/art',
        'split': 'validation',
        'num_samples': raw_results['total'],
        'raw_model': {
            'path': RAW_MODEL_PATH,
            'metrics': {
                'accuracy': raw_results['accuracy'],
                'precision_macro': raw_results['precision_macro'],
                'recall_macro': raw_results['recall_macro'],
                'f1_macro': raw_results['f1_macro'],
                'precision_weighted': raw_results['precision_weighted'],
                'recall_weighted': raw_results['recall_weighted'],
                'f1_weighted': raw_results['f1_weighted']
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
                'precision_macro': finetuned_results['precision_macro'],
                'recall_macro': finetuned_results['recall_macro'],
                'f1_macro': finetuned_results['f1_macro'],
                'precision_weighted': finetuned_results['precision_weighted'],
                'recall_weighted': finetuned_results['recall_weighted'],
                'f1_weighted': finetuned_results['f1_weighted']
            },
            'correct': finetuned_results['correct'],
            'total': finetuned_results['total'],
            'failed_extractions': finetuned_results['failed_extractions']
        },
        'comparison': {
            'accuracy_improvement': improvement,
            'accuracy_relative_improvement_percent': relative_improvement,
            'f1_improvement': f1_improvement,
            'precision_improvement': precision_improvement,
            'recall_improvement': recall_improvement,
            'overall_improved': improvement > 0
        }
    }
    
    summary_file = os.path.join(output_dir, f"comparison_summary_{timestamp}.json")
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
    print("🚀 ART DATASET EVALUATION: ALL CHECKPOINTS")
    print("="*80)
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Batch Size: {args.batch_size}")
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
        raw_results = evaluate_on_art(raw_model, raw_tokenizer, args.max_samples, "Raw Model", args.batch_size)
        del raw_model
        torch.cuda.empty_cache()
        print(f"\n✅ Raw model evaluation complete")
        print(f"   Accuracy: {raw_results['accuracy']:.4f} ({raw_results['accuracy']*100:.2f}%)")
    
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
            finetuned_results = evaluate_on_art(
                finetuned_model, finetuned_tokenizer, args.max_samples, 
                f"{ckpt_name}", args.batch_size
            )
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
            
            # Show improvement vs raw model if available
            if raw_results:
                acc_improvement = finetuned_results['accuracy'] - raw_results['accuracy']
                precision_improvement = finetuned_results['precision_macro'] - raw_results['precision_macro']
                recall_improvement = finetuned_results['recall_macro'] - raw_results['recall_macro']
                f1_improvement = finetuned_results['f1_macro'] - raw_results['f1_macro']
                print(f"   📈 Improvement vs Raw:")
                print(f"      Accuracy:  {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
                print(f"      Precision: {precision_improvement:+.4f} ({precision_improvement*100:+.2f}%)")
                print(f"      Recall:    {recall_improvement:+.4f} ({recall_improvement*100:+.2f}%)")
                print(f"      F1-Score:  {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)")
            
        except Exception as e:
            print(f"❌ Error evaluating {ckpt_name}: {e}")
            continue
    
    # Save all results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create summary comparison
    print("\n" + "="*80)
    print("📊 SUMMARY: ALL CHECKPOINTS COMPARISON")
    print("="*80)
    
    if raw_results:
        print(f"\n🤖 RAW MODEL:")
        print(f"   Accuracy:  {raw_results['accuracy']:.4f} ({raw_results['accuracy']*100:.2f}%)")
        print(f"   Precision: {raw_results['precision_macro']:.4f} (macro)")
        print(f"   Recall:    {raw_results['recall_macro']:.4f} (macro)")
        print(f"   F1-Score:  {raw_results['f1_macro']:.4f} (macro)")
    
    print(f"\n🎯 FINE-TUNED CHECKPOINTS:")
    if raw_results:
        print(f"   {'Checkpoint':<20} {'Accuracy':<15} {'F1-Score':<12} {'Acc Δ':<12} {'F1 Δ':<12}")
        print(f"   {'-'*75}")
    else:
        print(f"   {'Checkpoint':<20} {'Accuracy':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"   {'-'*75}")
    
    for ckpt_info in all_checkpoint_results:
        res = ckpt_info['results']
        if raw_results:
            acc_delta = res['accuracy'] - raw_results['accuracy']
            f1_delta = res['f1_macro'] - raw_results['f1_macro']
            print(f"   {ckpt_info['checkpoint_name']:<20} "
                  f"{res['accuracy']:.4f} ({res['accuracy']*100:5.2f}%) "
                  f"{res['f1_macro']:.4f}      "
                  f"{acc_delta:+.4f}      "
                  f"{f1_delta:+.4f}")
        else:
            print(f"   {ckpt_info['checkpoint_name']:<20} "
                  f"{res['accuracy']:.4f} ({res['accuracy']*100:5.2f}%) "
                  f"{res['precision_macro']:.4f}       "
                  f"{res['recall_macro']:.4f}       "
                  f"{res['f1_macro']:.4f}")
    
    # Find best checkpoint
    if all_checkpoint_results:
        best_ckpt = max(all_checkpoint_results, key=lambda x: x['results']['accuracy'])
        print(f"\n🏆 BEST CHECKPOINT: {best_ckpt['checkpoint_name']}")
        print(f"   Accuracy: {best_ckpt['results']['accuracy']:.4f} ({best_ckpt['results']['accuracy']*100:.2f}%)")
        print(f"   F1-Score: {best_ckpt['results']['f1_macro']:.4f} (macro)")
        
        if raw_results:
            best_acc_imp = best_ckpt['results']['accuracy'] - raw_results['accuracy']
            best_f1_imp = best_ckpt['results']['f1_macro'] - raw_results['f1_macro']
            best_rel_imp = (best_acc_imp / raw_results['accuracy'] * 100) if raw_results['accuracy'] > 0 else 0
            print(f"   📈 Improvement vs Raw: Accuracy {best_acc_imp:+.4f} ({best_acc_imp*100:+.2f}%), Relative {best_rel_imp:+.2f}%")
            print(f"                          F1-Score {best_f1_imp:+.4f} ({best_f1_imp*100:+.2f}%)")
    
    # Save detailed results to JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    summary_data = {
        'evaluation_time': timestamp,
        'checkpoint_directory': checkpoint_dir,
        'num_checkpoints_evaluated': len(all_checkpoint_results),
        'raw_model': {
            'path': RAW_MODEL_PATH,
            'results': raw_results if raw_results else 'not_evaluated'
        },
        'checkpoints': [
            {
                'name': ckpt_info['checkpoint_name'],
                'path': ckpt_info['checkpoint_path'],
                'metrics': {
                    'accuracy': ckpt_info['results']['accuracy'],
                    'precision_macro': ckpt_info['results']['precision_macro'],
                    'recall_macro': ckpt_info['results']['recall_macro'],
                    'f1_macro': ckpt_info['results']['f1_macro'],
                    'precision_weighted': ckpt_info['results']['precision_weighted'],
                    'recall_weighted': ckpt_info['results']['recall_weighted'],
                    'f1_weighted': ckpt_info['results']['f1_weighted']
                },
                'improvements_vs_raw': {
                    'accuracy_delta': ckpt_info['results']['accuracy'] - raw_results['accuracy'] if raw_results else None,
                    'f1_delta': ckpt_info['results']['f1_macro'] - raw_results['f1_macro'] if raw_results else None,
                    'precision_delta': ckpt_info['results']['precision_macro'] - raw_results['precision_macro'] if raw_results else None,
                    'recall_delta': ckpt_info['results']['recall_macro'] - raw_results['recall_macro'] if raw_results else None
                } if raw_results else None
            }
            for ckpt_info in all_checkpoint_results
        ]
    }
    
    summary_file = os.path.join(OUTPUT_DIR, f"all_checkpoints_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n💾 All results saved to: {summary_file}")
    print("="*80 + "\n")

def print_comparison(summary):
    """Print formatted comparison results."""
    print("\n" + "="*80)
    print("📊 ART DATASET EVALUATION: RAW vs FINE-TUNED MODEL")
    print("="*80)
    
    raw_metrics = summary['raw_model']['metrics']
    ft_metrics = summary['finetuned_model']['metrics']
    
    print("\n🤖 RAW MODEL:")
    print(f"   Accuracy:  {raw_metrics['accuracy']:.4f} ({raw_metrics['accuracy']*100:.2f}%) - {summary['raw_model']['correct']}/{summary['raw_model']['total']} correct")
    print(f"   Precision: {raw_metrics['precision_macro']:.4f} (macro), {raw_metrics['precision_weighted']:.4f} (weighted)")
    print(f"   Recall:    {raw_metrics['recall_macro']:.4f} (macro), {raw_metrics['recall_weighted']:.4f} (weighted)")
    print(f"   F1-Score:  {raw_metrics['f1_macro']:.4f} (macro), {raw_metrics['f1_weighted']:.4f} (weighted)")
    
    print("\n🎯 FINE-TUNED MODEL:")
    print(f"   Checkpoint: {os.path.basename(summary['finetuned_model']['checkpoint'])}")
    val_score = summary['finetuned_model']['validation_score']
    val_score_str = f"{val_score:.4f}" if isinstance(val_score, (int, float)) else str(val_score)
    print(f"   Validation Score: {val_score_str}")
    print(f"   Accuracy:  {ft_metrics['accuracy']:.4f} ({ft_metrics['accuracy']*100:.2f}%) - {summary['finetuned_model']['correct']}/{summary['finetuned_model']['total']} correct")
    print(f"   Precision: {ft_metrics['precision_macro']:.4f} (macro), {ft_metrics['precision_weighted']:.4f} (weighted)")
    print(f"   Recall:    {ft_metrics['recall_macro']:.4f} (macro), {ft_metrics['recall_weighted']:.4f} (weighted)")
    print(f"   F1-Score:  {ft_metrics['f1_macro']:.4f} (macro), {ft_metrics['f1_weighted']:.4f} (weighted)")
    
    print("\n📈 IMPROVEMENTS:")
    comp = summary['comparison']
    acc_imp = comp['accuracy_improvement']
    acc_rel = comp['accuracy_relative_improvement_percent']
    
    print(f"   Accuracy:  {acc_imp:+.4f} ({acc_imp*100:+.2f}%) | Relative: {acc_rel:+.2f}%")
    print(f"   Precision: {comp['precision_improvement']:+.4f} ({comp['precision_improvement']*100:+.2f}%)")
    print(f"   Recall:    {comp['recall_improvement']:+.4f} ({comp['recall_improvement']*100:+.2f}%)")
    print(f"   F1-Score:  {comp['f1_improvement']:+.4f} ({comp['f1_improvement']*100:+.2f}%)")
    
    print("\n" + "-"*80)
    
    if comp['overall_improved']:
        print("✅ RESULT: Fine-tuning on your dataset IMPROVED performance on ART!")
        print(f"   • Accuracy improved by {acc_rel:.2f}% (relative)")
        print(f"   • F1-Score improved by {comp['f1_improvement']*100:+.2f}%")
        print(f"   The model shows better generalization to abductive reasoning tasks.")
    elif acc_imp < 0:
        print("⚠️  RESULT: Fine-tuning on your dataset DECREASED performance on ART.")
        print(f"   • Accuracy decreased by {acc_rel:.2f}% (relative)")
        print(f"   • This suggests potential overfitting to your training data.")
    else:
        print("➖ RESULT: Fine-tuning had NO SIGNIFICANT IMPACT on ART performance.")
        print(f"   The model maintained baseline abductive reasoning ability.")
    
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Evaluate raw vs fine-tuned model on ART dataset')
    parser.add_argument('--max_samples', type=int, default=None, 
        help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test', 'validation'],
        help='Dataset split to use (default: train). Note: AIME 2025 dataset may only have "train" split.')
    parser.add_argument('--cuda_device', type=str, default='0',
        help='CUDA device to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1,
        help='Batch size for evaluation. Higher values (4-16) are faster but use more GPU memory (default: 1)')
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.checkpoint_path and args.checkpoint_dir:
        print("❌ Error: Cannot use both --checkpoint_path and --checkpoint_dir")
        print("   Use --checkpoint_path for a single checkpoint")
        print("   Use --checkpoint_dir to evaluate all checkpoints in a directory")
        return
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    
    # If checkpoint_dir is provided, evaluate all checkpoints
    if args.checkpoint_dir:
        evaluate_all_checkpoints(args)
        return
    
    print("="*70)
    print("🚀 ART DATASET EVALUATION: RAW vs FINE-TUNED")
    print("="*70)
    print(f"Raw Model: {RAW_MODEL_PATH}")
    print(f"Training Dir: {TRAINING_DIR}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Batch Size: {args.batch_size}")
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
        raw_results = evaluate_on_art(raw_model, raw_tokenizer, args.max_samples, "Raw Model", args.batch_size)
        del raw_model  # Free memory
        torch.cuda.empty_cache()
    else:
        raw_results = None
        print("\n⏭️  Skipping raw model evaluation")
    
    # Evaluate fine-tuned model
    if not args.skip_finetuned:
        finetuned_model, finetuned_tokenizer = load_finetuned_model(best_checkpoint_info['path'], args.cuda_device)
        finetuned_results = evaluate_on_art(finetuned_model, finetuned_tokenizer, args.max_samples, "Fine-tuned Model", args.batch_size)
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

