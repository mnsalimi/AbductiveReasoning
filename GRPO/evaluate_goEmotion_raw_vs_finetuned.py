#!/usr/bin/env python3
"""
GoEmotions Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the GoEmotions emotion classification dataset.

Usage:
    python evaluate_goemotion_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
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

# GoEmotions emotion labels (27 emotions + neutral)
GOEMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

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

def create_goemotion_prompt(text):
    """Create a prompt for GoEmotions emotion classification.
    
    Args:
        text: The input text to classify
    
    Returns:
        system_prompt, user_prompt
    """
    emotions_list = ", ".join(GOEMOTION_LABELS)
    
    system_prompt = f"""You are an expert emotion classifier. Analyze the given text and identify the emotion(s) expressed.

Available emotions: {emotions_list}

IMPORTANT:
- Identify ALL emotions present in the text (can be multiple)
- Use only the emotions from the list above
- Format your answer as: "Emotions: emotion1, emotion2, ..." or "Emotion: emotion1"
- Be precise and consider the context"""
    
    user_prompt = f"""Text: "{text}"

What emotion(s) are expressed in this text? Provide your answer in the format "Emotions: [list]" or "Emotion: [emotion]"."""
    
    return system_prompt, user_prompt

def extract_emotions(response, valid_emotions=GOEMOTION_LABELS):
    """Extract emotion labels from model response.
    
    Returns:
        list: List of extracted emotion labels, or None if extraction fails
    """
    response = response.strip().lower()
    
    # Try various patterns
    patterns = [
        r'emotions?:\s*([^\n]+)',
        r'the emotions? (?:is|are)\s*:?\s*([^\n]+)',
        r'detected emotions?:\s*([^\n]+)',
        r'(?:i detect|i identify|i sense)\s*(?:the emotions?)?\s*:?\s*([^\n]+)',
    ]
    
    extracted_text = None
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()
            break
    
    if not extracted_text:
        # Try to find any valid emotion words in the response
        found_emotions = []
        for emotion in valid_emotions:
            if emotion.lower() in response:
                found_emotions.append(emotion.lower())
        
        if found_emotions:
            return list(set(found_emotions))  # Remove duplicates
        return None
    
    # Parse the extracted text
    # Remove common separators and clean up
    extracted_text = extracted_text.replace(' and ', ', ')
    extracted_text = extracted_text.replace('&', ',')
    extracted_text = re.sub(r'[.!?;]', '', extracted_text)
    
    # Split by commas
    emotion_candidates = [e.strip() for e in extracted_text.split(',')]
    
    # Filter to valid emotions
    valid_emotions_lower = [e.lower() for e in valid_emotions]
    detected_emotions = []
    
    for candidate in emotion_candidates:
        candidate_clean = candidate.strip().lower()
        # Remove quotes
        candidate_clean = candidate_clean.replace('"', '').replace("'", '')
        
        # Check if it's a valid emotion
        if candidate_clean in valid_emotions_lower:
            detected_emotions.append(candidate_clean)
        else:
            # Check for partial matches
            for valid_emotion in valid_emotions_lower:
                if valid_emotion in candidate_clean or candidate_clean in valid_emotion:
                    detected_emotions.append(valid_emotion)
                    break
    
    if not detected_emotions:
        return None
    
    return list(set(detected_emotions))  # Remove duplicates

def evaluate_on_goemotion(model, tokenizer, max_samples=None, model_name="Model", batch_size=1, split="test"):
    """Evaluate model on GoEmotions dataset with batch processing support."""
    print(f"\n🔍 Evaluating {model_name} on GoEmotions dataset (split: {split})...")
    print(f"   Batch size: {batch_size}")
    
    # Load GoEmotions dataset from HuggingFace
    print("Loading GoEmotions dataset...")
    
    try:
        # Load the simplified version of GoEmotions
        dataset = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
        print(f"Loaded {len(dataset)} samples from GoEmotions (simplified) dataset")
        
    except Exception as e1:
        print(f"Failed to load GoEmotions (simplified): {e1}")
        try:
            # Try raw version
            dataset = load_dataset("google-research-datasets/go_emotions", "raw", split=split)
            print(f"Loaded {len(dataset)} samples from GoEmotions (raw) dataset")
        except Exception as e2:
            print(f"❌ Error loading dataset: {e2}")
            print("\n💡 Make sure you have internet connection and the dataset is accessible.")
            return None
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Evaluating on {len(dataset)} samples (limited)")
    else:
        print(f"Evaluating on {len(dataset)} samples (full {split} set)")
    
    results = []
    all_true_labels = []
    all_pred_labels = []
    failed_extractions = 0
    
    # Process in batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {model_name}"):
        # Get batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]
        
        # Handle both single sample and batch cases
        if not isinstance(batch['text'], list):
            batch = {k: [v] for k, v in batch.items()}
        
        batch_size_actual = len(batch['text'])
        
        # Prepare prompts for batch
        formatted_prompts = []
        true_labels_batch = []
        batch_data = []
        
        for i in range(batch_size_actual):
            text = batch['text'][i]
            
            # Get true labels
            # GoEmotions has 'labels' field which is a list of label indices
            true_label_indices = batch['labels'][i]
            if not isinstance(true_label_indices, list):
                true_label_indices = [true_label_indices]
            
            # Convert indices to emotion names
            true_emotions = [GOEMOTION_LABELS[idx] for idx in true_label_indices]
            
            # Create prompt
            system_prompt, user_prompt = create_goemotion_prompt(text)
            
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
            true_labels_batch.append(true_emotions)
            batch_data.append({
                'text': text,
                'true_emotions': true_emotions,
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
                max_new_tokens=128,
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
            
            # Extract emotions from response
            predicted_emotions = extract_emotions(response)
            
            if predicted_emotions is None:
                failed_extractions += 1
                predicted_emotions = []
            
            true_emotions = true_labels_batch[i]
            
            # Convert to multi-hot vectors for evaluation
            true_vector = [1 if label in [e.lower() for e in true_emotions] else 0 
                          for label in [e.lower() for e in GOEMOTION_LABELS]]
            pred_vector = [1 if label in predicted_emotions else 0 
                          for label in [e.lower() for e in GOEMOTION_LABELS]]
            
            all_true_labels.append(true_vector)
            all_pred_labels.append(pred_vector)
            
            # Calculate exact match for this sample
            exact_match = set([e.lower() for e in true_emotions]) == set(predicted_emotions)
            
            # Store result
            results.append({
                'sample_id': batch_data[i]['id'],
                'text': batch_data[i]['text'],
                'true_emotions': true_emotions,
                'predicted_emotions': predicted_emotions,
                'response': response,
                'exact_match': exact_match
            })
    
    # Calculate metrics
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    
    # Exact match accuracy (all labels must match)
    exact_match_count = sum(1 for r in results if r['exact_match'])
    exact_match_accuracy = exact_match_count / len(results) if results else 0.0
    
    # Hamming accuracy (fraction of correct labels per sample, averaged)
    hamming_accuracy = accuracy_score(all_true_labels, all_pred_labels)
    
    # Micro/Macro F1 scores
    micro_f1 = f1_score(all_true_labels, all_pred_labels, average='micro', zero_division=0)
    macro_f1 = f1_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
    
    # Precision and recall
    micro_precision = precision_score(all_true_labels, all_pred_labels, average='micro', zero_division=0)
    micro_recall = recall_score(all_true_labels, all_pred_labels, average='micro', zero_division=0)
    
    # Extraction rate
    extraction_rate = (len(results) - failed_extractions) / len(results) if results else 0.0
    
    print(f"\n📊 {model_name} Results:")
    print(f"   Exact Match Accuracy: {exact_match_accuracy:.4f} ({exact_match_accuracy*100:.2f}%) - {exact_match_count}/{len(results)} exact matches")
    print(f"   Hamming Accuracy:     {hamming_accuracy:.4f} ({hamming_accuracy*100:.2f}%)")
    print(f"   Micro F1:             {micro_f1:.4f}")
    print(f"   Macro F1:             {macro_f1:.4f}")
    print(f"   Micro Precision:      {micro_precision:.4f}")
    print(f"   Micro Recall:         {micro_recall:.4f}")
    print(f"   Extraction Rate:      {extraction_rate:.4f} ({extraction_rate*100:.2f}%)")
    print(f"   Failed extractions:   {failed_extractions}/{len(results)} ({failed_extractions/len(results)*100:.1f}%)")
    
    return {
        'exact_match_accuracy': exact_match_accuracy,
        'hamming_accuracy': hamming_accuracy,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'extraction_rate': extraction_rate,
        'exact_match_count': exact_match_count,
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
        'dataset': 'GoEmotions',
        'metrics': {
            'exact_match_accuracy': raw_results['exact_match_accuracy'],
            'hamming_accuracy': raw_results['hamming_accuracy'],
            'micro_f1': raw_results['micro_f1'],
            'macro_f1': raw_results['macro_f1'],
            'extraction_rate': raw_results['extraction_rate']
        },
        'exact_match_count': raw_results['exact_match_count'],
        'total': raw_results['total'],
        'failed_extractions': raw_results['failed_extractions'],
        'detailed_results': raw_results['results'][:100]  # Save first 100 for space
    }
    
    raw_file = os.path.join(output_dir, f"raw_model_goemotion_results_{timestamp}.json")
    with open(raw_file, 'w') as f:
        json.dump(raw_output, f, indent=2)
    print(f"\n💾 Raw model results saved to: {raw_file}")
    
    # Save fine-tuned model results
    finetuned_output = {
        'base_model': RAW_MODEL_PATH,
        'checkpoint': best_checkpoint_info['path'],
        'validation_score': best_checkpoint_info['score'],
        'evaluation_time': timestamp,
        'dataset': 'GoEmotions',
        'metrics': {
            'exact_match_accuracy': finetuned_results['exact_match_accuracy'],
            'hamming_accuracy': finetuned_results['hamming_accuracy'],
            'micro_f1': finetuned_results['micro_f1'],
            'macro_f1': finetuned_results['macro_f1'],
            'extraction_rate': finetuned_results['extraction_rate']
        },
        'exact_match_count': finetuned_results['exact_match_count'],
        'total': finetuned_results['total'],
        'failed_extractions': finetuned_results['failed_extractions'],
        'detailed_results': finetuned_results['results'][:100]  # Save first 100 for space
    }
    
    finetuned_file = os.path.join(output_dir, f"finetuned_model_goemotion_results_{timestamp}.json")
    with open(finetuned_file, 'w') as f:
        json.dump(finetuned_output, f, indent=2)
    print(f"💾 Fine-tuned model results saved to: {finetuned_file}")
    
    # Save comparison summary
    improvement_exact = finetuned_results['exact_match_accuracy'] - raw_results['exact_match_accuracy']
    improvement_f1 = finetuned_results['micro_f1'] - raw_results['micro_f1']
    
    summary = {
        'evaluation_time': timestamp,
        'dataset': 'GoEmotions',
        'split': 'test',
        'num_samples': raw_results['total'],
        'raw_model': {
            'path': RAW_MODEL_PATH,
            'metrics': {
                'exact_match_accuracy': raw_results['exact_match_accuracy'],
                'hamming_accuracy': raw_results['hamming_accuracy'],
                'micro_f1': raw_results['micro_f1'],
                'macro_f1': raw_results['macro_f1'],
                'extraction_rate': raw_results['extraction_rate']
            }
        },
        'finetuned_model': {
            'base_model': RAW_MODEL_PATH,
            'checkpoint': best_checkpoint_info['path'],
            'validation_score': best_checkpoint_info['score'],
            'metrics': {
                'exact_match_accuracy': finetuned_results['exact_match_accuracy'],
                'hamming_accuracy': finetuned_results['hamming_accuracy'],
                'micro_f1': finetuned_results['micro_f1'],
                'macro_f1': finetuned_results['macro_f1'],
                'extraction_rate': finetuned_results['extraction_rate']
            }
        },
        'comparison': {
            'exact_match_improvement': improvement_exact,
            'micro_f1_improvement': improvement_f1,
            'overall_improved': improvement_f1 > 0
        }
    }
    
    summary_file = os.path.join(output_dir, f"goemotion_comparison_summary_{timestamp}.json")
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
    print("🚀 GOEMOTION EVALUATION: ALL CHECKPOINTS")
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
        raw_results = evaluate_on_goemotion(raw_model, raw_tokenizer, args.max_samples, "Raw Model", args.batch_size, args.split)
        if raw_results is None:
            print("❌ Failed to evaluate raw model")
            return
        del raw_model
        torch.cuda.empty_cache()
        print(f"\n✅ Raw model evaluation complete")
        print(f"   Exact Match Accuracy: {raw_results['exact_match_accuracy']:.4f}")
        print(f"   Micro F1: {raw_results['micro_f1']:.4f}")
    
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
            finetuned_results = evaluate_on_goemotion(
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
            print(f"   Exact Match: {finetuned_results['exact_match_accuracy']:.4f} ({finetuned_results['exact_match_accuracy']*100:.2f}%)")
            print(f"   Micro F1: {finetuned_results['micro_f1']:.4f}")
            
            # Show improvement vs raw model if available
            if raw_results:
                f1_improvement = finetuned_results['micro_f1'] - raw_results['micro_f1']
                exact_improvement = finetuned_results['exact_match_accuracy'] - raw_results['exact_match_accuracy']
                print(f"   📈 Improvement vs Raw: F1 {f1_improvement:+.4f}, Exact Match {exact_improvement:+.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {ckpt_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("📊 SUMMARY: ALL CHECKPOINTS COMPARISON (GoEmotions)")
    print("="*80)
    
    if raw_results:
        print(f"\n🤖 RAW MODEL:")
        print(f"   Exact Match: {raw_results['exact_match_accuracy']:.4f}")
        print(f"   Micro F1:    {raw_results['micro_f1']:.4f}")
    
    print(f"\n🎯 FINE-TUNED CHECKPOINTS:")
    if raw_results:
        print(f"   {'Checkpoint':<20} {'Exact Match':<15} {'Micro F1':<15} {'EM Δ':<12} {'F1 Δ':<12}")
        print(f"   {'-'*80}")
        
        for ckpt_info in all_checkpoint_results:
            res = ckpt_info['results']
            em_delta = res['exact_match_accuracy'] - raw_results['exact_match_accuracy']
            f1_delta = res['micro_f1'] - raw_results['micro_f1']
            
            print(f"   {ckpt_info['checkpoint_name']:<20} "
                  f"{res['exact_match_accuracy']:.4f}         "
                  f"{res['micro_f1']:.4f}         "
                  f"{em_delta:+.4f}      "
                  f"{f1_delta:+.4f}")
    
    # Find best checkpoint
    if all_checkpoint_results:
        best_ckpt = max(all_checkpoint_results, key=lambda x: x['results']['micro_f1'])
        print(f"\n🏆 BEST CHECKPOINT: {best_ckpt['checkpoint_name']}")
        print(f"   Micro F1: {best_ckpt['results']['micro_f1']:.4f}")
        print(f"   Exact Match: {best_ckpt['results']['exact_match_accuracy']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    summary_data = {
        'evaluation_time': timestamp,
        'dataset': 'GoEmotions',
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
                    'exact_match_accuracy': ckpt_info['results']['exact_match_accuracy'],
                    'micro_f1': ckpt_info['results']['micro_f1'],
                    'macro_f1': ckpt_info['results']['macro_f1']
                }
            }
            for ckpt_info in all_checkpoint_results
        ]
    }
    
    summary_file = os.path.join(OUTPUT_DIR, f"goemotion_all_checkpoints_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n💾 All results saved to: {summary_file}")
    print("="*80 + "\n")

def print_comparison(summary):
    """Print formatted comparison results."""
    print("\n" + "="*80)
    print("📊 GOEMOTION EVALUATION: RAW vs FINE-TUNED MODEL")
    print("="*80)
    
    raw_metrics = summary['raw_model']['metrics']
    ft_metrics = summary['finetuned_model']['metrics']
    
    print("\n🤖 RAW MODEL:")
    print(f"   Exact Match:  {raw_metrics['exact_match_accuracy']:.4f} ({raw_metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"   Micro F1:     {raw_metrics['micro_f1']:.4f}")
    print(f"   Macro F1:     {raw_metrics['macro_f1']:.4f}")
    
    print("\n🎯 FINE-TUNED MODEL:")
    print(f"   Checkpoint: {os.path.basename(summary['finetuned_model']['checkpoint'])}")
    print(f"   Exact Match:  {ft_metrics['exact_match_accuracy']:.4f} ({ft_metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"   Micro F1:     {ft_metrics['micro_f1']:.4f}")
    print(f"   Macro F1:     {ft_metrics['macro_f1']:.4f}")
    
    print("\n📈 IMPROVEMENTS:")
    comp = summary['comparison']
    em_imp = comp['exact_match_improvement']
    f1_imp = comp['micro_f1_improvement']
    
    print(f"   Exact Match:  {em_imp:+.4f} ({em_imp*100:+.2f}%)")
    print(f"   Micro F1:     {f1_imp:+.4f} ({f1_imp*100:+.2f}%)")
    
    print("\n" + "-"*80)
    
    if comp['overall_improved']:
        print("✅ RESULT: Fine-tuning IMPROVED emotion classification performance!")
        print(f"   • F1 score improved by {f1_imp:.4f}")
    else:
        print("⚠️  RESULT: Fine-tuning did not improve emotion classification.")
    
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Evaluate raw vs fine-tuned model on GoEmotions dataset')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum number of samples to evaluate (default: all samples)')
    parser.add_argument('--cuda_device', type=str, default='0',
                       help='CUDA device to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation (default: 4)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'validation'],
                       help='Dataset split to use (default: test)')
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
    print("🚀 GOEMOTION EVALUATION: RAW vs FINE-TUNED")
    print("="*70)
    print(f"Raw Model: {RAW_MODEL_PATH}")
    print(f"Training Dir: {TRAINING_DIR}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Split: {args.split}")
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
        raw_results = evaluate_on_goemotion(raw_model, raw_tokenizer, args.max_samples, "Raw Model", args.batch_size, args.split)
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
        finetuned_results = evaluate_on_goemotion(finetuned_model, finetuned_tokenizer, args.max_samples, "Fine-tuned Model", args.batch_size, args.split)
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
