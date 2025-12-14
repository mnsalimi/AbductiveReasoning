"""Shared utility functions for evaluation scripts."""

import os
import json
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM
from vllm.lora.request import LoRARequest

# Global state for LoRA requests
CURRENT_LORA_REQUEST = None
CURRENT_LORA_INT_ID = 0


def sanitize_name(name: str) -> str:
    """Convert a string into a safe identifier-style name."""
    return re.sub(r'\W|^(?=\d)', '_', name).upper()


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


def load_raw_model(raw_model_path, model_type, device):
    """Load the raw/base model."""
    print(f"\n🤖 Loading raw model from: {raw_model_path}")
    print(f"- Model type: {model_type}")
    
    tokenizer = AutoTokenizer.from_pretrained(raw_model_path, trust_remote_code=True)
    
    if model_type == "hf":
        model = AutoModelForCausalLM.from_pretrained(
            raw_model_path,
            torch_dtype=torch.float16,
            device_map={"": f"cuda:0"},
            trust_remote_code=True,
            load_in_4bit=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
    elif model_type == "vllm":
        model = LLM(
            model=raw_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            quantization="bitsandbytes"
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    print("✅ Raw model loaded successfully")
    
    return model, tokenizer


def load_finetuned_model(raw_model_path, checkpoint_path, model_type, device):
    """
    Load the fine-tuned model with LoRA adapter.
    For vLLM, creates and saves a global LoRARequest with a meaningful name.
    """
    global CURRENT_LORA_REQUEST, CURRENT_LORA_INT_ID

    print(f"\nLoading fine-tuned model (MODEL_TYPE={model_type})")
    print(f"Checkpoint path: {checkpoint_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(raw_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create meaningful lora_name
    raw_model_name = sanitize_name(Path(raw_model_path).stem)
    checkpoint_name = sanitize_name(Path(checkpoint_path).stem)
    meaningful_lora_name = f"LORA_{raw_model_name}_{checkpoint_name}"

    if model_type.lower() == "hf":
        base_model = AutoModelForCausalLM.from_pretrained(
            raw_model_path,
            torch_dtype=torch.float16,
            device_map={"": f'cuda:0'},
            trust_remote_code=True,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.eval()
        print("✅ HF LoRA model loaded successfully")
        return model, tokenizer

    elif model_type.lower() == "vllm":
        llm = LLM(
            model=raw_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            quantization="bitsandbytes",
            enable_lora=True,
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
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")


def get_lora_request():
    """Get the current LoRA request for vLLM."""
    return CURRENT_LORA_REQUEST

