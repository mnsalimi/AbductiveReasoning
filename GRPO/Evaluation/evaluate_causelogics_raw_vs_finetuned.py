#!/usr/bin/env python3
"""
CauseLogics Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the CauseLogics abductive logical decision dataset.

Usage:
    python evaluate_causelogics_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from datasets import load_dataset
from base_evaluator import BaseEvaluator

np.random.seed(42)


def _normalize_label(raw_label) -> str:
    """Normalize CauseLogics labels to TRUE/FALSE."""
    if isinstance(raw_label, bool):
        return "TRUE" if raw_label else "FALSE"
    if isinstance(raw_label, int):
        return "TRUE" if raw_label == 1 else "FALSE"
    if isinstance(raw_label, str):
        x = raw_label.strip().lower()
        if x in {"true", "t", "yes", "y", "1"}:
            return "TRUE"
        if x in {"false", "f", "no", "n", "0"}:
            return "FALSE"
    return "FALSE"


def _resolve_causelogics_dir(data_dir: Union[str, Path]) -> Path:
    """Resolve CauseLogics directory path."""
    p = Path(data_dir).expanduser().resolve()
    if (p / "CauseLogics").is_dir():
        return (p / "CauseLogics").resolve()
    return p


def _find_data_files(root: Path, split: str = "test", level: Optional[Union[int, str]] = None) -> List[Path]:
    """Find CauseLogics data files."""
    split = (split or "test").lower()
    split_aliases = {
        "train": ["train"],
        "test": ["test"],
        "validation": ["validation", "valid", "val", "dev"],
        "dev": ["dev", "valid", "val", "validation"],
        "all": [],
    }
    tokens = split_aliases.get(split, [split])
    
    level_tokens = []
    if level is not None:
        lvl = str(level).lower().replace("level", "").strip()
        level_tokens = [f"level{lvl}", f"lv{lvl}", f"level_{lvl}", f"level-{lvl}", f"c{lvl}", f"levelc{lvl}"]
    
    exts = ["*.jsonl", "*.json", "*.jsonl.gz", "*.json.gz"]
    candidates = []
    
    for ext in exts:
        for fp in root.rglob(ext):
            name = fp.as_posix().lower()
            if level_tokens and not any(tok in name for tok in level_tokens):
                continue
            if tokens and not any(tok in name for tok in tokens):
                continue
            candidates.append(fp)
    
    if not candidates:
        for ext in exts:
            for fp in root.rglob(ext):
                name = fp.as_posix().lower()
                if level_tokens and not any(tok in name for tok in level_tokens):
                    continue
                candidates.append(fp)
    
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any .json/.jsonl data files under: {root}\n"
            f"Tip: pass data_dir pointing to the repo root or the CauseLogics folder."
        )
    
    return candidates


def load_causelogics_dataset(data_dir: str, split: str = "test", level: Optional[Union[int, str]] = None):
    """Load CauseLogics from local files."""
    root = _resolve_causelogics_dir(data_dir)
    files = _find_data_files(root, split=split, level=level)
    
    split_key = split.lower()
    if split_key == "all":
        split_key = "test"
    
    ds = load_dataset(
        "json",
        data_files={split_key: [str(p) for p in files]},
        split=split_key
    )
    return ds


class CauseLogicsEvaluator(BaseEvaluator):
    """Evaluator for CauseLogics dataset."""
    
    def __init__(self, *args, data_dir=None, level=None, **kwargs):
        """Initialize with optional data_dir and level parameters."""
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir or "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/CauseJudger"
        self.level = level or 3
    
    def get_dataset_name(self):
        return "causelogics"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/causelogics_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for CauseLogics (abductive logical decision)."""
        system_prompt = """
You are an expert logician and careful reasoning assistant.
You will be given:
- a set of Premises (facts),
- a set of Rules (implications),
- an observed Phenomenon,
- and a Possible Cause (a hypothesis).

Your task:
1. Assume the Possible Cause is added as an additional premise.
2. Using ONLY the given Premises + Rules (+ the Possible Cause), reason forward.
3. Decide whether the Phenomenon can be logically inferred.
   - If the Phenomenon can be inferred, the Possible Cause is TRUE.
   - If the Phenomenon cannot be inferred, the Possible Cause is FALSE.
4. Provide step-by-step reasoning referencing which premises/rules you used.
5. Output the final label.

Your entire output MUST use exactly the following format and nothing else:

<reasoning>
[Your step-by-step analysis]
</reasoning>
<answer>
[Output exactly one of these two options: TRUE, FALSE]
</answer>
"""
        
        # Handle different field name variations
        premises_raw = example.get("Premises") or example.get("premises")
        rules_raw = example.get("Rules") or example.get("rules")
        phenomenon = example.get("Phenomenon") or example.get("phenomenon")
        possible_cause = example.get("PossibleCause") or example.get("possible_cause")
        
        # Convert to text format
        if isinstance(premises_raw, list):
            premises_text = "\n".join([f"- {x}" for x in premises_raw])
        else:
            premises_text = f"- {premises_raw}" if premises_raw is not None else ""
        
        if isinstance(rules_raw, list):
            rules_text = "\n".join([f"- {x}" for x in rules_raw])
        else:
            rules_text = f"- {rules_raw}" if rules_raw is not None else ""
        
        user_prompt = f"""
Premises:
{premises_text}

Rules:
{rules_text}

Phenomenon:
{phenomenon}

Possible Cause:
{possible_cause}

Determine whether the Possible Cause is TRUE or FALSE.
"""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the label from the <answer>...</answer> block."""
        if not response:
            return None

        match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
        if match:
            clean_answer = match.group(1).strip().upper()
            clean_answer = clean_answer.rstrip('.')
            if clean_answer in {"TRUE", "FALSE"}:
                return clean_answer
        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load CauseLogics dataset."""
        print(f"Loading CauseLogics dataset (split={split}, level={self.level})...")
        dataset = load_causelogics_dataset(data_dir=self.data_dir, split=split, level=self.level)
        
        # CauseLogics uses random sampling of 300 examples per level
        indices = np.random.choice(len(dataset), int(300), replace=False)
        dataset = dataset.select(indices)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example."""
        label_raw = example.get("Label") or example.get("label")
        return _normalize_label(label_raw)
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer."""
        if predicted_answer is None or true_answer is None:
            return False
        return predicted_answer == true_answer
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return "FAILED"
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "Phenomenon"
    
    def evaluate_on_dataset(self, model, tokenizer, max_samples=None, model_name="Model", 
                           batch_size=1, split='test'):
        """Override to handle CauseLogics-specific evaluation."""
        from base_evaluator import BaseEvaluator
        import torch
        from tqdm import tqdm
        import time
        
        print(f"\n🔍 Evaluating {model_name} on {self.get_dataset_name()} dataset...")
        print(f"   Batch size: {batch_size}")
        print(f"   Split: {split} | Level: {self.level}")
        
        dataset = self.load_dataset(split, max_samples)
        
        if max_samples:
            print(f"Evaluating on {len(dataset)} samples (limited)")
        else:
            print(f"Evaluating on {len(dataset)} samples (full dataset)")
        
        results = []
        correct = 0
        total = 0
        failed_extractions = 0
        
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        btime = time.time()
        
        first_key = list(dataset[0].keys())[0]
        
        for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {model_name}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = dataset[start_idx:end_idx]
            
            if not isinstance(batch[first_key], list):
                batch = {k: [v] for k, v in batch.items()}
            
            batch_size_actual = len(batch[first_key])
            
            formatted_prompts = []
            true_answers = []
            batch_data = []
            
            for i in range(batch_size_actual):
                example = {k: batch[k][i] for k in batch.keys()}
                true_answer = self.get_true_answer(example)
                
                system_prompt, user_prompt = self.create_prompt(example)
                
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
                    formatted_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                formatted_prompts.append(formatted_prompt)
                true_answers.append(true_answer)
                batch_data.append({
                    'example': example,
                    'id': start_idx + i
                })
            
            if self.model_type.lower() == 'hf':
                inputs = tokenizer(
                    formatted_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                    )
                
                for i in range(batch_size_actual):
                    prompt_len = int(inputs["attention_mask"][i].sum().item())
                    response = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
                    
                    predicted_answer = self.extract_answer(response) or self._get_failed_answer()
                    
                    if predicted_answer == self._get_failed_answer():
                        failed_extractions += 1
                    
                    is_correct = (predicted_answer == true_answers[i])
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    results.append({
                        'problem_id': batch_data[i]['id'],
                        'problem': self._get_problem_text(batch_data[i]['example']),
                        'true_answer': true_answers[i],
                        'predicted_answer': predicted_answer,
                        'reasoning': response,
                        'correct': is_correct
                    })
            
            elif self.model_type.lower() == 'vllm':
                from vllm import SamplingParams
                from evaluator_utils import get_lora_request
                sampling = SamplingParams(max_tokens=256, temperature=0.0, top_p=1.0)
                vllm_outputs = model.generate(
                    formatted_prompts, 
                    sampling_params=sampling, 
                    lora_request=get_lora_request()
                )
                for i, out in enumerate(vllm_outputs):
                    response = out.outputs[0].text
                    predicted_answer = self.extract_answer(response) or self._get_failed_answer()
                    
                    if predicted_answer == self._get_failed_answer():
                        failed_extractions += 1
                    
                    is_correct = (predicted_answer == true_answers[i])
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    results.append({
                        'problem_id': batch_data[i]['id'],
                        'problem': self._get_problem_text(batch_data[i]['example']),
                        'true_answer': true_answers[i],
                        'predicted_answer': predicted_answer,
                        'reasoning': response,
                        'correct': is_correct
                    })
            else:
                raise ValueError(f"Unsupported MODEL_TYPE={self.model_type}")
        
        etime = time.time()
        print(f"Batch processing time: {etime - btime:.2f} seconds")
        accuracy = correct / total if total > 0 else 0.0
        extraction_rate = (total - failed_extractions) / total if total > 0 else 0.0
        
        print(f"\n📊 {model_name} Results:")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%) - {correct}/{total} correct")
        print(f"   Extraction Rate: {extraction_rate:.4f} ({extraction_rate*100:.2f}%) - {total - failed_extractions}/{total} extracted")
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


def main():
    evaluator = CauseLogicsEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
