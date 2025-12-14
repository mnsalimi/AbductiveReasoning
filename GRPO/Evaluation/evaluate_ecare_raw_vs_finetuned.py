#!/usr/bin/env python3
"""
e-CARE Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the e-CARE causal reasoning dataset.

Usage:
    python evaluate_ecare_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
import numpy as np
from datasets import load_dataset
from base_evaluator import BaseEvaluator

np.random.seed(42)


class ECareEvaluator(BaseEvaluator):
    """Evaluator for e-CARE dataset."""
    
    def get_dataset_name(self):
        return "ecare"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/ecare_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for e-CARE Causal Reasoning (multiple-choice)."""
        system_prompt = """
You are an expert causal reasoner and careful multiple-choice evaluator.
You will be given a Premise, a Question Type (cause/effect), and two candidate Choices.

Your task:
1. Read the Premise and Question Type carefully.
2. Decide which choice (CHOICE1 or CHOICE2) is the more plausible answer to the question:
   - If Question Type is "cause": pick the choice that best causes the Premise.
   - If Question Type is "effect": pick the choice that is the most likely result of the Premise.
3. Provide step-by-step reasoning that compares both choices.
4. Provide the final label.

Your entire output MUST use exactly the following format and nothing else:

<reasoning>
[Your step-by-step causal analysis comparing CHOICE1 vs CHOICE2]
</reasoning>
<answer>
[Output exactly one of these two options: CHOICE1, CHOICE2]
</answer>
""".strip()
        
        premise = example.get('premise', '')
        qtype = example.get('question', '')  # "cause" or "effect"
        choice1 = example.get('choice1', '')
        choice2 = example.get('choice2', '')
        
        evidence_text = f"""Question Type: {qtype}

CHOICE1:
{choice1}

CHOICE2:
{choice2}""".strip()
        
        user_prompt = f"""
Premise:
{premise}

{evidence_text}

Choose the correct answer based on causal reasoning.
""".strip()
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the label from the <answer>...</answer> block."""
        if not response:
            return None

        match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
        if not match:
            return None

        raw = match.group(1).strip().upper()
        raw = raw.rstrip('.')

        # Normalize common ways models might answer
        if raw in {"CHOICE1", "CHOICE 1", "OPTION1", "OPTION 1", "A", "1"}:
            return "CHOICE1"
        if raw in {"CHOICE2", "CHOICE 2", "OPTION2", "OPTION 2", "B", "2"}:
            return "CHOICE2"

        # If the model writes something like "The answer is CHOICE1"
        if "CHOICE1" in raw or "OPTION1" in raw:
            return "CHOICE1"
        if "CHOICE2" in raw or "OPTION2" in raw:
            return "CHOICE2"

        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load e-CARE dataset with random sampling."""
        print(f"Loading 12ml/e-CARE dataset (split={split})...")
        dataset = load_dataset("12ml/e-CARE", split="validation")
        
        # e-CARE uses random sampling of 1000 examples
        indices = np.random.choice(len(dataset), int(1000), replace=False)
        dataset = dataset.select(indices)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (label 0/1 -> CHOICE1/CHOICE2)."""
        label_id = int(example.get('label', 0))
        LABEL_MAP = {0: "CHOICE1", 1: "CHOICE2"}
        return LABEL_MAP.get(label_id, "CHOICE1")
    
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
        return "premise"
    
    def evaluate_on_dataset(self, model, tokenizer, max_samples=None, model_name="Model", 
                           batch_size=1, split='validation'):
        """Override to handle e-CARE-specific evaluation with left padding."""
        from base_evaluator import BaseEvaluator
        import torch
        from tqdm import tqdm
        import time
        
        print(f"\n🔍 Evaluating {model_name} on {self.get_dataset_name()} dataset...")
        print(f"   Batch size: {batch_size}")
        print(f"   Split: {split}")
        
        dataset = self.load_dataset(split, max_samples)
        
        if max_samples:
            print(f"Evaluating on {len(dataset)} samples (limited)")
        else:
            print(f"Evaluating on {len(dataset)} samples (full dataset)")
        
        # e-CARE uses left padding
        try:
            tokenizer.padding_side = "left"
        except Exception:
            pass
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        
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
                    'id': example.get('idx', start_idx + i)
                })
            
            if self.model_type.lower() == 'hf':
                inputs = tokenizer(
                    formatted_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    )
                
                seq_len = inputs["input_ids"].shape[1]
                for i in range(batch_size_actual):
                    response = tokenizer.decode(outputs[i][seq_len:], skip_special_tokens=True)
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
    evaluator = ECareEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
