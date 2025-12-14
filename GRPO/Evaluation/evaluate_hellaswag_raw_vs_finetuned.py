#!/usr/bin/env python3
"""
HellaSwag Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the HellaSwag commonsense reasoning dataset.

Usage:
    python evaluate_hellaswag_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
import numpy as np
from datasets import load_dataset
from base_evaluator import BaseEvaluator

np.random.seed(42)


class HellaSwagEvaluator(BaseEvaluator):
    """Evaluator for HellaSwag dataset."""
    
    def get_dataset_name(self):
        return "hellaswag"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/hellaswag_evaluation_results"
    
    def create_prompt(self, example):
    """Create a prompt for HellaSwag (4-way multiple choice)."""
    system_prompt = """
    You are an expert at commonsense reasoning.
    You will be given a short Context and four candidate Endings (A, B, C, D).

    Your task:
    1. Read the Context and the four Endings carefully.
    2. Select the single most plausible Ending that best completes the Context.
    3. Provide brief reasoning.
    4. Provide the final choice letter.

    Your entire output MUST use exactly the following format and nothing else:

    <reasoning>
    [Brief explanation of why the chosen ending best fits the context]
    </reasoning>
    <answer>
    [Output exactly one of these four options: A, B, C, D]
    </answer>
    """

        ctx = example.get('ctx', '')
        endings = example.get('endings', [])
        
    user_prompt = f"""
    Context:
    {ctx}

    Endings (append one ending to the context):
    A) {endings[0]}
    B) {endings[1]}
    C) {endings[2]}
    D) {endings[3]}

    Which ending best completes the context?
    """

    return system_prompt, user_prompt

    def extract_answer(self, response):
        """Extract the label from the <answer>...</answer> block."""
    if not response:
        return None

    match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
    if match:
        clean_answer = match.group(1).strip().upper()
        # normalize common trailing punctuation
        clean_answer = clean_answer.rstrip('.').rstrip(')').rstrip(':').strip()
            return self._normalize_hellaswag_choice(clean_answer)
    return None

    def _normalize_hellaswag_choice(self, ans):
        """Normalize extracted answers to one of {A,B,C,D}."""
    if ans is None:
        return None

    ans = ans.strip().upper()

    # Direct letter
    if ans in {"A", "B", "C", "D"}:
        return ans

    # Numeric index
    if ans in {"0", "1", "2", "3"}:
        return {"0": "A", "1": "B", "2": "C", "3": "D"}[ans]

    # Starts with a letter
    if len(ans) > 0 and ans[0] in {"A", "B", "C", "D"}:
        return ans[0]

    # Starts with a digit
    if len(ans) > 0 and ans[0] in {"0", "1", "2", "3"}:
        return {"0": "A", "1": "B", "2": "C", "3": "D"}[ans[0]]

    return None

    def load_dataset(self, split, max_samples=None):
        """Load HellaSwag dataset with special sampling."""
    print(f"Loading Rowan/hellaswag dataset (split={split})...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    
        # HellaSwag uses random sampling of 1000 examples
    indices = np.random.choice(len(dataset), int(1000), replace=False)
    dataset = dataset.select(indices)

    if max_samples:
            dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
        
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (label 0..3 => A..D)."""
        label_id = int(example.get('label', 0)) if example.get('label') is not None else None
    LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
        return LABEL_MAP.get(label_id, None)
    
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
        return "ctx"
    
    def evaluate_on_dataset(self, model, tokenizer, max_samples=None, model_name="Model", 
                           batch_size=1, split='validation'):
        """Override to handle HellaSwag-specific evaluation with smaller max_new_tokens."""
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
                    'id': example.get('ind', start_idx + i)
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
                        max_new_tokens=128,  # Smaller for MCQ
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )

        for i in range(batch_size_actual):
            input_length = inputs["input_ids"][i].shape[0]
            response = tokenizer.decode(outputs[i][input_length:], skip_special_tokens=True)

                    extracted = self.extract_answer(response)
                    predicted_answer = extracted if extracted is not None else self._get_failed_answer()

                    if predicted_answer == self._get_failed_answer():
                failed_extractions += 1

            is_correct = (true_answers[i] is not None) and (predicted_answer == true_answers[i])
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
                sampling = SamplingParams(max_tokens=128, temperature=0.0, top_p=1.0)
                vllm_outputs = model.generate(
                    formatted_prompts, 
                    sampling_params=sampling, 
                    lora_request=get_lora_request()
                )
                for i, out in enumerate(vllm_outputs):
                    response = out.outputs[0].text
                    extracted = self.extract_answer(response)
                    predicted_answer = extracted if extracted is not None else self._get_failed_answer()
                    
                    if predicted_answer == self._get_failed_answer():
                        failed_extractions += 1
                    
                    is_correct = (true_answers[i] is not None) and (predicted_answer == true_answers[i])
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
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Extraction Rate: {extraction_rate:.4f} ({extraction_rate*100:.2f}%)")
    print(f"   Time: {etime - btime:.1f}s")

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
    evaluator = HellaSwagEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
