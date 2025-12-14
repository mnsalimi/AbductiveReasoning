#!/usr/bin/env python3
"""
StrategyQA Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the StrategyQA dataset with evidence handling.

Usage:
    python evaluate_strategyqa_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
import json
import numpy as np
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download
from base_evaluator import BaseEvaluator

np.random.seed(42)

_PARAGRAPH_ID_RE = re.compile(r".+-\d+$")  # e.g., "Genghis Khan-15"


def _normalize_for_arrow(ex: dict) -> dict:
    """Make columns Arrow-friendly by forcing inconsistent/nested fields into JSON strings."""
    ex = dict(ex)
    if "evidence" in ex:
        ex["evidence"] = json.dumps(ex["evidence"], ensure_ascii=False)
    for k in ["decomposition", "facts"]:
        if k in ex:
            ex[k] = json.dumps(ex[k], ensure_ascii=False)
    if "qid" not in ex and "id" in ex:
        ex["qid"] = ex["id"]
    return ex


def _get_evidence_obj(example: dict):
    """Convert evidence JSON string back to python object when needed."""
    ev = example.get("evidence", None)
    if ev is None:
        return None
    if isinstance(ev, str):
        try:
            return json.loads(ev)
        except Exception:
            return ev
    return ev


def _extract_paragraph_ids(obj):
    """Recursively extract paragraph ids from StrategyQA 'evidence' field."""
    seen = set()
    ordered = []

    def rec(x):
        if isinstance(x, str):
            if x in {"no_evidence", "operation"}:
                return
            if _PARAGRAPH_ID_RE.match(x):
                if x not in seen:
                    seen.add(x)
                    ordered.append(x)
        elif isinstance(x, list) or isinstance(x, tuple):
            for y in x:
                rec(y)
        elif isinstance(x, dict):
            for y in x.values():
                rec(y)

    rec(obj)
    return ordered


class StrategyQAEvaluator(BaseEvaluator):
    """Evaluator for StrategyQA dataset."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paragraphs_dict = None
    
    def get_dataset_name(self):
        return "strategyqa"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/strategyqa_evaluation_results"
    
    def _build_evidence_text(self, example, max_paragraphs=8, max_chars_per_paragraph=900):
        """Build evidence text from example."""
        chunks = []
        evidence_obj = _get_evidence_obj(example)

        if self.paragraphs_dict is not None and evidence_obj is not None:
            para_ids = _extract_paragraph_ids(evidence_obj)
            para_ids = para_ids[:max_paragraphs]
            for pid in para_ids:
                p = self.paragraphs_dict.get(pid)
                if not p:
                    continue
                title = p.get("title", "")
                content = (p.get("content", "") or "").strip().replace("\n", " ")
                if max_chars_per_paragraph and len(content) > max_chars_per_paragraph:
                    content = content[:max_chars_per_paragraph].rstrip() + "…"
                if title:
                    chunks.append(f"- ({pid}) {title}: {content}")
                else:
                    chunks.append(f"- ({pid}) {content}")

        # fallback to facts
        if not chunks and "facts" in example and example["facts"]:
            facts_obj = example["facts"]
            if isinstance(facts_obj, str):
                try:
                    facts_obj = json.loads(facts_obj)
                except Exception:
                    facts_obj = [facts_obj]
            if isinstance(facts_obj, list):
                for f in facts_obj[:max_paragraphs]:
                    chunks.append(f"- {str(f).strip()}")

        if not chunks:
            return "- (No evidence provided.)"
        return "\n".join(chunks)
    
    def create_prompt(self, example):
        """Create a prompt for StrategyQA (Yes/No QA with evidence)."""
        system_prompt = """
You are an expert at answering yes/no questions using provided evidence.

You will be given:
- A Question
- A list of Evidence paragraphs

Your task:
1. Read the Question and the provided Evidence carefully.
2. Decide whether the correct answer is YES or NO.
3. Provide step-by-step reasoning that uses specific parts of the evidence.
4. Provide the final label.

Your entire output MUST use exactly the following format and nothing else:

<reasoning>
[Your step-by-step analysis grounded in the evidence]
</reasoning>
<answer>
[Output exactly one of these two options: YES, NO]
</answer>
""".strip()
        
        question = example.get('question', '')
        evidence_text = self._build_evidence_text(example)
        
        user_prompt = f"""
Question:
{question}

Evidence:
{evidence_text}

Based on the evidence provided, answer the question.
""".strip()
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the label from the <answer>...</answer> block. Normalizes common variants."""
        if not response:
            return None

        match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
        if not match:
            return None

        clean_answer = match.group(1).strip().upper()
        clean_answer = clean_answer.rstrip(".")

        # Map common variants to YES/NO
        if clean_answer in {"TRUE", "T"}:
            clean_answer = "YES"
        elif clean_answer in {"FALSE", "F"}:
            clean_answer = "NO"
        elif clean_answer in {"Y"}:
            clean_answer = "YES"
        elif clean_answer in {"N"}:
            clean_answer = "NO"

        if clean_answer not in {"YES", "NO"}:
            return None

        return clean_answer
    
    def load_dataset(self, split, max_samples=None):
        """Load StrategyQA dataset with special handling."""
        print(f"Loading voidful/StrategyQA (split={split})...")
        
        try:
            filename = "strategyqa_train.json" if split.startswith("train") else "strategyqa_test.json"
            local_path = hf_hub_download(repo_id="voidful/StrategyQA", repo_type="dataset", filename=filename)
            
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Normalize each row so Arrow sees consistent scalar (string) columns
            data = [_normalize_for_arrow(ex) for ex in data]
            dataset = Dataset.from_list(data)
        except Exception as e:
            print(f"load_dataset failed ({type(e).__name__}: {e}). Falling back to hf_hub_download + Dataset.from_list.")
            filename = "strategyqa_train.json" if split.startswith("train") else "strategyqa_test.json"
            local_path = hf_hub_download(repo_id="voidful/StrategyQA", repo_type="dataset", filename=filename)
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
        
        # StrategyQA uses random sampling of 1000 examples
        indices = np.random.choice(len(dataset), int(1000), replace=False)
        dataset = dataset.select(indices)
        
        # Load paragraphs store (for evidence text)
        print("Loading strategyqa_train_paragraphs.json (paragraph evidence store)...")
        para_path = hf_hub_download(
            repo_id="voidful/StrategyQA",
            repo_type="dataset",
            filename="strategyqa_train_paragraphs.json",
        )
        with open(para_path, "r", encoding="utf-8") as f:
            self.paragraphs_dict = json.load(f)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (answer is bool, convert to YES/NO)."""
        answer = example.get('answer', None)
        if answer is None:
            return None
        return "YES" if bool(answer) else "NO"
    
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
        return "question"
    
    def evaluate_on_dataset(self, model, tokenizer, max_samples=None, model_name="Model", 
                           batch_size=1, split='train'):
        """Override to handle StrategyQA-specific evaluation with attention_mask."""
        import torch
        from tqdm import tqdm
        import time
        
        print(f"\n🔍 Evaluating {model_name} on StrategyQA...")
        print(f"   Split: {split}")
        print(f"   Batch size: {batch_size}")
        
        dataset = self.load_dataset(split, max_samples)
        
        if max_samples:
            print(f"Evaluating on {len(dataset)} samples (limited)")
        else:
            print(f"Evaluating on {len(dataset)} samples (full split)")
        
        results = []
        correct = 0
        total = 0
        failed_extractions = 0
        
        # StrategyQA labels are bools in train; test has no labels
        has_labels = "answer" in dataset.column_names
        
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
                question = example.get('question', '')
                qid = example.get('qid', start_idx + i)
                
                # True label (if available)
                if has_labels:
                    true_answer = "YES" if bool(example.get('answer', False)) else "NO"
                else:
                    true_answer = None
                
                system_prompt, user_prompt = self.create_prompt(example)
                
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    formatted_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                formatted_prompts.append(formatted_prompt)
                true_answers.append(true_answer)
                batch_data.append({"qid": qid, "question": question})
            
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
                        max_new_tokens=512,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                for i in range(batch_size_actual):
                    # Use attention_mask to get the true per-sample prompt length
                    prompt_len = int(inputs["attention_mask"][i].sum().item())
                    response = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
                    
                    predicted_answer = self.extract_answer(response)
                    if predicted_answer is None:
                        failed_extractions += 1
                        predicted_answer = self._get_failed_answer()
                    
                    is_correct = (true_answers[i] is not None) and (predicted_answer == true_answers[i])
                    if is_correct:
                        correct += 1
                    
                    total += 1
                    
                    results.append({
                        'problem_id': batch_data[i]['qid'],
                        'problem': batch_data[i]['question'],
                        'true_answer': true_answers[i],
                        'predicted_answer': predicted_answer,
                        'reasoning': response,
                        'correct': is_correct if true_answers[i] is not None else None
                    })
            
            elif self.model_type.lower() == 'vllm':
                from vllm import SamplingParams
                from evaluator_utils import get_lora_request
                sampling = SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0)
                vllm_outputs = model.generate(
                    formatted_prompts, 
                    sampling_params=sampling, 
                    lora_request=get_lora_request()
                )
                for i, out in enumerate(vllm_outputs):
                    response = out.outputs[0].text
                    predicted_answer = self.extract_answer(response)
                    if predicted_answer is None:
                        failed_extractions += 1
                        predicted_answer = self._get_failed_answer()
                    
                    is_correct = (true_answers[i] is not None) and (predicted_answer == true_answers[i])
                    if is_correct:
                        correct += 1
                    
                    total += 1
                    
                    results.append({
                        'problem_id': batch_data[i]['qid'],
                        'problem': batch_data[i]['question'],
                        'true_answer': true_answers[i],
                        'predicted_answer': predicted_answer,
                        'reasoning': response,
                        'correct': is_correct if true_answers[i] is not None else None
                    })
            else:
                raise ValueError(f"Unsupported MODEL_TYPE={self.model_type}")
        
        etime = time.time()
        
        # If no labels (e.g., test split), accuracy isn't defined
        if has_labels:
            accuracy = correct / total if total > 0 else 0.0
        else:
            accuracy = None
        
        extraction_rate = (total - failed_extractions) / total if total > 0 else 0.0
        
        print(f"\n📊 {model_name} Results:")
        if accuracy is not None:
            print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        else:
            print("   Accuracy:  N/A (no labels in this split)")
        print(f"   Extraction Rate: {extraction_rate:.4f} ({extraction_rate*100:.2f}%)")
        
        return {
            "accuracy": accuracy,
            "correct": correct if has_labels else None,
            "total": total,
            "failed_extractions": failed_extractions,
            "extraction_rate": extraction_rate,
            "time": etime - btime,
            "results": results,
        }


def main():
    evaluator = StrategyQAEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
