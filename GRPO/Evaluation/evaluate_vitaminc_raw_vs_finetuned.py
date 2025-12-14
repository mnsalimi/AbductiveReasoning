#!/usr/bin/env python3
"""
VitaminC Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the VitaminC fact verification dataset.

Usage:
    python evaluate_vitaminc_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
import numpy as np
from datasets import load_dataset
from base_evaluator import BaseEvaluator

np.random.seed(42)


class VitaminCEvaluator(BaseEvaluator):
    """Evaluator for VitaminC dataset."""
    
    def get_dataset_name(self):
        return "vitaminc"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/vitaminc_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for VitaminC Fact Verification."""
        system_prompt = """
You are an expert fact-checker.
You will be given a Claim and a specific piece of Evidence.

The VitaminC dataset focuses on "contrastive evidence"—small changes in evidence can flip the label. 
Pay close attention to negations, numbers, and specific entities.

Your task:
1. Read the Claim and the Evidence carefully.
2. Determine if the Evidence SUPPORTS, REFUTES, or provides NOT ENOUGH INFO for the Claim.
3. Provide step-by-step reasoning.
4. Provide the final label.

Your entire output MUST use exactly the following format and nothing else:

<reasoning>
[Your step-by-step analysis]
</reasoning>
<answer>
[Output exactly one: SUPPORTS, REFUTES, NOT ENOUGH INFO]
</answer>
"""
        
        claim = example.get('claim', '')
        evidence_text = example.get('evidence', '')
        
        user_prompt = f"""
Claim:
{claim}

Evidence:
{evidence_text}

Based on the evidence provided, determine the veracity of the claim.
"""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the label from the <answer>...</answer> block."""
        if not response:
            return None

        match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
        
        if match:
            clean_answer = match.group(1).strip().upper()
            clean_answer = clean_answer.rstrip('.').rstrip('!')
            # Normalize common variants
            if clean_answer in {"SUPPORTS", "SUPPORT", "SUPPORTED"}:
                return "SUPPORTS"
            elif clean_answer in {"REFUTES", "REFUTE", "REFUTED"}:
                return "REFUTES"
            elif clean_answer in {"NOT ENOUGH INFO", "NOT ENOUGH INFORMATION", "INSUFFICIENT", "NEI"}:
                return "NOT ENOUGH INFO"
            return clean_answer
        
        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load VitaminC dataset with random sampling."""
        print(f"Loading tals/vitaminc dataset (split={split})...")
        dataset = load_dataset("tals/vitaminc", split="test")
        
        # VitaminC uses random sampling of 1000 examples
        indices = np.random.choice(len(dataset), int(1000), replace=False)
        dataset = dataset.select(indices)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (label 0..2 => SUPPORTS/REFUTES/NOT ENOUGH INFO)."""
        label_id = example.get('label', 2)
        LABEL_MAP = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
        return LABEL_MAP.get(label_id, "NOT ENOUGH INFO")
    
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
        return "claim"


def main():
    evaluator = VitaminCEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
