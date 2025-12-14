#!/usr/bin/env python3
"""
Climate-FEVER Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the Climate-FEVER fact verification dataset.

Usage:
    python evaluate_climate_fever_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class ClimateFeverEvaluator(BaseEvaluator):
    """Evaluator for Climate-FEVER dataset."""
    
    def get_dataset_name(self):
        return "climate_fever"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/climate_fever_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for Climate-FEVER Fact Verification."""
        system_prompt = """
You are an expert climate scientist and professional fact-checker.
You will be given a specific Claim and a list of Evidences.

Your task:
1. Read the Claim and the provided Evidence carefully.
2. Determine if the Evidence SUPPORTS or REFUTES the Claim, or if there is NOT ENOUGH INFO.
3. Provide step-by-step reasoning explaining which specific parts of the evidence support your decision.
4. Provide the final label.

Your entire output MUST use exactly the following format and nothing else:

<reasoning>
[Your step-by-step analysis of how the evidence relates to the claim]
</reasoning>
<answer>
[Output exactly one of these three options: SUPPORTS, REFUTES, NOT ENOUGH INFO]
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
            clean_answer = clean_answer.rstrip('.')
            # Normalize common variants
            if clean_answer in {"SUPPORTS", "SUPPORT", "SUPPORTED"}:
                return "SUPPORTS"
            elif clean_answer in {"REFUTES", "REFUTE", "REFUTED"}:
                return "REFUTES"
            elif clean_answer in {"NOT ENOUGH INFO", "NOT ENOUGH INFORMATION", "INSUFFICIENT", "NEI"}:
                return "NOT ENOUGH INFO"
            elif clean_answer == "DISPUTED":
                return "NOT ENOUGH INFO"  # Map DISPUTED to NOT ENOUGH INFO
            return clean_answer
        
        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load Climate-FEVER dataset."""
        print(f"Loading tdiggelm/climate_fever dataset (split={split})...")
        dataset = load_dataset("tdiggelm/climate_fever", split=split)
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (label 0..3 => SUPPORTS/REFUTES/NOT ENOUGH INFO/DISPUTED)."""
        label_id = example.get('label', 2)
        LABEL_MAP = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO", 3: "DISPUTED"}
        return LABEL_MAP.get(label_id, "NOT ENOUGH INFO")
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer."""
        if predicted_answer is None or true_answer is None:
            return False
        # Map DISPUTED to NOT ENOUGH INFO for comparison
        if true_answer == "DISPUTED":
            true_answer = "NOT ENOUGH INFO"
        return predicted_answer == true_answer
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return "FAILED"
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "claim"


def main():
    evaluator = ClimateFeverEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
