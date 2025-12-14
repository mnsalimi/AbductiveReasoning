#!/usr/bin/env python3
"""
AIME 2025 Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the AIME 2025 math competition dataset.
AIME answers are integers from 0-999.

Usage:
    python evaluate_aime_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class AIMEEvaluator(BaseEvaluator):
    """Evaluator for AIME 2025 dataset."""
    
    def get_dataset_name(self):
        return "aime"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/aime_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for AIME math problem."""
        system_prompt = """You are an expert mathematician. Solve the following AIME (American Invitational Mathematics Examination) problem.

AIME answers are always integers between 0 and 999.

First, read the problem carefully and solve it step by step. Then give the final answer as a single integer between 0 and 999.

Your entire output MUST use exactly the following format and nothing else (no text before, between, or after these tags):

<reasoning>
[here you write your chain-of-thought reasoning and intermediate steps]
</reasoning>
<answer>
[here you output ONLY the final integer answer between 0 and 999, with no extra words]
</answer>"""
        
        problem = example.get('problem', '')
        user_prompt = f"""Problem: {problem}

Solve this problem step by step, then provide your final answer."""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the AIME numerical answer (integer 0–999) from the <answer>...</answer> block."""
        if not response:
            return None

        # Find the content inside <answer>...</answer>
        tag_match = re.search(
            r'<answer>\s*(.*?)\s*</answer>',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if not tag_match:
            return None

        answer_content = tag_match.group(1)
        # Clean common wrappers/symbols
        answer_content = answer_content.replace('$', '').strip()

        # Look for a 1–3 digit integer
        num_match = re.search(r'\b(\d{1,3})\b', answer_content)
        if not num_match:
            return None

        num = int(num_match.group(1))
        if 0 <= num <= 999:
            return num

        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load AIME 2025 dataset."""
        print(f"Loading AIME 2025 dataset (split={split})...")
        dataset = load_dataset("yentinglin/aime_2025", split=split)
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example."""
        return int(example.get('answer', 0))
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer."""
        return predicted_answer == true_answer
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "problem"


def main():
    evaluator = AIMEEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
