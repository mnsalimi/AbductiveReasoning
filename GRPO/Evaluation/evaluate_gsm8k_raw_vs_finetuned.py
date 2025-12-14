#!/usr/bin/env python3
"""
GSM8K Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the GSM8K (Grade School Math 8K) dataset.

Usage:
    python evaluate_gsm8k_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class GSM8KEvaluator(BaseEvaluator):
    """Evaluator for GSM8K dataset."""
    
    def get_dataset_name(self):
        return "gsm8k"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/gsm8k_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for GSM8K math problem."""
        system_prompt = """You are an expert mathematician.

First, think step by step and explain your mathematical reasoning in just one paragraph. Then compute the final numerical answer.

Your entire output MUST use exactly the following format and nothing else (no text before, between, or after these tags):

<reasoning>
[here you write your chain-of-thought reasoning and intermediate steps]
</reasoning>
<answer>
[here you output ONLY the final numeric answer, e.g. 42 or 3.14]
</answer>"""
        
        problem = example.get('question', '')
        user_prompt = f"""Problem: {problem}

Solve this problem step by step, then provide your final numerical answer."""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the numerical answer from the <answer>...</answer> block.

        Handles integers, decimals, and numbers with commas.
        """
        if not response:
            return None

        # Find the content inside <answer>...</answer>
        tag_match = re.search(r'<answer>\s*(.*?)\s*</answer>',
                              response,
                              re.IGNORECASE | re.DOTALL)
        if not tag_match:
            return None

        answer_content = tag_match.group(1).strip()

        # Look for a number inside the answer content
        num_match = re.search(r'[-+]?\d+(?:,\d+)*(?:\.\d+)?', answer_content)
        if not num_match:
            return None

        num_str = num_match.group(0).replace(',', '')

        try:
            # Convert to int if it's a whole number, else float
            if '.' in num_str:
                return float(num_str)
            else:
                return int(num_str)
        except ValueError:
            return None
    
    def load_dataset(self, split, max_samples=None):
        """Load GSM8K dataset."""
        print("Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main", split=split)
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from GSM8K format (answer_text contains '#### number')."""
        answer_text = example.get('answer', '')
        # Look for #### pattern
        match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', answer_text)
        if match:
            # Remove commas and convert to float
            num_str = match.group(1).replace(',', '')
            try:
                # Try to convert to int if it's a whole number
                if '.' in num_str:
                    return float(num_str)
                else:
                    return int(num_str)
            except ValueError:
                pass
        return None
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer (handles floating point)."""
        if predicted_answer is None or true_answer is None:
            return False
        # Handle floating point comparison
        if isinstance(predicted_answer, float) or isinstance(true_answer, float):
            return abs(predicted_answer - true_answer) < 0.01
        else:
            return predicted_answer == true_answer
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return -999999
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "question"


def main():
    evaluator = GSM8KEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
