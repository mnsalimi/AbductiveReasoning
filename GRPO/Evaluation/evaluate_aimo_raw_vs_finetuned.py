#!/usr/bin/env python3
"""
AIMO Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the AIMO math competition dataset.

Usage:
    python evaluate_aimo_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class AIMOEvaluator(BaseEvaluator):
    """Evaluator for AIMO dataset."""
    
    def get_dataset_name(self):
        return "aimo"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/aimo_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for AIMO problem - handles LaTeX properly."""
        system_prompt = """You are an expert mathematician specializing in competition mathematics (AMC, AIME, etc.).

First, read the problem carefully, including all LaTeX mathematical notation, and solve it step by step. Then give the final answer as a single number.

Your entire output MUST use exactly the following format and nothing else (no text before, between, or after these tags):

<reasoning>
[here you write your chain-of-thought reasoning and intermediate steps]
</reasoning>
<answer>
[here you output ONLY the final answer as a number, decimal, or fraction a/b, with no extra words]
</answer>"""
        
        problem = example.get('problem', '')
        user_prompt = f"""Problem:
{problem}

Solve this problem and provide your final numerical answer."""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the numerical answer from the <answer>...</answer> block.
        
        Supports integers, decimals, and simple fractions (a/b).
        Returns the answer as a cleaned string, or None if not found.
        """
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

        answer_content = tag_match.group(1).strip()

        # Handle possible \boxed{...} inside the answer block
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer_content)
        if boxed_match:
            answer_content = boxed_match.group(1).strip()

        # Clean common wrappers/symbols
        answer_content = answer_content.replace('$', '').replace(',', '').strip()

        # Look for a number / decimal / fraction inside the answer content
        num_match = re.search(r'[+-]?\d+(?:\.\d+)?(?:/\d+)?', answer_content)
        if num_match:
            return num_match.group(0).strip()

        return None
    
    def normalize_answer(self, ans):
        """Normalize answers for comparison - handles integers, decimals, and fractions."""
        if ans is None:
            return None
        
        ans = str(ans).strip().lower()
        
        # Remove dollar signs, spaces, commas
        ans = ans.replace('$', '').replace(' ', '').replace(',', '')
        
        # Handle LaTeX fractions: \frac{a}{b} -> a/b
        ans = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', ans)
        
        # Remove other LaTeX commands
        ans = ans.replace('\\', '')
        
        # Try to evaluate fractions to compare as floats
        if '/' in ans:
            try:
                parts = ans.split('/')
                if len(parts) == 2:
                    numerator = float(parts[0])
                    denominator = float(parts[1])
                    if denominator != 0:
                        return str(numerator / denominator)
            except:
                pass
        
        return ans
    
    def load_dataset(self, split, max_samples=None):
        """Load AIMO dataset."""
        print(f"Loading AIMO dataset (split={split})...")
        dataset = load_dataset("yentinglin/aimo", split=split)
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example."""
        return str(example.get('answer', ''))
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer (handles fractions and decimals)."""
        if predicted_answer is None or true_answer is None:
            return False
        
        pred_norm = self.normalize_answer(predicted_answer)
        true_norm = self.normalize_answer(true_answer)
        
        if pred_norm is None or true_norm is None:
            return False
        
        try:
            pred_val = float(pred_norm)
            true_val = float(true_norm)
            # Use small tolerance for floating point comparison
            return abs(pred_val - true_val) < 0.001
        except:
            # Fallback to string comparison
            return pred_norm == true_norm
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return "FAILED"
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "problem"


def main():
    evaluator = AIMOEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
