#!/usr/bin/env python3
"""
NeuLR Deductive Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the NeuLR deductive reasoning dataset.

Usage:
    python evaluate_neulr_deductive_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class NeuLRDeductiveEvaluator(BaseEvaluator):
    """Evaluator for NeuLR deductive reasoning dataset."""
    
    def get_dataset_name(self):
        return "neulr_deductive"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/neulr_deductive_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for deductive reasoning task."""
        system_prompt = """
You are a brilliant detective specializing in symbolic logic and pattern recognition.
You will be given a context containing logical rules regarding specific alphanumeric codes and a resulting question.

Your task:
1. Carefully parse the context to identify facts (who is what) and rules (who is afraid of whom).
2. Perform step-by-step deductive reasoning to trace the relationship from the subject in the question to the final answer.
3. Give the correct answer as the EXACT alphanumeric code from the text.

Your entire output MUST use exactly the following format and nothing else (no text before, between, or after these tags):

<reasoning>
[here you write your chain-of-thought reasoning, explicitly linking the individual to their group and the group to the object of their fear]
</reasoning>
<answer>
[here you output ONLY the exact alphanumeric code answer]
</answer>
"""
        
        context = example.get('context', '')
        problem = example.get('problem', '')
        
        user_prompt = f"""
Context:
{context}

Problem:
{problem}

Solve this problem step by step using detective reasoning to find the logical connection, then provide your final answer in one word ONLY.
"""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the alphanumeric code string from the <answer>...</answer> block."""
        if not response:
            return None

        match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
        
        if match:
            clean_answer = match.group(1).strip()
            clean_answer = clean_answer.rstrip('.')
            return clean_answer
        
        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load NeuLR deductive dataset."""
        print(f"Loading neulr_deductive dataset (split={split})...")
        dataset = load_dataset("json", data_files="/home/moein_salimi/users/amirmo/AbductiveReasoning/datasets/NeuLR/deductive_neutral.json")["train"]
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (label is alphanumeric code)."""
        return example.get('label', '').strip()
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer (exact match)."""
        if predicted_answer is None or true_answer is None:
            return False
        return predicted_answer.strip() == true_answer.strip()
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return ""
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "problem"


def main():
    evaluator = NeuLRDeductiveEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
