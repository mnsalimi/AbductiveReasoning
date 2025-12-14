#!/usr/bin/env python3
"""
NeuLR Inductive Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the NeuLR inductive reasoning dataset.

Usage:
    python evaluate_neulr_inductive_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class NeuLRInductiveEvaluator(BaseEvaluator):
    """Evaluator for NeuLR inductive reasoning dataset."""
    
    def get_dataset_name(self):
        return "neulr_inductive"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/neulr_inductive_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for inductive reasoning task."""
        system_prompt = """
You are a brilliant detective specializing in symbolic logic and pattern recognition.
You will be given a context containing facts about entities, their group memberships, and their specific properties.

Your task:
1. Carefully parse the context to identify which group the target entity belongs to.
2. Look for other entities in that same group to see what properties they possess.
3. Perform step-by-step reasoning to deduce the property of the target entity based on these shared group characteristics.
4. Give the correct answer as the EXACT alphanumeric code from the text.

Your entire output MUST use exactly the following format and nothing else (no text before, between, or after these tags):

<reasoning>
[here you write your chain-of-thought reasoning, explicitly linking the target entity to a group, finding a sibling entity in that group, and transferring the property]
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

Solve this problem step by step using detective reasoning to find the shared property, then provide your final answer in one word ONLY.
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
        """Load NeuLR inductive dataset."""
        print(f"Loading neulr_inductive dataset (split={split})...")
        dataset = load_dataset("json", data_files="/home/moein_salimi/users/amirmo/AbductiveReasoning/datasets/NeuLR/inductive_neutral.json")["train"]
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
    evaluator = NeuLRInductiveEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
