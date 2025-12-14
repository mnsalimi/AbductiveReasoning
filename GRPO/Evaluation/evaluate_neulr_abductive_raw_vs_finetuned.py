#!/usr/bin/env python3
"""
NeuLR Abductive Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the NeuLR abductive reasoning dataset.

Usage:
    python evaluate_neulr_abductive_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class NeuLRAbductiveEvaluator(BaseEvaluator):
    """Evaluator for NeuLR abductive reasoning dataset."""
    
    def get_dataset_name(self):
        return "neulr_abductive"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/neulr_abductive_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for an abductive reasoning task."""
        system_prompt = """
You are an expert Forensic Logic Analyst.

Task: Abductive Reasoning (Find the Missing Fact).
You are given:
1. A list of 'Logical Rules and Known Facts'.
2. A 'Target Conclusion' (Observed Fact).

The 'Target Conclusion' is currently unprovable with the provided facts alone. 
Your goal is to identify the single MISSING FACT (premise) that, when added to the known facts, makes the Target Conclusion true based on the Rules.

Output Format:
<reasoning>
[Step-by-step logic: Identify the rule triggered by the Conclusion, trace backwards to find what condition is missing.]
</reasoning>
<answer>
[The missing fact as a complete sentence. Example: NPsw0v0k is ADP37scy8.]
</answer>
"""
        
        problem = example.get('problem', '')
        context = example.get('context', '')
        
        # Separate the provided Rules/Facts from the Target Conclusion
        if "The fact is:" in context:
            rules_block, target_fact = context.split("The fact is:", 1)
            rules_block = rules_block.strip()
            target_fact = target_fact.strip()
        else:
            rules_block = context.strip()
            target_fact = problem.strip() if problem else ""
        
        user_prompt = f"""
Logical Rules and Known Facts:
{rules_block}

Target Conclusion:
{target_fact}

Question: What missing fact is required to conclude the Target Conclusion?
"""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the full sentence answer from the <answer> block."""
        if not response:
            return None

        match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load NeuLR abductive dataset."""
        print(f"Loading neulr_abductive dataset (split={split})...")
        dataset = load_dataset("json", data_files="/home/moein_salimi/users/amirmo/AbductiveReasoning/datasets/NeuLR/abductive_neutral.json")["train"]
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (label field)."""
        return example.get('label', '').strip()
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer (exact match for sentences)."""
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
    evaluator = NeuLRAbductiveEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
