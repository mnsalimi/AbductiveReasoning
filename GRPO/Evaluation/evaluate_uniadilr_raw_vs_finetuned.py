#!/usr/bin/env python3
"""
UniADILR Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the UniADILR dataset.

Usage:
    python evaluate_uniadilr_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator

SYSTEM_PROMPT_UniADILR = """
You are an expert in logical reasoning and evidence identification.

Given a Context (multiple sentences) and a Hypothesis, your task is to identify which sentence(s) from the Context provide the necessary evidence to support the Hypothesis.

Your entire output MUST use exactly the following format and nothing else:

<reasoning>
[Step-by-step analysis of which sentences support the hypothesis]
</reasoning>
<answer>
[Output the sentence numbers (e.g., 1, 3, 5) separated by commas or spaces]
</answer>
"""


class UniADILREvaluator(BaseEvaluator):
    """Evaluator for UniADILR dataset."""
    
    def get_dataset_name(self):
        return "uniadilr"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/uniadilr_evaluation_results"
    
    def create_prompt(self, example):
        """Build system + user prompt for a UniADILR example."""
        context = example.get("context", {})
        hypothesis = example.get("hypothesis", "")
        
        # Preserve the original order of context items
        context_lines = [f"{k}: {v}" for k, v in context.items()]
        context_str = "\n".join(context_lines)
        
        system_prompt = SYSTEM_PROMPT_UniADILR
        
        user_prompt = f"""Context:
{context_str}

Hypothesis:
{hypothesis}

Based on the context and hypothesis above, identify which sentence(s) provide the necessary evidence for the hypothesis."""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract sentence numbers from model output. Returns set of integers as string representation."""
        # Try to find answer tags
        answer_match = re.search(
            r"<answer>\s*([^<]+?)\s*</answer>", response, re.IGNORECASE | re.DOTALL
        )
        
        if answer_match:
            answer_content = answer_match.group(1)
            numbers = re.findall(r"\b(\d+)\b", answer_content)
            if numbers:
                # Return as sorted comma-separated string for comparison
                return ', '.join(sorted(set(str(n) for n in numbers)))
        return None
    
    def parse_proof(self, proof_str):
        """Parse ground-truth proof string to sentence indices set."""
        if "->" in proof_str:
            proof_str = proof_str.split("->")[0]
        numbers = re.findall(r"sent(\d+)", proof_str)
        if numbers:
            return ', '.join(sorted(set(str(n) for n in numbers)))
        return ''
    
    def load_dataset(self, split, max_samples=None):
        """Load UniADILR dataset."""
        print(f"Loading UniADILR dataset (split={split})...")
        # Adjust path as needed
        dataset = load_dataset("json", data_files="/path/to/uniadilr.json")["train"]
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (proof field)."""
        proof_str = example.get("proof", "")
        return self.parse_proof(proof_str)
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer (set comparison)."""
        if predicted_answer is None or true_answer is None:
            return False
        
        # Convert to sets for comparison
        pred_set = set([n.strip() for n in str(predicted_answer).split(',') if n.strip()])
        true_set = set([n.strip() for n in str(true_answer).split(',') if n.strip()])
        
        return pred_set == true_set
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return ""
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "hypothesis"


def main():
    evaluator = UniADILREvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
