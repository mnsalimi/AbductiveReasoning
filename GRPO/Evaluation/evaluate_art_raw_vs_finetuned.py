#!/usr/bin/env python3
"""
ART Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the ART (Abductive Reasoning Task) dataset.

Usage:
    python evaluate_art_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class ARTEvaluator(BaseEvaluator):
    """Evaluator for ART dataset."""
    
    def get_dataset_name(self):
        return "art"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/art_evaluation_results"
    
    def create_prompt(self, example):
        """Create prompt for ART task."""
        system_prompt = """You are an expert in abductive reasoning. Given two observations and two hypotheses, select which hypothesis (1 or 2) best explains what happened between the observations.

First, think step by step and explain your abductive reasoning in just one paragraph. Then decide which hypothesis (1 or 2) is better.

Your entire output MUST use exactly the following format and nothing else (no text before, between, or after these tags):

<reasoning>
[here you write your chain-of-thought reasoning about which hypothesis is better]
</reasoning>
<answer>
[here you output ONLY the number 1 or 2]
</answer>"""
        
        obs1 = example.get('observation_1', '')
        obs2 = example.get('observation_2', '')
        hyp1 = example.get('hypothesis_1', '')
        hyp2 = example.get('hypothesis_2', '')
        
        user_prompt = f"""Observation 1: {obs1}
Observation 2: {obs2}

Hypothesis 1: {hyp1}
Hypothesis 2: {hyp2}

Which hypothesis better explains the transition from Observation 1 to Observation 2? Answer with just the number 1 or 2."""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the hypothesis number (1 or 2) from model response."""
        # First try to extract <answer>...</answer> tags
        tag_match = re.search(r'<answer>\s*([12])\s*</answer>', response, re.IGNORECASE)
        if tag_match:
            return int(tag_match.group(1))
        
        # Fallback: try various patterns
        response_clean = response.strip().lower()
        
        patterns = [
            r'^\s*(\d)\s*$',  # Just a number
            r'(?:hypothesis\s*)?(\d)',  # "hypothesis 1" or just "1"
            r'(?:answer|select|choose)(?:\s+is)?\s*(\d)',  # "answer is 1"
            r'(?:^|\s)(\d)(?:\s|$|\.)',  # Number with spaces
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_clean)
            if match:
                num = match.group(1)
                if num in ['1', '2']:
                    return int(num)
        
        # Check if response starts with 1 or 2
        if response_clean.startswith('1'):
            return 1
        if response_clean.startswith('2'):
            return 2
        
        # Look for "first" or "second"
        if 'first' in response_clean[:20]:
            return 1
        if 'second' in response_clean[:20]:
            return 2
        
        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load ART dataset."""
        print("Loading ART dataset...")
        dataset = load_dataset("allenai/art", split=split)
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (label is 1 or 2)."""
        return int(example.get('label', 1))
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer."""
        if predicted_answer is None or true_answer is None:
            return False
        return int(predicted_answer) == int(true_answer)
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return -1
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "observation_1"


def main():
    evaluator = ARTEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
