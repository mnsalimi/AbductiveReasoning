#!/usr/bin/env python3
"""
COPA Guess Effect Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the COPA dataset focusing on identifying the EFFECT given a CAUSE.

Usage:
    python evaluate_copa_raw_vs_finetuned_guess_effect.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class COPAGuessEffectEvaluator(BaseEvaluator):
    """Evaluator for COPA guess effect task."""
    
    def get_dataset_name(self):
        return "copa_guess_effect"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/copa_evaluation_results_guess_effect"
    
    def create_prompt(self, example):
        """Create a prompt for COPA causal reasoning task (identify effect given cause)."""
        system_prompt = """You are an expert in causal reasoning. Given a CAUSE and two possible EFFECT options, select which option (1 or 2) is the most plausible direct effect.

First, think step by step and explain your causal reasoning in just one paragraph. Then decide which option (1 or 2) is better.

Your entire output MUST use exactly the following format and nothing else (no text before, between, or after these tags):

<reasoning>
[here you write your chain-of-thought reasoning about which effect is more plausible]
</reasoning>
<answer>
[here you output ONLY the number 1 or 2]
</answer>"""
        
        premise = example.get('premise', '')
        choice1 = example.get('choice1', '')
        choice2 = example.get('choice2', '')
        
        user_prompt = f"""Cause: {premise}

Which of the following is the most plausible EFFECT of this cause?

Option 1: {choice1}
Option 2: {choice2}

Think step by step about which option is the most likely effect, then provide your answer in <answer></answer> tags."""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract answer from model response. Returns 1 or 2 (1-indexed)."""
        # First try to extract <answer>...</answer> tags
        tag_match = re.search(r'<answer>\s*([12])\s*</answer>', response, re.IGNORECASE)
        if tag_match:
            return int(tag_match.group(1))
        
        # Fallback patterns
        fallback_patterns = [
            r'(?:answer|choice)[\s:]+(\d+)',
            r'option\s+(\d+)',
            r'(?:^|\s)([12])(?:\s|$|\.|,)'
        ]
        
        for pattern in fallback_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                answer = matches[-1].strip()
                if answer in ['1', '2']:
                    return int(answer)
        
        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load COPA dataset and filter for effect questions."""
        print("Loading COPA dataset...")
        try:
            dataset = load_dataset("pkavumba/balanced-copa", split="train")
            # Filter for "effect" questions only (where we're given the cause)
            dataset = dataset.filter(lambda x: x.get('question', '') == 'effect')
            print(f"Loaded {len(dataset)} effect questions from COPA dataset")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (label is 0 or 1, convert to 1-indexed)."""
        label = int(example.get('label', 0))
        return label + 1  # Convert 0-indexed to 1-indexed
    
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
        return "premise"


def main():
    evaluator = COPAGuessEffectEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
