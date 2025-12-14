#!/usr/bin/env python3
"""
Defeasible NLI Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the Defeasible NLI (Thinking Like a Skeptic) dataset.

Usage:
    python evaluate_defeasible_nli_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
import numpy as np
from datasets import load_dataset
from base_evaluator import BaseEvaluator

np.random.seed(42)


class DefeasibleNLIEvaluator(BaseEvaluator):
    """Evaluator for Defeasible NLI dataset."""
    
    def get_dataset_name(self):
        return "defeasible_nli"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/defeasible_nli_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for Defeasible NLI (Thinking Like a Skeptic)."""
        system_prompt = """
You are an expert in defeasible reasoning and skepticism.
You will be given a Hypothesis (a tentative conclusion) and an Update (new information).
Sometimes, a Premise (context) is also provided.

Your task:
1. Consider the Hypothesis in the context of the Premise (if available).
2. Analyze how the new Update affects the likelihood of the Hypothesis.
3. Determine if the Update STRENGTHENS (makes it more likely) or WEAKENS (makes it less likely) the Hypothesis.

Your entire output MUST use exactly the following format and nothing else:

<reasoning>
[Your step-by-step analysis of how the update affects the hypothesis]
</reasoning>
<answer>
[Output exactly one word: STRENGTHENS or WEAKENS]
</answer>
"""
        
        premise = example.get('premise', '')
        hypothesis = example.get('hypothesis', '')
        update = example.get('update', '')
        
        # Handle cases where Premise might be empty
        if premise and isinstance(premise, str) and len(premise.strip()) > 0:
            context_block = f"Premise:\n{premise}\n\nHypothesis:\n{hypothesis}"
        else:
            context_block = f"Hypothesis:\n{hypothesis}"
        
        user_prompt = f"""
{context_block}

Update:
{update}

Does this Update STRENGTHEN or WEAKEN the Hypothesis?
"""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the label from the <answer>...</answer> block."""
        if not response:
            return None

        match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
        
        if match:
            clean_answer = match.group(1).strip().upper()
            # Remove punctuation
            clean_answer = clean_answer.rstrip('.').rstrip('!')
            
            # Normalize to standard output labels
            if "STRENGTH" in clean_answer:
                return "STRENGTHENS"
            if "WEAK" in clean_answer:
                return "WEAKENS"
            
            return clean_answer
        
        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load Defeasible NLI dataset with random sampling."""
        print(f"Loading tasksource/defeasible-nli dataset (split=social)...")
        dataset = load_dataset("tasksource/defeasible-nli", "social")["test"]
        
        # Defeasible NLI uses random sampling of 1000 examples
        indices = np.random.choice(len(dataset), int(1000), replace=False)
        dataset = dataset.select(indices)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (label is STRENGTHENS or WEAKENS)."""
        label = example.get('label', '')
        # Map numeric labels if needed
        if isinstance(label, int):
            return "STRENGTHENS" if label == 0 else "WEAKENS"
        return str(label).upper()
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer."""
        if predicted_answer is None or true_answer is None:
            return False
        return str(predicted_answer).upper() == str(true_answer).upper()
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return "FAILED"
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "hypothesis"


def main():
    evaluator = DefeasibleNLIEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
