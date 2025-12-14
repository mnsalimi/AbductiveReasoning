#!/usr/bin/env python3
"""
WinoGrande Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the WinoGrande commonsense reasoning dataset.

Usage:
    python evaluate_winogrande_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class WinoGrandeEvaluator(BaseEvaluator):
    """Evaluator for WinoGrande dataset."""
    
    def get_dataset_name(self):
        return "winogrande"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/winogrande_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for WinoGrande-style commonsense pronoun resolution."""
        system_prompt = """
You are an expert in commonsense reasoning and pronoun resolution.

You will be given:
- A sentence containing a blank represented by an underscore character: _
- Two candidate options (Option 1 and Option 2)

Your task:
1. Decide which option best fills the blank to make the sentence coherent and logically correct.
2. Provide step-by-step reasoning.
3. Provide the final answer as the option number.

Your entire output MUST use exactly the following format and nothing else:

<reasoning>
[Your step-by-step analysis of which option best completes the sentence]
</reasoning>
<answer>
[Output exactly one of these two options: 1 or 2]
</answer>
""".strip()
        
        sentence = example.get('sentence', '')
        option1 = example.get('option1', '')
        option2 = example.get('option2', '')
        
        user_prompt = f"""
Sentence:
{sentence}

Option 1:
{option1}

Option 2:
{option2}

Which option correctly fills the blank "_" in the sentence?
""".strip()
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the label from the <answer>...</answer> block. Returns "1" or "2" (strings) or None."""
        if not response:
            return None

        match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if not match:
            return None

        clean_answer = match.group(1).strip().upper()
        clean_answer = clean_answer.rstrip(".")

        # Accept a few common variants but normalize to "1"/"2"
        if clean_answer in {"1", "OPTION1", "OPTION 1", "A"}:
            return "1"
        if clean_answer in {"2", "OPTION2", "OPTION 2", "B"}:
            return "2"

        # If the model outputs extra text like "1 (Option 1)", grab first 1/2 digit
        m = re.search(r"([12])", clean_answer)
        if m:
            return m.group(1)

        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load WinoGrande dataset."""
        print(f"Loading allenai/winogrande (split={split})...")
        dataset = load_dataset("allenai/winogrande", "winogrande_xl", split=split)
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (answer field contains "1" or "2")."""
        answer = example.get('answer', '')
        return str(answer) if answer else None
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer."""
        if predicted_answer is None or true_answer is None:
            return False
        return str(predicted_answer) == str(true_answer)
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return "FAILED"
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "sentence"


def main():
    evaluator = WinoGrandeEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
