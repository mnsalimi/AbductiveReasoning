#!/usr/bin/env python3
"""
MuSR Object Placements Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the MuSR object placements dataset.

Usage:
    python evaluate_musr_object_placements_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class MuSRObjectPlacementsEvaluator(BaseEvaluator):
    """Evaluator for MuSR object placements dataset."""
    
    def get_dataset_name(self):
        return "musr_object_placements"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/musr_object_placements_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for a detective-style multiple-choice reasoning question."""
        system_prompt = """
You are a brilliant detective analyzing clues to solve a mystery. 
You will be given context and a multiple-choice question.

Your task:
1. Carefully read the context and the question.
2. Perform step-by-step detective reasoning.
3. Select the **index number** (0, 1, 2, ...) of the correct choice.

Your entire output MUST use exactly the following format and nothing else (no text before, between, or after these tags):

<reasoning>
[here you write your chain-of-thought reasoning and intermediate steps]
</reasoning>
<answer>
[here you output ONLY the index number of the correct choice, with no extra words]
</answer>
"""
        
        context = example.get('context', '')
        problem = example.get('problem', '')
        
        user_prompt = f"""
Context:
{context}

Problem:
{problem}

Solve this problem step by step using detective reasoning, then provide your final answer (the index number of the correct choice).
"""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the index-based answer (0, 1, 2, ...) from the <answer>...</answer> block."""
        if not response:
            return None

        tag_match = re.search(
            r'<answer>\s*(.*?)\s*</answer>',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if not tag_match:
            return None

        answer_content = tag_match.group(1).strip()
        answer_content = answer_content.replace('$', '').strip()

        index_match = re.fullmatch(r'[0-9]+', answer_content)
        if not index_match:
            return None

        return int(answer_content)
    
    def load_dataset(self, split, max_samples=None):
        """Load MuSR object placements dataset."""
        print(f"Loading musr_object dataset (split={split})...")
        dataset = load_dataset("json", data_files="/home/moein_salimi/users/amirmo/AbductiveReasoning/datasets/object_placements.json")["train"]
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (label is index)."""
        return int(example.get('label', 0))
    
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
        return "problem"


def main():
    evaluator = MuSRObjectPlacementsEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
