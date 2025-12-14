#!/usr/bin/env python3
"""
MedQA Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the MedQA multiple-choice medical question dataset.

Usage:
    python evaluate_medqa_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class MedQAEvaluator(BaseEvaluator):
    """Evaluator for MedQA dataset."""
    
    def get_dataset_name(self):
        return "MedQA"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/MedQA_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for a MedQA multiple-choice medical question."""
        system_prompt = """You are an expert medical clinician. Solve the following MedQA multiple-choice problem.

The final answer must be exactly one of the following letters: A, B, C, or D.

First, read the question carefully and analyze it step by step using clinical reasoning. Then select the correct option among A, B, C, or D.

Your entire output MUST use exactly the following format and nothing else (no text before, between, or after these tags):

<reasoning>
[here you write your chain-of-thought reasoning and intermediate steps]
</reasoning>
<answer>
[here you output ONLY one letter: A, B, C, or D, with no extra words]
</answer>"""
        
        question = example.get('question', '')
        options = example.get('options', '')
        problem = question + "\n" + str(options)
        
        user_prompt = f"""Problem: {problem}

Solve this problem step by step, then provide your final answer."""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the MedQA multiple-choice answer (A, B, C, or D) from the <answer>...</answer> block."""
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

        choice_match = re.fullmatch(r'[A-D]', answer_content, re.IGNORECASE)
        if not choice_match:
            return None

        return answer_content.upper()
    
    def load_dataset(self, split, max_samples=None):
        """Load MedQA dataset."""
        print(f"Loading MedQA dataset (split={split})...")
        # MedQA uses a local JSONL file
        dataset = load_dataset("json", data_files="/home/moein_salimi/users/amirmo/AbductiveReasoning/datasets/data_clean/questions/US/test.jsonl")["train"]
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (answer_idx as string)."""
        return str(example.get('answer_idx', ''))
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer."""
        return predicted_answer == true_answer
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return -1
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "question"


def main():
    evaluator = MedQAEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
