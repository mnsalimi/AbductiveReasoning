#!/usr/bin/env python3
"""
CLUTRR Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the CLUTRR family relation reasoning dataset.

Usage:
    python evaluate_clutrr_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class CLUTRREvaluator(BaseEvaluator):
    """Evaluator for CLUTRR dataset."""
    
    def get_dataset_name(self):
        return "clutrr"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/clutrr_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for CLUTRR (Family Relation Reasoning)."""
        system_prompt = """
You are a logic expert specializing in genealogy and family trees.
You will be given a short Story describing family relationships and a Query about the relationship between two specific people.

Your task:
1. Read the story carefully to build a mental family tree.
2. Trace the path from the first person to the second person in the Query.
3. deduce the exact kinship relation (e.g., father, aunt, grandson, son-in-law).
4. Output ONLY the relation keyword.

Your entire output MUST use exactly the following format:

<reasoning>
[Step-by-step deduction of the family tree path]
</reasoning>
<answer>
[The exact kinship relation word, e.g., grandmother]
</answer>
"""
        
        story = example.get('story', '')
        query = example.get('query', '')
        
        # If query is a tuple/list ['Alice', 'Bob'], format it
        if isinstance(query, list) or isinstance(query, tuple):
            entity_1 = query[0]
            entity_2 = query[1]
            formatted_query = f"What is the relationship of {entity_1} to {entity_2}?"
        else:
            formatted_query = query
        
        user_prompt = f"""
Story:
{story}

Query:
{formatted_query}

Relation:
"""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the relation keyword from the answer tag."""
        if not response:
            return None

        match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
        
        if match:
            clean_answer = match.group(1).strip().lower()
            # Remove punctuation like periods
            clean_answer = clean_answer.rstrip('.').rstrip('!')
            return clean_answer
        
        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load CLUTRR dataset."""
        print(f"Loading CLUTRR dataset (using 'CLUTRR/v1' as default source)...")
        try:
            dataset = load_dataset("CLUTRR/v1", "gen_train234_test2to10", split=split)
        except:
            print("Warning: Could not load specific config, trying generic load...")
            dataset = load_dataset("clutrr", split=split)
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (target relation)."""
        return example.get('target', '').lower().strip()
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer (case-insensitive)."""
        if predicted_answer is None or true_answer is None:
            return False
        return predicted_answer.lower().strip() == true_answer.lower().strip()
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return "FAILED"
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "story"


def main():
    evaluator = CLUTRREvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
