#!/usr/bin/env python3
"""
INABHYD Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the INABHYD inductive/abductive reasoning dataset.

Usage:
    python evaluate_inabhyd_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
import pickle
from datasets import Dataset
from base_evaluator import BaseEvaluator

_HYP_SPLIT_RE = re.compile(r'[.\n;]+')


def _split_hypotheses(text):
    """Turn a blob of hypotheses text into a list of canonical clauses."""
    if not text:
        return []
    
    # Remove simple bullet / numbering markers like "- ", "1) ", "1. "
    lines = [
        re.sub(r'^\s*[\-\*\d]+\s*[.)]?\s*', '', ln)
        for ln in text.splitlines()
    ]
    joined = "\n".join(lines)
    
    raw_clauses = _HYP_SPLIT_RE.split(joined)
    clauses = [
        c.strip().lower()
        for c in raw_clauses
        if c.strip()
    ]
    
    return sorted(set(clauses))


class INABHYDEvaluator(BaseEvaluator):
    """Evaluator for INABHYD dataset."""
    
    def get_dataset_name(self):
        return "inabhyd"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/inabhyd_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for INABHYD task.
        
        NOTE: `claim` is actually the INABHYD world model / theories,
        and `evidence_text` is the observations.
        """
        system_prompt = """
You are an expert logician specializing in inductive and abductive reasoning
over synthetic first-order logic worlds.

You will be given:
- Theories: axioms describing an (incomplete) fictional world model.
- Observations: facts that must be explained.

Your task:
1. Propose one or more hypotheses that, when added to the Theories, make all Observations
   deductively follow.
2. Each hypothesis must be a simple sentence in the form:
      - "A is B"
      - "A is not B"
      - "All A are B"
      - "All A are not B"
3. Make hypotheses as short and general as possible (prefer parsimonious explanations).
   Do NOT restate the observations as hypotheses unless absolutely necessary.
4. First, think step by step.
5. Then output ONLY your final hypotheses inside an <answer>...</answer> block,
   one hypothesis per line, with no commentary.

Your entire output MUST use exactly the following format and nothing else:

<reasoning>
[Your step-by-step analysis of how candidate hypotheses explain all observations]
</reasoning>
<answer>
[Your final hypotheses only, one per line, no bullet symbols or numbering]
</answer>
""".strip()
        
        # In INABHYD, 'claim' is theories and 'evidence_text' is observations
        theories = example.get('theories', example.get('claim', ''))
        observations = example.get('observations', example.get('evidence_text', ''))
        
        user_prompt = f"""
Theories:
{theories}

Observations:
{observations}

Based on the theories and observations, propose hypotheses that explain all observations.
Remember: output ONLY hypotheses (no explanations) in the <answer> block.
""".strip()
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the content from the <answer>...</answer> block.
        
        For INABHYD this block is expected to contain one hypothesis per line.
        """
        if not response:
            return None

        match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
        
        if match:
            clean_answer = match.group(1).strip()
            return clean_answer
        
        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load INABHYD dataset from pickle file."""
        print(f"Loading INABHYD dataset (split={split})...")
        dataset_path = "/home/moein_salimi/users/amirmo/1hop_0shot_membership_ontology_property.pkl"
        
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        # Convert to list of dicts if needed
        if isinstance(data, list):
            dataset = Dataset.from_list(data)
        else:
            dataset = Dataset.from_list([data] if not isinstance(data, dict) else [data])
        
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (hypotheses field)."""
        hypotheses = example.get('hypotheses', '')
        if isinstance(hypotheses, str):
            return hypotheses
        elif isinstance(hypotheses, list):
            return '\n'.join(hypotheses)
        return ''
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer (set comparison for hypotheses)."""
        if predicted_answer is None or true_answer is None:
            return False
        
        # Split into canonical clauses and compare sets
        pred_clauses = set(_split_hypotheses(str(predicted_answer)))
        true_clauses = set(_split_hypotheses(str(true_answer)))
        
        return pred_clauses == true_clauses
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return ""
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "theories"


def main():
    evaluator = INABHYDEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
