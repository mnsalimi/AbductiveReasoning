#!/usr/bin/env python3
"""
[Dataset Name] Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the [Dataset Name] dataset.

Usage:
    python evaluate_[dataset]_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator


class [Dataset]Evaluator(BaseEvaluator):
    """Evaluator for [Dataset Name] dataset."""
    
    def get_dataset_name(self):
        """Return the name of the dataset."""
        return "[dataset_name]"
    
    def _get_default_output_dir(self):
        """Return the default output directory for this dataset."""
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/[dataset]_evaluation_results"
    
    def create_prompt(self, example):
        """Create system and user prompts from a dataset example.
        
        Args:
            example: A single example from the dataset
            
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        system_prompt = """[Your system prompt here]"""
        
        # Extract relevant fields from example
        # field1 = example.get('field1', '')
        # field2 = example.get('field2', '')
        
        user_prompt = f"""[Your user prompt here with {example.get('field', '')}]"""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract the answer from model response.
        
        Args:
            response: The model's generated response string
            
        Returns:
            The extracted answer, or None if extraction failed
        """
        if not response:
            return None

        # Example: Extract from <answer>...</answer> tags
        tag_match = re.search(
            r'<answer>\s*(.*?)\s*</answer>',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if not tag_match:
            return None

        answer_content = tag_match.group(1).strip()
        
        # Process answer_content based on your dataset's answer format
        # For numeric answers:
        #   num_match = re.search(r'\b(\d+)\b', answer_content)
        #   if num_match:
        #       return int(num_match.group(1))
        # For text answers:
        #   return answer_content.upper()  # or normalize as needed
        
        return answer_content
    
    def load_dataset(self, split, max_samples=None):
        """Load the dataset for evaluation.
        
        Args:
            split: Dataset split to use (e.g., 'train', 'test')
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            The loaded dataset
        """
        print(f"Loading [Dataset Name] dataset (split={split})...")
        # Adjust dataset path and configuration as needed
        dataset = load_dataset("[dataset_path]", split=split)
        return dataset
    
    def get_true_answer(self, example):
        """Extract the true answer from a dataset example.
        
        Args:
            example: A single example from the dataset
            
        Returns:
            The true answer for this example
        """
        # Adjust based on your dataset's answer field
        return example.get('answer', None)
        # For multiple choice (A, B, C, D):
        #   label_id = int(example.get('label', 0))
        #   return {0: "A", 1: "B", 2: "C", 3: "D"}[label_id]
        # For boolean:
        #   return "YES" if example.get('answer', False) else "NO"
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer.
        
        Args:
            predicted_answer: The predicted answer
            true_answer: The true answer
            
        Returns:
            bool: True if correct, False otherwise
        """
        if predicted_answer is None or true_answer is None:
            return False
        
        # For exact match:
        return predicted_answer == true_answer
        
        # For numeric with tolerance:
        # if isinstance(predicted_answer, float) or isinstance(true_answer, float):
        #     return abs(predicted_answer - true_answer) < 0.01
        # else:
        #     return predicted_answer == true_answer
    
    def get_problem_field(self):
        """Return the field name in results that contains the problem/question."""
        return "problem"  # or "question", "ctx", etc.
    
    # Optional: Override if you need a different failed answer sentinel
    # def _get_failed_answer(self):
    #     return "FAILED"  # or -1, -999999, etc.


def main():
    evaluator = [Dataset]Evaluator()
    evaluator.main()


if __name__ == '__main__':
    main()

