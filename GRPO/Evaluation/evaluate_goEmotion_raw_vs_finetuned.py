#!/usr/bin/env python3
"""
GoEmotions Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the GoEmotions emotion classification dataset.

Usage:
    python evaluate_goEmotion_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import re
from datasets import load_dataset
from base_evaluator import BaseEvaluator

# GoEmotions emotion labels (27 emotions + neutral)
GOEMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


class GoEmotionEvaluator(BaseEvaluator):
    """Evaluator for GoEmotions dataset."""
    
    def get_dataset_name(self):
        return "goEmotion"
    
    def _get_default_output_dir(self):
        return "/home/moein_salimi/users/amirmo/AbductiveReasoning/GRPO/Evaluation/goEmotion_evaluation_results"
    
    def create_prompt(self, example):
        """Create a prompt for GoEmotions emotion classification."""
        emotions_list = ", ".join(GOEMOTION_LABELS)
        
        system_prompt = f"""You are an expert emotion classifier. Given a text and a list of possible emotions, identify all emotions expressed in the text.

Available emotions: {emotions_list}

First, think step by step and explain your reasoning about which emotions are present, considering context and nuance, in just one paragraph. Then list all and only the emotions that apply.

Your entire output MUST use exactly the following format and nothing else (no text before, between, or after these tags):

<reasoning>
[here you write your chain-of-thought reasoning about which emotions are present and why]
</reasoning>
<answer>
[here you output ONLY the emotion names from the available list, separated by commas if there are multiple; e.g. "joy, surprise" or "anger"]
</answer>"""
        
        text = example.get('text', '')
        user_prompt = f"""Text: "{text}"

What emotion(s) are expressed in this text?"""
        
        return system_prompt, user_prompt
    
    def extract_answer(self, response):
        """Extract emotion labels from model response. Returns comma-separated string."""
        # First try to extract <answer>...</answer> tags
        tag_match = re.search(r'<answer>\s*([^<]+)\s*</answer>', response, re.IGNORECASE)
        if tag_match:
            answer_text = tag_match.group(1).strip()
            emotions = [e.strip().lower() for e in answer_text.split(',') if e.strip()]
            # Filter to valid emotions only
            valid_emotions = [e for e in emotions if e.lower() in [l.lower() for l in GOEMOTION_LABELS]]
            if valid_emotions:
                return ', '.join(sorted(set(valid_emotions)))  # Sort and deduplicate
        
        # Fallback: try to find any valid emotion words in the response
        response_lower = response.lower()
        found_emotions = []
        for emotion in GOEMOTION_LABELS:
            if emotion.lower() in response_lower:
                found_emotions.append(emotion.lower())
        
        if found_emotions:
            return ', '.join(sorted(set(found_emotions)))
        
        return None
    
    def load_dataset(self, split, max_samples=None):
        """Load GoEmotions dataset."""
        print(f"Loading go_emotions dataset (split={split})...")
        dataset = load_dataset("go_emotions", split=split)
        return dataset
    
    def get_true_answer(self, example):
        """Extract true answer from example (labels are indices, convert to emotion names)."""
        label_indices = example.get('labels', [])
        if isinstance(label_indices, list):
            true_emotions = [GOEMOTION_LABELS[idx] for idx in label_indices if idx < len(GOEMOTION_LABELS)]
            return ', '.join(sorted([e.lower() for e in true_emotions]))
        return ''
    
    def is_correct(self, predicted_answer, true_answer):
        """Check if predicted answer matches true answer (set comparison for multi-label)."""
        if predicted_answer is None or true_answer is None:
            return False
        
        # Convert to sets for comparison (case-insensitive)
        pred_set = set([e.strip().lower() for e in str(predicted_answer).split(',') if e.strip()])
        true_set = set([e.strip().lower() for e in str(true_answer).split(',') if e.strip()])
        
        return pred_set == true_set
    
    def _get_failed_answer(self):
        """Return the sentinel value for failed answer extraction."""
        return ""
    
    def get_problem_field(self):
        """Return the field name containing the problem."""
        return "text"


def main():
    evaluator = GoEmotionEvaluator()
    evaluator.main()


if __name__ == '__main__':
    main()
