from prompting import create_prompt_step2, parse_model_answer_step2
from api_handler import get_model_response
import time
import re

def step2(sample, idx, model_name, api_key, max_tokens, temperature, thinking, sleep_time, dataset_name, step1_result):
    """
    Step 2: Refine the BN Schema from Step 1 by adding missing nodes and edges.
    
    Args:
        sample: Original data sample
        idx: Sample index
        model_name: AI model to use
        api_key: API key for authentication
        max_tokens: Maximum response tokens
        temperature: Model temperature
        thinking: Whether model supports <think> blocks
        sleep_time: Delay between API calls
        dataset_name: "medqa" or "uniadilr"
        step1_result: Result dictionary from step1 containing the original BN Schema
    
    Returns:
        dict: Result dictionary with refined BN Schema
    """
    time.sleep(sleep_time)

    # Check if step1 was successful and has a valid format
    if not step1_result.get("successful_api_call") or not step1_result.get("right_format"):
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "model_output": None,
            "model_answer": None,
            "correct_answer": sample.get("answer_idx"),
            "input_text": None,
            "idx": idx,
            "error": "Step 1 failed or had invalid format - cannot proceed with Step 2",
            "token_usage": None,
            "step1_result": step1_result
        }

    input_text = create_prompt_step2(dataset_name, sample, step1_result)
    
    error_message = None
    model_output = None
    successful_api_call = False

    try:
        model_output, usage = get_model_response(model_name, api_key, input_text, max_tokens, temperature)
        successful_api_call = True
    except Exception as e:
        error_message = f"API call failed for sample {idx} in Step 2: {e}"
    
    result = parse_model_answer_step2(sample, model_output, successful_api_call, thinking, step1_result)

    result["input_text"] = input_text
    result["idx"] = idx
    result["error"] = error_message
    result["token_usage"] = usage if successful_api_call else None
    result["step1_result"] = step1_result

    return result

