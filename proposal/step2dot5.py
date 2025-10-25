from prompting import create_prompt_step2dot5, parse_model_answer_step2dot5
from api_handler import get_model_response
import time

def step2dot5(sample, idx, model_name, api_key, max_tokens, temperature, thinking, sleep_time, dataset_name, step2_result):
    """
    Step 2.5: Refine DAG to ensure answer choices/options exist as exactly one categorical node.
    
    This step takes the refined BN Schema from Step 2 and ensures that the answer choices
    (options A, B, C, D for MedQA or hypothesis-related conclusions for UniADILR) are
    represented as a single categorical node in the DAG with categories being exactly the options.
    
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
        step2_result: Result dictionary from step2 containing the refined BN Schema
    
    Returns:
        dict: Result dictionary with further refined BN Schema ensuring proper option representation
    """
    time.sleep(sleep_time)

    # Check if step2 was successful and has a valid format
    if not step2_result.get("successful_api_call") or not step2_result.get("right_format"):
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "model_output": None,
            "model_answer": None,
            "correct_answer": sample.get("answer_idx"),
            "input_text": None,
            "idx": idx,
            "error": "Step 2 failed or had invalid format - cannot proceed with Step 2.5",
            "token_usage": None
        }

    input_text = create_prompt_step2dot5(dataset_name, sample, step2_result)
    
    error_message = None
    model_output = None
    successful_api_call = False

    try:
        model_output, usage = get_model_response(model_name, api_key, input_text, max_tokens, temperature)
        successful_api_call = True
    except Exception as e:
        error_message = f"API call failed for sample {idx} in Step 2.5: {e}"
    
    result = parse_model_answer_step2dot5(sample, model_output, successful_api_call, thinking, step2_result, dataset_name)

    result["input_text"] = input_text
    result["idx"] = idx
    result["error"] = error_message
    result["token_usage"] = usage if successful_api_call else None

    return result

