from prompting import create_prompt_step1, parse_model_answer_step1
from api_handler import get_model_response
import time

def step1(sample, idx, model_name, api_key, max_tokens, temperature, thinking, sleep_time, dataset_name):
    """
    # TODO
    """
    time.sleep(sleep_time)

    input_text = create_prompt_step1(dataset_name, sample)
    
    error_message = None
    model_output = None
    successful_api_call = False

    try:
        model_output, usage = get_model_response(model_name, api_key, input_text, max_tokens, temperature)
        successful_api_call = True
    except Exception as e:
        error_message = f"API call failed for sample {idx}: {e}"
    
    result = parse_model_answer_step1(sample, model_output, successful_api_call, thinking)

    result["input_text"] = input_text
    result["idx"] = idx
    result["error"] = error_message
    result["token_usage"] = usage if successful_api_call else None

    return result