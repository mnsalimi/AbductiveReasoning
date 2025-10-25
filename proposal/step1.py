from prompting import create_prompt_step1, parse_model_answer_step1
from api_handler import get_model_response
import time

def step1(sample, idx, model_name, api_key, max_tokens, temperature, thinking, sleep_time, dataset_name):
    """
    # TODO
    """
    time.sleep(sleep_time)

    input_text = create_prompt_step1(dataset_name, sample)
    
    # Retry logic: up to 3 attempts
    max_retries = 3
    retry_count = 0
    last_error = None
    all_attempts = []
    
    while retry_count < max_retries:
        retry_count += 1
        
        error_message = None
        model_output = None
        successful_api_call = False

        try:
            print(f"\n  🔄 Step 1 - Attempt {retry_count}/{max_retries}...")
            model_output, usage = get_model_response(model_name, api_key, input_text, max_tokens, temperature)
            successful_api_call = True
        except Exception as e:
            error_message = f"API call failed for sample {idx}: {e}"
            last_error = error_message
            all_attempts.append({
                "attempt": retry_count,
                "error": error_message,
                "stage": "api_call"
            })
            print(f"      ❌ API call failed: {e}")
            if retry_count < max_retries:
                print(f"      🔄 Retrying...")
                time.sleep(sleep_time)
            continue
        
        # Parse the model response
        result = parse_model_answer_step1(sample, model_output, successful_api_call, thinking)
        
        # Record this attempt
        all_attempts.append({
            "attempt": retry_count,
            "successful_api_call": successful_api_call,
            "right_format": result.get("right_format", False)
        })
        
        # Check if parsing was successful
        if result.get("right_format"):
            print(f"      ✅ Step 1 successful!")
            result["input_text"] = input_text
            result["idx"] = idx
            result["error"] = None
            result["token_usage"] = usage
            result["attempts"] = all_attempts
            return result
        else:
            last_error = result.get("error", "Unknown parsing error")
            print(f"      ❌ Parsing failed: {last_error}")
            
            if retry_count < max_retries:
                print(f"      🔄 Retrying...")
                time.sleep(sleep_time)
    
    # All retries exhausted
    result = parse_model_answer_step1(sample, None, False, thinking)
    result["input_text"] = input_text
    result["idx"] = idx
    result["error"] = f"Step 1 failed after {max_retries} attempts. Last error: {last_error}"
    result["token_usage"] = None
    result["attempts"] = all_attempts
    return result