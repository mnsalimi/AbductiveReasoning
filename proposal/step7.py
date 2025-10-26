import time
import os
import json
from typing import Dict, Any
from api_handler import get_model_response
from prompting import create_prompt_step7, parse_model_answer_step7


def step7(sample: Dict[str, Any], idx: int, model_name: str, api_key: str, 
          max_tokens: int, temperature: float, thinking: bool, sleep_time: float, 
          dataset_name: str, step6_result: Dict[str, Any], step5_result: Dict[str, Any], options_node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 7: Answer Extraction - Extract the final answer from Bayesian Network analysis.
    
    This step takes the complete variable assignments (observed + MPE inferred) from the
    Bayesian Network analysis and asks the LLM to identify what answer is indicated by
    this analysis. The LLM does NOT solve the problem - it only identifies the answer
    that the analysis indicates.
    
    Args:
        sample: Original data sample (contains context and question)
        idx: Sample index
        model_name: Name of the LLM model to use
        api_key: API key for the LLM service
        max_tokens: Maximum tokens for LLM response
        temperature: Temperature for LLM sampling
        thinking: Whether model supports <think> blocks
        sleep_time: Delay before API call
        dataset_name: "medqa" or "uniadilr"
        step6_result: Result dictionary from step6 containing MPE results
        step5_result: Result dictionary from step5 containing the Bayesian Network
        options_node: Options node from step2.5
    Returns:
        dict: Result dictionary with extracted answer and metadata
    """
    time.sleep(sleep_time)
    
    # Check if step6 was successful
    if not step6_result.get("successful_api_call") or not step6_result.get("right_format"):
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "extracted_answer": None,
            "model_output": None,
            "correct_answer": sample.get("answer_idx"),
            "is_correct": False,
            "idx": idx,
            "error": "Step 6 failed or had invalid format - cannot proceed with Step 7",
            "step2dot5_result": step6_result.get("step2dot5_result"),
            "step3dot5_result": step6_result.get("step3dot5_result"),
        }
    
    # Create the prompt with context, question, and all variable assignments
    print(f"\n  📝 Creating prompt with variable assignments...")
    input_text = create_prompt_step7(dataset_name, sample, step6_result, step5_result, options_node)
    
    # Retry logic: up to 3 attempts
    max_retries = 3
    retry_count = 0
    last_error = None
    all_attempts = []
    
    while retry_count < max_retries:
        retry_count += 1
        
        try:
            # Call the LLM API
            print(f"\n  🔄 Step 7 - Attempt {retry_count}/{max_retries}...")
            print(f"  📞 Calling LLM API to extract answer...")
            model_output, token_usage = get_model_response(
                model_name=model_name,
                api_key=api_key,
                input_text=input_text,
                max_tokens=max_tokens,
                temperature=temperature,
                print_for_debug=True
            )
            
            # Parse the model's response
            print(f"  🔍 Parsing extracted answer...")
            parsed_result = parse_model_answer_step7(
                sample=sample,
                model_output=model_output,
                successful_api_call=True,
                thinking=thinking,
                dataset_name=dataset_name
            )
            
            # Record this attempt
            all_attempts.append({
                "attempt": retry_count,
                "successful_api_call": True,
                "right_format": parsed_result.get("right_format", False)
            })
            
            # Check if parsing was successful
            if parsed_result.get("right_format"):
                print(f"      ✅ Step 7 successful!")
                
                # Add additional metadata
                parsed_result["input_text"] = input_text
                parsed_result["token_usage"] = token_usage
                parsed_result["idx"] = idx
                parsed_result["step1_result"] = step6_result.get("step1_result")
                parsed_result["step2_result"] = step6_result.get("step2_result")
                parsed_result["step2dot5_result"] = step6_result.get("step2dot5_result")
                parsed_result["step3_result"] = step6_result.get("step3_result")
                parsed_result["step3dot5_result"] = step6_result.get("step3dot5_result")
                parsed_result["step4_result"] = step6_result.get("step4_result")
                parsed_result["step5_result"] = step6_result.get("step5_result")
                parsed_result["step6_result"] = step6_result
                parsed_result["attempts"] = all_attempts
                
                # Check if the extracted answer is correct
                if parsed_result.get("extracted_answer"):
                    extracted_answer = parsed_result["extracted_answer"]
                    correct_answer = sample.get("answer_idx")
                    
                    if dataset_name == "medqa":
                        # For MedQA, compare option letters
                        extracted_option = extracted_answer.get("option", "").upper()
                        is_correct = (extracted_option == correct_answer)
                    elif dataset_name == "uniadilr":
                        # For UniADILR, we would need to compare with the proof sentences
                        # This is more complex, so for now we'll leave it as None
                        is_correct = None
                    else:
                        is_correct = None
                    
                    parsed_result["is_correct"] = is_correct
                
                return parsed_result
            else:
                last_error = parsed_result.get("error", "Unknown parsing error")
                print(f"      ❌ Parsing failed: {last_error}")
                
                if retry_count < max_retries:
                    print(f"      🔄 Retrying...")
                    time.sleep(sleep_time)
            
        except Exception as e:
            last_error = f"Step 7 failed: {str(e)}"
            all_attempts.append({
                "attempt": retry_count,
                "error": last_error,
                "stage": "api_call"
            })
            print(f"      ❌ API call failed: {e}")
            
            if retry_count < max_retries:
                print(f"      🔄 Retrying...")
                time.sleep(sleep_time)
    
    # All retries exhausted
    return {
        "raw_data": sample,
        "successful_api_call": False,
        "right_format": False,
        "extracted_answer": None,
        "model_output": None,
        "correct_answer": sample.get("answer_idx"),
        "is_correct": False,
        "idx": idx,
        "error": f"Step 7 failed after {max_retries} attempts. Last error: {last_error}",
        "step2dot5_result": step6_result.get("step2dot5_result"),
        "step3dot5_result": step6_result.get("step3dot5_result"),
        "attempts": all_attempts
    }


def _convert_tuple_keys_to_strings(obj):
    """
    Recursively convert tuple keys to strings for JSON serialization.
    
    Args:
        obj: Object to convert (can be dict, list, tuple, or primitive)
    
    Returns:
        Converted object with tuple keys as strings
    """
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            # Convert tuple keys to string representation
            if isinstance(key, tuple):
                new_key = str(key)
            else:
                new_key = key
            new_dict[new_key] = _convert_tuple_keys_to_strings(value)
        return new_dict
    elif isinstance(obj, list):
        return [_convert_tuple_keys_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(obj)  # Convert tuples to lists
    else:
        return obj


def _strip_nested_steps(step_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove nested step results from a step result dictionary.
    
    Each step result contains all previous step results nested inside it
    (e.g., step6_result contains step5_result, which contains step4_result, etc.).
    This function creates a copy with only the step's own data, removing the nested
    step_X_result keys to avoid exponential duplication.
    
    Args:
        step_result: Full step result dictionary with nested previous steps
    
    Returns:
        dict: Step result with only its own data (no nested step results)
    """
    if not step_result:
        return None
    
    # Create a copy of the result
    clean_result = {}
    
    # Copy all keys EXCEPT nested step results
    for key, value in step_result.items():
        # Skip keys that are nested step results
        if key.startswith("step") and key.endswith("_result"):
            continue
        # Keep everything else
        clean_result[key] = value
    
    return clean_result


def save_step7_result(result: Dict[str, Any], dataset_name: str, model_name: str, 
                      output_dir: str = None) -> None:
    """
    Save step7 result to results.jsonl file.
    
    This function appends the result to a JSONL file, creating the directory
    structure if it doesn't exist. It strips out nested step results from each
    step's data to avoid exponential duplication (e.g., step6 normally contains
    step5, which contains step4, etc.). Each step's result will contain only
    its own specific data.
    
    Args:
        result: Step7 result dictionary
        dataset_name: "medqa" or "uniadilr"
        model_name: Name of the model used
        output_dir: Optional custom output directory (defaults to proposal/results/)
    """
    # Determine output directory
    if output_dir is None:
        # Default to proposal/results/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "results", dataset_name, model_name)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to results.jsonl
    results_file = os.path.join(output_dir, "results.jsonl")
    
    # Prepare the result for saving
    # Strip out nested step results from each step to avoid exponential duplication
    # (e.g., step6 contains step5, which contains step4, etc.)
    save_result = {
        "idx": result["idx"],
        "raw_data": result["raw_data"],
        "successful_api_call": result["successful_api_call"],
        "right_format": result["right_format"],
        "extracted_answer": result.get("extracted_answer"),
        "correct_answer": result.get("correct_answer"),
        "is_correct": result.get("is_correct"),
        "model_output": result.get("model_output"),
        "error": result.get("error"),
        "token_usage": result.get("token_usage"),
        "step1_result": _strip_nested_steps(result.get("step1_result")),
        "step2_result": _strip_nested_steps(result.get("step2_result")),
        "step2dot5_result": _strip_nested_steps(result.get("step2dot5_result")),
        "step3_result": _strip_nested_steps(result.get("step3_result")),
        "step3dot5_result": _strip_nested_steps(result.get("step3dot5_result")),
        "step4_result": _strip_nested_steps(result.get("step4_result")),
        "step5_result": _strip_nested_steps(result.get("step5_result")),
        "step6_result": _strip_nested_steps(result.get("step6_result"))
    }
    
    # Convert tuple keys to strings for JSON serialization
    save_result = _convert_tuple_keys_to_strings(save_result)
    
    # Append to JSONL file
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(save_result, ensure_ascii=False) + "\n")
    
    print(f"  💾 Result saved to: {results_file}")


def create_step7_summary(results_file: str, output_dir: str) -> Dict[str, Any]:
    """
    Create a summary report of step7 results from the results.jsonl file.
    
    Args:
        results_file: Path to results.jsonl
        output_dir: Directory to save summary report
    
    Returns:
        dict: Summary statistics
    """
    if not os.path.exists(results_file):
        return None
    
    # Read all results
    results = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    # Calculate statistics
    total_samples = len(results)
    successful_api_calls = sum(1 for r in results if r.get("successful_api_call"))
    right_format = sum(1 for r in results if r.get("right_format"))
    
    # Count correct answers (only for samples with valid format)
    valid_results = [r for r in results if r.get("right_format") and r.get("is_correct") is not None]
    correct_answers = sum(1 for r in valid_results if r.get("is_correct"))
    
    summary = {
        "total_samples": total_samples,
        "successful_api_calls": successful_api_calls,
        "right_format": right_format,
        "valid_results": len(valid_results),
        "correct_answers": correct_answers,
        "accuracy": correct_answers / len(valid_results) if valid_results else 0.0
    }
    
    # Save summary to file
    summary_file = os.path.join(output_dir, "step7_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n  📊 Summary saved to: {summary_file}")
    print(f"  📈 Accuracy: {summary['accuracy']:.2%} ({correct_answers}/{len(valid_results)})")
    
    return summary

