from prompting import create_prompt_step3dot5, parse_model_answer_step3dot5
from api_handler import get_model_response
import time

def step3dot5(sample, idx, model_name, api_key, max_tokens, temperature, thinking, sleep_time, dataset_name, step3_result):
    """
    Step 3.5: Identify Visible Nodes - Determine which nodes have known values from the question/context.
    
    This step takes the registered DAG from Step 3 and uses an AI model to identify which nodes
    in the DAG have values that are explicitly mentioned or can be directly inferred from the
    question and context.
    
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
        step3_result: Result dictionary from step3 containing the registered DAG
    
    Returns:
        dict: Result dictionary with visible nodes (node_id -> value) and metadata
    """
    time.sleep(sleep_time)
    
    # Check if step3 was successful and has a valid format
    if not step3_result.get("successful_api_call") or not step3_result.get("right_format"):
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "visible_nodes": None,
            "visible_nodes_metadata": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": "Step 3 failed or had invalid format - cannot proceed with Step 3.5",
            "step1_result": step3_result.get("step1_result"),
            "step2_result": step3_result.get("step2_result"),
            "step3_result": step3_result
        }
    
    registered_dag = step3_result.get("registered_dag")
    if not registered_dag:
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "visible_nodes": None,
            "visible_nodes_metadata": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": "No registered DAG found in step3 result",
            "step1_result": step3_result.get("step1_result"),
            "step2_result": step3_result.get("step2_result"),
            "step3_result": step3_result
        }
    
    # Retry logic: up to 3 attempts
    max_retries = 3
    retry_count = 0
    last_error = None
    all_attempts = []
    
    while retry_count < max_retries:
        retry_count += 1
        
        try:
            print(f"\n  🔍 Attempt {retry_count}/{max_retries} to identify visible nodes...")
            
            # Create prompt for visible node identification
            input_text = create_prompt_step3dot5(dataset_name, sample, step3_result)
            
            error_message = None
            model_output = None
            successful_api_call = False
            
            try:
                model_output, usage = get_model_response(model_name, api_key, input_text, max_tokens, temperature)
                successful_api_call = True
            except Exception as e:
                error_message = f"API call failed for sample {idx} in Step 3.5: {e}"
                last_error = error_message
                all_attempts.append({
                    "attempt": retry_count,
                    "error": error_message,
                    "stage": "api_call"
                })
                print(f"      ❌ API call failed: {e}")
                continue
            
            # Parse the model response
            result = parse_model_answer_step3dot5(sample, model_output, successful_api_call, thinking, step3_result)
            
            # Record this attempt
            all_attempts.append({
                "attempt": retry_count,
                "successful_api_call": successful_api_call,
                "right_format": result.get("right_format", False),
                "num_visible_nodes": len(result.get("visible_nodes", {})),
                "validation_issues": result.get("validation_issues", [])
            })
            
            # Check if parsing was successful
            if result.get("right_format"):
                print(f"      ✅ Successfully identified {len(result.get('visible_nodes', {}))} visible node(s)")
                
                # Create metadata
                visible_nodes_metadata = _create_visible_nodes_metadata(
                    result.get("visible_nodes", {}),
                    registered_dag,
                    retry_count,
                    all_attempts
                )
                
                return {
                    "raw_data": sample,
                    "successful_api_call": True,
                    "right_format": True,
                    "visible_nodes": result.get("visible_nodes", {}),
                    "visible_nodes_metadata": visible_nodes_metadata,
                    "model_output": model_output,
                    "reasoning": result.get("reasoning", ""),
                    "input_text": input_text,
                    "correct_answer": sample.get("answer_idx"),
                    "idx": idx,
                    "error": None,
                    "token_usage": usage if successful_api_call else None,
                    "attempts": all_attempts,
                    "step1_result": step3_result.get("step1_result"),
                    "step2_result": step3_result.get("step2_result"),
                    "step3_result": step3_result
                }
            else:
                # Parsing failed, record error and retry
                last_error = result.get("error", "Unknown parsing error")
                print(f"      ❌ Parsing failed: {last_error}")
                
                if retry_count < max_retries:
                    print(f"      🔄 Retrying...")
                    time.sleep(sleep_time)  # Small delay before retry
                
        except Exception as e:
            last_error = f"Unexpected error in Step 3.5 attempt {retry_count}: {e}"
            all_attempts.append({
                "attempt": retry_count,
                "error": last_error,
                "stage": "processing"
            })
            print(f"      ❌ Unexpected error: {e}")
            
            if retry_count < max_retries:
                print(f"      🔄 Retrying...")
                time.sleep(sleep_time)
    
    # All retries exhausted
    return {
        "raw_data": sample,
        "successful_api_call": False,
        "right_format": False,
        "visible_nodes": None,
        "visible_nodes_metadata": None,
        "correct_answer": sample.get("answer_idx"),
        "idx": idx,
        "error": f"Step 3.5 failed after {max_retries} attempts. Last error: {last_error}",
        "attempts": all_attempts,
        "step1_result": step3_result.get("step1_result"),
        "step2_result": step3_result.get("step2_result"),
        "step3_result": step3_result
    }


def _create_visible_nodes_metadata(visible_nodes, registered_dag, retry_count, all_attempts):
    """
    Create metadata about the visible nodes identification process.
    
    Args:
        visible_nodes: Dictionary mapping node_id -> value
        registered_dag: The registered DAG structure
        retry_count: Number of attempts needed
        all_attempts: List of all attempt results
    
    Returns:
        dict: Metadata about visible nodes
    """
    # Analyze visible nodes
    visible_node_details = []
    node_type_counts = {"binary": 0, "categorical": 0}
    
    for node_id, value in visible_nodes.items():
        if node_id in registered_dag["nodes"]:
            node_info = registered_dag["nodes"][node_id]
            node_type_counts[node_info["type"]] += 1
            
            visible_node_details.append({
                "node_id": node_id,
                "node_name": node_info["name"],
                "node_type": node_info["type"],
                "assigned_value": value,
                "possible_values": _get_node_states_from_node_info(node_info)
            })
    
    # Calculate coverage
    total_nodes = len(registered_dag["nodes"])
    visible_count = len(visible_nodes)
    hidden_count = total_nodes - visible_count
    visibility_ratio = visible_count / total_nodes if total_nodes > 0 else 0
    
    return {
        "identification_timestamp": time.time(),
        "attempts_needed": retry_count,
        "total_attempts": len(all_attempts),
        "node_statistics": {
            "total_nodes": total_nodes,
            "visible_nodes": visible_count,
            "hidden_nodes": hidden_count,
            "visibility_ratio": visibility_ratio,
            "visible_binary_nodes": node_type_counts["binary"],
            "visible_categorical_nodes": node_type_counts["categorical"]
        },
        "visible_node_details": visible_node_details,
        "all_attempts": all_attempts
    }


def _get_node_states_from_node_info(node_info):
    """
    Get the possible states for a node.
    
    This matches the logic from step4.py's _get_node_states function.
    
    Args:
        node_info: Node information dictionary
    
    Returns:
        list: List of possible states
    """
    if node_info["type"] == "binary":
        return ["yes", "no"]
    elif node_info["type"] == "categorical":
        categories = node_info.get("categories", [])
        return categories if isinstance(categories, list) and categories else []
    else:
        return ["unknown"]

