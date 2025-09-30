import time
import itertools
from typing import Dict, Any, List, Tuple, Optional
from api_handler import get_model_response

def step4(sample: Dict[str, Any], idx: int, model_name: str, api_key: str, max_tokens: int, 
          temperature: float, thinking: bool, sleep_time: float, dataset_name: str, 
          step3_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 4: CPT Creator - Generate Conditional Probability Tables for all nodes using LLMs.
    
    This step takes the registered DAG from Step 3 and creates CPTs for each node
    by querying LLMs with qualitative probability estimates that are then converted
    to numerical values.
    
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
        dict: Result dictionary with CPTs for all nodes
    """
    
    # Check if step3 was successful and has a valid format
    if not step3_result.get("successful_api_call") or not step3_result.get("right_format"):
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "cpts": None,
            "cpt_metadata": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": "Step 3 failed or had invalid format - cannot proceed with Step 4",
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
            "cpts": None,
            "cpt_metadata": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": "No registered DAG found in step3 result",
            "step1_result": step3_result.get("step1_result"),
            "step2_result": step3_result.get("step2_result"),
            "step3_result": step3_result
        }
    
    try:
        # Generate CPTs for all nodes
        all_cpts = {}
        cpt_generation_log = []
        total_api_calls = 0
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Process each node in the DAG
        for node_id, node_info in registered_dag["nodes"].items():
            time.sleep(sleep_time)
            
            print(f"Generating CPT for node {node_id}: {node_info['name']}")
            
            # Generate CPT for this node
            cpt_result = _generate_node_cpt(
                sample, node_info, registered_dag, model_name, api_key, 
                max_tokens, temperature, thinking, dataset_name
            )
            
            if cpt_result["success"]:
                all_cpts[node_id] = cpt_result["cpt"]
                cpt_generation_log.append({
                    "node_id": node_id,
                    "node_name": node_info["name"],
                    "success": True,
                    "token_usage": cpt_result.get("token_usage")
                })
                
                # Accumulate token usage
                if cpt_result.get("token_usage"):
                    for key in total_token_usage:
                        total_token_usage[key] += cpt_result["token_usage"].get(key, 0)
                
                total_api_calls += 1
            else:
                cpt_generation_log.append({
                    "node_id": node_id,
                    "node_name": node_info["name"],
                    "success": False,
                    "error": cpt_result["error"]
                })
                
                # If any CPT generation fails, return error
                return {
                    "raw_data": sample,
                    "successful_api_call": False,
                    "right_format": False,
                    "cpts": None,
                    "cpt_metadata": None,
                    "correct_answer": sample.get("answer_idx"),
                    "idx": idx,
                    "error": f"CPT generation failed for node {node_id}: {cpt_result['error']}",
                    "cpt_generation_log": cpt_generation_log,
                    "step1_result": step3_result.get("step1_result"),
                    "step2_result": step3_result.get("step2_result"),
                    "step3_result": step3_result
                }
        
        # Create metadata about CPT generation
        cpt_metadata = _create_cpt_metadata(all_cpts, registered_dag, cpt_generation_log, total_api_calls, total_token_usage)
        
        return {
            "raw_data": sample,
            "successful_api_call": True,
            "right_format": True,
            "cpts": all_cpts,
            "cpt_metadata": cpt_metadata,
            "cpt_generation_log": cpt_generation_log,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": None,
            "token_usage": total_token_usage,
            "step1_result": step3_result.get("step1_result"),
            "step2_result": step3_result.get("step2_result"),
            "step3_result": step3_result
        }
        
    except Exception as e:
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "cpts": None,
            "cpt_metadata": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": f"CPT generation failed for sample {idx}: {e}",
            "step1_result": step3_result.get("step1_result"),
            "step2_result": step3_result.get("step2_result"),
            "step3_result": step3_result
        }


def _generate_node_cpt(sample: Dict[str, Any], node_info: Dict[str, Any], registered_dag: Dict[str, Any],
                      model_name: str, api_key: str, max_tokens: int, temperature: float, 
                      thinking: bool, dataset_name: str) -> Dict[str, Any]:
    """
    Generate CPT for a single node using LLM with row-by-row approach.
    
    Args:
        sample: Original data sample
        node_info: Information about the target node
        registered_dag: The complete registered DAG
        model_name, api_key, max_tokens, temperature: LLM parameters
        thinking: Whether model supports <think> blocks
        dataset_name: "medqa" or "uniadilr"
    
    Returns:
        dict: Result with CPT or error information
    """
    try:
        # Get parent information
        parent_info = _get_parent_info(node_info, registered_dag)
        node_states = _get_node_states(node_info)
        
        # Generate all parent combinations
        if parent_info["has_parents"]:
            parent_combinations = _generate_parent_combinations(parent_info)
        else:
            # Root node - only one "combination" (no parents)
            parent_combinations = ["NO_PARENTS"]
        
        print(f"Generating CPT for {node_info['name']} with {len(parent_combinations)} combinations")
        
        # Generate CPT row by row
        qualitative_cpt = {}
        numerical_cpt = {}
        all_raw_responses = []
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        for combination in parent_combinations:
            # Generate CPT row for this specific parent combination
            row_result = _generate_cpt_row(
                sample, node_info, parent_info, combination, registered_dag,
                model_name, api_key, max_tokens, temperature, thinking, dataset_name
            )
            
            if row_result["success"]:
                # Store the row result
                qualitative_cpt[combination] = row_result["qualitative_probs"]
                numerical_cpt[combination] = row_result["numerical_probs"]
                all_raw_responses.append({
                    "condition": combination,
                    "response": row_result["raw_response"]
                })
                
                # Accumulate token usage
                if row_result.get("token_usage"):
                    for key in total_token_usage:
                        total_token_usage[key] += row_result["token_usage"].get(key, 0)
            else:
                return {
                    "success": False,
                    "error": f"Failed to generate CPT row for condition {combination}: {row_result['error']}",
                    "raw_responses": all_raw_responses
                }
        
        return {
            "success": True,
            "cpt": {
                "node_id": node_info["id"],
                "node_name": node_info["name"],
                "node_type": node_info["type"],
                "states": node_states,
                "parents": parent_info,
                "qualitative_cpt": qualitative_cpt,
                "numerical_cpt": numerical_cpt,
                "raw_responses": all_raw_responses,
                "generation_method": "row_by_row"
            },
            "token_usage": total_token_usage
        }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"CPT generation failed: {e}"
        }


def _generate_parent_combinations(parent_info: Dict[str, Any]) -> List:
    """Generate all possible combinations of parent states."""
    import itertools
    
    if not parent_info["has_parents"]:
        return ["NO_PARENTS"]
    
    # Get all parent states
    parent_states = []
    parent_order = []
    
    for parent_id, parent_data in parent_info["parents"].items():
        parent_order.append(parent_id)
        parent_states.append(parent_data["states"])
    
    # Generate all combinations
    combinations = []
    for combo in itertools.product(*parent_states):
        if len(combo) == 1:
            combinations.append(combo[0])
        else:
            combinations.append(combo)
    
    return combinations


def _generate_cpt_row(sample: Dict[str, Any], node_info: Dict[str, Any], parent_info: Dict[str, Any],
                     condition: Any, registered_dag: Dict[str, Any], model_name: str, api_key: str,
                     max_tokens: int, temperature: float, thinking: bool, dataset_name: str) -> Dict[str, Any]:
    """Generate a single CPT row for a specific parent condition."""
    try:
        # Create prompt for this specific condition
        prompt = _create_cpt_row_prompt(sample, node_info, parent_info, condition, registered_dag, dataset_name)
        
        # Get LLM response
        model_output, usage = get_model_response(model_name, api_key, prompt, max_tokens, temperature)
        
        # Parse the response
        parsed_row = _parse_cpt_row_response(model_output, node_info, condition, thinking)
        
        if parsed_row["success"]:
            # Convert qualitative probabilities to numerical values
            numerical_probs = _convert_qualitative_row_to_numerical(parsed_row["qualitative_probs"])
            
            return {
                "success": True,
                "qualitative_probs": parsed_row["qualitative_probs"],
                "numerical_probs": numerical_probs,
                "raw_response": model_output,
                "token_usage": usage
            }
        else:
            return {
                "success": False,
                "error": f"Failed to parse CPT row response: {parsed_row['error']}",
                "raw_response": model_output
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"CPT row generation failed: {e}"
        }


def _create_cpt_row_prompt(sample: Dict[str, Any], node_info: Dict[str, Any], parent_info: Dict[str, Any],
                          condition: Any, registered_dag: Dict[str, Any], dataset_name: str) -> str:
    """Create prompt for generating a single CPT row."""
    from prompting import _load_prompt_template, _parse_context_and_question
    
    # Load the appropriate template
    prompt_template = _load_prompt_template(dataset_name, "step4", "v1")
    
    # Get context and question
    context, question = _parse_context_and_question(dataset_name, sample)
    
    # Get node states
    node_states = _get_node_states(node_info)
    
    # Build the node information section
    node_section = f"""
TARGET NODE INFORMATION:
- Node Name: {node_info['name']}
- Node Type: {node_info['type']}
- Possible States: {', '.join(node_states)}
"""
    
    # Build parent information and specific condition
    if parent_info["has_parents"]:
        parent_section = f"""
PARENT NODES INFORMATION:
"""
        for parent_id, parent_data in parent_info["parents"].items():
            parent_section += f"- {parent_data['name']} ({parent_data['type']}): {', '.join(parent_data['states'])}\n"
        
        # Format the specific condition
        if condition == "NO_PARENTS":
            condition_section = f"""
SPECIFIC PARENT CONDITION TO ANALYZE:
- NO_PARENTS (this is a root node)
"""
        else:
            if isinstance(condition, tuple):
                # Multiple parents
                parent_names = [parent_data['name'] for parent_data in parent_info["parents"].values()]
                condition_pairs = [f"{name} = {state}" for name, state in zip(parent_names, condition)]
                condition_str = ", ".join(condition_pairs)
            else:
                # Single parent
                parent_name = list(parent_info["parents"].values())[0]['name']
                condition_str = f"{parent_name} = {condition}"
            
            condition_section = f"""
SPECIFIC PARENT CONDITION TO ANALYZE:
- {condition_str}
"""
    else:
        parent_section = "\nPARENT NODES INFORMATION:\n- This node has no parents (root node)\n"
        condition_section = f"""
SPECIFIC PARENT CONDITION TO ANALYZE:
- NO_PARENTS (this is a root node)
"""
    
    # Build context section
    context_section = f"""
ORIGINAL CONTEXT:
{context}

QUESTION:
{question}
"""
    
    # Build complete prompt
    complete_prompt = f"""{prompt_template}

{context_section}

{node_section}

{parent_section}

{condition_section}

Please provide the conditional probability distribution for '{node_info['name']}' given the specific parent condition above."""
    
    return complete_prompt


def _parse_cpt_row_response(model_output: str, node_info: Dict[str, Any], condition: Any, thinking: bool) -> Dict[str, Any]:
    """Parse the LLM response for a single CPT row."""
    import re
    
    try:
        # Clean the output if it contains <think> blocks
        if thinking:
            cleaned_text = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL).strip()
        else:
            cleaned_text = model_output.strip()
        
        # Extract reasoning and CPD row sections
        reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
        cpd_row_pattern = r"<cpd_row>(.*?)</cpd_row>"
        
        reasoning_match = re.search(reasoning_pattern, cleaned_text, flags=re.DOTALL)
        cpd_row_match = re.search(cpd_row_pattern, cleaned_text, flags=re.DOTALL)
        
        if not reasoning_match or not cpd_row_match:
            return {"success": False, "error": "Missing reasoning or cpd_row sections"}
        
        reasoning = reasoning_match.group(1).strip()
        cpd_row_content = cpd_row_match.group(1).strip()
        
        # Parse CPD row content
        row_data = _parse_cpd_row_content(cpd_row_content, node_info, condition)
        
        if row_data["success"]:
            return {
                "success": True,
                "qualitative_probs": row_data["probabilities"],
                "reasoning": reasoning
            }
        else:
            return {"success": False, "error": row_data["error"]}
            
    except Exception as e:
        return {"success": False, "error": f"Parsing error: {e}"}


def _parse_cpd_row_content(cpd_row_content: str, node_info: Dict[str, Any], condition: Any) -> Dict[str, Any]:
    """Parse the CPD row content section."""
    import re
    
    try:
        lines = [line.strip() for line in cpd_row_content.split('\n') if line.strip()]
        
        node_name = None
        parsed_condition = None
        states = []
        probabilities = {}
        
        for line in lines:
            if line.startswith("NODE:"):
                node_name = line.split("NODE:", 1)[1].strip()
            elif line.startswith("CONDITION:"):
                parsed_condition = line.split("CONDITION:", 1)[1].strip()
            elif line.startswith("STATES:"):
                states_str = line.split("STATES:", 1)[1].strip()
                states = [s.strip() for s in states_str.split(",")]
            elif line.startswith("PROBABILITIES:"):
                continue  # Skip header
            elif ":" in line and not line.startswith(("NODE:", "CONDITION:", "STATES:")):
                # Parse probability line
                state, prob_level = line.split(":", 1)
                probabilities[state.strip()] = prob_level.strip()
        
        return {
            "success": True,
            "node_name": node_name,
            "condition": parsed_condition,
            "states": states,
            "probabilities": probabilities
        }
        
    except Exception as e:
        return {"success": False, "error": f"CPD row content parsing error: {e}"}


def _convert_qualitative_row_to_numerical(qualitative_probs: Dict[str, str]) -> Dict[str, float]:
    """Convert qualitative probabilities in a single row to numerical values."""
    
    # Mapping from qualitative terms to numerical values
    qualitative_mapping = {
        "very_low": 0.05,
        "low": 0.20,
        "moderate": 0.40,
        "high": 0.70,
        "very_high": 0.90
    }
    
    numerical_probs = {}
    raw_probs = {}
    total_raw = 0
    
    # Convert qualitative to raw numerical
    for state, qual_prob in qualitative_probs.items():
        numerical_value = qualitative_mapping.get(qual_prob.lower(), 0.4)  # Default to moderate
        raw_probs[state] = numerical_value
        total_raw += numerical_value
    
    # Normalize to ensure probabilities sum to 1
    if total_raw > 0:
        for state, raw_prob in raw_probs.items():
            numerical_probs[state] = round(raw_prob / total_raw, 4)
    else:
        # Fallback: equal distribution
        num_states = len(qualitative_probs)
        for state in qualitative_probs.keys():
            numerical_probs[state] = round(1.0 / num_states, 4)
    
    return numerical_probs


def _create_cpt_prompt(sample: Dict[str, Any], node_info: Dict[str, Any], 
                      registered_dag: Dict[str, Any], dataset_name: str) -> str:
    """Create prompt for CPT generation."""
    from prompting import _load_prompt_template, _parse_context_and_question
    
    # Load the appropriate template
    prompt_template = _load_prompt_template(dataset_name, "step4", "v1")
    
    # Get context and question
    context, question = _parse_context_and_question(dataset_name, sample)
    
    # Get node states
    node_states = _get_node_states(node_info)
    
    # Get parent information
    parent_info = _get_parent_info(node_info, registered_dag)
    
    # Build the complete prompt
    node_section = f"""
TARGET NODE INFORMATION:
- Node Name: {node_info['name']}
- Node Type: {node_info['type']}
- Possible States: {', '.join(node_states)}
"""
    
    if parent_info["has_parents"]:
        parent_section = f"""
PARENT NODES INFORMATION:
"""
        for parent_id, parent_data in parent_info["parents"].items():
            parent_section += f"- {parent_data['name']} ({parent_data['type']}): {', '.join(parent_data['states'])}\n"
    else:
        parent_section = "\nPARENT NODES INFORMATION:\n- This node has no parents (root node)\n"
    
    context_section = f"""
ORIGINAL CONTEXT:
{context}

QUESTION:
{question}
"""
    
    complete_prompt = f"""{prompt_template}

{context_section}

{node_section}

{parent_section}

Please provide the conditional probability distribution for the target node '{node_info['name']}' given its parent configuration."""
    
    return complete_prompt


def _get_node_states(node_info: Dict[str, Any]) -> List[str]:
    """Get the possible states for a node."""
    if node_info["type"] == "binary":
        return ["yes", "no"]  # Standard binary states
    elif node_info["type"] == "categorical":
        categories = node_info.get("categories", [])
        # Ensure we always return a list, even if categories is False or None
        return categories if isinstance(categories, list) and categories else []
    else:
        return ["unknown"]


def _get_parent_info(node_info: Dict[str, Any], registered_dag: Dict[str, Any]) -> Dict[str, Any]:
    """Get information about parent nodes."""
    node_id = node_info["id"]
    parent_ids = registered_dag["adjacency_list"][node_id]["parents"]
    
    if not parent_ids:
        return {"has_parents": False, "parents": {}}
    
    parents = {}
    for parent_id in parent_ids:
        parent_node = registered_dag["nodes"][parent_id]
        parents[parent_id] = {
            "name": parent_node["name"],
            "type": parent_node["type"],
            "states": _get_node_states(parent_node)
        }
    
    return {"has_parents": True, "parents": parents}


def _parse_cpt_response(model_output: str, node_info: Dict[str, Any], 
                       registered_dag: Dict[str, Any], thinking: bool) -> Dict[str, Any]:
    """Parse the LLM response to extract CPT information."""
    import re
    
    try:
        # Clean the output if it contains <think> blocks
        if thinking:
            cleaned_text = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL).strip()
        else:
            cleaned_text = model_output.strip()
        
        # Extract reasoning and CPD sections
        reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
        cpd_pattern = r"<cpd>(.*?)</cpd>"
        
        reasoning_match = re.search(reasoning_pattern, cleaned_text, flags=re.DOTALL)
        cpd_match = re.search(cpd_pattern, cleaned_text, flags=re.DOTALL)
        
        if not reasoning_match or not cpd_match:
            return {"success": False, "error": "Missing reasoning or cpd sections"}
        
        reasoning = reasoning_match.group(1).strip()
        cpd_content = cpd_match.group(1).strip()
        
        # Parse CPD content
        cpt_data = _parse_cpd_content(cpd_content, node_info, registered_dag)
        
        if cpt_data["success"]:
            return {
                "success": True,
                "cpt_data": {
                    "reasoning": reasoning,
                    "node_name": cpt_data["node_name"],
                    "parents": cpt_data["parents"],
                    "states": cpt_data["states"],
                    "conditional_probabilities": cpt_data["conditional_probabilities"]
                }
            }
        else:
            return {"success": False, "error": cpt_data["error"]}
            
    except Exception as e:
        return {"success": False, "error": f"Parsing error: {e}"}


def _parse_cpd_content(cpd_content: str, node_info: Dict[str, Any], 
                      registered_dag: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the CPD content section."""
    import re
    
    try:
        lines = [line.strip() for line in cpd_content.split('\n') if line.strip()]
        
        node_name = None
        parents = []
        states = []
        conditional_probabilities = {}
        
        for line in lines:
            if line.startswith("NODE:"):
                node_name = line.split("NODE:", 1)[1].strip()
            elif line.startswith("PARENTS:"):
                parents_str = line.split("PARENTS:", 1)[1].strip()
                if parents_str and parents_str != "None":
                    parents = [p.strip() for p in parents_str.split(",")]
            elif line.startswith("STATES:"):
                states_str = line.split("STATES:", 1)[1].strip()
                states = [s.strip() for s in states_str.split(",")]
            elif line.startswith("CONDITIONAL_PROBABILITIES:"):
                continue  # Skip header
            elif "=>" in line:
                # Parse conditional probability line
                condition, probabilities = line.split("=>", 1)
                condition = condition.strip()
                probabilities = probabilities.strip()
                
                # Parse condition (parent states)
                if condition == "NO_PARENTS":
                    condition_key = "NO_PARENTS"
                else:
                    condition_key = tuple(s.strip() for s in condition.split(","))
                
                # Parse probabilities for each state
                prob_dict = {}
                prob_pairs = probabilities.split(",")
                for pair in prob_pairs:
                    if ":" in pair:
                        state, prob_level = pair.split(":", 1)
                        prob_dict[state.strip()] = prob_level.strip()
                
                conditional_probabilities[condition_key] = prob_dict
        
        return {
            "success": True,
            "node_name": node_name,
            "parents": parents,
            "states": states,
            "conditional_probabilities": conditional_probabilities
        }
        
    except Exception as e:
        return {"success": False, "error": f"CPD content parsing error: {e}"}


def _convert_qualitative_to_numerical(cpt_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert qualitative probability levels to numerical values."""
    
    # Mapping from qualitative terms to numerical values
    #TODO: set proper values
    qualitative_mapping = {
        "very_low": 0.1,
        "low": 0.30,
        "moderate": 0.50,
        "high": 0.70,
        "very_high": 0.90
    }
    
    numerical_cpt = {}
    
    for condition, prob_dict in cpt_data["conditional_probabilities"].items():
        numerical_probs = {}
        
        # Convert qualitative to numerical
        raw_probs = {}
        total_raw = 0
        
        for state, qual_prob in prob_dict.items():
            numerical_value = qualitative_mapping.get(qual_prob.lower(), 0.4)  # Default to moderate
            raw_probs[state] = numerical_value
            total_raw += numerical_value
        
        # Normalize to ensure probabilities sum to 1
        if total_raw > 0:
            for state, raw_prob in raw_probs.items():
                numerical_probs[state] = round(raw_prob / total_raw, 4)
        else:
            # Fallback: equal distribution
            num_states = len(prob_dict)
            for state in prob_dict.keys():
                numerical_probs[state] = round(1.0 / num_states, 4)
        
        numerical_cpt[condition] = numerical_probs
    
    return numerical_cpt


def _create_cpt_metadata(all_cpts: Dict[str, Any], registered_dag: Dict[str, Any], 
                        generation_log: List[Dict], total_api_calls: int, 
                        total_token_usage: Dict[str, int]) -> Dict[str, Any]:
    """Create metadata about the CPT generation process."""
    
    # Analyze CPT characteristics
    node_types = {"binary": 0, "categorical": 0}
    total_parameters = 0
    
    for node_id, cpt in all_cpts.items():
        node_types[cpt["node_type"]] += 1
        
        # Count parameters (number of probability values)
        for condition_probs in cpt["numerical_cpt"].values():
            total_parameters += len(condition_probs)
    
    # Success statistics
    successful_generations = sum(1 for log in generation_log if log["success"])
    failed_generations = len(generation_log) - successful_generations
    
    return {
        "generation_timestamp": time.time(),
        "dag_info": {
            "total_nodes": registered_dag["num_nodes"],
            "nodes_with_cpts": len(all_cpts)
        },
        "cpt_statistics": {
            "total_cpts": len(all_cpts),
            "binary_node_cpts": node_types["binary"],
            "categorical_node_cpts": node_types["categorical"],
            "total_parameters": total_parameters
        },
        "generation_statistics": {
            "total_api_calls": total_api_calls,
            "successful_generations": successful_generations,
            "failed_generations": failed_generations,
            "success_rate": successful_generations / len(generation_log) if generation_log else 0
        },
        "token_usage": total_token_usage,
        "generation_log": generation_log
    }
