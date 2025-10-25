import yaml
from typing import Dict, Any, Tuple
import re
import os

import re
from typing import Dict, Any, Optional, Tuple

def _parse_dag_content_strictly(dag_content: str) -> Tuple[Optional[list], Optional[list]]:
    """
    Parses the content of a <dag> block strictly.
    Requires NODES: header, then EDGES: header.
    Returns (parsed_nodes, parsed_edges) tuple.
    
    Now handles both binary and categorical nodes:
    - "node_name (binary)" 
    - "node_name (categorical: cat1, cat2, cat3)"
    """
    parsed_nodes = None
    parsed_edges = None

    # 1. Check for the existence and order of NODES: and EDGES: headers.
    # We use re.IGNORECASE to be slightly tolerant of "nodes:" vs "NODES:".
    nodes_header_match = re.search(r"NODES:", dag_content, re.IGNORECASE)
    edges_header_match = re.search(r"EDGES:", dag_content, re.IGNORECASE)

    if not nodes_header_match or not edges_header_match:
        # If either header is missing, parsing fails.
        return None, None

    if nodes_header_match.start() >= edges_header_match.start():
        # If NODES: does not appear before EDGES:, parsing fails.
        return None, None
        
    # 2. Split the DAG content into two parts based on the headers.
    # The node_text is the content between "NODES:" and "EDGES:".
    node_text = dag_content[nodes_header_match.end():edges_header_match.start()]
    # The edge_text is the content after "EDGES:".
    edge_text = dag_content[edges_header_match.end():]

    # 3. Parse nodes from the node_text section with type information.
    # Pattern matches: "node1: node_name (binary)" or "node1: node_name (categorical: cat1, cat2, cat3)"
    nodes_pattern = r"node\d+:\s*(.*)"
    node_matches = re.findall(nodes_pattern, node_text)
    if node_matches:
        parsed_nodes = []
        for node_line in node_matches:
            node_line = node_line.strip()
            # Parse the node with type information
            node_info = _parse_node_with_type(node_line)
            if node_info:
                parsed_nodes.append(node_info)

    # 4. Parse edges from the edge_text section.
    edges_pattern = r"edge\d+:\s*node\d+\s*->\s*node\d+"
    edge_matches = re.findall(edges_pattern, edge_text)
    if edge_matches:
        # We need to extract just the 'nodeX -> nodeY' part.
        # A refined regex can do this in one step.
        edge_capture_pattern = r"edge\d+:\s*(node\d+\s*->\s*node\d+)"
        captured_edges = re.findall(edge_capture_pattern, edge_text)
        parsed_edges = [edge.strip() for edge in captured_edges]

    return parsed_nodes, parsed_edges

def _parse_node_with_type(node_line: str) -> Optional[dict]:
    """
    Parse a node line to extract name, type, and categories.
    
    Expected formats:
    - "node_name (binary)"
    - "node_name (categorical: cat1, cat2, cat3)"
    
    Returns dict with: {"name": str, "type": str, "categories": list or None}
    """
    # Try to match binary nodes first
    binary_match = re.match(r"^(.+?)\s*\(binary\)\s*$", node_line)
    if binary_match:
        return {
            "name": binary_match.group(1).strip(),
            "type": "binary",
            "categories": None
        }
    
    # Try to match categorical nodes
    categorical_match = re.match(r"^(.+?)\s*\(categorical:\s*(.+?)\)\s*$", node_line)
    if categorical_match:
        node_name = categorical_match.group(1).strip()
        categories_str = categorical_match.group(2).strip()
        # Split categories by comma and clean them
        categories = [cat.strip() for cat in categories_str.split(',')]
        return {
            "name": node_name,
            "type": "categorical",
            "categories": categories
        }
    
    # If neither pattern matches, return None (parsing failed)
    return None

def _parse_additions_content_strictly(additions_content: str) -> Tuple[Optional[list], Optional[list]]:
    """
    Parses the content of an <additions> block strictly.
    Requires NEW_NODES: header, then NEW_EDGES: header.
    Returns (parsed_new_nodes, parsed_new_edges) tuple.
    
    Handles both binary and categorical nodes, and also handles "None" values
    when there are no additions in a section.
    """
    parsed_new_nodes = []
    parsed_new_edges = []

    # 1. Check for the existence and order of NEW_NODES: and NEW_EDGES: headers.
    nodes_header_match = re.search(r"NEW_NODES:", additions_content, re.IGNORECASE)
    edges_header_match = re.search(r"NEW_EDGES:", additions_content, re.IGNORECASE)

    if not nodes_header_match or not edges_header_match:
        # If either header is missing, parsing fails.
        return None, None

    if nodes_header_match.start() >= edges_header_match.start():
        # If NEW_NODES: does not appear before NEW_EDGES:, parsing fails.
        return None, None
        
    # 2. Split the additions content into two parts based on the headers.
    node_text = additions_content[nodes_header_match.end():edges_header_match.start()]
    edge_text = additions_content[edges_header_match.end():]

    # 3. Parse new nodes from the node_text section with type information.
    # Check if the section contains "None" (case-insensitive)
    if re.search(r"^\s*None\s*$", node_text.strip(), re.IGNORECASE):
        parsed_new_nodes = []
    else:
        nodes_pattern = r"node\d+:\s*(.*)"
        node_matches = re.findall(nodes_pattern, node_text)
        if node_matches:
            for node_line in node_matches:
                node_line = node_line.strip()
                # Parse the node with type information
                node_info = _parse_node_with_type(node_line)
                if node_info:
                    parsed_new_nodes.append(node_info)

    # 4. Parse new edges from the edge_text section.
    # Check if the section contains "None" (case-insensitive)
    if re.search(r"^\s*None\s*$", edge_text.strip(), re.IGNORECASE):
        parsed_new_edges = []
    else:
        edge_capture_pattern = r"edge\d+:\s*(node\d+\s*->\s*node\d+)"
        captured_edges = re.findall(edge_capture_pattern, edge_text)
        parsed_new_edges = [edge.strip() for edge in captured_edges]

    return parsed_new_nodes, parsed_new_edges


def parse_model_answer_step1(sample: Dict[str, Any], model_output: str, successful_api_call: bool, thinking: bool) -> Dict[str, Any]:
    right_format = False
    extracted_reasoning = None
    parsed_nodes = None
    parsed_edges = None

    if not successful_api_call:
        return {
            "raw_data": sample, "successful_api_call": False, "right_format": False,
            "model_output": model_output, "model_answer": None,
            "correct_answer": sample.get("answer_idx"), "token_usage": None,
        }

    # 1. Clean the output if it contains <think> blocks
    if thinking:
        cleaned_text = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL).strip()
    else:
        cleaned_text = model_output.strip()

    # 2. Define regex patterns for the main blocks
    reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
    dag_pattern = r"<dag>(.*?)</dag>"

    # 3. Enforce order: Find <reasoning> first, then find <dag> *after* it.
    reasoning_match = re.search(reasoning_pattern, cleaned_text, flags=re.DOTALL)
    
    if reasoning_match:
        extracted_reasoning = reasoning_match.group(1).strip()
        
        # Search for the <dag> block *only in the text following the reasoning block*.
        text_after_reasoning = cleaned_text[reasoning_match.end():]
        dag_match = re.search(dag_pattern, text_after_reasoning, flags=re.DOTALL)

        if dag_match:
            dag_content = dag_match.group(1).strip()
            
            # 4. Use the new strict function to parse the <dag> content
            parsed_nodes, parsed_edges = _parse_dag_content_strictly(dag_content)

    # 5. Determine if the format is correct based on the strict parsing results
    # The format is correct only if all components were found in the correct order.
    if extracted_reasoning is not None and parsed_nodes and parsed_edges:
        right_format = True

    # 6. Structure the final model answer
    model_answer = None
    if right_format:
        model_answer = {
            "reasoning": extracted_reasoning,
            "nodes": parsed_nodes,
            "edges": parsed_edges,
        }
        
    correct_answer = sample.get("answer_idx")

    # 7. Return the comprehensive dictionary
    return {
        "raw_data": sample,
        "successful_api_call": successful_api_call,
        "right_format": right_format,
        "model_output": model_output,
        "model_answer": model_answer,
        "correct_answer": correct_answer,
    }

def _parse_context_and_question(dataset_name: str, sample: Dict[str, Any]) -> Tuple[str, str]:
    if dataset_name == "medqa":
        return sample["question"].replace(sample["question"].split(".")[-1], ""), sample["question"].split(".")[-1]
    elif dataset_name == "uniadilr":
        return sample["context"], sample["hypothesis"]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

def _get_answer_choices(dataset_name: str, sample: Dict[str, Any]) -> str:
    """
    Extract answer choices from the sample data.
    
    Args:
        dataset_name: "medqa" or "uniadilr"
        sample: Original data sample
    
    Returns:
        str: Formatted answer choices text (empty string if no choices available)
    """
    if dataset_name == "medqa":
        # MedQA has multiple choice options with A, B, C, D format
        options = sample.get("options", {})
        if options:
            choices_text = "Answer Choices:\n"
            for key, value in sorted(options.items()):
                choices_text += f"{key}. {value}\n"
            return choices_text.strip()
        return ""
    elif dataset_name == "uniadilr":
        # UniADILR doesn't have multiple choice options - it's a logical proof task
        return ""
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

def _load_prompt_template(dataset_name: str, step: str, type: str):
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for prompt in config["datasets"][dataset_name]["prompts"]:
        if prompt["type"] == type:
            template_name = prompt["template"]
            return config["prompt_templates"][template_name][step]
    raise ValueError(f"No prompt template found for dataset {dataset_name}, step {step}, type {type}")

def create_prompt_step1(dataset_name: str, sample: dict, type: str = "v1"):
    prompt_content = _load_prompt_template(dataset_name, "step1", type)
    context, _ = _parse_context_and_question(dataset_name, sample)
    
    if dataset_name == "medqa":
        # For MedQA: only context (no question in step 1)
        input_text = f"Context:\n{context}\n\n{prompt_content}"
    elif dataset_name == "uniadilr":
        # For UniADILR: only context (no hypothesis in step 1)
        # Format context sentences nicely
        if isinstance(context, dict):
            context_text = "Context:\n"
            for key, value in context.items():
                context_text += f"{key}: {value}\n"
            context_text = context_text.strip()
        else:
            context_text = f"Context:\n{str(context)}"
        
        input_text = f"{context_text}\n\n{prompt_content}"
    
    return input_text

def create_prompt_step2(dataset_name: str, sample: dict, step1_result: dict, type: str = "v1"):
    """
    Create prompt for Step 2: BN Schema refinement
    
    Args:
        dataset_name: "medqa" or "uniadilr"
        sample: Original data sample
        step1_result: Result from step1 containing the original BN Schema
        type: Template type (default "v1")
    
    Returns:
        str: Complete prompt for step2
    """
    prompt_content = _load_prompt_template(dataset_name, "step2", type)
    context, question = _parse_context_and_question(dataset_name, sample)
    answer_choices = _get_answer_choices(dataset_name, sample)
    
    # Build the step1 BN Schema text to include in the prompt
    step1_schema_text = ""
    if step1_result.get("model_answer"):
        model_answer = step1_result["model_answer"]
        nodes = model_answer.get("nodes", [])
        
        # Format nodes properly (handle both old string format and new dict format)
        formatted_nodes = []
        for i, node in enumerate(nodes):
            if isinstance(node, dict):
                # New format with type information
                if node["type"] == "binary":
                    formatted_nodes.append(f"node{i+1}: {node['name']} (binary)")
                elif node["type"] == "categorical":
                    categories_str = ", ".join(node["categories"])
                    formatted_nodes.append(f"node{i+1}: {node['name']} (categorical: {categories_str})")
            else:
                # Old format (backward compatibility) - assume binary
                formatted_nodes.append(f"node{i+1}: {node} (binary)")
        
        step1_schema_text = f"""
Original BN Schema from Step 1:
<reasoning>
{model_answer.get("reasoning", "")}
</reasoning>
<dag>
NODES: 
{chr(10).join(formatted_nodes)}
EDGES: 
{chr(10).join([f"edge{i+1}: {edge}" for i, edge in enumerate(model_answer.get("edges", []))])}
</dag>
"""
    
    # Build the complete input with clear labels
    if dataset_name == "medqa":
        # For MedQA: clearly labeled context, question, and answer choices
        input_parts = [
            f"Context:\n{context}",
            f"Question:\n{question}"
        ]
        if answer_choices:
            input_parts.append(f"Options:\n{answer_choices}")
        input_parts.extend([step1_schema_text, prompt_content])
        input_text = "\n\n".join(input_parts)
    elif dataset_name == "uniadilr":
        # For UniADILR: clearly labeled context and hypothesis
        # Format context sentences nicely
        if isinstance(context, dict):
            context_text = "Context:\n"
            for key, value in context.items():
                context_text += f"{key}: {value}\n"
            context_text = context_text.strip()
        else:
            context_text = f"Context:\n{str(context)}"
        
        input_parts = [
            context_text,
            f"Hypothesis:\n{str(question)}",
            step1_schema_text,
            prompt_content
        ]
        input_text = "\n\n".join(input_parts)
    
    return input_text

def parse_model_answer_step2(sample: Dict[str, Any], model_output: str, successful_api_call: bool, thinking: bool, step1_result: dict) -> dict:
    """
    Parse the model's response for Step 2 (BN Schema refinement with additions only)
    
    Args:
        sample: Original data sample
        model_output: Raw model response
        successful_api_call: Whether API call succeeded
        thinking: Whether model supports <think> blocks
        step1_result: Result from step1 to merge additions with
    
    Returns:
        dict: Parsed result with merged BN Schema (step1 + additions)
    """
    right_format = False
    extracted_analysis = None
    parsed_new_nodes = None
    parsed_new_edges = None
    merged_nodes = None
    merged_edges = None

    if not successful_api_call:
        return {
            "raw_data": sample, "successful_api_call": False, "right_format": False,
            "model_output": model_output, "model_answer": None,
            "correct_answer": sample.get("answer_idx"), "token_usage": None,
        }

    # 1. Clean the output if it contains <think> blocks
    if thinking:
        cleaned_text = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL).strip()
    else:
        cleaned_text = model_output.strip()

    # 2. Define regex patterns for the main blocks
    analysis_pattern = r"<analysis>(.*?)</analysis>"
    additions_pattern = r"<additions>(.*?)</additions>"

    # 3. Enforce order: Find <analysis> first, then find <additions> *after* it.
    analysis_match = re.search(analysis_pattern, cleaned_text, flags=re.DOTALL)
    
    if analysis_match:
        extracted_analysis = analysis_match.group(1).strip()
        
        # Search for the <additions> block *only in the text following the analysis block*.
        text_after_analysis = cleaned_text[analysis_match.end():]
        additions_match = re.search(additions_pattern, text_after_analysis, flags=re.DOTALL)

        if additions_match:
            additions_content = additions_match.group(1).strip()
            
            # 4. Use the strict function to parse the <additions> content
            parsed_new_nodes, parsed_new_edges = _parse_additions_content_strictly(additions_content)

    # 5. Determine if the format is correct based on the strict parsing results
    # The format is correct if both parsed lists are not None (they can be empty lists though)
    if extracted_analysis is not None and parsed_new_nodes is not None and parsed_new_edges is not None:
        right_format = True

    # 6. Merge additions with original schema from step1
    validation_passed = False
    if right_format and step1_result.get("model_answer"):
        original_nodes = step1_result["model_answer"].get("nodes", [])
        original_edges = step1_result["model_answer"].get("edges", [])
        
        # Merge nodes: original + new
        merged_nodes = original_nodes.copy()
        merged_nodes.extend(parsed_new_nodes)
        
        # Merge edges: original + new
        merged_edges = original_edges.copy()
        merged_edges.extend(parsed_new_edges)
        
        # Validation is successful if we have merged schema (even if no additions were made)
        validation_passed = True

    # 7. Structure the final model answer
    model_answer = None
    if right_format and validation_passed:
        original_nodes_count = len(step1_result["model_answer"].get("nodes", []))
        original_edges_count = len(step1_result["model_answer"].get("edges", []))
        
        # Calculate node type statistics for merged schema
        binary_nodes = [node for node in merged_nodes if node["type"] == "binary"]
        categorical_nodes = [node for node in merged_nodes if node["type"] == "categorical"]
        
        model_answer = {
            "analysis": extracted_analysis,
            "nodes": merged_nodes,  # Complete merged schema
            "edges": merged_edges,  # Complete merged schema
            "new_nodes": parsed_new_nodes,  # Only new additions
            "new_edges": parsed_new_edges,  # Only new additions
            "original_nodes_count": original_nodes_count,
            "original_edges_count": original_edges_count,
            "added_nodes_count": len(parsed_new_nodes),
            "added_edges_count": len(parsed_new_edges),
            "total_nodes_count": len(merged_nodes),
            "total_edges_count": len(merged_edges),
            "binary_nodes_count": len(binary_nodes),
            "categorical_nodes_count": len(categorical_nodes),
            "node_type_distribution": {
                "binary": len(binary_nodes),
                "categorical": len(categorical_nodes)
            }
        }
        
    correct_answer = sample.get("answer_idx")

    # 8. Return the comprehensive dictionary
    return {
        "raw_data": sample,
        "successful_api_call": successful_api_call,
        "right_format": right_format,
        "validation_passed": validation_passed,
        "model_output": model_output,
        "model_answer": model_answer,
        "correct_answer": correct_answer,
    }


def create_prompt_step3dot5(dataset_name: str, sample: dict, step3_result: dict, type: str = "v1"):
    """
    Create prompt for Step 3.5: Identify Visible Nodes
    
    This prompt asks the model to identify which nodes in the DAG have values that are
    explicitly mentioned or can be directly inferred from the question and context.
    
    Args:
        dataset_name: "medqa" or "uniadilr"
        sample: Original data sample
        step3_result: Result from step3 containing the registered DAG
        type: Template type (default "v1")
    
    Returns:
        str: Complete prompt for step 3.5
    """
    # Load the prompt template from config
    prompt_template = _load_prompt_template(dataset_name, "step3dot5", type)
    
    context, question = _parse_context_and_question(dataset_name, sample)
    answer_choices = _get_answer_choices(dataset_name, sample)
    
    # Extract DAG information from step3_result
    registered_dag = step3_result.get("registered_dag", {})
    nodes = registered_dag.get("nodes", {})
    
    # Build node information section with IDs, names, types, and possible states
    node_info_text = "DAG NODES:\n"
    for node_id, node_data in sorted(nodes.items(), key=lambda x: x[1]["index"]):
        node_name = node_data["name"]
        node_type = node_data["type"]
        
        # Get possible states/values
        if node_type == "binary":
            states = "yes, no"
        elif node_type == "categorical":
            categories = node_data.get("categories", [])
            states = ", ".join(categories) if categories else "unknown"
        else:
            states = "unknown"
        
        node_info_text += f"- {node_id}: {node_name} (type: {node_type}, possible values: {states})\n"
    
    # Build the complete prompt based on dataset type
    if dataset_name == "medqa":
        input_text = f"""QUESTION AND CONTEXT:
{context}
{question}

{answer_choices}

{node_info_text}

{prompt_template}"""
    
    elif dataset_name == "uniadilr":
        # Format context nicely for UniADILR
        if isinstance(context, dict):
            context_text = "CONTEXT:\n"
            for key, value in context.items():
                context_text += f"{key}: {value}\n"
            context_text = context_text.strip()
        else:
            context_text = f"CONTEXT:\n{str(context)}"
        
        input_text = f"""{context_text}

HYPOTHESIS:
{question}

{node_info_text}

{prompt_template}"""
    
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    
    return input_text


def parse_model_answer_step3dot5(sample: Dict[str, Any], model_output: str, successful_api_call: bool, 
                                  thinking: bool, step3_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the model's response for Step 3.5: Identify Visible Nodes
    
    Args:
        sample: Original data sample
        model_output: Raw output from the model
        successful_api_call: Whether the API call was successful
        thinking: Whether model uses <think> blocks
        step3_result: Result from step3 (needed for validation)
    
    Returns:
        dict: Parsed result with visible_nodes mapping and validation info
    """
    right_format = False
    extracted_reasoning = None
    visible_nodes = {}
    validation_issues = []
    
    if not successful_api_call:
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "model_output": model_output,
            "visible_nodes": {},
            "reasoning": None,
            "correct_answer": sample.get("answer_idx"),
            "error": "API call was not successful"
        }
    
    # Get registered DAG for validation
    registered_dag = step3_result.get("registered_dag", {})
    nodes = registered_dag.get("nodes", {})
    
    # 1. Clean the output if it contains <think> blocks
    if thinking:
        cleaned_text = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL).strip()
    else:
        cleaned_text = model_output.strip()
    
    # 2. Extract reasoning and visible_nodes sections
    reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
    visible_nodes_pattern = r"<visible_nodes>(.*?)</visible_nodes>"
    
    reasoning_match = re.search(reasoning_pattern, cleaned_text, flags=re.DOTALL)
    visible_nodes_match = re.search(visible_nodes_pattern, cleaned_text, flags=re.DOTALL)
    
    if not reasoning_match or not visible_nodes_match:
        return {
            "raw_data": sample,
            "successful_api_call": True,
            "right_format": False,
            "model_output": model_output,
            "visible_nodes": {},
            "reasoning": None,
            "correct_answer": sample.get("answer_idx"),
            "error": "Missing <reasoning> or <visible_nodes> sections in model output"
        }
    
    extracted_reasoning = reasoning_match.group(1).strip()
    visible_nodes_content = visible_nodes_match.group(1).strip()
    
    # 3. Parse visible nodes content
    # Check if it's "None" (no visible nodes)
    if re.search(r"^\s*None\s*$", visible_nodes_content, re.IGNORECASE):
        visible_nodes = {}
        right_format = True
    else:
        # Parse NODE: VALUE: pairs
        lines = visible_nodes_content.split('\n')
        current_node = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("NODE:"):
                current_node = line.split("NODE:", 1)[1].strip()
            elif line.startswith("VALUE:") and current_node:
                value = line.split("VALUE:", 1)[1].strip()
                
                # Validate that the node exists in the DAG
                if current_node not in nodes:
                    validation_issues.append(f"Node {current_node} not found in DAG")
                    current_node = None
                    continue
                
                # Validate that the value is valid for this node's type
                node_info = nodes[current_node]
                valid_values = _get_node_states_for_validation(node_info)
                
                # Case-insensitive matching
                value_lower = value.lower()
                valid_values_lower = [v.lower() for v in valid_values]
                
                if value_lower in valid_values_lower:
                    # Find the correctly-cased value
                    correct_value = valid_values[valid_values_lower.index(value_lower)]
                    visible_nodes[current_node] = correct_value
                else:
                    validation_issues.append(
                        f"Invalid value '{value}' for node {current_node} "
                        f"(type: {node_info['type']}, valid values: {valid_values})"
                    )
                
                current_node = None
        
        # If we successfully parsed at least something or had no issues, consider format correct
        if len(validation_issues) == 0:
            right_format = True
        else:
            # If there were validation issues, format is incorrect
            right_format = False
    
    return {
        "raw_data": sample,
        "successful_api_call": True,
        "right_format": right_format,
        "model_output": model_output,
        "visible_nodes": visible_nodes,
        "reasoning": extracted_reasoning,
        "validation_issues": validation_issues,
        "correct_answer": sample.get("answer_idx"),
        "error": "; ".join(validation_issues) if validation_issues else None
    }


def _get_node_states_for_validation(node_info: Dict[str, Any]) -> list:
    """
    Get the possible states for a node (for validation in step3dot5).
    
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


def create_prompt_step7(dataset_name: str, sample: dict, step6_result: dict, type: str = "v1"):
    """
    Create prompt for Step 7: Answer Extraction
    
    This prompt presents the LLM with the complete Bayesian Network analysis results
    (observed variables + MPE inferred variables) and asks it to identify what answer
    is indicated by the analysis.
    
    Args:
        dataset_name: "medqa" or "uniadilr"
        sample: Original data sample
        step6_result: Result from step6 containing MPE assignments
        type: Template type (default "v1")
    
    Returns:
        str: Complete prompt for step7
    """
    # Load the prompt template from config
    prompt_template = _load_prompt_template(dataset_name, "step7", type)
    
    context, question = _parse_context_and_question(dataset_name, sample)
    answer_choices = _get_answer_choices(dataset_name, sample)
    
    # Extract variable assignments from step6_result
    variable_assignments_text = _format_variable_assignments(step6_result)
    
    # Build the complete prompt based on dataset type
    if dataset_name == "medqa":
        input_text = f"""MEDICAL QUESTION AND CONTEXT:
{context}
{question}

ANSWER CHOICES:
{answer_choices}

BAYESIAN NETWORK ANALYSIS RESULTS:
{variable_assignments_text}

{prompt_template}"""
    
    elif dataset_name == "uniadilr":
        # Format context nicely for UniADILR
        if isinstance(context, dict):
            context_text = "CONTEXT SENTENCES:\n"
            for key, value in context.items():
                context_text += f"{key}: {value}\n"
            context_text = context_text.strip()
        else:
            context_text = f"CONTEXT:\n{str(context)}"
        
        input_text = f"""{context_text}

HYPOTHESIS:
{question}

BAYESIAN NETWORK ANALYSIS RESULTS:
{variable_assignments_text}

{prompt_template}"""
    
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    
    return input_text


def _format_variable_assignments(step6_result: Dict[str, Any]) -> str:
    """
    Format variable assignments from step6 for display in step7 prompt.
    
    Combines observed variables (from step3.5) and inferred variables (from MPE in step6).
    
    Args:
        step6_result: Result from step6 containing MPE and visible nodes
    
    Returns:
        str: Formatted text showing all variable assignments
    """
    lines = []
    
    # Get the MPE assignment (contains all variables)
    mpe_assignment = step6_result.get("mpe_result", {}).get("mpe_assignment", {})
    
    # Get the observed variables (visible nodes from step3.5)
    step3dot5_result = step6_result.get("step3dot5_result", {})
    visible_nodes = step3dot5_result.get("visible_nodes", {})
    
    # Get node information from step5 (Bayesian Network)
    step5_result = step6_result.get("step5_result", {})
    bayesian_network = step5_result.get("bayesian_network", {})
    nodes = bayesian_network.get("nodes", {})
    
    if not mpe_assignment:
        lines.append("No variable assignments available.")
        return "\n".join(lines)
    
    # Add header
    lines.append("Variable Assignments (Observed = directly from question/context, Inferred = from MPE):")
    lines.append("")
    
    # Sort nodes by their index for consistent ordering
    sorted_node_ids = sorted(nodes.keys(), key=lambda x: nodes[x].get("index", 0))
    
    for node_id in sorted_node_ids:
        if node_id in mpe_assignment:
            node_info = nodes.get(node_id, {})
            node_name = node_info.get("name", node_id)
            assigned_value = mpe_assignment[node_id]
            
            # Mark if this is an observed variable
            if node_id in visible_nodes:
                status = "[OBSERVED]"
            else:
                status = "[INFERRED]"
            
            lines.append(f"  {status} {node_name}: {assigned_value}")
    
    return "\n".join(lines)


def parse_model_answer_step7(sample: Dict[str, Any], model_output: str, 
                              successful_api_call: bool, thinking: bool,
                              dataset_name: str) -> Dict[str, Any]:
    """
    Parse the model's response for Step 7: Answer Extraction
    
    Args:
        sample: Original data sample
        model_output: Raw output from the model
        successful_api_call: Whether the API call was successful
        thinking: Whether model uses <think> blocks
        dataset_name: "medqa" or "uniadilr"
    
    Returns:
        dict: Parsed result with extracted answer and validation info
    """
    right_format = False
    extracted_answer = None
    
    if not successful_api_call:
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "model_output": model_output,
            "extracted_answer": None,
            "correct_answer": sample.get("answer_idx"),
            "error": "API call was not successful"
        }
    
    # 1. Clean the output if it contains <think> blocks
    if thinking:
        cleaned_text = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL).strip()
    else:
        cleaned_text = model_output.strip()
    
    # 2. Extract the <answer> block
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, cleaned_text, flags=re.DOTALL)
    
    if not answer_match:
        return {
            "raw_data": sample,
            "successful_api_call": True,
            "right_format": False,
            "model_output": model_output,
            "extracted_answer": None,
            "correct_answer": sample.get("answer_idx"),
            "error": "Missing <answer> block in model output"
        }
    
    answer_content = answer_match.group(1).strip()
    
    # 3. Parse based on dataset type
    if dataset_name == "medqa":
        extracted_answer, right_format, error = _parse_medqa_answer(answer_content, sample)
    elif dataset_name == "uniadilr":
        extracted_answer, right_format, error = _parse_uniadilr_answer(answer_content, sample)
    else:
        return {
            "raw_data": sample,
            "successful_api_call": True,
            "right_format": False,
            "model_output": model_output,
            "extracted_answer": None,
            "correct_answer": sample.get("answer_idx"),
            "error": f"Invalid dataset name: {dataset_name}"
        }
    
    return {
        "raw_data": sample,
        "successful_api_call": True,
        "right_format": right_format,
        "model_output": model_output,
        "extracted_answer": extracted_answer,
        "correct_answer": sample.get("answer_idx"),
        "error": error
    }


def _parse_medqa_answer(answer_content: str, sample: Dict[str, Any]) -> Tuple[Optional[Dict[str, str]], bool, Optional[str]]:
    """
    Parse MedQA answer from the <answer> block.
    
    Expected format:
    OPTION: A
    VALUE: text of option A
    
    Args:
        answer_content: Content inside <answer> tags
        sample: Original sample (for validation)
    
    Returns:
        Tuple of (extracted_answer dict, right_format bool, error message)
    """
    lines = answer_content.split('\n')
    option = None
    value = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("OPTION:"):
            option = line.split("OPTION:", 1)[1].strip().upper()
        elif line.startswith("VALUE:"):
            value = line.split("VALUE:", 1)[1].strip()
    
    # Validate
    if not option:
        return None, False, "Missing OPTION field in answer"
    
    if not value:
        return None, False, "Missing VALUE field in answer"
    
    # Check if option is a valid letter (A, B, C, or D)
    if option not in ["A", "B", "C", "D"]:
        return None, False, f"Invalid option '{option}' - must be A, B, C, or D"
    
    # Optionally validate that value matches the option in the sample
    sample_options = sample.get("options", {})
    if sample_options and option in sample_options:
        expected_value = sample_options[option]
        # We won't enforce exact match since LLM might paraphrase
        # Just check that value is not empty
        pass
    
    extracted_answer = {
        "option": option,
        "value": value
    }
    
    return extracted_answer, True, None


def _parse_uniadilr_answer(answer_content: str, sample: Dict[str, Any]) -> Tuple[Optional[Dict[str, str]], bool, Optional[str]]:
    """
    Parse UniADILR answer from the <answer> block.
    
    Expected format:
    CONCLUSION: text describing the conclusion
    
    Args:
        answer_content: Content inside <answer> tags
        sample: Original sample (for validation)
    
    Returns:
        Tuple of (extracted_answer dict, right_format bool, error message)
    """
    lines = answer_content.split('\n')
    conclusion = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("CONCLUSION:"):
            conclusion = line.split("CONCLUSION:", 1)[1].strip()
            # Continue reading subsequent lines as part of conclusion
            break
    
    # If conclusion starts on the line with CONCLUSION:, get the rest of the lines too
    if conclusion is not None:
        # Find the index of the line with CONCLUSION:
        for i, line in enumerate(lines):
            if line.strip().startswith("CONCLUSION:"):
                # Get all subsequent lines as part of conclusion
                subsequent_lines = [l.strip() for l in lines[i+1:] if l.strip()]
                if subsequent_lines:
                    conclusion = conclusion + " " + " ".join(subsequent_lines)
                break
    
    # Validate
    if not conclusion:
        return None, False, "Missing CONCLUSION field in answer"
    
    extracted_answer = {
        "conclusion": conclusion
    }
    
    return extracted_answer, True, None
