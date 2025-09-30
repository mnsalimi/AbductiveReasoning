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

def _parse_refined_schema_content_strictly(schema_content: str) -> Tuple[Optional[list], Optional[list]]:
    """
    Parses the content of a <refined_schema> block strictly.
    Requires NODES: header, then EDGES: header.
    Returns (parsed_nodes, parsed_edges) tuple.
    
    Now handles both binary and categorical nodes like _parse_dag_content_strictly.
    """
    parsed_nodes = None
    parsed_edges = None

    # 1. Check for the existence and order of NODES: and EDGES: headers.
    # We use re.IGNORECASE to be slightly tolerant of "nodes:" vs "NODES:".
    nodes_header_match = re.search(r"NODES:", schema_content, re.IGNORECASE)
    edges_header_match = re.search(r"EDGES:", schema_content, re.IGNORECASE)

    if not nodes_header_match or not edges_header_match:
        # If either header is missing, parsing fails.
        return None, None

    if nodes_header_match.start() >= edges_header_match.start():
        # If NODES: does not appear before EDGES:, parsing fails.
        return None, None
        
    # 2. Split the schema content into two parts based on the headers.
    # The node_text is the content between "NODES:" and "EDGES:".
    node_text = schema_content[nodes_header_match.end():edges_header_match.start()]
    # The edge_text is the content after "EDGES:".
    edge_text = schema_content[edges_header_match.end():]

    # 3. Parse nodes from the node_text section with type information.
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
    Parse the model's response for Step 2 (BN Schema refinement)
    
    Args:
        sample: Original data sample
        model_output: Raw model response
        successful_api_call: Whether API call succeeded
        thinking: Whether model supports <think> blocks
        step1_result: Result from step1 to validate refinement
    
    Returns:
        dict: Parsed result with refined BN Schema
    """
    right_format = False
    extracted_analysis = None
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
    analysis_pattern = r"<analysis>(.*?)</analysis>"
    refined_schema_pattern = r"<refined_schema>(.*?)</refined_schema>"

    # 3. Enforce order: Find <analysis> first, then find <refined_schema> *after* it.
    analysis_match = re.search(analysis_pattern, cleaned_text, flags=re.DOTALL)
    
    if analysis_match:
        extracted_analysis = analysis_match.group(1).strip()
        
        # Search for the <refined_schema> block *only in the text following the analysis block*.
        text_after_analysis = cleaned_text[analysis_match.end():]
        schema_match = re.search(refined_schema_pattern, text_after_analysis, flags=re.DOTALL)

        if schema_match:
            schema_content = schema_match.group(1).strip()
            
            # 4. Use the strict function to parse the <refined_schema> content
            parsed_nodes, parsed_edges = _parse_refined_schema_content_strictly(schema_content)

    # 5. Determine if the format is correct based on the strict parsing results
    # The format is correct only if all components were found in the correct order.
    if extracted_analysis is not None and parsed_nodes and parsed_edges:
        right_format = True

    # 6. Validate that the refined schema includes all original nodes/edges from step1
    validation_passed = False
    if right_format and step1_result.get("model_answer"):
        original_nodes = step1_result["model_answer"].get("nodes", [])
        original_edges = step1_result["model_answer"].get("edges", [])
        
        # For nodes, we need to compare node names since they now have type information
        if parsed_nodes and original_nodes:
            # Extract names from parsed_nodes (which are now dicts)
            parsed_node_names = [node["name"] for node in parsed_nodes]
            # original_nodes might be strings (from old format) or dicts (from new format)
            if original_nodes and isinstance(original_nodes[0], dict):
                original_node_names = [node["name"] for node in original_nodes]
            else:
                original_node_names = original_nodes  # backward compatibility
            
            nodes_preserved = all(node_name in parsed_node_names for node_name in original_node_names)
        else:
            nodes_preserved = False
            
        # Check if all original edges are present in refined schema
        edges_preserved = all(edge in parsed_edges for edge in original_edges) if parsed_edges and original_edges else False
        
        validation_passed = nodes_preserved and edges_preserved

    # 7. Structure the final model answer
    model_answer = None
    if right_format and validation_passed:
        original_nodes_count = len(step1_result["model_answer"].get("nodes", []))
        original_edges_count = len(step1_result["model_answer"].get("edges", []))
        
        # Calculate node type statistics
        binary_nodes = [node for node in parsed_nodes if node["type"] == "binary"]
        categorical_nodes = [node for node in parsed_nodes if node["type"] == "categorical"]
        
        model_answer = {
            "analysis": extracted_analysis,
            "nodes": parsed_nodes,
            "edges": parsed_edges,
            "original_nodes_count": original_nodes_count,
            "original_edges_count": original_edges_count,
            "added_nodes_count": len(parsed_nodes) - original_nodes_count,
            "added_edges_count": len(parsed_edges) - original_edges_count,
            "total_nodes_count": len(parsed_nodes),
            "total_edges_count": len(parsed_edges),
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
    