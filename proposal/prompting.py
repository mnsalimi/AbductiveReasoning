import yaml
from typing import Dict, Any, Tuple, Optional, List
import re
import os

def _parse_dag_content_strictly(dag_content: str) -> Tuple[Optional[list], Optional[list], Optional[str]]:
    """
    Parses the content of a <dag> block strictly.
    Requires NODES: header, then EDGES: header.
    Returns (parsed_nodes, parsed_edges, error_message) tuple.
    
    Now handles both binary and categorical nodes:
    - "node_name (binary)" 
    - "node_name (categorical: cat1, cat2, cat3)"
    """
    parsed_nodes = None
    parsed_edges = None

    # 1. Check for the existence of NODES: and EDGES: headers (order-agnostic)
    nodes_header_match = re.search(r"NODES:", dag_content, re.IGNORECASE)
    edges_header_match = re.search(r"EDGES:", dag_content, re.IGNORECASE)

    if not nodes_header_match:
        return None, None, "Missing 'NODES:' header in DAG"
    if not edges_header_match:
        return None, None, "Missing 'EDGES:' header in DAG"

    # 2. Build sections regardless of order
    headers = [
        {"name": "NODES", "start": nodes_header_match.start(), "end": nodes_header_match.end()},
        {"name": "EDGES", "start": edges_header_match.start(), "end": edges_header_match.end()},
    ]
    headers.sort(key=lambda h: h["start"])
    sections = {}
    for i, h in enumerate(headers):
        section_start = h["end"]
        section_end = headers[i + 1]["start"] if i + 1 < len(headers) else len(dag_content)
        sections[h["name"]] = dag_content[section_start:section_end]

    node_text = sections.get("NODES", "")
    edge_text = sections.get("EDGES", "")

    # 3. Parse nodes from the node_text section with type information.
    # Pattern matches: "node1: node_name (binary)" or "node1: node_name (categorical: cat1, cat2, cat3)"
    nodes_pattern = r"node\d+:\s*(.*)"
    node_matches = re.findall(nodes_pattern, node_text)
    
    if not node_matches:
        return None, None, "No nodes found in NODES section. Expected format: 'node1: NodeName (binary)' or 'node1: NodeName (categorical: cat1, cat2)'"
    
    parsed_nodes = []
    failed_nodes = []
    for i, node_line in enumerate(node_matches, 1):
        node_line = node_line.strip()
        # Parse the node with type information
        node_info = _parse_node_with_type(node_line)
        if node_info:
            parsed_nodes.append(node_info)
        else:
            failed_nodes.append(f"node{i}: '{node_line}'")
    
    if failed_nodes:
        return None, None, f"Failed to parse {len(failed_nodes)} node(s). Invalid format: {', '.join(failed_nodes[:3])}{'...' if len(failed_nodes) > 3 else ''}. Expected: 'NodeName (binary)' or 'NodeName (categorical: cat1, cat2, ...)'"

    if not parsed_nodes:
        return None, None, "No valid nodes parsed from NODES section"

    # 4. Parse edges from the edge_text section.
    edges_pattern = r"edge\d+:\s*node\d+\s*->\s*node\d+"
    edge_matches = re.findall(edges_pattern, edge_text)
    
    if edge_matches:
        # We need to extract just the 'nodeX -> nodeY' part.
        # A refined regex can do this in one step.
        edge_capture_pattern = r"edge\d+:\s*(node\d+\s*->\s*node\d+)"
        captured_edges = re.findall(edge_capture_pattern, edge_text)
        parsed_edges = [edge.strip() for edge in captured_edges]
    else:
        # It's okay to have no edges for some cases, but return empty list instead of None
        parsed_edges = []

    return parsed_nodes, parsed_edges, None

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
        categories = [cat.strip() for cat in categories_str.split(',') if cat.strip()]
        return {
            "name": node_name,
            "type": "categorical",
            "categories": categories
        }
    
    # If neither pattern matches, return None (parsing failed)
    return None


def _fix_invalid_nodes(nodes: List[Dict]) -> List[Dict]:
    """
    Post-process parsed nodes to fix common issues:
    - Convert categorical nodes with only 1 category to binary
    - Remove empty categories
    
    Args:
        nodes: List of node dictionaries
    
    Returns:
        List of fixed node dictionaries
    """
    fixed_nodes = []
    for node in nodes:
        if node["type"] == "categorical":
            categories = node.get("categories", [])
            # Filter out empty categories
            categories = [c for c in categories if c]
            
            if len(categories) <= 1:
                # Convert to binary (1 or fewer categories should be binary)
                print(f"  🔧 Auto-fixing: Converting '{node['name']}' from categorical to binary (had {len(categories)} category/ies)")
                fixed_nodes.append({
                    "name": node["name"],
                    "type": "binary",
                    "categories": None
                })
            else:
                # Keep as categorical with cleaned categories
                fixed_nodes.append({
                    "name": node["name"],
                    "type": "categorical",
                    "categories": categories
                })
        else:
            fixed_nodes.append(node)
    return fixed_nodes

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

    # 1. Check for the existence of NEW_NODES: and NEW_EDGES: headers (order-agnostic)
    nodes_header_match = re.search(r"NEW_NODES:", additions_content, re.IGNORECASE)
    edges_header_match = re.search(r"NEW_EDGES:", additions_content, re.IGNORECASE)

    if not nodes_header_match or not edges_header_match:
        # If either header is missing, parsing fails.
        return None, None

    # 2. Build sections regardless of order
    header_matches = sorted([
        {"name": "NEW_NODES", "start": nodes_header_match.start(), "end": nodes_header_match.end()},
        {"name": "NEW_EDGES", "start": edges_header_match.start(), "end": edges_header_match.end()},
    ], key=lambda x: x["start"])

    sections: Dict[str, str] = {}
    for i, h in enumerate(header_matches):
        section_start = h["end"]
        section_end = header_matches[i + 1]["start"] if i + 1 < len(header_matches) else len(additions_content)
        sections[h["name"]] = additions_content[section_start:section_end]

    node_text = sections.get("NEW_NODES", "")
    edge_text = sections.get("NEW_EDGES", "")

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
    error_message = None

    if not successful_api_call:
        return {
            "raw_data": sample, "successful_api_call": False, "right_format": False,
            "model_output": model_output, "model_answer": None,
            "correct_answer": sample.get("answer_idx"), "token_usage": None,
            "error": "API call failed"
        }

    # 1. Clean the output if it contains <think> blocks
    if thinking:
        cleaned_text = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL).strip()
    else:
        cleaned_text = model_output.strip()

    # 2. Define regex patterns for the main blocks
    reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
    dag_pattern = r"<dag>(.*?)</dag>"

    # 3. Order-agnostic extraction of <reasoning> and <dag>
    reasoning_match = re.search(reasoning_pattern, cleaned_text, flags=re.DOTALL)
    dag_match = re.search(dag_pattern, cleaned_text, flags=re.DOTALL)

    if not reasoning_match:
        error_message = "Missing <reasoning> tag in model output"
    if not dag_match:
        error_message = "Missing <dag> tag in model output"

    if reasoning_match:
        extracted_reasoning = reasoning_match.group(1).strip()
    if dag_match:
        dag_content = dag_match.group(1).strip()
        # 4. Use the strict function to parse the <dag> content (now order-agnostic inside)
        parsed_nodes, parsed_edges, parse_error = _parse_dag_content_strictly(dag_content)
        if parse_error:
            error_message = f"DAG parsing error: {parse_error}"
        elif parsed_nodes:
            # 4.5. Auto-fix invalid nodes (e.g., categorical with 1 category -> binary)
            parsed_nodes = _fix_invalid_nodes(parsed_nodes)

    # 5. Determine if the format is correct based on the strict parsing results
    # The format is correct only if all components were found in the correct order.
    if extracted_reasoning is not None and parsed_nodes and parsed_edges is not None:
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
        "error": error_message
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

    # 3. Order-agnostic extraction of <analysis> and <additions>
    analysis_match = re.search(analysis_pattern, cleaned_text, flags=re.DOTALL)
    additions_match = re.search(additions_pattern, cleaned_text, flags=re.DOTALL)

    if analysis_match:
        extracted_analysis = analysis_match.group(1).strip()
    if additions_match:
        additions_content = additions_match.group(1).strip()
        # 4. Use the strict function to parse the <additions> content
        parsed_new_nodes, parsed_new_edges = _parse_additions_content_strictly(additions_content)
        # 4.5. Auto-fix invalid nodes in additions
        if parsed_new_nodes:
            parsed_new_nodes = _fix_invalid_nodes(parsed_new_nodes)

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
    # Extract all node IDs for explicit listing
    available_node_ids = sorted(nodes.keys(), key=lambda x: int(x.replace('node', '')))
    
    node_info_text = f"""AVAILABLE NODE IDs: {', '.join(available_node_ids)}

⚠️ CRITICAL: Use ONLY the node IDs listed above. These are the ONLY valid node IDs in this DAG.
⚠️ Node IDs may not be sequential due to previous processing steps.
⚠️ When specifying a node, use the EXACT node ID from the list above (e.g., '{available_node_ids[0]}' not '1' or 'Node 1').

DAG NODES (detailed information):
"""
    
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


def _normalize_node_id(raw_id: str) -> str:
    """
    Normalize various node ID formats to standard 'nodeX' format.
    
    Handles:
    - "1" → "node1"
    - "Node 1" → "node1"
    - "NODE1" → "node1"
    - "node1" → "node1"
    - "node 1" → "node1"
    
    Args:
        raw_id: Raw node ID string from model output
    
    Returns:
        Normalized node ID in format "nodeX"
    """
    # Remove whitespace
    cleaned = raw_id.strip().lower()
    
    # If it's just a number, prepend "node"
    if cleaned.isdigit():
        return f"node{cleaned}"
    
    # If it starts with "node" (case-insensitive), normalize it
    if cleaned.startswith("node"):
        # Remove "node" prefix, extract number, rebuild
        number_part = cleaned.replace("node", "").strip()
        if number_part.isdigit():
            return f"node{number_part}"
    
    # If all else fails, return as-is (lowercase)
    return cleaned


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
            
            # Match NODE: with case-insensitive
            node_match = re.match(r"(?i)NODE:\s*(.+)", line)
            if node_match:
                raw_node_id = node_match.group(1).strip()
                # Normalize node ID: handle "1", "Node 1", "node1", "NODE1", etc.
                current_node = _normalize_node_id(raw_node_id)
            elif re.match(r"(?i)VALUE:", line) and current_node:
                value_match = re.match(r"(?i)VALUE:\s*(.+)", line)
                if value_match:
                    value = value_match.group(1).strip()
                    
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
    
    # Get the options node from step2.5
    step2dot5_result = step6_result.get("step2dot5_result", {})
    options_node = step2dot5_result.get("options_node")
    options_node_name = options_node.get("name") if options_node else None
    
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
            
            # Special highlight for options node
            if options_node_name and node_name == options_node_name:
                lines.append(f"  {status} 🎯 **{node_name}**: {assigned_value}  ← THIS IS THE ANSWER OPTIONS NODE")
            else:
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


def create_prompt_step2dot5(dataset_name: str, sample: dict, step2_result: dict, type: str = "v1"):
    """
    Create prompt for Step 2.5: Refine DAG to ensure answer choices/options are properly represented.
    
    Args:
        dataset_name: "medqa" or "uniadilr"
        sample: Original data sample
        step2_result: Result from step2 containing the refined BN Schema
        type: Template type (default "v1")
    
    Returns:
        str: Complete prompt for step2.5
    """
    prompt_content = _load_prompt_template(dataset_name, "step2dot5", type)
    context, question = _parse_context_and_question(dataset_name, sample)
    answer_choices = _get_answer_choices(dataset_name, sample)
    
    # Build the step2 BN Schema text to include in the prompt
    step2_schema_text = ""
    if step2_result.get("model_answer"):
        model_answer = step2_result["model_answer"]
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
        
        step2_schema_text = f"""
Current BN Schema from Step 2:
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
            input_parts.append(f"Answer Choices:\n{answer_choices}")
        input_parts.extend([step2_schema_text, prompt_content])
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
            step2_schema_text,
            prompt_content
        ]
        input_text = "\n\n".join(input_parts)
    
    return input_text


def parse_model_answer_step2dot5(sample: Dict[str, Any], model_output: str, successful_api_call: bool, 
                                  thinking: bool, step2_result: dict, dataset_name: str) -> dict:
    """
    Parse the model's response for Step 2.5 (DAG refinement for proper option representation)
    
    Args:
        sample: Original data sample
        model_output: Raw model response
        successful_api_call: Whether API call succeeded
        thinking: Whether model supports <think> blocks
        step2_result: Result from step2 to apply modifications to
        dataset_name: "medqa" or "uniadilr"
    
    Returns:
        dict: Parsed result with modified BN Schema and options_node separately identified
    """
    right_format = False
    extracted_analysis = None
    nodes_to_remove = None
    nodes_to_add = None
    options_node = None
    other_nodes_to_add = None
    edges_to_remove = None
    edges_to_add = None
    merged_nodes = None
    merged_edges = None
    error_message = None

    if not successful_api_call:
        return {
            "raw_data": sample, "successful_api_call": False, "right_format": False,
            "model_output": model_output, "model_answer": None,
            "correct_answer": sample.get("answer_idx"), "token_usage": None,
            "error": "API call failed"
        }

    # 1. Clean the output if it contains <think> blocks
    if thinking:
        cleaned_text = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL).strip()
    else:
        cleaned_text = model_output.strip()

    # 2. Define regex patterns for the main blocks
    analysis_pattern = r"<analysis>(.*?)</analysis>"
    modifications_pattern = r"<modifications>(.*?)</modifications>"

    # 3. Order-agnostic extraction of <analysis> and <modifications>
    analysis_match = re.search(analysis_pattern, cleaned_text, flags=re.DOTALL)
    modifications_match = re.search(modifications_pattern, cleaned_text, flags=re.DOTALL)

    if not analysis_match:
        error_message = "Missing <analysis> tag in model output"
    if not modifications_match:
        error_message = "Missing <modifications> tag in model output"

    if analysis_match:
        extracted_analysis = analysis_match.group(1).strip()
    if modifications_match:
        modifications_content = modifications_match.group(1).strip()
        # 4. Parse the <modifications> content (now extracting options_node separately)
        parsed_result = _parse_modifications_content_step2dot5(modifications_content, sample, dataset_name)
        if parsed_result["success"]:
            nodes_to_remove = parsed_result["nodes_to_remove"]
            options_node = parsed_result["options_node"]
            other_nodes_to_add = parsed_result["other_nodes_to_add"]
            nodes_to_add = parsed_result["nodes_to_add"]  # All nodes combined
            edges_to_remove = parsed_result["edges_to_remove"]
            edges_to_add = parsed_result["edges_to_add"]
        else:
            error_message = f"Modifications parsing error: {parsed_result.get('error', 'Unknown error')}"

    # 5. Determine if the format is correct based on the strict parsing results
    if (extracted_analysis is not None and nodes_to_remove is not None and 
        nodes_to_add is not None and edges_to_remove is not None and edges_to_add is not None):
        right_format = True

    # 6. Apply modifications to the schema from step2
    validation_passed = False
    if right_format and step2_result.get("model_answer"):
        original_nodes = step2_result["model_answer"].get("nodes", [])
        original_edges = step2_result["model_answer"].get("edges", [])
        
        # Apply modifications
        try:
            modified_result = _apply_dag_modifications(
                original_nodes, original_edges, 
                nodes_to_remove, nodes_to_add, 
                edges_to_remove, edges_to_add
            )
            
            merged_nodes = modified_result["nodes"]
            merged_edges = modified_result["edges"]
            validation_passed = True
        except Exception as e:
            validation_passed = False
            right_format = False

    # 7. Structure the final model answer
    model_answer = None
    if right_format and validation_passed:
        # Calculate node type statistics for merged schema
        binary_nodes = [node for node in merged_nodes if node["type"] == "binary"]
        categorical_nodes = [node for node in merged_nodes if node["type"] == "categorical"]
        
        model_answer = {
            "analysis": extracted_analysis,
            "nodes": merged_nodes,  # Complete modified schema
            "edges": merged_edges,  # Complete modified schema
            "options_node": options_node,  # The specific node representing answer options
            "nodes_removed": nodes_to_remove,  # Nodes that were removed
            "nodes_added": nodes_to_add,  # All nodes that were added
            "other_nodes_added": other_nodes_to_add,  # Other nodes added (not the options node)
            "edges_removed": edges_to_remove,  # Edges that were removed
            "edges_added": edges_to_add,  # Edges that were added
            "total_nodes_count": len(merged_nodes),
            "total_edges_count": len(merged_edges),
            "binary_nodes_count": len(binary_nodes),
            "categorical_nodes_count": len(categorical_nodes),
            "modifications_summary": {
                "nodes_removed_count": len(nodes_to_remove) if nodes_to_remove and nodes_to_remove != ["None"] else 0,
                "nodes_added_count": len(nodes_to_add) if nodes_to_add else 0,
                "edges_removed_count": len(edges_to_remove) if edges_to_remove and edges_to_remove != ["None"] else 0,
                "edges_added_count": len(edges_to_add) if edges_to_add else 0
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
        "options_node": options_node,  # Make it available at top level too
        "correct_answer": correct_answer,
        "error": error_message
    }


def _parse_modifications_content_step2dot5(modifications_content: str, sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """
    Parse the content of a <modifications> block strictly for step2.5.
    Extracts the options node separately from other nodes.
    
    Returns dict with:
        - success: bool
        - nodes_to_remove: list of node IDs
        - options_node: dict for the node representing answer options
        - other_nodes_to_add: list of other node dicts
        - nodes_to_add: list of all node dicts (options + others)
        - edges_to_remove: list of edge strings
        - edges_to_add: list of edge strings
        - error: error message if parsing failed
    """
    nodes_to_remove = []
    nodes_to_add = []
    options_node = None
    other_nodes_to_add = []
    edges_to_remove = []
    edges_to_add = []

    # 1) Find all section headers regardless of order and slice each section independently
    header_patterns = {
        "NODES_TO_REMOVE": r"NODES_TO_REMOVE:",
        "OPTIONS_NODE": r"OPTIONS_NODE:",
        "NODES_TO_ADD": r"NODES_TO_ADD:",
        "EDGES_TO_REMOVE": r"EDGES_TO_REMOVE:",
        "EDGES_TO_ADD": r"EDGES_TO_ADD:"
    }

    header_matches = []
    for name, pattern in header_patterns.items():
        m = re.search(pattern, modifications_content, re.IGNORECASE)
        if m:
            header_matches.append({"name": name, "start": m.start(), "end": m.end()})

    # OPTIONS_NODE is mandatory for step2.5
    if not any(h["name"] == "OPTIONS_NODE" for h in header_matches):
        return {
            "success": False,
            "error": "Missing required header: OPTIONS_NODE:"
        }

    # Sort headers by their position and build section text map
    header_matches = sorted(header_matches, key=lambda x: x["start"])
    sections: Dict[str, str] = {}
    for i, h in enumerate(header_matches):
        section_start = h["end"]
        section_end = header_matches[i + 1]["start"] if i + 1 < len(header_matches) else len(modifications_content)
        sections[h["name"]] = modifications_content[section_start:section_end]

    # Pull out each section text (missing sections become empty strings)
    nodes_remove_text = sections.get("NODES_TO_REMOVE", "")
    options_node_text = sections.get("OPTIONS_NODE", "")
    nodes_add_text = sections.get("NODES_TO_ADD", "")
    edges_remove_text = sections.get("EDGES_TO_REMOVE", "")
    edges_add_text = sections.get("EDGES_TO_ADD", "")

    # 3. Parse nodes to remove
    if not nodes_remove_text or re.search(r"^\s*None\s*$", nodes_remove_text.strip(), re.IGNORECASE):
        nodes_to_remove = ["None"]
    else:
        # Match pattern: "node{number}"
        node_remove_pattern = r"(node\d+)"
        nodes_to_remove = re.findall(node_remove_pattern, nodes_remove_text)

    # 4. Parse OPTIONS_NODE (the node that represents answer options)
    if not re.search(r"^\s*None\s*$", options_node_text.strip(), re.IGNORECASE):
        nodes_pattern = r"node\d+:\s*(.*)"
        node_matches = re.findall(nodes_pattern, options_node_text)
        if not node_matches or len(node_matches) == 0:
            return {
                "success": False,
                "error": "OPTIONS_NODE section is not 'None' but contains no valid node definition. Expected format: 'node{N}: NodeName (categorical: option1, option2, ...)'"
            }
        
        node_line = node_matches[0].strip()
        # Parse the node with type information
        options_node = _parse_node_with_type(node_line)
        if not options_node:
            return {
                "success": False,
                "error": f"Failed to parse OPTIONS_NODE. Invalid format: '{node_line}'. Expected: 'NodeName (categorical: option1, option2, ...)'"
            }
        
        # Validate that options node is categorical
        if options_node.get("type") != "categorical":
            return {
                "success": False,
                "error": f"OPTIONS_NODE must be categorical, got: {options_node.get('type')}"
            }
        
        # Validate that it has at least 2 categories
        categories = options_node.get("categories", [])
        if len(categories) < 2:
            return {
                "success": False,
                "error": f"OPTIONS_NODE must have at least 2 categories, got {len(categories)}: {categories}"
            }
        
        nodes_to_add.append(options_node)

    # 5. Parse other nodes to add (if NODES_TO_ADD section exists)
    if nodes_add_text and not re.search(r"^\s*None\s*$", nodes_add_text.strip(), re.IGNORECASE):
        nodes_pattern = r"node\d+:\s*(.*)"
        node_matches = re.findall(nodes_pattern, nodes_add_text)
        if node_matches:
            failed_additional_nodes = []
            for node_line in node_matches:
                node_line = node_line.strip()
                # Parse the node with type information
                node_info = _parse_node_with_type(node_line)
                if node_info:
                    other_nodes_to_add.append(node_info)
                    nodes_to_add.append(node_info)
                else:
                    failed_additional_nodes.append(node_line)
            
            if failed_additional_nodes:
                return {
                    "success": False,
                    "error": f"Failed to parse {len(failed_additional_nodes)} node(s) in NODES_TO_ADD section. Invalid format: {failed_additional_nodes[0]}"
                }

    # 6. Parse edges to remove
    if not edges_remove_text or re.search(r"^\s*None\s*$", edges_remove_text.strip(), re.IGNORECASE):
        edges_to_remove = ["None"]
    else:
        # Match pattern: "edge{number}: node{x} -> node{y}"
        edge_remove_pattern = r"edge\d+:\s*(node\d+\s*->\s*node\d+)"
        edges_to_remove = re.findall(edge_remove_pattern, edges_remove_text)

    # 7. Parse edges to add
    if not edges_add_text or re.search(r"^\s*None\s*$", edges_add_text.strip(), re.IGNORECASE):
        edges_to_add = []
    else:
        edge_capture_pattern = r"edge\d+:\s*(node\d+\s*->\s*node\d+)"
        edges_to_add = re.findall(edge_capture_pattern, edges_add_text)

    return {
        "success": True,
        "nodes_to_remove": nodes_to_remove,
        "options_node": options_node,
        "other_nodes_to_add": other_nodes_to_add,
        "nodes_to_add": nodes_to_add,
        "edges_to_remove": edges_to_remove,
        "edges_to_add": edges_to_add
    }


def _apply_dag_modifications(original_nodes: List, original_edges: List,
                            nodes_to_remove: List[str], nodes_to_add: List[Dict],
                            edges_to_remove: List[str], edges_to_add: List[str]) -> Dict[str, Any]:
    """
    Apply modifications to a DAG (remove and add nodes/edges).
    
    Args:
        original_nodes: Original list of nodes
        original_edges: Original list of edges
        nodes_to_remove: List of node IDs to remove (or ["None"])
        nodes_to_add: List of node dicts to add
        edges_to_remove: List of edge strings to remove (or ["None"])
        edges_to_add: List of edge strings to add
    
    Returns:
        dict: Modified DAG with "nodes" and "edges"
    """
    import re
    
    # Create working copies
    modified_nodes = original_nodes.copy()
    modified_edges = original_edges.copy()
    
    # 1. Remove nodes (if not "None")
    old_to_new_mapping = {}  # Maps old node IDs to new node IDs after removal
    indices_to_remove = []
    
    if nodes_to_remove and nodes_to_remove != ["None"]:
        # Build node ID to index mapping for the ORIGINAL nodes
        node_id_to_index = {}
        for i, node in enumerate(original_nodes):
            node_id = f"node{i+1}"
            node_id_to_index[node_id] = i
        
        # Get indices of nodes to remove
        for node_id in nodes_to_remove:
            if node_id in node_id_to_index:
                indices_to_remove.append(node_id_to_index[node_id])
        
        # Sort for consistent removal
        indices_to_remove = sorted(indices_to_remove)
        
        # Remove nodes from modified_nodes (in reverse order to maintain indices)
        for idx in sorted(indices_to_remove, reverse=True):
            modified_nodes.pop(idx)
    
    # Build mapping from old node IDs to new node IDs
    # Only create mappings for nodes that were NOT removed
    new_index = 0
    for old_index in range(len(original_nodes)):
        if old_index not in indices_to_remove:
            old_node_id = f"node{old_index + 1}"
            new_node_id = f"node{new_index + 1}"
            old_to_new_mapping[old_node_id] = new_node_id
            new_index += 1
        # Note: We don't create mappings for removed nodes - they should not appear in edges!
    
    # 2. Remove edges (if not "None")
    if edges_to_remove and edges_to_remove != ["None"]:
        # Normalize edge strings for comparison
        edges_to_remove_normalized = [edge.replace(" ", "") for edge in edges_to_remove]
        
        # Filter out edges to remove
        modified_edges = [
            edge for edge in modified_edges 
            if edge.replace(" ", "") not in edges_to_remove_normalized
        ]
    
    # 3. Renumber existing edges based on node removal
    def renumber_edge(edge_str: str, mapping: dict) -> str:
        """Renumber node references in an edge string using the provided mapping."""
        def replace_node(match):
            old_node = match.group(0)
            return mapping.get(old_node, old_node)
        
        return re.sub(r'node\d+', replace_node, edge_str)
    
    # Renumber and filter existing edges - remove any edges that reference removed nodes
    renumbered_edges = []
    for edge in modified_edges:
        # Check if edge references any removed nodes
        referenced_nodes = re.findall(r'node\d+', edge)
        if all(node in old_to_new_mapping for node in referenced_nodes):
            # All referenced nodes still exist, renumber the edge
            renumbered_edges.append(renumber_edge(edge, old_to_new_mapping))
        # else: skip this edge as it references removed nodes
    
    modified_edges = renumbered_edges
    
    # 4. Add new nodes
    num_nodes_before_adding = len(modified_nodes)
    if nodes_to_add:
        modified_nodes.extend(nodes_to_add)
    
    # 5. Add new edges with proper node ID remapping
    if edges_to_add:
        # Build mapping for newly added nodes
        # New nodes should be numbered starting from num_nodes_before_adding + 1
        new_node_mapping = {}
        for i, new_node in enumerate(nodes_to_add):
            # The LLM may have generated edges with node IDs beyond the current count
            # We need to map these to the actual new node IDs
            old_new_node_id = f"node{len(original_nodes) + i + 1}"
            actual_new_node_id = f"node{num_nodes_before_adding + i + 1}"
            new_node_mapping[old_new_node_id] = actual_new_node_id
        
        # Combine the mappings
        combined_mapping = {**old_to_new_mapping, **new_node_mapping}
        
        # Renumber and filter the edges to add - skip edges that reference removed nodes
        renumbered_edges_to_add = []
        skipped_edges = []
        for edge in edges_to_add:
            referenced_nodes = re.findall(r'node\d+', edge)
            if all(node in combined_mapping for node in referenced_nodes):
                renumbered_edges_to_add.append(renumber_edge(edge, combined_mapping))
            else:
                skipped_edges.append(edge)
        
        # Print warning if edges were skipped
        if skipped_edges:
            print(f"  ⚠️  Warning: {len(skipped_edges)} edge(s) skipped (referenced removed nodes)")
        
        modified_edges.extend(renumbered_edges_to_add)
    
    return {
        "nodes": modified_nodes,
        "edges": modified_edges
    }