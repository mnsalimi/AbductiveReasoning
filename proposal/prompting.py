import yaml
from typing import Dict, Any, Tuple
import re

import re
from typing import Dict, Any, Optional, Tuple

def _parse_dag_content_strictly(dag_content: str) -> Tuple[Optional[list], Optional[list]]:
    """
    Parses the content of a <dag> block strictly.
    Requires NODES: header, then EDGES: header.
    Returns (parsed_nodes, parsed_edges) tuple.
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

    # 3. Parse nodes from the node_text section.
    nodes_pattern = r"node\d+:\s*(.*)"
    node_matches = re.findall(nodes_pattern, node_text)
    if node_matches:
        # Store node names, cleaning up each one.
        parsed_nodes = [name.strip() for name in node_matches]

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


def parse_model_answer_step1(sample: Dict[str, Any], model_output: str, successful_api_call: bool, thinking: bool) -> str:
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

def _load_prompt_template(dataset_name: str, type: str):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    for prompt in config["datasets"][dataset_name]["prompts"]:
        if prompt["type"] == type:
            return prompt["template"]
    raise ValueError(f"No prompt template found for dataset {dataset_name}")

def create_prompt_step1(dataset_name: str, sample: dict, type: str):
    prompt_content = _load_prompt_template(dataset_name, type)
    context, _ = _parse_context_and_question(dataset_name, sample)
    if dataset_name == "medqa":
        input_text = context + "\n" + prompt_content
    elif dataset_name == "uniadilr":
        input_text = str(context) + "\n" + prompt_content
    