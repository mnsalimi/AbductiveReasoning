import time
from typing import Dict, Any, Optional, List, Tuple

def step3(sample: Dict[str, Any], idx: int, step2_result: Dict[str, Any], sleep_time: float = 0.0) -> Dict[str, Any]:
    """
    Step 3: Register DAG from Step 2 result.
    
    This step takes the refined BN Schema from Step 2 and converts it into a structured
    DAG representation that can be used by subsequent pipeline steps (CPT Creator,
    Bayesian Network Constructor, etc.).
    
    Args:
        sample: Original data sample
        idx: Sample index
        step2_result: Result dictionary from step2 containing the refined BN Schema
        sleep_time: Delay between operations (for consistency with other steps)
    
    Returns:
        dict: Result dictionary with registered DAG structure
    """
    time.sleep(sleep_time)
    
    # Check if step2 was successful and has a valid format
    if not step2_result.get("successful_api_call") or not step2_result.get("right_format"):
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "registered_dag": None,
            "dag_metadata": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": "Step 2 failed or had invalid format - cannot proceed with Step 3",
            "step1_result": step2_result.get("step1_result"),
            "step2_result": step2_result
        }
    
    try:
        # Extract the refined schema from step2 result
        model_answer = step2_result.get("model_answer")
        if not model_answer:
            raise ValueError("No model answer found in step2 result")
        
        # Register the DAG by creating a structured representation
        registered_dag = _register_dag_structure(model_answer)
        
        # Create metadata about the DAG
        dag_metadata = _create_dag_metadata(registered_dag, step2_result)
        
        # Validate the registered DAG
        validation_result = _validate_registered_dag(registered_dag)
        
        if not validation_result["is_valid"]:
            raise ValueError(f"DAG validation failed: {validation_result['errors']}")
        
        return {
            "raw_data": sample,
            "successful_api_call": True,
            "right_format": True,
            "registered_dag": registered_dag,
            "dag_metadata": dag_metadata,
            "validation_result": validation_result,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": None,
            "step1_result": step2_result.get("step1_result"),
            "step2_result": step2_result
        }
        
    except Exception as e:
        return {
            "raw_data": sample,
            "successful_api_call": True,  # API calls succeeded in previous steps
            "right_format": False,  # But registration failed
            "registered_dag": None,
            "dag_metadata": None,
            "validation_result": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": f"DAG registration failed for sample {idx}: {e}",
            "step1_result": step2_result.get("step1_result"),
            "step2_result": step2_result
        }


def _register_dag_structure(model_answer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the model answer into a structured DAG representation.
    
    Args:
        model_answer: Parsed model answer from step2 containing nodes and edges
    
    Returns:
        dict: Structured DAG representation
    """
    nodes = model_answer.get("nodes", [])
    edges = model_answer.get("edges", [])
    
    # Create node registry with proper indexing
    node_registry = {}
    indexed_nodes = {}
    
    for i, node in enumerate(nodes):
        node_id = f"node{i+1}"
        
        if isinstance(node, dict):
            # New format with type information
            node_info = {
                "id": node_id,
                "name": node["name"],
                "type": node["type"],
                "categories": node.get("categories"),
                "index": i
            }
        else:
            # Old format (backward compatibility) - assume binary
            node_info = {
                "id": node_id,
                "name": str(node),
                "type": "binary",
                "categories": None,
                "index": i
            }
        
        node_registry[node_id] = node_info
        indexed_nodes[i] = node_info
    
    # Process edges and create adjacency structures
    edge_list = []
    adjacency_list = {node_id: {"parents": [], "children": []} for node_id in node_registry.keys()}
    
    for i, edge in enumerate(edges):
        edge_info = _parse_edge(edge, i)
        if edge_info:
            edge_list.append(edge_info)
            
            # Update adjacency lists
            source_id = edge_info["source"]
            target_id = edge_info["target"]
            
            if source_id in adjacency_list and target_id in adjacency_list:
                adjacency_list[source_id]["children"].append(target_id)
                adjacency_list[target_id]["parents"].append(source_id)
    
    # Create adjacency matrix
    num_nodes = len(nodes)
    adjacency_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    
    for edge_info in edge_list:
        source_idx = node_registry[edge_info["source"]]["index"]
        target_idx = node_registry[edge_info["target"]]["index"]
        adjacency_matrix[source_idx][target_idx] = 1
    
    return {
        "nodes": node_registry,
        "indexed_nodes": indexed_nodes,
        "edges": edge_list,
        "adjacency_list": adjacency_list,
        "adjacency_matrix": adjacency_matrix,
        "num_nodes": num_nodes,
        "num_edges": len(edge_list)
    }


def _parse_edge(edge_str: str, edge_index: int) -> Optional[Dict[str, Any]]:
    """
    Parse an edge string into structured format.
    
    Args:
        edge_str: Edge string like "node1 -> node2"
        edge_index: Index of the edge
    
    Returns:
        dict: Structured edge information or None if parsing fails
    """
    import re
    
    # Match pattern like "node1 -> node2"
    edge_pattern = r"(node\d+)\s*->\s*(node\d+)"
    match = re.match(edge_pattern, edge_str.strip())
    
    if match:
        return {
            "id": f"edge{edge_index + 1}",
            "source": match.group(1),
            "target": match.group(2),
            "index": edge_index,
            "raw": edge_str.strip()
        }
    
    return None


def _create_dag_metadata(registered_dag: Dict[str, Any], step2_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create metadata about the registered DAG.
    
    Args:
        registered_dag: The registered DAG structure
        step2_result: Result from step2
    
    Returns:
        dict: DAG metadata
    """
    # Analyze node types
    node_types = {"binary": 0, "categorical": 0}
    categorical_node_details = []
    
    for node_info in registered_dag["nodes"].values():
        node_types[node_info["type"]] += 1
        if node_info["type"] == "categorical":
            categorical_node_details.append({
                "id": node_info["id"],
                "name": node_info["name"],
                "categories": node_info["categories"],
                "num_categories": len(node_info["categories"]) if node_info["categories"] else 0
            })
    
    # Analyze graph structure
    node_degrees = {}
    for node_id, adj_info in registered_dag["adjacency_list"].items():
        node_degrees[node_id] = {
            "in_degree": len(adj_info["parents"]),
            "out_degree": len(adj_info["children"]),
            "total_degree": len(adj_info["parents"]) + len(adj_info["children"])
        }
    
    # Find root and leaf nodes
    root_nodes = [node_id for node_id, deg in node_degrees.items() if deg["in_degree"] == 0]
    leaf_nodes = [node_id for node_id, deg in node_degrees.items() if deg["out_degree"] == 0]
    
    return {
        "registration_timestamp": time.time(),
        "source_steps": {
            "step1_successful": step2_result.get("step1_result", {}).get("successful_api_call", False),
            "step2_successful": step2_result.get("successful_api_call", False)
        },
        "node_statistics": {
            "total_nodes": registered_dag["num_nodes"],
            "binary_nodes": node_types["binary"],
            "categorical_nodes": node_types["categorical"],
            "categorical_details": categorical_node_details
        },
        "edge_statistics": {
            "total_edges": registered_dag["num_edges"],
            "avg_node_degree": sum(deg["total_degree"] for deg in node_degrees.values()) / len(node_degrees) if node_degrees else 0
        },
        "graph_structure": {
            "root_nodes": root_nodes,
            "leaf_nodes": leaf_nodes,
            "node_degrees": node_degrees
        }
    }


def _validate_registered_dag(registered_dag: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the registered DAG structure.
    
    Args:
        registered_dag: The registered DAG structure
    
    Returns:
        dict: Validation result with is_valid flag and any errors
    """
    errors = []
    warnings = []
    
    # Check basic structure
    required_keys = ["nodes", "edges", "adjacency_list", "adjacency_matrix", "num_nodes", "num_edges"]
    for key in required_keys:
        if key not in registered_dag:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        return {"is_valid": False, "errors": errors, "warnings": warnings}
    
    # Validate node consistency
    if len(registered_dag["nodes"]) != registered_dag["num_nodes"]:
        errors.append(f"Node count mismatch: {len(registered_dag['nodes'])} != {registered_dag['num_nodes']}")
    
    # Validate edge consistency
    if len(registered_dag["edges"]) != registered_dag["num_edges"]:
        errors.append(f"Edge count mismatch: {len(registered_dag['edges'])} != {registered_dag['num_edges']}")
    
    # Validate edge references
    node_ids = set(registered_dag["nodes"].keys())
    for edge in registered_dag["edges"]:
        if edge["source"] not in node_ids:
            errors.append(f"Edge references non-existent source node: {edge['source']}")
        if edge["target"] not in node_ids:
            errors.append(f"Edge references non-existent target node: {edge['target']}")
    
    # Validate adjacency matrix dimensions
    expected_size = registered_dag["num_nodes"]
    adj_matrix = registered_dag["adjacency_matrix"]
    if len(adj_matrix) != expected_size:
        errors.append(f"Adjacency matrix row count mismatch: {len(adj_matrix)} != {expected_size}")
    else:
        for i, row in enumerate(adj_matrix):
            if len(row) != expected_size:
                errors.append(f"Adjacency matrix column count mismatch in row {i}: {len(row)} != {expected_size}")
    
    # Check for cycles (warning, not error, as cycles might be valid in some contexts)
    if _has_cycles(registered_dag):
        warnings.append("DAG contains cycles - this may be intentional for some Bayesian networks")
    
    # Validate categorical node categories
    for node_info in registered_dag["nodes"].values():
        if node_info["type"] == "categorical":
            if not node_info["categories"] or len(node_info["categories"]) < 2:
                errors.append(f"Categorical node {node_info['id']} must have at least 2 categories")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def _has_cycles(registered_dag: Dict[str, Any]) -> bool:
    """
    Check if the DAG has cycles using DFS.
    
    Args:
        registered_dag: The registered DAG structure
    
    Returns:
        bool: True if cycles are detected
    """
    # Simple cycle detection using DFS
    visited = set()
    rec_stack = set()
    
    def dfs(node_id):
        visited.add(node_id)
        rec_stack.add(node_id)
        
        for child in registered_dag["adjacency_list"][node_id]["children"]:
            if child not in visited:
                if dfs(child):
                    return True
            elif child in rec_stack:
                return True
        
        rec_stack.remove(node_id)
        return False
    
    for node_id in registered_dag["nodes"].keys():
        if node_id not in visited:
            if dfs(node_id):
                return True
    
    return False
