import time
import os
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import networkx as nx

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
    
    # Check if previous step was successful and has a valid format
    if not step2_result.get("successful_api_call") or not step2_result.get("right_format"):
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "registered_dag": None,
            "dag_metadata": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": "Previous step (Step 2 or 2.5) failed or had invalid format - cannot proceed with Step 3"
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
            # Format a detailed error message
            error_msg = _format_validation_error(validation_result, registered_dag, step2_result)
            raise ValueError(error_msg)
        
        return {
            "raw_data": sample,
            "successful_api_call": True,
            "right_format": True,
            "registered_dag": registered_dag,
            "dag_metadata": dag_metadata,
            "validation_result": validation_result,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": None
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
            "error": f"DAG registration failed for sample {idx}: {e}"
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
    
    print(f"\n  📋 Registering DAG structure:")
    print(f"    Total nodes to register: {len(nodes)}")
    print(f"    Total edges to register: {len(edges)}")
    
    # Create node registry with proper indexing
    node_registry = {}
    indexed_nodes = {}
    
    binary_count = 0
    categorical_count = 0
    
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
            
            if node_info["type"] == "binary":
                binary_count += 1
            elif node_info["type"] == "categorical":
                categorical_count += 1
                categories = node_info.get("categories")
                num_cats = len(categories) if isinstance(categories, list) else 0
                print(f"    {node_id}: '{node_info['name']}' (categorical, {num_cats} categories: {categories})")
        else:
            # Old format (backward compatibility) - assume binary
            node_info = {
                "id": node_id,
                "name": str(node),
                "type": "binary",
                "categories": None,
                "index": i
            }
            binary_count += 1
        
        node_registry[node_id] = node_info
        indexed_nodes[i] = node_info
    
    print(f"    Node types: {binary_count} binary, {categorical_count} categorical")
    
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
    detailed_errors = []  # Store detailed error information
    
    # Check basic structure
    required_keys = ["nodes", "edges", "adjacency_list", "adjacency_matrix", "num_nodes", "num_edges"]
    for key in required_keys:
        if key not in registered_dag:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        return {"is_valid": False, "errors": errors, "warnings": warnings, "detailed_errors": detailed_errors}
    
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
    
    # Validate categorical node categories (with enhanced error messages)
    for node_info in registered_dag["nodes"].values():
        if node_info["type"] == "categorical":
            categories = node_info.get("categories")
            num_categories = len(categories) if isinstance(categories, list) else 0
            
            if not categories or num_categories < 2:
                # Create detailed error message
                error_detail = {
                    "error_type": "insufficient_categories",
                    "node_id": node_info['id'],
                    "node_name": node_info['name'],
                    "node_type": node_info['type'],
                    "current_categories": categories if categories else [],
                    "num_categories": num_categories,
                    "required_minimum": 2,
                    "node_index": node_info['index']
                }
                detailed_errors.append(error_detail)
                
                # Create human-readable error message
                if not categories:
                    error_msg = (f"Categorical node {node_info['id']} ('{node_info['name']}', index={node_info['index']}) "
                                f"has NO categories defined. Categorical nodes require at least 2 categories. "
                                f"Current value: {categories}")
                elif num_categories == 0:
                    error_msg = (f"Categorical node {node_info['id']} ('{node_info['name']}', index={node_info['index']}) "
                                f"has EMPTY categories list. Categorical nodes require at least 2 categories. "
                                f"Current value: {categories}")
                elif num_categories == 1:
                    error_msg = (f"Categorical node {node_info['id']} ('{node_info['name']}', index={node_info['index']}) "
                                f"has only 1 category: {categories}. Categorical nodes require at least 2 categories.")
                else:
                    error_msg = (f"Categorical node {node_info['id']} ('{node_info['name']}', index={node_info['index']}) "
                                f"has {num_categories} categories: {categories}. This should not happen in this branch.")
                
                errors.append(error_msg)
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "detailed_errors": detailed_errors
    }


def _format_validation_error(validation_result: Dict[str, Any], registered_dag: Dict[str, Any], 
                            step2_result: Dict[str, Any]) -> str:
    """
    Format a detailed, human-readable validation error message.
    
    Args:
        validation_result: Result from _validate_registered_dag
        registered_dag: The DAG that failed validation
        step2_result: Result from step2 for context
        
    Returns:
        str: Formatted error message with full context
    """
    lines = ["\n" + "="*80]
    lines.append("DAG VALIDATION FAILED")
    lines.append("="*80)
    
    # Show the errors
    errors = validation_result.get("errors", [])
    lines.append(f"\nFound {len(errors)} error(s):\n")
    
    for i, error in enumerate(errors, 1):
        lines.append(f"{i}. {error}")
    
    # Show detailed information for each problematic node
    detailed_errors = validation_result.get("detailed_errors", [])
    if detailed_errors:
        lines.append("\n" + "-"*80)
        lines.append("DETAILED NODE INFORMATION:")
        lines.append("-"*80)
        
        for detail in detailed_errors:
            lines.append(f"\n🔴 Node: {detail['node_id']} ('{detail['node_name']}')")
            lines.append(f"   Type: {detail['node_type']}")
            lines.append(f"   Index: {detail['node_index']}")
            lines.append(f"   Current categories: {detail['current_categories']}")
            lines.append(f"   Number of categories: {detail['num_categories']}")
            lines.append(f"   Required minimum: {detail['required_minimum']}")
            
            # Try to trace where this node came from
            step2_nodes = step2_result.get("model_answer", {}).get("nodes", [])
            if detail['node_index'] < len(step2_nodes):
                original_node = step2_nodes[detail['node_index']]
                lines.append(f"   Original node from Step 2: {original_node}")
            
            # Check if it was also in step1
            step1_result = step2_result.get("step1_result", {})
            step1_nodes = step1_result.get("model_answer", {}).get("nodes", [])
            if detail['node_index'] < len(step1_nodes):
                step1_node = step1_nodes[detail['node_index']]
                lines.append(f"   Original node from Step 1: {step1_node}")
    
    # Show all nodes for context
    lines.append("\n" + "-"*80)
    lines.append("ALL NODES IN DAG:")
    lines.append("-"*80)
    
    for node_id, node_info in registered_dag.get("nodes", {}).items():
        status = "✅" if node_info["type"] == "binary" else "🔶"
        if node_info["type"] == "categorical":
            cats = node_info.get("categories", [])
            num_cats = len(cats) if isinstance(cats, list) else 0
            if num_cats < 2:
                status = "❌"
        
        lines.append(f"{status} {node_id}: '{node_info['name']}' ({node_info['type']})")
        if node_info["type"] == "categorical":
            lines.append(f"     Categories: {node_info.get('categories', 'None')}")
    
    lines.append("="*80 + "\n")
    
    return "\n".join(lines)


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


def plot_dag(registered_dag: Dict[str, Any], output_path: str, sample_idx: int = 0) -> None:
    """
    Plot the registered DAG and save it to a file.
    
    Args:
        registered_dag: The registered DAG structure from step3
        output_path: Path where to save the plot (without extension)
        sample_idx: Index of the sample being processed
    """
    from matplotlib.patches import FancyArrowPatch, Patch
    import numpy as np
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with their names
    node_labels = {}
    node_colors = []
    
    for node_id, node_info in registered_dag["nodes"].items():
        G.add_node(node_id)
        node_labels[node_id] = f"{node_info['name']}\n({node_info['type']})"
        
        # Color nodes based on type
        if node_info['type'] == 'binary':
            node_colors.append('#87CEEB')  # Sky blue for binary
        else:
            node_colors.append('#FFB347')  # Orange for categorical
    
    # Add edges
    for edge_info in registered_dag["edges"]:
        G.add_edge(edge_info["source"], edge_info["target"])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Use hierarchical layout for DAG visualization
    try:
        # Try to use a hierarchical layout if possible
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # Draw nodes
    node_size = 4000
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, 
                          alpha=0.9, edgecolors='black', linewidths=2.5, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, node_labels, font_size=9, font_weight='bold', ax=ax)
    
    # Draw edges manually with proper arrow positioning
    # Calculate node radius in data coordinates
    node_radius = 0.08  # Approximate radius based on node_size
    
    for edge in G.edges():
        source, target = edge
        x1, y1 = pos[source]
        x2, y2 = pos[target]
        
        # Calculate direction vector
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            # Normalize direction
            dx_norm = dx / dist
            dy_norm = dy / dist
            
            # Adjust start and end points to be outside the nodes
            x1_adj = x1 + dx_norm * node_radius
            y1_adj = y1 + dy_norm * node_radius
            x2_adj = x2 - dx_norm * node_radius
            y2_adj = y2 - dy_norm * node_radius
            
            # Draw arrow
            arrow = FancyArrowPatch(
                (x1_adj, y1_adj), (x2_adj, y2_adj),
                arrowstyle='-|>',
                mutation_scale=30,
                linewidth=2.5,
                color='#333333',
                alpha=0.7,
                connectionstyle='arc3,rad=0.15',
                zorder=1
            )
            ax.add_patch(arrow)
    
    # Add title and legend
    plt.title(f"Bayesian Network DAG - Sample {sample_idx}\n"
             f"Nodes: {registered_dag['num_nodes']}, Edges: {registered_dag['num_edges']}", 
             fontsize=16, fontweight='bold', pad=20)
    
    # Create legend
    legend_elements = [
        Patch(facecolor='#87CEEB', edgecolor='black', label='Binary Node'),
        Patch(facecolor='#FFB347', edgecolor='black', label='Categorical Node')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
    
    ax.axis('off')
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the figure (PNG only)
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 DAG visualization saved to: {output_path}.png")
