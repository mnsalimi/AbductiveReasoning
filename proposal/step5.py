import time
import statistics
import traceback
from typing import Dict, Any, List, Tuple, Optional, Set
import itertools

def _product(iterable):
    """Calculate the product of all elements in an iterable."""
    result = 1
    for x in iterable:
        result *= x
    return result

def step5(sample: Dict[str, Any], idx: int, step4_result: Dict[str, Any], sleep_time: float = 0.0) -> Dict[str, Any]:
    """
    Step 5: Bayesian Network Construction - Construct a complete Bayesian Network from CPTs.
    
    This step takes the CPTs from Step 4 and constructs a complete, operational Bayesian Network
    with proper data structures that can be used for probabilistic inference and reasoning.
    
    Args:
        sample: Original data sample
        idx: Sample index
        step4_result: Result dictionary from step4 containing CPTs for all nodes
        sleep_time: Delay between operations (for consistency with other steps)
    
    Returns:
        dict: Result dictionary with constructed Bayesian Network
    """
    time.sleep(sleep_time)
    
    # Check if step4 was successful and has valid CPTs
    if not step4_result.get("successful_api_call") or not step4_result.get("right_format"):
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "bayesian_network": None,
            "network_metadata": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": "Step 4 failed or had invalid format - cannot proceed with Step 5",
            "step1_result": step4_result.get("step1_result"),
            "step2_result": step4_result.get("step2_result"),
            "step3_result": step4_result.get("step3_result"),
            "step4_result": step4_result
        }
    
    cpts = step4_result.get("cpts")
    if not cpts:
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "bayesian_network": None,
            "network_metadata": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": "No CPTs found in step4 result",
            "step1_result": step4_result.get("step1_result"),
            "step2_result": step4_result.get("step2_result"),
            "step3_result": step4_result.get("step3_result"),
            "step4_result": step4_result
        }
    
    try:
        # Construct the Bayesian Network
        bayesian_network = _construct_bayesian_network(cpts, step4_result)
        
        # Validate the constructed network
        validation_result = _validate_bayesian_network(bayesian_network)
        
        if not validation_result["is_valid"]:
            raise ValueError(f"Bayesian Network validation failed: {validation_result['errors']}")
        
        # Create metadata about the network
        network_metadata = _create_network_metadata(bayesian_network, step4_result)
        
        return {
            "raw_data": sample,
            "successful_api_call": True,
            "right_format": True,
            "bayesian_network": bayesian_network,
            "network_metadata": network_metadata,
            "validation_result": validation_result,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": None,
            "step1_result": step4_result.get("step1_result"),
            "step2_result": step4_result.get("step2_result"),
            "step3_result": step4_result.get("step3_result"),
            "step4_result": step4_result
        }
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR IN STEP 5 - FULL TRACEBACK:")
        print("="*80)
        traceback.print_exc()
        print("="*80 + "\n")
        
        return {
            "raw_data": sample,
            "successful_api_call": True,  # Previous steps succeeded
            "right_format": False,  # But network construction failed
            "bayesian_network": None,
            "network_metadata": None,
            "validation_result": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": f"Bayesian Network construction failed for sample {idx}: {e}",
            "step1_result": step4_result.get("step1_result"),
            "step2_result": step4_result.get("step2_result"),
            "step3_result": step4_result.get("step3_result"),
            "step4_result": step4_result
        }


def _construct_bayesian_network(cpts: Dict[str, Any], step4_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct a complete Bayesian Network from CPTs.
    
    Args:
        cpts: Dictionary of CPTs for all nodes from step4
        step4_result: Complete step4 result for additional context
    
    Returns:
        dict: Complete Bayesian Network structure
    """
    # Get the original DAG structure from step3
    registered_dag = step4_result["step3_result"]["registered_dag"]
    
    # Create node registry with enhanced information
    nodes = {}
    for node_id, cpt in cpts.items():
        try:
            node_data = registered_dag["nodes"][node_id]
            
            # Debug: Check CPT states
            if not isinstance(cpt["states"], list):
                raise ValueError(f"CPT states is not a list! Type: {type(cpt['states'])}, Value: {cpt['states']}")
            
            # Create comprehensive node structure
            nodes[node_id] = {
                "id": node_id,
                "name": cpt["node_name"],
                "type": cpt["node_type"],
                "states": cpt["states"],
                "state_count": len(cpt["states"]),
                "index": node_data["index"],
                "parents": _extract_parent_info(cpt, registered_dag),
                "children": _extract_children_info(node_id, registered_dag),
                "cpt": cpt["numerical_cpt"],
                "cpt_shape": _calculate_cpt_shape(cpt),
                "is_root": len(registered_dag["adjacency_list"][node_id]["parents"]) == 0,
                "is_leaf": len(registered_dag["adjacency_list"][node_id]["children"]) == 0
            }
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR constructing node {node_id} ({cpt.get('node_name', 'unknown')})")
            print(f"Node type: {cpt.get('node_type', 'unknown')}")
            print(f"CPT states: {cpt.get('states', 'N/A')} (type: {type(cpt.get('states', 'N/A'))})")
            print(f"DAG node data: {node_data}")
            print(f"Original error: {e}")
            print(f"{'='*80}\n")
            raise ValueError(f"Failed to construct node {node_id} ({cpt.get('node_name', 'unknown')}). Error: {e}")
    
    # Create edge information with probability implications
    edges = _construct_network_edges(registered_dag, nodes)
    
    # Build adjacency structures for efficient inference
    adjacency_matrix = _build_enhanced_adjacency_matrix(registered_dag, nodes)
    
    # Create inference-ready data structures
    inference_structures = _prepare_inference_structures(nodes, edges)
    
    # Create joint probability computation helpers
    joint_prob_helpers = _create_joint_probability_helpers(nodes, edges)
    
    return {
        "nodes": nodes,
        "edges": edges,
        "adjacency_matrix": adjacency_matrix,
        "adjacency_list": registered_dag["adjacency_list"],
        "inference_structures": inference_structures,
        "joint_probability_helpers": joint_prob_helpers,
        "network_properties": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "root_nodes": [nid for nid, node in nodes.items() if node["is_root"]],
            "leaf_nodes": [nid for nid, node in nodes.items() if node["is_leaf"]],
            "max_parents": max(len(node["parents"]) for node in nodes.values()),
            "total_parameters": sum(_product(node["cpt_shape"]) for node in nodes.values())
        }
    }


def _extract_parent_info(cpt: Dict[str, Any], registered_dag: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and organize parent information for a node."""
    node_id = cpt["node_id"]
    
    # Handle the parent info structure from Step 4
    cpt_parent_info = cpt.get("parents", {})
    
    # Step 4 uses structure: {"has_parents": bool, "parents": {...}}
    if isinstance(cpt_parent_info, dict) and "has_parents" in cpt_parent_info:
        has_parents = cpt_parent_info["has_parents"]
        cpt_parents_dict = cpt_parent_info.get("parents", {})
    else:
        # Fallback for direct dict format
        has_parents = bool(cpt_parent_info)
        cpt_parents_dict = cpt_parent_info
    
    if not has_parents or not cpt_parents_dict:
        return {}
    
    # Use the parent information from the CPT which already has states
    parents = {}
    for parent_id, parent_data in cpt_parents_dict.items():
        parents[parent_id] = {
            "id": parent_id,
            "name": parent_data["name"],
            "type": parent_data["type"],
            "states": parent_data["states"],
            "index": registered_dag["nodes"][parent_id]["index"]
        }
    
    return parents


def _extract_children_info(node_id: str, registered_dag: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and organize children information for a node."""
    child_ids = registered_dag["adjacency_list"][node_id]["children"]
    
    if not child_ids:
        return {}
    
    children = {}
    for child_id in child_ids:
        child_node = registered_dag["nodes"][child_id]
        try:
            states = _get_node_states_from_dag(child_node)
        except Exception as e:
            raise ValueError(f"Failed to get states for child node {child_id} ({child_node.get('name', 'unknown')}): {e}. Node info: {child_node}")
        
        children[child_id] = {
            "id": child_id,
            "name": child_node["name"],
            "type": child_node["type"],
            "states": states,
            "index": child_node["index"]
        }
    
    return children


def _get_node_states_from_dag(node_info: Dict[str, Any]) -> List[str]:
    """Get the possible states for a node from DAG information."""
    if node_info["type"] == "binary":
        return ["yes", "no"]
    elif node_info["type"] == "categorical":
        categories = node_info.get("categories", [])
        # Ensure we always return a list, even if categories is False or None
        return categories if isinstance(categories, list) and categories else []
    else:
        return ["unknown"]


def _calculate_cpt_shape(cpt: Dict[str, Any]) -> Tuple[int, ...]:
    """Calculate the shape of the CPT tensor."""
    try:
        # Handle the parent info structure from Step 4
        parent_info = cpt.get("parents", {})
        
        # Step 4 uses structure: {"has_parents": bool, "parents": {...}}
        # We need to extract the actual parents dict
        if isinstance(parent_info, dict) and "has_parents" in parent_info:
            has_parents = parent_info["has_parents"]
            parents_dict = parent_info.get("parents", {})
        else:
            # Fallback for direct dict format
            has_parents = bool(parent_info)
            parents_dict = parent_info
        
        if not has_parents or not parents_dict:
            # Root node - just the number of states
            states = cpt["states"]
            if not isinstance(states, list):
                raise TypeError(f"CPT states is not a list! Type: {type(states)}, Value: {states}")
            return (len(states),)
        
        # Get parent state counts
        parent_shapes = []
        for parent_id, parent_data in parents_dict.items():
            parent_states = parent_data["states"]
            if not isinstance(parent_states, list):
                raise TypeError(f"Parent {parent_id} states is not a list! Type: {type(parent_states)}, Value: {parent_states}")
            parent_shapes.append(len(parent_states))
        
        # Add target node states
        states = cpt["states"]
        if not isinstance(states, list):
            raise TypeError(f"CPT states is not a list! Type: {type(states)}, Value: {states}")
        parent_shapes.append(len(states))
        
        return tuple(parent_shapes)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR in _calculate_cpt_shape for node {cpt.get('node_id', 'unknown')} ({cpt.get('node_name', 'unknown')})")
        print(f"CPT structure:")
        print(f"  - node_id: {cpt.get('node_id', 'N/A')}")
        print(f"  - node_name: {cpt.get('node_name', 'N/A')}")
        print(f"  - node_type: {cpt.get('node_type', 'N/A')}")
        print(f"  - states: {cpt.get('states', 'N/A')} (type: {type(cpt.get('states', 'N/A'))})")
        print(f"  - parents structure: {cpt.get('parents', 'N/A')}")
        print(f"Original error: {e}")
        print(f"{'='*80}\n")
        raise


def _construct_network_edges(registered_dag: Dict[str, Any], nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Construct enhanced edge information for the Bayesian Network."""
    edges = []
    
    for edge_info in registered_dag["edges"]:
        source_id = edge_info["source"]
        target_id = edge_info["target"]
        
        edge = {
            "id": edge_info["id"],
            "source_id": source_id,
            "target_id": target_id,
            "source_name": nodes[source_id]["name"],
            "target_name": nodes[target_id]["name"],
            "source_type": nodes[source_id]["type"],
            "target_type": nodes[target_id]["type"],
            "source_states": nodes[source_id]["states"],
            "target_states": nodes[target_id]["states"],
            "influence_strength": _calculate_influence_strength(source_id, target_id, nodes),
            "raw": edge_info["raw"]
        }
        
        edges.append(edge)
    
    return edges


def _calculate_influence_strength(source_id: str, target_id: str, nodes: Dict[str, Any]) -> float:
    """
    Calculate the influence strength between two nodes based on CPT analysis.
    This is a simplified metric for network analysis.
    """
    target_node = nodes[target_id]
    target_cpt = target_node["cpt"]
    
    if not target_cpt:
        return 0.0
    
    # For simplicity, calculate variance in probabilities as influence measure
    all_probs = []
    for condition_probs in target_cpt.values():
        all_probs.extend(condition_probs.values())
    
    if len(all_probs) < 2:
        return 0.0
    
    # Higher variance indicates stronger influence
    variance = statistics.variance(all_probs)
    return float(variance)


def _build_enhanced_adjacency_matrix(registered_dag: Dict[str, Any], nodes: Dict[str, Any]) -> List[List[float]]:
    """Build an enhanced adjacency matrix with influence weights."""
    num_nodes = len(nodes)
    adjacency_matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    
    for edge_info in registered_dag["edges"]:
        source_idx = nodes[edge_info["source"]]["index"]
        target_idx = nodes[edge_info["target"]]["index"]
        
        # Use influence strength as edge weight
        influence = _calculate_influence_strength(edge_info["source"], edge_info["target"], nodes)
        adjacency_matrix[source_idx][target_idx] = max(influence, 0.1)  # Minimum weight for existing edges
    
    return adjacency_matrix


def _prepare_inference_structures(nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare data structures optimized for probabilistic inference."""
    
    # Create topological ordering for efficient inference
    topological_order = _compute_topological_order(nodes, edges)
    
    # Create parent-child lookup tables
    parent_lookup = {}
    child_lookup = {}
    
    for node_id, node in nodes.items():
        parent_lookup[node_id] = list(node["parents"].keys())
        child_lookup[node_id] = list(node["children"].keys())
    
    # Create state space information
    state_spaces = {}
    for node_id, node in nodes.items():
        state_spaces[node_id] = {
            "states": node["states"],
            "state_to_index": {state: idx for idx, state in enumerate(node["states"])},
            "index_to_state": {idx: state for idx, state in enumerate(node["states"])}
        }
    
    # Create CPT access patterns for efficient lookup
    cpt_patterns = {}
    for node_id, node in nodes.items():
        if node["parents"]:
            # Multi-dimensional CPT
            parent_ids = list(node["parents"].keys())
            parent_states = [node["parents"][pid]["states"] for pid in parent_ids]
            
            # Create all parent combinations
            combinations = list(itertools.product(*parent_states))
            pattern = {}
            
            for combo in combinations:
                key = tuple(combo) if len(combo) > 1 else combo[0] if combo else "NO_PARENTS"
                if key in node["cpt"]:
                    pattern[combo] = node["cpt"][key]
                
            cpt_patterns[node_id] = {
                "type": "conditional",
                "parent_ids": parent_ids,
                "pattern": pattern,
                "combinations": combinations
            }
        else:
            # Root node - prior probability
            cpt_patterns[node_id] = {
                "type": "prior",
                "parent_ids": [],
                "pattern": node["cpt"].get("NO_PARENTS", {}),
                "combinations": []
            }
    
    return {
        "topological_order": topological_order,
        "parent_lookup": parent_lookup,
        "child_lookup": child_lookup,
        "state_spaces": state_spaces,
        "cpt_patterns": cpt_patterns
    }


def _compute_topological_order(nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> List[str]:
    """Compute topological ordering of nodes for efficient inference."""
    # Kahn's algorithm for topological sorting
    in_degree = {node_id: len(node["parents"]) for node_id, node in nodes.items()}
    queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
    result = []
    
    while queue:
        node_id = queue.pop(0)
        result.append(node_id)
        
        # Reduce in-degree of children
        for child_id in nodes[node_id]["children"]:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)
    
    return result


def _create_joint_probability_helpers(nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create helper structures for joint probability computation."""
    
    # Create variable ordering for joint probability calculations
    variable_order = list(nodes.keys())
    
    # Create state combination generators
    all_states = [nodes[node_id]["states"] for node_id in variable_order]
    total_combinations = 1
    for states in all_states:
        total_combinations *= len(states)
    
    # Create probability computation templates
    computation_templates = {}
    for node_id, node in nodes.items():
        if node["is_root"]:
            # Root nodes use prior probabilities
            computation_templates[node_id] = {
                "type": "prior",
                "formula": f"P({node['name']})"
            }
        else:
            # Child nodes use conditional probabilities
            parent_names = [node["parents"][pid]["name"] for pid in node["parents"]]
            parent_str = ", ".join(parent_names)
            computation_templates[node_id] = {
                "type": "conditional", 
                "formula": f"P({node['name']} | {parent_str})"
            }
    
    return {
        "variable_order": variable_order,
        "total_combinations": total_combinations,
        "computation_templates": computation_templates,
        "network_factorization": _create_network_factorization(nodes)
    }


def _create_network_factorization(nodes: Dict[str, Any]) -> str:
    """Create the mathematical factorization of the joint probability distribution."""
    factors = []
    
    for node_id, node in nodes.items():
        if node["is_root"]:
            factors.append(f"P({node['name']})")
        else:
            parent_names = [node["parents"][pid]["name"] for pid in node["parents"]]
            parent_str = ", ".join(parent_names)
            factors.append(f"P({node['name']} | {parent_str})")
    
    return "P(Network) = " + " × ".join(factors)


def _validate_bayesian_network(bayesian_network: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the constructed Bayesian Network."""
    errors = []
    warnings = []
    
    nodes = bayesian_network["nodes"]
    edges = bayesian_network["edges"]
    
    # Check basic structure
    if not nodes:
        errors.append("No nodes found in Bayesian Network")
    
    if not edges and len(nodes) > 1:
        warnings.append("No edges found - network consists of independent nodes")
    
    # Validate each node
    for node_id, node in nodes.items():
        # Check CPT completeness
        if not node.get("cpt"):
            errors.append(f"Node {node_id} ({node['name']}) has no CPT")
            continue
        
        # Validate CPT probabilities
        cpt_validation = _validate_node_cpt(node)
        if cpt_validation["errors"]:
            errors.extend([f"Node {node_id}: {err}" for err in cpt_validation["errors"]])
        if cpt_validation["warnings"]:
            warnings.extend([f"Node {node_id}: {warn}" for warn in cpt_validation["warnings"]])
    
    # Check network connectivity
    connectivity_check = _check_network_connectivity(bayesian_network)
    if connectivity_check["warnings"]:
        warnings.extend(connectivity_check["warnings"])
    
    # Validate inference structures
    inference_validation = _validate_inference_structures(bayesian_network)
    if inference_validation["errors"]:
        errors.extend(inference_validation["errors"])
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "node_count": len(nodes),
        "edge_count": len(edges)
    }


def _validate_node_cpt(node: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the CPT of a single node."""
    errors = []
    warnings = []
    
    cpt = node["cpt"]
    
    # Check probability distributions
    for condition, probs in cpt.items():
        if not isinstance(probs, dict):
            errors.append(f"Invalid probability format for condition {condition}")
            continue
        
        # Check if probabilities sum to approximately 1.0
        prob_sum = sum(probs.values())
        if abs(prob_sum - 1.0) > 0.01:  # Allow small numerical errors
            warnings.append(f"Probabilities for condition {condition} sum to {prob_sum:.4f}, not 1.0")
        
        # Check for negative probabilities
        for state, prob in probs.items():
            if prob < 0:
                errors.append(f"Negative probability {prob} for state {state} in condition {condition}")
            if prob > 1:
                errors.append(f"Probability {prob} > 1.0 for state {state} in condition {condition}")
    
    # Check completeness - all expected combinations should be present
    if node["parents"]:
        parent_states = [parent["states"] for parent in node["parents"].values()]
        expected_combinations = list(itertools.product(*parent_states))
        
        for combo in expected_combinations:
            key = tuple(combo) if len(combo) > 1 else combo[0] if combo else "NO_PARENTS"
            if key not in cpt:
                errors.append(f"Missing CPT entry for parent combination {combo}")
    
    return {"errors": errors, "warnings": warnings}


def _check_network_connectivity(bayesian_network: Dict[str, Any]) -> Dict[str, Any]:
    """Check network connectivity properties."""
    warnings = []
    
    nodes = bayesian_network["nodes"]
    root_nodes = bayesian_network["network_properties"]["root_nodes"]
    leaf_nodes = bayesian_network["network_properties"]["leaf_nodes"]
    
    # Check for isolated components
    if len(root_nodes) > 3:
        warnings.append(f"Many root nodes ({len(root_nodes)}) - network may be fragmented")
    
    if len(leaf_nodes) > 3:
        warnings.append(f"Many leaf nodes ({len(leaf_nodes)}) - network may be too branched")
    
    # Check for very deep or wide structures
    max_parents = bayesian_network["network_properties"]["max_parents"]
    if max_parents > 5:
        warnings.append(f"Node with {max_parents} parents - may cause inference complexity")
    
    return {"warnings": warnings}


def _validate_inference_structures(bayesian_network: Dict[str, Any]) -> Dict[str, Any]:
    """Validate inference-ready structures."""
    errors = []
    
    inference_structures = bayesian_network.get("inference_structures", {})
    
    required_structures = ["topological_order", "parent_lookup", "child_lookup", "state_spaces", "cpt_patterns"]
    for structure in required_structures:
        if structure not in inference_structures:
            errors.append(f"Missing inference structure: {structure}")
    
    # Validate topological order
    if "topological_order" in inference_structures:
        topo_order = inference_structures["topological_order"]
        if len(topo_order) != len(bayesian_network["nodes"]):
            errors.append("Topological order incomplete - not all nodes included")
    
    return {"errors": errors}


def _create_network_metadata(bayesian_network: Dict[str, Any], step4_result: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive metadata about the constructed Bayesian Network."""
    
    nodes = bayesian_network["nodes"]
    edges = bayesian_network["edges"]
    network_props = bayesian_network["network_properties"]
    
    # Analyze network structure
    node_analysis = {
        "total_nodes": len(nodes),
        "root_nodes": len(network_props["root_nodes"]),
        "leaf_nodes": len(network_props["leaf_nodes"]),
        "intermediate_nodes": len(nodes) - len(network_props["root_nodes"]) - len(network_props["leaf_nodes"]),
        "binary_nodes": sum(1 for node in nodes.values() if node["type"] == "binary"),
        "categorical_nodes": sum(1 for node in nodes.values() if node["type"] == "categorical")
    }
    
    # Analyze complexity
    complexity_analysis = {
        "total_parameters": network_props["total_parameters"],
        "max_parents": network_props["max_parents"],
        "avg_parents": sum(len(node["parents"]) for node in nodes.values()) / len(nodes),
        "total_edges": len(edges),
        "network_density": len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0
    }
    
    # CPT analysis
    cpt_analysis = {
        "total_cpts": len([n for n in nodes.values() if n["cpt"]]),
        "avg_cpt_size": sum(len(n["cpt"]) for n in nodes.values()) / len(nodes),
        "largest_cpt": max(len(n["cpt"]) for n in nodes.values()) if nodes else 0,
        "prior_distributions": len(network_props["root_nodes"])
    }
    
    return {
        "construction_timestamp": time.time(),
        "source_steps": {
            "step1_successful": step4_result.get("step1_result", {}).get("successful_api_call", False),
            "step2_successful": step4_result.get("step2_result", {}).get("successful_api_call", False),
            "step3_successful": step4_result.get("step3_result", {}).get("successful_api_call", False),
            "step4_successful": step4_result.get("successful_api_call", False)
        },
        "network_structure": node_analysis,
        "complexity_metrics": complexity_analysis,
        "cpt_analysis": cpt_analysis,
        "inference_readiness": {
            "has_topological_order": "topological_order" in bayesian_network.get("inference_structures", {}),
            "has_state_spaces": "state_spaces" in bayesian_network.get("inference_structures", {}),
            "has_cpt_patterns": "cpt_patterns" in bayesian_network.get("inference_structures", {}),
            "factorization": bayesian_network.get("joint_probability_helpers", {}).get("network_factorization", "")
        }
    }
