import time
import math
from typing import Dict, Any, List, Tuple, Optional, Set
import itertools

def step6(sample: Dict[str, Any], idx: int, step5_result: Dict[str, Any], sleep_time: float = 0.0, step3dot5_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Step 6: MPE Algorithm - Apply Most Probable Explanation algorithm on the Bayesian Network.
    
    This step takes the constructed Bayesian Network from Step 5 and applies MPE (Most Probable
    Explanation) inference to find the most likely explanation/assignment of all variables
    given the available evidence.
    
    Args:
        sample: Original data sample (contains context and question for evidence extraction)
        idx: Sample index
        step5_result: Result dictionary from step5 containing the constructed Bayesian Network
        sleep_time: Delay between operations (for consistency with other steps)
    
    Returns:
        dict: Result dictionary with MPE results and explanations
    """
    time.sleep(sleep_time)
    
    # Check if step5 was successful and has a valid Bayesian Network
    if not step5_result.get("successful_api_call") or not step5_result.get("right_format"):
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "mpe_result": None,
            "mpe_metadata": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": "Step 5 failed or had invalid format - cannot proceed with Step 6",
            "step3dot5_result": step3dot5_result,
        }
    
    bayesian_network = step5_result.get("bayesian_network")
    if not bayesian_network:
        return {
            "raw_data": sample,
            "successful_api_call": False,
            "right_format": False,
            "mpe_result": None,
            "mpe_metadata": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": "No Bayesian Network found in step5 result",
            "step3dot5_result": step3dot5_result,
        }
    
    try:
        # Extract evidence from the sample (context and question) and step3dot5 visible nodes
        evidence = _extract_evidence_from_sample(sample, bayesian_network, step3dot5_result)
        
        # Apply MPE algorithm
        mpe_result = _apply_mpe_algorithm(bayesian_network, evidence)
        
        # Validate MPE result
        validation_result = _validate_mpe_result(mpe_result, bayesian_network)
        
        if not validation_result["is_valid"]:
            raise ValueError(f"MPE result validation failed: {validation_result['errors']}")
        
        # Create metadata about the MPE computation
        mpe_metadata = _create_mpe_metadata(mpe_result, bayesian_network, evidence)
        
        return {
            "raw_data": sample,
            "successful_api_call": True,
            "right_format": True,
            "mpe_result": mpe_result,
            "mpe_metadata": mpe_metadata,
            "validation_result": validation_result,
            "evidence": evidence,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": None,
            "step3dot5_result": step3dot5_result,
        }
        
    except Exception as e:
        return {
            "raw_data": sample,
            "successful_api_call": True,  # Previous steps succeeded
            "right_format": False,  # But MPE failed
            "mpe_result": None,
            "mpe_metadata": None,
            "validation_result": None,
            "evidence": None,
            "correct_answer": sample.get("answer_idx"),
            "idx": idx,
            "error": f"MPE algorithm failed for sample {idx}: {e}",
            "step3dot5_result": step3dot5_result,
        }


def _extract_evidence_from_sample(sample: Dict[str, Any], bayesian_network: Dict[str, Any], 
                                 step3dot5_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Extract evidence from the sample context and question.
    
    This function analyzes the sample to identify what evidence is available
    and maps it to the Bayesian Network nodes. If step3dot5_result is provided,
    it uses the visible nodes identified by the AI model as evidence.
    
    Args:
        sample: Original data sample
        bayesian_network: Constructed Bayesian Network from step5
        step3dot5_result: Optional result from step3dot5 containing visible nodes
    
    Returns:
        dict: Evidence mapping and metadata
    """
    context = sample.get("context", "")
    question = sample.get("question", "") or sample.get("hypothesis", "")
    
    # Initialize evidence structure
    evidence = {
        "observed_variables": {},  # node_id -> observed_state
        "query_variables": [],     # Variables we want to find MPE for
        "evidence_source": "unknown",
        "raw_context": context,
        "raw_question": question
    }
    
    # Get network nodes for analysis
    nodes = bayesian_network["nodes"]
    
    # Check if step3dot5 provided visible nodes
    if (step3dot5_result and 
        step3dot5_result.get("successful_api_call") and 
        step3dot5_result.get("right_format") and 
        step3dot5_result.get("visible_nodes")):
        
        # Use visible nodes from step3dot5 as evidence
        visible_nodes = step3dot5_result.get("visible_nodes", {})
        
        print(f"\n  📊 Using {len(visible_nodes)} visible nodes from Step 3.5 as evidence")
        
        for node_id, value in visible_nodes.items():
            if node_id in nodes:
                evidence["observed_variables"][node_id] = value
                print(f"      • {node_id}: {value}")
            else:
                print(f"      ⚠️  Node {node_id} from visible nodes not found in Bayesian Network")
        
        evidence["evidence_source"] = "step3dot5_visible_nodes"
        
        # If no visible nodes were found, fall back to heuristic extraction
        if not evidence["observed_variables"]:
            print(f"      ⚠️  No valid visible nodes found, falling back to heuristic extraction")
            evidence_extraction = _heuristic_evidence_extraction(context, question, nodes)
            evidence.update(evidence_extraction)
            evidence["evidence_source"] = "heuristic_fallback"
    else:
        # Fall back to heuristic evidence extraction
        if step3dot5_result:
            print(f"\n  ⚠️  Step 3.5 did not complete successfully, using heuristic evidence extraction")
        else:
            print(f"\n  ℹ️  No Step 3.5 result available, using heuristic evidence extraction")
        
        evidence_extraction = _heuristic_evidence_extraction(context, question, nodes)
        evidence.update(evidence_extraction)
        evidence["evidence_source"] = "heuristic"
    
    # If no specific evidence found, treat all non-query variables as hidden
    if not evidence["observed_variables"] and not evidence["query_variables"]:
        # Default strategy: find the most likely complete assignment
        evidence["query_variables"] = list(nodes.keys())
        evidence["inference_type"] = "complete_mpe"
    else:
        evidence["inference_type"] = "conditional_mpe"
    
    return evidence


def _heuristic_evidence_extraction(context: str, question: str, nodes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use heuristics to extract evidence from text context.
    
    This is a simplified evidence extraction that looks for keyword matches
    with node names and states.
    """
    observed_variables = {}
    query_variables = []
    
    # Convert to lowercase for matching
    context_lower = context.lower()
    question_lower = question.lower()
    combined_text = f"{context_lower} {question_lower}"
    
    # Look for evidence in context (what's observed)
    for node_id, node in nodes.items():
        node_name_lower = node["name"].lower()
        
        # Check if node is mentioned in context (might be observed)
        if any(keyword in combined_text for keyword in [node_name_lower, node_name_lower.replace(" ", "")]):
            
            # Try to find which state is observed
            observed_state = None
            for state in node["states"]:
                state_lower = state.lower()
                # Look for exact state mentions or related keywords
                if state_lower in combined_text:
                    observed_state = state
                    break
                
                # Check for binary states with common synonyms
                if node["type"] == "binary":
                    if state_lower == "yes" and any(word in combined_text for word in ["present", "positive", "exists", "has", "shows"]):
                        observed_state = state
                        break
                    elif state_lower == "no" and any(word in combined_text for word in ["absent", "negative", "lacks", "without", "none"]):
                        observed_state = state
                        break
            
            if observed_state:
                observed_variables[node_id] = observed_state
            else:
                # If node is mentioned but state unclear, add to query variables
                if node_id not in query_variables:
                    query_variables.append(node_id)
    
    # Look for what we're trying to diagnose/find (query variables)
    diagnosis_keywords = ["diagnosis", "diagnose", "condition", "disease", "likely", "probable", "most", "what is"]
    if any(keyword in question_lower for keyword in diagnosis_keywords):
        # Find diagnosis-related nodes
        for node_id, node in nodes.items():
            node_name_lower = node["name"].lower()
            if any(keyword in node_name_lower for keyword in ["diagnosis", "condition", "disease", "outcome"]):
                if node_id not in query_variables and node_id not in observed_variables:
                    query_variables.append(node_id)
    
    # If no specific query variables found, assume we want MPE for all unobserved variables
    if not query_variables:
        query_variables = [node_id for node_id in nodes.keys() if node_id not in observed_variables]
    
    return {
        "observed_variables": observed_variables,
        "query_variables": query_variables
    }


def _apply_mpe_algorithm(bayesian_network: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply the MPE (Most Probable Explanation) algorithm to the Bayesian Network.
    
    This implements a variable elimination-based MPE algorithm that finds the
    most likely assignment to all query variables given the evidence.
    """
    nodes = bayesian_network["nodes"]
    inference_structures = bayesian_network["inference_structures"]
    
    observed_vars = evidence["observed_variables"]
    query_vars = evidence["query_variables"]
    
    # Get topological order for efficient computation
    topo_order = inference_structures["topological_order"]
    
    # Initialize MPE computation
    mpe_computation = _initialize_mpe_computation(nodes, observed_vars, query_vars, topo_order)
    
    # Apply MPE algorithm using dynamic programming approach
    if evidence["inference_type"] == "complete_mpe":
        # Find most likely complete assignment
        mpe_assignment = _compute_complete_mpe(nodes, mpe_computation)
    else:
        # Find most likely assignment given evidence
        mpe_assignment = _compute_conditional_mpe(nodes, mpe_computation, observed_vars)
    
    # Calculate the probability of the MPE assignment
    mpe_probability = _calculate_mpe_probability(mpe_assignment, nodes)
    
    # Create explanation of the MPE result
    explanation = _create_mpe_explanation(mpe_assignment, mpe_probability, nodes, evidence)
    
    return {
        "mpe_assignment": mpe_assignment,
        "mpe_probability": mpe_probability,
        "log_probability": math.log(mpe_probability) if mpe_probability > 0 else float('-inf'),
        "explanation": explanation,
        "computation_details": mpe_computation,
        "algorithm_type": "variable_elimination_mpe"
    }


def _initialize_mpe_computation(nodes: Dict[str, Any], observed_vars: Dict[str, str], 
                               query_vars: List[str], topo_order: List[str]) -> Dict[str, Any]:
    """Initialize data structures for MPE computation."""
    
    # Create variable domains (possible values for each variable)
    variable_domains = {}
    for node_id, node in nodes.items():
        if node_id in observed_vars:
            # Observed variables have fixed values
            variable_domains[node_id] = [observed_vars[node_id]]
        else:
            # Unobserved variables can take any of their states
            variable_domains[node_id] = node["states"]
    
    # Create factor list from CPTs
    factors = []
    for node_id in topo_order:
        node = nodes[node_id]
        factor = _create_factor_from_cpt(node_id, node, variable_domains)
        factors.append(factor)
    
    return {
        "variable_domains": variable_domains,
        "factors": factors,
        "elimination_order": _compute_elimination_order(query_vars, topo_order),
        "observed_vars": observed_vars,
        "query_vars": query_vars
    }


def _create_factor_from_cpt(node_id: str, node: Dict[str, Any], variable_domains: Dict[str, List[str]]) -> Dict[str, Any]:
    """Create a factor from a node's CPT for MPE computation."""
    
    cpt = node["cpt"]
    
    if node["is_root"]:
        # Root node - prior probability
        factor_vars = [node_id]
        factor_values = {}
        
        prior_probs = cpt.get("NO_PARENTS", {})
        for state in variable_domains[node_id]:
            factor_values[(state,)] = prior_probs.get(state, 0.0)
    
    else:
        # Child node - conditional probability
        parent_ids = list(node["parents"].keys())
        factor_vars = parent_ids + [node_id]
        factor_values = {}
        
        # Generate all combinations of parent and child states
        parent_domains = [variable_domains[pid] for pid in parent_ids]
        child_domain = variable_domains[node_id]
        
        for parent_combo in itertools.product(*parent_domains):
            # Create key for CPT lookup
            if len(parent_combo) == 1:
                cpt_key = parent_combo[0]
            else:
                cpt_key = parent_combo
            
            conditional_probs = cpt.get(cpt_key, {})
            
            for child_state in child_domain:
                full_assignment = parent_combo + (child_state,)
                factor_values[full_assignment] = conditional_probs.get(child_state, 0.0)
    
    return {
        "variables": factor_vars,
        "values": factor_values,
        "source_node": node_id
    }


def _compute_elimination_order(query_vars: List[str], topo_order: List[str]) -> List[str]:
    """Compute an efficient elimination order for MPE."""
    # For MPE, we eliminate variables in reverse topological order
    # but keep query variables for last
    
    elimination_order = []
    
    # First eliminate non-query variables in reverse topological order
    for node_id in reversed(topo_order):
        if node_id not in query_vars:
            elimination_order.append(node_id)
    
    # Then add query variables (these won't actually be eliminated, but tracked for MPE)
    elimination_order.extend(query_vars)
    
    return elimination_order


def _compute_complete_mpe(nodes: Dict[str, Any], mpe_computation: Dict[str, Any]) -> Dict[str, str]:
    """Compute MPE for complete assignment (no evidence)."""
    
    factors = mpe_computation["factors"]
    variable_domains = mpe_computation["variable_domains"]
    
    # Find the assignment that maximizes the joint probability
    best_assignment = {}
    best_probability = float('-inf')
    
    # Generate all possible complete assignments
    all_vars = list(nodes.keys())
    all_domains = [variable_domains[var_id] for var_id in all_vars]
    
    for assignment_tuple in itertools.product(*all_domains):
        assignment = dict(zip(all_vars, assignment_tuple))
        
        # Calculate probability of this assignment
        prob = _calculate_assignment_probability(assignment, factors)
        
        if prob > best_probability:
            best_probability = prob
            best_assignment = assignment
    
    return best_assignment


def _compute_conditional_mpe(nodes: Dict[str, Any], mpe_computation: Dict[str, Any], 
                           observed_vars: Dict[str, str]) -> Dict[str, str]:
    """Compute MPE given evidence (conditional MPE)."""
    
    factors = mpe_computation["factors"]
    variable_domains = mpe_computation["variable_domains"]
    query_vars = mpe_computation["query_vars"]
    
    # Find the assignment to query variables that maximizes P(query_vars | evidence)
    best_assignment = {}
    best_probability = float('-inf')
    
    # Get all variables (both observed and unobserved)
    all_vars = list(nodes.keys())
    unobserved_vars = [var_id for var_id in all_vars if var_id not in observed_vars]
    
    # Generate all possible assignments to unobserved variables
    unobserved_domains = [variable_domains[var_id] for var_id in unobserved_vars]
    
    for unobserved_assignment_tuple in itertools.product(*unobserved_domains):
        # Create complete assignment
        assignment = observed_vars.copy()
        for i, var_id in enumerate(unobserved_vars):
            assignment[var_id] = unobserved_assignment_tuple[i]
        
        # Calculate probability of this assignment
        prob = _calculate_assignment_probability(assignment, factors)
        
        if prob > best_probability:
            best_probability = prob
            best_assignment = assignment
    
    return best_assignment


def _calculate_assignment_probability(assignment: Dict[str, str], factors: List[Dict[str, Any]]) -> float:
    """Calculate the probability of a complete variable assignment."""
    
    total_prob = 1.0
    
    for factor in factors:
        # Extract the relevant part of assignment for this factor
        factor_assignment = []
        for var_id in factor["variables"]:
            factor_assignment.append(assignment[var_id])
        
        factor_key = tuple(factor_assignment)
        factor_prob = factor["values"].get(factor_key, 0.0)
        
        total_prob *= factor_prob
        
        # Early termination if probability becomes 0
        if total_prob == 0.0:
            break
    
    return total_prob


def _calculate_mpe_probability(mpe_assignment: Dict[str, str], nodes: Dict[str, Any]) -> float:
    """Calculate the probability of the MPE assignment using the original CPTs."""
    
    total_prob = 1.0
    
    for node_id, node in nodes.items():
        if node["is_root"]:
            # Prior probability
            state = mpe_assignment[node_id]
            prior_probs = node["cpt"].get("NO_PARENTS", {})
            node_prob = prior_probs.get(state, 0.0)
        
        else:
            # Conditional probability
            parent_ids = list(node["parents"].keys())
            parent_states = [mpe_assignment[pid] for pid in parent_ids]
            
            # Create key for CPT lookup
            if len(parent_states) == 1:
                cpt_key = parent_states[0]
            else:
                cpt_key = tuple(parent_states)
            
            conditional_probs = node["cpt"].get(cpt_key, {})
            child_state = mpe_assignment[node_id]
            node_prob = conditional_probs.get(child_state, 0.0)
        
        total_prob *= node_prob
        
        if total_prob == 0.0:
            break
    
    return total_prob


def _create_mpe_explanation(mpe_assignment: Dict[str, str], mpe_probability: float,
                          nodes: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Create a human-readable explanation of the MPE result."""
    
    explanation = {
        "summary": f"Most probable explanation with probability {mpe_probability:.6f}",
        "assignment_details": {},
        "reasoning_chain": [],
        "confidence_level": _calculate_confidence_level(mpe_probability),
        "evidence_summary": evidence
    }
    
    # Create detailed assignment explanation
    for node_id, assigned_state in mpe_assignment.items():
        node = nodes[node_id]
        
        assignment_detail = {
            "variable": node["name"],
            "assigned_state": assigned_state,
            "variable_type": node["type"],
            "is_observed": node_id in evidence["observed_variables"],
            "is_query": node_id in evidence["query_variables"]
        }
        
        # Add reasoning for this assignment
        if node["is_root"]:
            prior_probs = node["cpt"].get("NO_PARENTS", {})
            assignment_detail["reasoning"] = f"Prior probability: {prior_probs.get(assigned_state, 0.0):.4f}"
        else:
            parent_ids = list(node["parents"].keys())
            parent_states = [mpe_assignment[pid] for pid in parent_ids]
            parent_names = [nodes[pid]["name"] for pid in parent_ids]
            
            parent_description = ", ".join([f"{name}={state}" for name, state in zip(parent_names, parent_states)])
            
            if len(parent_states) == 1:
                cpt_key = parent_states[0]
            else:
                cpt_key = tuple(parent_states)
                
            conditional_probs = node["cpt"].get(cpt_key, {})
            prob = conditional_probs.get(assigned_state, 0.0)
            
            assignment_detail["reasoning"] = f"Given {parent_description}, probability: {prob:.4f}"
        
        explanation["assignment_details"][node_id] = assignment_detail
    
    # Create reasoning chain
    topo_order = _get_topological_order_from_nodes(nodes)
    for node_id in topo_order:
        if node_id in mpe_assignment:
            node = nodes[node_id]
            assigned_state = mpe_assignment[node_id]
            detail = explanation["assignment_details"][node_id]
            
            reasoning_step = f"{node['name']} = {assigned_state} ({detail['reasoning']})"
            explanation["reasoning_chain"].append(reasoning_step)
    
    return explanation


def _get_topological_order_from_nodes(nodes: Dict[str, Any]) -> List[str]:
    """Compute topological order from nodes (simplified version)."""
    # Simple topological sort based on parent relationships
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


def _calculate_confidence_level(probability: float) -> str:
    """Calculate confidence level based on probability."""
    if probability >= 0.8:
        return "very_high"
    elif probability >= 0.6:
        return "high"
    elif probability >= 0.4:
        return "moderate"
    elif probability >= 0.2:
        return "low"
    else:
        return "very_low"


def _validate_mpe_result(mpe_result: Dict[str, Any], bayesian_network: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the MPE result."""
    errors = []
    warnings = []
    
    mpe_assignment = mpe_result.get("mpe_assignment", {})
    mpe_probability = mpe_result.get("mpe_probability", 0.0)
    
    # Check if all variables are assigned
    nodes = bayesian_network["nodes"]
    for node_id in nodes.keys():
        if node_id not in mpe_assignment:
            errors.append(f"Variable {node_id} not assigned in MPE result")
    
    # Check if assignments are valid
    for node_id, assigned_state in mpe_assignment.items():
        if node_id in nodes:
            valid_states = nodes[node_id]["states"]
            if assigned_state not in valid_states:
                errors.append(f"Invalid state {assigned_state} for variable {node_id}")
    
    # Check probability validity
    if mpe_probability < 0 or mpe_probability > 1:
        errors.append(f"Invalid probability {mpe_probability} (must be between 0 and 1)")
    
    if mpe_probability == 0:
        warnings.append("MPE probability is 0 - may indicate inconsistent evidence")
    
    # Verify probability calculation
    calculated_prob = _calculate_mpe_probability(mpe_assignment, nodes)
    if abs(calculated_prob - mpe_probability) > 1e-6:
        warnings.append(f"Probability mismatch: calculated {calculated_prob}, reported {mpe_probability}")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "assignment_count": len(mpe_assignment),
        "probability_valid": 0 <= mpe_probability <= 1
    }


def _create_mpe_metadata(mpe_result: Dict[str, Any], bayesian_network: Dict[str, Any], 
                        evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive metadata about the MPE computation."""
    
    nodes = bayesian_network["nodes"]
    mpe_assignment = mpe_result["mpe_assignment"]
    
    # Analyze assignment characteristics
    assignment_analysis = {
        "total_variables": len(mpe_assignment),
        "observed_variables": len(evidence["observed_variables"]),
        "query_variables": len(evidence["query_variables"]),
        "root_assignments": sum(1 for node_id in mpe_assignment if nodes[node_id]["is_root"]),
        "leaf_assignments": sum(1 for node_id in mpe_assignment if nodes[node_id]["is_leaf"])
    }
    
    # Analyze probability characteristics
    probability_analysis = {
        "mpe_probability": mpe_result["mpe_probability"],
        "log_probability": mpe_result["log_probability"],
        "confidence_level": mpe_result["explanation"]["confidence_level"],
        "entropy": _calculate_assignment_entropy(mpe_assignment, nodes)
    }
    
    # Computational complexity analysis
    complexity_analysis = {
        "inference_type": evidence["inference_type"],
        "algorithm_used": mpe_result["algorithm_type"],
        "network_complexity": bayesian_network["network_properties"]["total_parameters"],
        "search_space_size": _calculate_search_space_size(nodes, evidence)
    }
    
    return {
        "computation_timestamp": time.time(),
        "source_steps": {
            "step1_successful": bayesian_network.get("step1_result", {}).get("successful_api_call", False),
            "step2_successful": bayesian_network.get("step2_result", {}).get("successful_api_call", False),
            "step3_successful": bayesian_network.get("step3_result", {}).get("successful_api_call", False),
            "step4_successful": bayesian_network.get("step4_result", {}).get("successful_api_call", False),
            "step5_successful": bayesian_network.get("step5_result", {}).get("successful_api_call", False)
        },
        "assignment_analysis": assignment_analysis,
        "probability_analysis": probability_analysis,
        "complexity_analysis": complexity_analysis,
        "evidence_analysis": {
            "evidence_strength": len(evidence["observed_variables"]) / len(nodes),
            "query_coverage": len(evidence["query_variables"]) / len(nodes),
            "evidence_source": evidence["evidence_source"]
        }
    }


def _calculate_assignment_entropy(assignment: Dict[str, str], nodes: Dict[str, Any]) -> float:
    """Calculate the entropy of the MPE assignment."""
    total_entropy = 0.0
    
    for node_id, assigned_state in assignment.items():
        node = nodes[node_id]
        
        if node["is_root"]:
            prior_probs = node["cpt"].get("NO_PARENTS", {})
            prob = prior_probs.get(assigned_state, 0.0)
        else:
            # For simplicity, use uniform entropy for conditional nodes
            prob = 1.0 / len(node["states"])
        
        if prob > 0:
            total_entropy -= prob * math.log2(prob)
    
    return total_entropy


def _calculate_search_space_size(nodes: Dict[str, Any], evidence: Dict[str, Any]) -> int:
    """Calculate the size of the search space for MPE."""
    search_space = 1
    
    for node_id, node in nodes.items():
        if node_id not in evidence["observed_variables"]:
            search_space *= len(node["states"])
    
    return search_space
