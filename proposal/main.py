import yaml
import os
import sys
import builtins
import time
from loader import load_med_qa_dataset, load_uniadilr_hgc_dataset
from step1 import step1
from step2 import step2
from step2dot5 import step2dot5
from step3 import step3, plot_dag
from step3dot5 import step3dot5
from step4 import step4
from step5 import step5
from step6 import step6
from step7 import step7, save_step7_result

def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def print_separator(title=""):
    """Print a visual separator with optional title"""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'-'*80}\n")

def main():
    # Load configuration
    config = load_config()
    
    # Configuration parameters
    # model_name = "Llama4-Scout-17B-16E"
    model_name = "GPT-OSS-120B"

    dataset_name = "medqa"  # or "uniadilr"
    n_samples = 100
    
    # Get model-specific configuration
    model_config = config["models"][model_name]
    api_key = config["api"]["api_key"]
    sleep_time = config["sleep_time"]
    sample_timeout = config.get("sample_timeout", 3600)  # Default to 1 hour if not specified
    thinking = model_config["thinking"]
    temperature = model_config["temperature"]
    max_tokens = model_config["max_tokens_by_prompt_type"].get("CPT Creation", 4096)
    
    # Load dataset
    print_separator("LOADING DATASET")
    if dataset_name == "medqa":
        dataset = load_med_qa_dataset(n_samples=n_samples)
        print(f"✓ Loaded MedQA dataset: {len(dataset)} samples")
    else:
        dataset = load_uniadilr_hgc_dataset(n_samples=n_samples)
        print(f"✓ Loaded UniADILR dataset: {len(dataset)} samples")
    
    print(f"✓ Model: {model_name}")
    print(f"✓ Temperature: {temperature}")
    print(f"✓ Thinking mode: {thinking}")
    print(f"✓ Sleep time: {sleep_time}s")
    if sample_timeout > 0:
        print(f"✓ Sample timeout: {sample_timeout}s ({sample_timeout / 3600:.1f} hour)")
    else:
        print(f"✓ Sample timeout: disabled")
    
    # Process each sample through the complete pipeline
    for idx, sample in enumerate(dataset):
        # Record start time for timeout tracking
        sample_start_time = time.time()
        
        # Prepare per-sample log file and duplicate prints into it (simple tee)
        logs_dir = os.path.join(os.path.dirname(__file__), "results", dataset_name, model_name, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"sample_{idx + 1:04d}.log")
        original_print = builtins.print
        log_file = open(log_path, "w", encoding="utf-8")
        
        def print_both(*args, **kwargs):
            original_print(*args, **kwargs)
            try:
                print_kwargs = dict(kwargs)
                file_obj = print_kwargs.pop("file", None)
                end_val = print_kwargs.pop("end", "\n")
                sep_val = print_kwargs.pop("sep", " ")
                text = sep_val.join(str(a) for a in args) + end_val
                log_file.write(text)
                log_file.flush()
            except Exception:
                pass
        
        # Function to check if timeout has been exceeded
        def is_timeout_exceeded():
            if sample_timeout <= 0:  # Timeout disabled
                return False
            elapsed = time.time() - sample_start_time
            return elapsed > sample_timeout
        
        builtins.print = print_both
        try:
            print_separator(f"PROCESSING SAMPLE {idx + 1}/{len(dataset)}")
        
            # Print sample information
            print("📄 SAMPLE INFORMATION:")
            if dataset_name == "medqa":
                print(f"  Context: {sample.get('context', '')}")
                print(f"  Question: {sample.get('question', '')}")
                print(f"  Correct Answer: {sample.get('answer_idx', 'N/A')}")
            else:
                print(f"  Context: {sample.get('context', '')[:200]}...")
                print(f"  Hypothesis: {sample.get('hypothesis', '')[:200]}...")
        
            # Check timeout before starting
            if is_timeout_exceeded():
                elapsed = time.time() - sample_start_time
                print(f"\n⏱️  TIMEOUT EXCEEDED: Sample took {elapsed:.2f}s (limit: {sample_timeout}s)")
                print(f"⏭️  Skipping sample {idx + 1}")
                continue
            
            # STEP 1: Create BN Schema
            print_separator("STEP 1: Create BN Schema")
            step1_result = step1(sample, idx, model_name, api_key, max_tokens, temperature, 
                               thinking, sleep_time, dataset_name)
        
            if step1_result.get("successful_api_call") and step1_result.get("right_format"):
                print("✓ Step 1 completed successfully")
                nodes = step1_result["model_answer"].get("nodes", [])
                edges = step1_result["model_answer"].get("edges", [])
                print(f"  📊 Nodes created: {len(nodes)}")
                print(f"  🔗 Edges created: {len(edges)}")
                print(f"  💭 Reasoning length: {len(step1_result['model_answer'].get('reasoning', ''))} chars")
                if step1_result.get("token_usage"):
                    print(f"  🔢 Tokens used: {step1_result['token_usage'].get('total_tokens', 0)}")
            else:
                print(f"✗ Step 1 failed: {step1_result.get('error', 'Unknown error')}")
                continue
            
            # Check timeout before step 2
            if is_timeout_exceeded():
                elapsed = time.time() - sample_start_time
                print(f"\n⏱️  TIMEOUT EXCEEDED: Sample took {elapsed:.2f}s (limit: {sample_timeout}s)")
                print(f"⏭️  Skipping sample {idx + 1} after Step 1")
                continue
        
            # STEP 2: Refine BN Schema
            print_separator("STEP 2: Refine BN Schema")
            step2_result = step2(sample, idx, model_name, api_key, max_tokens, temperature,
                               thinking, sleep_time, dataset_name, step1_result)
        
            if step2_result.get("successful_api_call") and step2_result.get("right_format"):
                print("✓ Step 2 completed successfully")
                refined_nodes = step2_result["model_answer"].get("nodes", [])
                refined_edges = step2_result["model_answer"].get("edges", [])
                
                # Compare with Step 1 to find additions/modifications using node names
                old_node_names = {node['name'] for node in nodes}
                new_node_names = {node['name'] for node in refined_nodes}
                added_node_names = new_node_names - old_node_names
                removed_node_names = old_node_names - new_node_names
                
                # Find modified nodes (same name but different properties)
                modified_node_names = set()
                old_nodes_dict = {node['name']: node for node in nodes}
                new_nodes_dict = {node['name']: node for node in refined_nodes}
                for node_name in old_node_names & new_node_names:
                    old_node = old_nodes_dict[node_name]
                    new_node = new_nodes_dict[node_name]
                    if old_node != new_node:
                        modified_node_names.add(node_name)
                
                # Compare edges (edges are strings like "node1 -> node2")
                old_edges_set = set(edges)
                new_edges_set = set(refined_edges)
                added_edges = new_edges_set - old_edges_set
                removed_edges = old_edges_set - new_edges_set
                
                print(f"  📊 Refined nodes: {len(refined_nodes)} (added {len(added_node_names)}, modified {len(modified_node_names)}, removed {len(removed_node_names)})")
                print(f"  🔗 Refined edges: {len(refined_edges)} (added {len(added_edges)}, removed {len(removed_edges)})")
                
                # Display all nodes with their types and categories
                print(f"\n  📋 All nodes after refinement:")
                for idx_node, node in enumerate(refined_nodes, 1):
                    node_id = f"node{idx_node}"
                    node_name = node['name']
                    node_type = node['type']
                    status = ""
                    if node_name in added_node_names:
                        status = " [NEW]"
                    elif node_name in modified_node_names:
                        status = " [MODIFIED]"
                    
                    if node_type == "binary":
                        print(f"      • {node_id}: '{node_name}' (binary){status}")
                    else:
                        categories = node.get('categories', [])
                        print(f"      • {node_id}: '{node_name}' (categorical, {len(categories)} categories: {categories}){status}")
                
                # Display edge changes if any
                if added_edges or removed_edges:
                    print(f"\n  🔗 Edge changes:")
                    if added_edges:
                        print(f"      Added edges:")
                        for edge in sorted(added_edges):
                            print(f"        ➕ {edge}")
                    if removed_edges:
                        print(f"      Removed edges:")
                        for edge in sorted(removed_edges):
                            print(f"        ➖ {edge}")
                
                if step2_result.get("token_usage"):
                    print(f"\n  🔢 Tokens used: {step2_result['token_usage'].get('total_tokens', 0)}")
            else:
                print(f"✗ Step 2 failed: {step2_result.get('error', 'Unknown error')}")
                continue
            
            # Check timeout before step 2.5
            if is_timeout_exceeded():
                elapsed = time.time() - sample_start_time
                print(f"\n⏱️  TIMEOUT EXCEEDED: Sample took {elapsed:.2f}s (limit: {sample_timeout}s)")
                print(f"⏭️  Skipping sample {idx + 1} after Step 2")
                continue
        
            # STEP 2.5: Refine DAG to ensure answer options are properly represented
            print_separator("STEP 2.5: Refine DAG for Answer Options")
            step2dot5_result = step2dot5(sample, idx, model_name, api_key, max_tokens, temperature,
                                          thinking, sleep_time, dataset_name, step2_result)
        
            if step2dot5_result.get("successful_api_call") and step2dot5_result.get("right_format"):
                print("✓ Step 2.5 completed successfully")
                modified_nodes = step2dot5_result["model_answer"].get("nodes", [])
                modified_edges = step2dot5_result["model_answer"].get("edges", [])
                modifications = step2dot5_result["model_answer"].get("modifications_summary", {})
                options_node = step2dot5_result.get("options_node")
                
                # Compare with Step 2 to find additions/modifications/deletions using node names
                old_node_names = {node['name'] for node in refined_nodes}
                new_node_names = {node['name'] for node in modified_nodes}
                added_node_names = new_node_names - old_node_names
                removed_node_names = old_node_names - new_node_names
                
                # Find modified nodes (same name but different properties)
                modified_node_names = set()
                old_nodes_dict = {node['name']: node for node in refined_nodes}
                new_nodes_dict = {node['name']: node for node in modified_nodes}
                for node_name in old_node_names & new_node_names:
                    old_node = old_nodes_dict[node_name]
                    new_node = new_nodes_dict[node_name]
                    if old_node != new_node:
                        modified_node_names.add(node_name)
                
                # Compare edges (edges are strings like "node1 -> node2")
                old_edges_set = set(refined_edges)
                new_edges_set = set(modified_edges)
                added_edges = new_edges_set - old_edges_set
                removed_edges = old_edges_set - new_edges_set
                
                print(f"  📊 Final nodes: {len(modified_nodes)} (added {len(added_node_names)}, modified {len(modified_node_names)}, removed {len(removed_node_names)})")
                print(f"  🔗 Final edges: {len(modified_edges)} (added {len(added_edges)}, removed {len(removed_edges)})")
                
                # Display all nodes with their types and categories
                print(f"\n  📋 All nodes after option refinement:")
                for idx_node, node in enumerate(modified_nodes, 1):
                    node_id = f"node{idx_node}"
                    node_name = node['name']
                    node_type = node['type']
                    status = ""
                    if node_name in added_node_names:
                        status = " [NEW]"
                    elif node_name in modified_node_names:
                        status = " [MODIFIED]"
                    
                    # Highlight the options node
                    is_options = options_node and node_name == options_node.get('name')
                    if is_options:
                        status += " 🎯"
                    
                    if node_type == "binary":
                        print(f"      • {node_id}: '{node_name}' (binary){status}")
                    else:
                        categories = node.get('categories', [])
                        print(f"      • {node_id}: '{node_name}' (categorical, {len(categories)} categories: {categories}){status}")
                
                # Display edge changes if any
                if added_edges or removed_edges:
                    print(f"\n  🔗 Edge changes:")
                    if removed_edges:
                        print(f"      Removed edges:")
                        for edge in sorted(removed_edges):
                            print(f"        ➖ {edge}")
                    if added_edges:
                        print(f"      Added edges:")
                        for edge in sorted(added_edges):
                            print(f"        ➕ {edge}")
                
                if options_node:
                    options_node_idx = next((i+1 for i, n in enumerate(modified_nodes) if n['name'] == options_node['name']), None)
                    options_node_id = f"node{options_node_idx}" if options_node_idx else "unknown"
                    print(f"\n  🎯 Options node: '{options_node['name']}' ({options_node_id}) with {len(options_node.get('categories', []))} categories")
                
                if step2dot5_result.get("token_usage"):
                    print(f"  🔢 Tokens used: {step2dot5_result['token_usage'].get('total_tokens', 0)}")
            else:
                print(f"✗ Step 2.5 failed: {step2dot5_result.get('error', 'Unknown error')}")
                continue
            
            # Check timeout before step 3
            if is_timeout_exceeded():
                elapsed = time.time() - sample_start_time
                print(f"\n⏱️  TIMEOUT EXCEEDED: Sample took {elapsed:.2f}s (limit: {sample_timeout}s)")
                print(f"⏭️  Skipping sample {idx + 1} after Step 2.5")
                continue
        
            # STEP 3: Register DAG
            print_separator("STEP 3: Register DAG")
            step3_result = step3(sample, idx, step2dot5_result, sleep_time)
        
            if step3_result.get("successful_api_call") and step3_result.get("right_format"):
                print("✓ Step 3 completed successfully")
                dag = step3_result.get("registered_dag", {})
                metadata = step3_result.get("dag_metadata", {})
                
                print(f"  📊 Total nodes: {dag.get('num_nodes', 0)}")
                print(f"  🔗 Total edges: {dag.get('num_edges', 0)}")
                
                node_stats = metadata.get("node_statistics", {})
                print(f"  🔵 Binary nodes: {node_stats.get('binary_nodes', 0)}")
                print(f"  🔶 Categorical nodes: {node_stats.get('categorical_nodes', 0)}")
                
                # Show categorical node details
                categorical_details = node_stats.get("categorical_details", [])
                if categorical_details:
                    print(f"  📋 Categorical nodes:")
                    for cat_node in categorical_details:
                        print(f"      • {cat_node['id']} ('{cat_node['name']}'): {cat_node['num_categories']} categories")
                
                graph_structure = metadata.get("graph_structure", {})
                print(f"  🌳 Root nodes: {len(graph_structure.get('root_nodes', []))}")
                print(f"  🍃 Leaf nodes: {len(graph_structure.get('leaf_nodes', []))}")
                
                validation = step3_result.get("validation_result", {})
                if validation.get("warnings"):
                    print(f"  ⚠️  Warnings: {len(validation['warnings'])}")
                    for warning in validation.get("warnings", []):
                        print(f"      - {warning}")
                
                # Plot and save the DAG visualization
                output_dir = os.path.join(os.path.dirname(__file__), "dag_visualizations", dataset_name, model_name)
                output_path = os.path.join(output_dir, f"dag_sample_{idx+1}")
                plot_dag(dag, output_path, idx)
            else:
                error_msg = step3_result.get('error', 'Unknown error')
                print(f"✗ Step 3 failed")
                print(f"\n{error_msg}")
                
                # Show validation details if available
                validation = step3_result.get("validation_result", {})
                if validation and validation.get("detailed_errors"):
                    print("\n  📋 Quick Summary:")
                    for detail in validation["detailed_errors"]:
                        print(f"      ❌ {detail['node_id']} ('{detail['node_name']}'): "
                              f"{detail['num_categories']} categories (need {detail['required_minimum']})")
                continue
            
            # Check timeout before step 3.5
            if is_timeout_exceeded():
                elapsed = time.time() - sample_start_time
                print(f"\n⏱️  TIMEOUT EXCEEDED: Sample took {elapsed:.2f}s (limit: {sample_timeout}s)")
                print(f"⏭️  Skipping sample {idx + 1} after Step 3")
                continue
        
            # STEP 3.5: Identify Visible Nodes
            print_separator("STEP 3.5: Identify Visible Nodes")
            step3dot5_result = step3dot5(sample, idx, model_name, api_key, max_tokens, temperature,
                                          thinking, sleep_time, dataset_name, step3_result, options_node)
        
            if step3dot5_result.get("successful_api_call") and step3dot5_result.get("right_format"):
                print("✓ Step 3.5 completed successfully")
                visible_nodes = step3dot5_result.get("visible_nodes", {})
                metadata = step3dot5_result.get("visible_nodes_metadata", {})
                
                print(f"  👁️  Visible nodes identified: {len(visible_nodes)}")
                
                node_stats = metadata.get("node_statistics", {})
                total_nodes = node_stats.get("total_nodes", 0)
                visibility_ratio = node_stats.get("visibility_ratio", 0)
                print(f"  📊 Visibility ratio: {visibility_ratio:.2%} ({len(visible_nodes)}/{total_nodes})")
                print(f"  🔵 Binary visible: {node_stats.get('visible_binary_nodes', 0)}")
                print(f"  🔶 Categorical visible: {node_stats.get('visible_categorical_nodes', 0)}")
                
                # Show visible node details
                visible_node_details = metadata.get("visible_node_details", [])
                if visible_node_details:
                    print(f"  📋 Visible nodes:")
                    for detail in visible_node_details[:5]:  # Show first 5
                        print(f"      • {detail['node_id']} ('{detail['node_name']}'): {detail['assigned_value']}")
                    if len(visible_node_details) > 5:
                        print(f"      ... and {len(visible_node_details) - 5} more")
                
                attempts_needed = metadata.get("attempts_needed", 1)
                if attempts_needed > 1:
                    print(f"  🔄 Attempts needed: {attempts_needed}")
                
                if step3dot5_result.get("token_usage"):
                    print(f"  🔢 Tokens used: {step3dot5_result['token_usage'].get('total_tokens', 0)}")
            else:
                error_msg = step3dot5_result.get('error', 'Unknown error')
                print(f"✗ Step 3.5 failed: {error_msg}")
                
                # Show attempt details if available
                attempts = step3dot5_result.get("attempts", [])
                if attempts:
                    print(f"\n  📊 Attempts made: {len(attempts)}")
                    for i, attempt in enumerate(attempts, 1):
                        if attempt.get("error"):
                            print(f"      Attempt {i}: Failed - {attempt.get('stage', 'unknown')} error")
                        else:
                            print(f"      Attempt {i}: Parsing failed")
                continue
            
            # Check timeout before step 4
            if is_timeout_exceeded():
                elapsed = time.time() - sample_start_time
                print(f"\n⏱️  TIMEOUT EXCEEDED: Sample took {elapsed:.2f}s (limit: {sample_timeout}s)")
                print(f"⏭️  Skipping sample {idx + 1} after Step 3.5")
                continue
        
            # STEP 4: CPT Creator
            print_separator("STEP 4: CPT Creator")
            step4_result = step4(sample, idx, model_name, api_key, max_tokens, temperature,
                               thinking, sleep_time, dataset_name, step3_result, step3dot5_result, batch_size=5,
                               sample_start_time=sample_start_time, sample_timeout=sample_timeout)
        
            if step4_result.get("successful_api_call") and step4_result.get("right_format"):
                print("✓ Step 4 completed successfully")
                cpts = step4_result.get("cpts", {})
                cpt_metadata = step4_result.get("cpt_metadata", {})
                
                print(f"  📊 CPTs generated: {len(cpts)}")
                
                cpt_stats = cpt_metadata.get("cpt_statistics", {})
                print(f"  🔢 Total parameters: {cpt_stats.get('total_parameters', 0)}")
                print(f"  🔵 Binary node CPTs: {cpt_stats.get('binary_node_cpts', 0)}")
                print(f"  🔶 Categorical node CPTs: {cpt_stats.get('categorical_node_cpts', 0)}")
                
                # Show optimization statistics if any combinations were skipped
                opt_stats = cpt_metadata.get("optimization_statistics", {})
                if opt_stats.get("optimization_enabled"):
                    skipped = opt_stats.get("skipped_combinations", 0)
                    queried = cpt_stats.get("total_queried_rows", 0)
                    total_possible = queried + skipped
                    savings_pct = (skipped / total_possible * 100) if total_possible > 0 else 0
                    print(f"  ⚡ Optimization: queried {queried} rows, skipped {skipped} ({savings_pct:.1f}% reduction)")
                
                gen_stats = cpt_metadata.get("generation_statistics", {})
                print(f"  📞 API calls: {gen_stats.get('total_api_calls', 0)}")
                print(f"  ✅ Success rate: {gen_stats.get('success_rate', 0):.2%}")
                
                if step4_result.get("token_usage"):
                    total_tokens = step4_result['token_usage'].get('total_tokens', 0)
                    print(f"  🔢 Total tokens used: {total_tokens}")
            else:
                error_msg = step4_result.get('error', 'Unknown error')
                is_timeout = step4_result.get('timeout', False)
                
                if is_timeout:
                    print(f"⏱️  Step 4 timed out")
                    print(f"\n{error_msg}")
                else:
                    print(f"✗ Step 4 failed")
                    print(f"\n{error_msg}")
                
                # Show generation log summary if available
                gen_log = step4_result.get("cpt_generation_log", [])
                if gen_log:
                    successful = sum(1 for log in gen_log if log.get('success', False))
                    print(f"\n  📊 Progress: {successful}/{len(gen_log)} nodes completed before failure")
                continue
            
            # Check timeout before step 5
            if is_timeout_exceeded():
                elapsed = time.time() - sample_start_time
                print(f"\n⏱️  TIMEOUT EXCEEDED: Sample took {elapsed:.2f}s (limit: {sample_timeout}s)")
                print(f"⏭️  Skipping sample {idx + 1} after Step 4")
                continue
        
            # STEP 5: Bayesian Network Construction
            print_separator("STEP 5: Bayesian Network Construction")
            step5_result = step5(sample, idx, step4_result, step3_result, sleep_time, step3dot5_result)
        
            if step5_result.get("successful_api_call") and step5_result.get("right_format"):
                print("✓ Step 5 completed successfully")
                bn = step5_result.get("bayesian_network", {})
                bn_metadata = step5_result.get("network_metadata", {})
                
                network_structure = bn_metadata.get("network_structure", {})
                print(f"  📊 Total nodes: {network_structure.get('total_nodes', 0)}")
                print(f"  🌳 Root nodes: {network_structure.get('root_nodes', 0)}")
                print(f"  🍃 Leaf nodes: {network_structure.get('leaf_nodes', 0)}")
                print(f"  🔄 Intermediate nodes: {network_structure.get('intermediate_nodes', 0)}")
                
                complexity = bn_metadata.get("complexity_metrics", {})
                print(f"  🔗 Total edges: {complexity.get('total_edges', 0)}")
                print(f"  📈 Network density: {complexity.get('network_density', 0):.3f}")
                print(f"  🔢 Total parameters: {complexity.get('total_parameters', 0)}")
                print(f"  👨‍👩‍👧‍👦 Max parents per node: {complexity.get('max_parents', 0)}")
                
                inference_readiness = bn_metadata.get("inference_readiness", {})
                print(f"  ✅ Topological order: {inference_readiness.get('has_topological_order', False)}")
                print(f"  ✅ State spaces: {inference_readiness.get('has_state_spaces', False)}")
                print(f"  ✅ CPT patterns: {inference_readiness.get('has_cpt_patterns', False)}")
                
                factorization = inference_readiness.get("factorization", "")
                if factorization:
                    print(f"  📐 Factorization: {factorization[:100]}..." if len(factorization) > 100 else f"  📐 Factorization: {factorization}")
                
                # Show validation warnings if any
                validation = step5_result.get("validation_result", {})
                if validation and validation.get("warnings"):
                    print(f"  ⚠️  Warnings: {len(validation['warnings'])}")
                    for warning in validation.get("warnings", [])[:3]:  # Show first 3
                        print(f"      - {warning}")
            else:
                error_msg = step5_result.get('error', 'Unknown error')
                print(f"✗ Step 5 failed")
                print(f"\n{error_msg}")
                
                # Show validation details if available
                validation = step5_result.get("validation_result", {})
                if validation and validation.get("errors"):
                    print(f"\n  📋 Validation errors: {len(validation['errors'])}")
                continue
            
            # Check timeout before step 6
            if is_timeout_exceeded():
                elapsed = time.time() - sample_start_time
                print(f"\n⏱️  TIMEOUT EXCEEDED: Sample took {elapsed:.2f}s (limit: {sample_timeout}s)")
                print(f"⏭️  Skipping sample {idx + 1} after Step 5")
                continue
        
            # STEP 6: MPE Algorithm
            print_separator("STEP 6: MPE Algorithm (Most Probable Explanation)")
            step6_result = step6(sample, idx, step5_result, sleep_time, step3dot5_result)
        
            if step6_result.get("successful_api_call") and step6_result.get("right_format"):
                print("✓ Step 6 completed successfully")
                mpe_result = step6_result.get("mpe_result", {})
                mpe_metadata = step6_result.get("mpe_metadata", {})
                evidence = step6_result.get("evidence", {})
                
                print(f"  🔍 Inference type: {evidence.get('inference_type', 'N/A')}")
                print(f"  👁️  Observed variables: {len(evidence.get('observed_variables', {}))}")
                print(f"  ❓ Query variables: {len(evidence.get('query_variables', []))}")
                
                print(f"\n  📊 MPE RESULT:")
                mpe_assignment = mpe_result.get("mpe_assignment", {})
                mpe_probability = mpe_result.get("mpe_probability", 0.0)
                log_probability = mpe_result.get("log_probability", 0.0)
                
                print(f"  🎯 MPE Probability: {mpe_probability:.6f}")
                print(f"  📈 Log Probability: {log_probability:.6f}")
                
                explanation = mpe_result.get("explanation", {})
                confidence = explanation.get("confidence_level", "N/A")
                print(f"  💪 Confidence Level: {confidence}")
                
                print(f"\n  📋 VARIABLE ASSIGNMENTS:")
                assignment_details = explanation.get("assignment_details", {})
                for node_id, detail in list(assignment_details.items())[:5]:  # Show first 5
                    var_name = detail.get("variable", "Unknown")
                    state = detail.get("assigned_state", "Unknown")
                    is_observed = "👁️ " if detail.get("is_observed") else ""
                    is_query = "❓" if detail.get("is_query") else ""
                    print(f"    {is_observed}{is_query} {var_name} = {state}")
                
                if len(assignment_details) > 5:
                    print(f"    ... and {len(assignment_details) - 5} more variables")
                
                print(f"\n  🧠 REASONING CHAIN:")
                reasoning_chain = explanation.get("reasoning_chain", [])
                for i, step in enumerate(reasoning_chain[:3]):  # Show first 3 steps
                    print(f"    {i+1}. {step}")
                if len(reasoning_chain) > 3:
                    print(f"    ... and {len(reasoning_chain) - 3} more reasoning steps")
                # Log-only: full reasoning chain
                if reasoning_chain:
                    pass
                
                complexity_analysis = mpe_metadata.get("complexity_analysis", {})
                search_space = complexity_analysis.get("search_space_size", 0)
                print(f"\n  🔬 COMPLEXITY:")
                print(f"    Search space size: {search_space:,}")
                print(f"    Algorithm: {mpe_result.get('algorithm_type', 'N/A')}")
                
                prob_analysis = mpe_metadata.get("probability_analysis", {})
                entropy = prob_analysis.get("entropy", 0.0)
                print(f"    Entropy: {entropy:.4f}")
                
                # Show validation warnings if any
                validation = step6_result.get("validation_result", {})
                if validation and validation.get("warnings"):
                    print(f"\n  ⚠️  Warnings: {len(validation['warnings'])}")
                    for warning in validation.get("warnings", [])[:3]:  # Show first 3
                        print(f"      - {warning}")
            else:
                error_msg = step6_result.get('error', 'Unknown error')
                print(f"✗ Step 6 failed: {error_msg}")
                
                # Show validation details if available
                validation = step6_result.get("validation_result", {})
                if validation and validation.get("errors"):
                    print(f"\n  📋 Validation errors:")
                    for error in validation.get("errors", [])[:5]:  # Show first 5
                        print(f"      • {error}")
                continue
            
            # Check timeout before step 7
            if is_timeout_exceeded():
                elapsed = time.time() - sample_start_time
                print(f"\n⏱️  TIMEOUT EXCEEDED: Sample took {elapsed:.2f}s (limit: {sample_timeout}s)")
                print(f"⏭️  Skipping sample {idx + 1} after Step 6")
                continue
        
            # STEP 7: Answer Extraction
            print_separator("STEP 7: Answer Extraction")
            # Use max_tokens from Chain of Thought or default to 2048 for step7
            max_tokens_step7 = model_config["max_tokens_by_prompt_type"].get("Chain of Thought", 2048)
            step7_result = step7(sample, idx, model_name, api_key, max_tokens_step7, temperature,
                               thinking, sleep_time, dataset_name, step6_result, step5_result, options_node)
        
            if step7_result.get("successful_api_call") and step7_result.get("right_format"):
                print("✓ Step 7 completed successfully")
                extracted_answer = step7_result.get("extracted_answer", {})
                
                if dataset_name == "medqa":
                    option = extracted_answer.get("option", "N/A")
                    value = extracted_answer.get("value", "N/A")
                    print(f"  🎯 Extracted Answer: {option}")
                    print(f"  📝 Answer Text: {value}")
                elif dataset_name == "uniadilr":
                    conclusion = extracted_answer.get("conclusion", "N/A")
                    print(f"  🎯 Extracted Conclusion: {conclusion}")
                
                is_correct = step7_result.get("is_correct")
                if is_correct is not None:
                    correct_indicator = "✅" if is_correct else "❌"
                    print(f"  {correct_indicator} Correctness: {'Correct' if is_correct else 'Incorrect'}")
                    print(f"  📊 Ground Truth: {step7_result.get('correct_answer', 'N/A')}")
                
                if step7_result.get("token_usage"):
                    print(f"  🔢 Tokens used: {step7_result['token_usage'].get('total_tokens', 0)}")
                
                # Save step7 result to file with all step results aggregated
                # Create a combined result with all steps
                combined_result = {
                    **step7_result,
                    "step1_result": step1_result,
                    "step2_result": step2_result,
                    "step2dot5_result": step2dot5_result,
                    "step3_result": step3_result,
                    "step3dot5_result": step3dot5_result,
                    "step4_result": step4_result,
                    "step5_result": step5_result,
                    "step6_result": step6_result
                }
                save_step7_result(combined_result, dataset_name, model_name)
            else:
                error_msg = step7_result.get('error', 'Unknown error')
                print(f"✗ Step 7 failed: {error_msg}")
                # Still save the result even if it failed with all step results aggregated
                combined_result = {
                    **step7_result,
                    "step1_result": step1_result,
                    "step2_result": step2_result,
                    "step2dot5_result": step2dot5_result,
                    "step3_result": step3_result,
                    "step3dot5_result": step3dot5_result,
                    "step4_result": step4_result,
                    "step5_result": step5_result,
                    "step6_result": step6_result
                }
                save_step7_result(combined_result, dataset_name, model_name)
            
            print_separator("PIPELINE COMPLETED FOR SAMPLE")
            sample_elapsed_time = time.time() - sample_start_time
            print(f"✓ All steps completed for sample {idx + 1}")
            print(f"⏱️  Total time: {sample_elapsed_time:.2f}s ({sample_elapsed_time / 60:.2f} minutes)")
            
            # Print summary
            print("\n📊 SUMMARY:")
            print(f"  Step 1: {len(nodes)} nodes, {len(edges)} edges")
            print(f"  Step 2: {len(refined_nodes)} nodes, {len(refined_edges)} edges (refined)")
            print(f"  Step 2.5: {len(modified_nodes)} nodes, {len(modified_edges)} edges (options refined)")
            print(f"  Step 3: DAG registered with {dag.get('num_nodes', 0)} nodes")
            print(f"  Step 4: {len(cpts)} CPTs created")
            print(f"  Step 5: Bayesian Network constructed")
            print(f"  Step 6: MPE probability = {mpe_probability:.6f}, confidence = {confidence}")
            if step7_result.get("right_format"):
                if dataset_name == "medqa":
                    extracted_option = step7_result.get("extracted_answer", {}).get("option", "N/A")
                    print(f"  Step 7: Extracted answer = {extracted_option}")
                else:
                    print(f"  Step 7: Answer extracted successfully")
        finally:
            builtins.print = original_print
            try:
                log_file.flush()
            finally:
                log_file.close()
        
    print_separator("ALL SAMPLES PROCESSED")
    print(f"✓ Successfully processed {len(dataset)} samples through the complete pipeline")

if __name__ == "__main__":
    main()
