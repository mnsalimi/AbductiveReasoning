import yaml
import os
from loader import load_med_qa_dataset, load_uniadilr_hgc_dataset
from step1 import step1
from step2 import step2
from step3 import step3
from step4 import step4
from step5 import step5
from step6 import step6

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
    model_name = "Llama4-Scout-17B-16E"
    dataset_name = "medqa"  # or "uniadilr"
    n_samples = 1
    
    # Get model-specific configuration
    model_config = config["models"][model_name]
    api_key = config["api"]["api_key"]
    sleep_time = config["sleep_time"]
    thinking = model_config["thinking"]
    temperature = model_config["temperature"]
    max_tokens = model_config["max_tokens_by_prompt_type"].get("CPT Creation", 1024)
    
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
    
    # Process each sample through the complete pipeline
    for idx, sample in enumerate(dataset):
        print_separator(f"PROCESSING SAMPLE {idx + 1}/{len(dataset)}")
        
        # Print sample information
        print("📄 SAMPLE INFORMATION:")
        if dataset_name == "medqa":
            print("Context:",sample["question"].replace(sample["question"].split(".")[-1], "")[:200])
            print("Question:",sample["question"].split(".")[-1][:200])
            print(f"  Correct Answer: {sample.get('answer_idx', 'N/A')}")
        else:
            print(f"  Context: {sample.get('context', '')[:200]}...")
            print(f"  Hypothesis: {sample.get('hypothesis', '')[:200]}...")
        
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
        
        # STEP 2: Refine BN Schema
        print_separator("STEP 2: Refine BN Schema")
        step2_result = step2(sample, idx, model_name, api_key, max_tokens, temperature,
                           thinking, sleep_time, dataset_name, step1_result)
        
        if step2_result.get("successful_api_call") and step2_result.get("right_format"):
            print("✓ Step 2 completed successfully")
            refined_nodes = step2_result["model_answer"].get("nodes", [])
            refined_edges = step2_result["model_answer"].get("edges", [])
            print(f"  📊 Refined nodes: {len(refined_nodes)} (added {len(refined_nodes) - len(nodes)})")
            print(f"  🔗 Refined edges: {len(refined_edges)} (added {len(refined_edges) - len(edges)})")
            if step2_result.get("token_usage"):
                print(f"  🔢 Tokens used: {step2_result['token_usage'].get('total_tokens', 0)}")
        else:
            print(f"✗ Step 2 failed: {step2_result.get('error', 'Unknown error')}")
            continue
        
        # STEP 3: Register DAG
        print_separator("STEP 3: Register DAG")
        step3_result = step3(sample, idx, step2_result, sleep_time)
        
        if step3_result.get("successful_api_call") and step3_result.get("right_format"):
            print("✓ Step 3 completed successfully")
            dag = step3_result.get("registered_dag", {})
            metadata = step3_result.get("dag_metadata", {})
            
            print(f"  📊 Total nodes: {dag.get('num_nodes', 0)}")
            print(f"  🔗 Total edges: {dag.get('num_edges', 0)}")
            
            node_stats = metadata.get("node_statistics", {})
            print(f"  🔵 Binary nodes: {node_stats.get('binary_nodes', 0)}")
            print(f"  🔶 Categorical nodes: {node_stats.get('categorical_nodes', 0)}")
            
            graph_structure = metadata.get("graph_structure", {})
            print(f"  🌳 Root nodes: {len(graph_structure.get('root_nodes', []))}")
            print(f"  🍃 Leaf nodes: {len(graph_structure.get('leaf_nodes', []))}")
            
            validation = step3_result.get("validation_result", {})
            if validation.get("warnings"):
                print(f"  ⚠️  Warnings: {len(validation['warnings'])}")
        else:
            print(f"✗ Step 3 failed: {step3_result.get('error', 'Unknown error')}")
            continue
        
        # STEP 4: CPT Creator
        print_separator("STEP 4: CPT Creator")
        step4_result = step4(sample, idx, model_name, api_key, max_tokens, temperature,
                           thinking, sleep_time, dataset_name, step3_result)
        
        if step4_result.get("successful_api_call") and step4_result.get("right_format"):
            print("✓ Step 4 completed successfully")
            cpts = step4_result.get("cpts", {})
            cpt_metadata = step4_result.get("cpt_metadata", {})
            
            print(f"  📊 CPTs generated: {len(cpts)}")
            
            cpt_stats = cpt_metadata.get("cpt_statistics", {})
            print(f"  🔢 Total parameters: {cpt_stats.get('total_parameters', 0)}")
            print(f"  🔵 Binary node CPTs: {cpt_stats.get('binary_node_cpts', 0)}")
            print(f"  🔶 Categorical node CPTs: {cpt_stats.get('categorical_node_cpts', 0)}")
            
            gen_stats = cpt_metadata.get("generation_statistics", {})
            print(f"  📞 API calls: {gen_stats.get('total_api_calls', 0)}")
            print(f"  ✅ Success rate: {gen_stats.get('success_rate', 0):.2%}")
            
            if step4_result.get("token_usage"):
                total_tokens = step4_result['token_usage'].get('total_tokens', 0)
                print(f"  🔢 Total tokens used: {total_tokens}")
        else:
            print(f"✗ Step 4 failed: {step4_result.get('error', 'Unknown error')}")
            continue
        
        # STEP 5: Bayesian Network Construction
        print_separator("STEP 5: Bayesian Network Construction")
        step5_result = step5(sample, idx, step4_result, sleep_time)
        
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
        else:
            print(f"✗ Step 5 failed: {step5_result.get('error', 'Unknown error')}")
            continue
        
        # STEP 6: MPE Algorithm
        print_separator("STEP 6: MPE Algorithm (Most Probable Explanation)")
        step6_result = step6(sample, idx, step5_result, sleep_time)
        
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
            
            complexity_analysis = mpe_metadata.get("complexity_analysis", {})
            search_space = complexity_analysis.get("search_space_size", 0)
            print(f"\n  🔬 COMPLEXITY:")
            print(f"    Search space size: {search_space:,}")
            print(f"    Algorithm: {mpe_result.get('algorithm_type', 'N/A')}")
            
            prob_analysis = mpe_metadata.get("probability_analysis", {})
            entropy = prob_analysis.get("entropy", 0.0)
            print(f"    Entropy: {entropy:.4f}")
            
        else:
            print(f"✗ Step 6 failed: {step6_result.get('error', 'Unknown error')}")
            continue
        
        print_separator("PIPELINE COMPLETED FOR SAMPLE")
        print(f"✓ All 6 steps completed successfully for sample {idx + 1}")
        
        # Print summary
        print("\n📊 SUMMARY:")
        print(f"  Step 1: {len(nodes)} nodes, {len(edges)} edges")
        print(f"  Step 2: {len(refined_nodes)} nodes, {len(refined_edges)} edges (refined)")
        print(f"  Step 3: DAG registered with {dag.get('num_nodes', 0)} nodes")
        print(f"  Step 4: {len(cpts)} CPTs created")
        print(f"  Step 5: Bayesian Network constructed")
        print(f"  Step 6: MPE probability = {mpe_probability:.6f}, confidence = {confidence}")
        
    print_separator("ALL SAMPLES PROCESSED")
    print(f"✓ Successfully processed {len(dataset)} samples through the complete pipeline")

if __name__ == "__main__":
    main()
