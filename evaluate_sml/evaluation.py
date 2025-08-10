import os
import re
import json
import yaml
import time
import shutil
from datetime import datetime
from tqdm import tqdm

from loader import load_med_qa_dataset
from api_handler import get_model_response

def process_sample(sample, idx, model_name, api_key, max_tokens, temperature, format_prompt, sleep_time):
    """
    Processes a single data sample: formats prompt, calls API, parses result.
    This function runs sequentially.
    """
    # Sleep to respect API rate limits
    time.sleep(sleep_time)
    
    input_text = sample["question"] + "\n" + str(sample["options"]) + "\n" + format_prompt
    
    error_message = None  # To store potential error messages
    model_output = None
    successful_api_call = False

    try:
        model_output = get_model_response(model_name, api_key, input_text, max_tokens, temperature)
        successful_api_call = True
    except Exception as e:
        # Capture the error message
        error_message = f"API call failed for sample {idx}: {e}"

    # Initialize parsing-related variables
    right_format = False
    extracted_letter = None

    # Only attempt to parse if the API call was successful
    if successful_api_call:
        pattern = r"<a>([A-D])</a>"
        match = re.search(pattern, model_output)
        if match:
            extracted_letter = match.group(1)
            right_format = True
    
    return {
        "idx": idx,
        "raw_data": sample,
        "successful_api_call": successful_api_call,
        "right_format": right_format,
        "input_text": input_text,
        "model_output": model_output,
        "model_answer": extracted_letter,
        "correct_answer": sample["answer_idx"],
        "error": error_message, 
    }

def evaluate_model(
    dataset_name: str, 
    model_name: str, 
    api_key: str, 
    max_tokens: int, 
    temperature: float,
    use_cache: bool
):
    """
    Main function to run the model evaluation pipeline sequentially.
    """
    start_time = time.time()
    run_timestamp = datetime.now()

    with open("evaluate_sml/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    sleep_time = config['sleep_time']
    dataset_config = config["datasets"][dataset_name]
    format_prompt = dataset_config["format_prompt"]
    
    # --- Experiment Setup ---
    output_base_dir = dataset_config["output_dir"]
    os.makedirs(output_base_dir, exist_ok=True)
    
    existing_experiments = [int(d) for d in os.listdir(output_base_dir) if d.isdigit()]
    experiment_id = max(existing_experiments) + 1 if existing_experiments else 1
    
    experiment_dir = os.path.join(output_base_dir, str(experiment_id))
    os.makedirs(experiment_dir, exist_ok=True)
    
    results_file = os.path.join(experiment_dir, "results.jsonl")
    run_details_file = os.path.join(experiment_dir, "run_details.json")
    
    print(f"Starting Experiment ID: {experiment_id}")
    print(f"Results will be saved in: {experiment_dir}")

    # --- FIX 1: WRITE INITIAL RUN DETAILS AT THE START ---
    # This file signals that a run has started and stores its configuration.
    # It allows us to check the config of a crashed run.
    initial_run_details = {
        "experiment_id": experiment_id,
        "status": "running",
        "run_time_start": run_timestamp.isoformat(),
        "config": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "format_prompt": format_prompt,
        }
    }
    with open(run_details_file, "w") as f:
        json.dump(initial_run_details, f, indent=4)

    # --- Caching Logic (FIXED) ---
    results = []
    processed_indices = set()
    cached_exp_dir_to_delete = None
    
    if use_cache:
        latest_exp_id = experiment_id - 1
        if latest_exp_id > 0:
            prev_exp_dir = os.path.join(output_base_dir, str(latest_exp_id))
            prev_run_details_file = os.path.join(prev_exp_dir, "run_details.json")
            prev_results_file = os.path.join(prev_exp_dir, "results.jsonl")

            # Check if the previous run's details and results exist
            if os.path.exists(prev_run_details_file) and os.path.exists(prev_results_file):
                with open(prev_run_details_file, "r") as f:
                    prev_run_details = json.load(f)
                
                # Compare the config from the previous run with the current one
                is_same_config = (
                    prev_run_details["config"].get("model_name") == model_name and
                    prev_run_details["config"].get("max_tokens") == max_tokens and
                    prev_run_details["config"].get("temperature") == temperature and 
                    prev_run_details["config"].get("format_prompt") == format_prompt
                )

                if is_same_config:
                    print(f"Found matching cache in experiment {latest_exp_id}. Resuming...")
                    # Consolidate cache by copying old results into the new experiment directory
                    shutil.copy(prev_results_file, results_file)
                    
                    with open(results_file, "r") as f:
                        for line in f:
                            cached_result = json.loads(line)
                            # Only trust successful results from the cache
                            if cached_result.get("successful_api_call") and cached_result.get("right_format"):
                                results.append(cached_result)
                                processed_indices.add(cached_result["idx"])
                    
                    # Mark the old directory for deletion upon successful completion
                    cached_exp_dir_to_delete = prev_exp_dir
                    print(f"Loaded {len(results)} valid results from cache. Failed/invalid samples will be re-processed.")
                else:
                    print("Previous run has a different config. Starting from scratch.")
            else:
                print("No valid cache found from the previous run. Starting from scratch.")
    else:
        print("`use_cache` is False. Starting from scratch.")

    # --- Data Loading ---
    if dataset_name == "medqa":
        dataset = list(load_med_qa_dataset())
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    unprocessed_samples = [(sample, idx) for idx, sample in enumerate(dataset) if idx not in processed_indices]
    print(f"Total samples: {len(dataset)}. Processed from cache: {len(processed_indices)}. Remaining: {len(unprocessed_samples)}")

    if not unprocessed_samples and cached_exp_dir_to_delete:
        print("All samples were processed in the cached run.")
        # The finalization step below will handle summary and cleanup
    
    # --- Sequential Processing ---
    progress_bar = tqdm(unprocessed_samples, desc="Processing Samples", unit="sample")
    for sample, idx in progress_bar:
        try:
            result = process_sample(
                sample, idx, model_name, api_key, max_tokens, temperature, format_prompt, sleep_time
            )
            
            if result.get("error"):
                tqdm.write(result["error"])

            # Add to the in-memory list for final summary calculation
            results.append(result)

            # --- FIX 2: SAVE EACH RESULT IMMEDIATELY TO THE FILE ---
            # This ensures progress is saved even if the script crashes.
            result_to_write = result.copy()
            result_to_write.pop('error', None) # Don't save temporary error key to file
            
            with open(results_file, "a") as f:
                f.write(json.dumps(result_to_write) + "\n")

        except Exception as exc:
            tqdm.write(f'Sample {idx} generated an unhandled exception: {exc}')

    # --- Finalization and Summary ---
    execution_time = time.time() - start_time
    
    # Sort results for consistency before creating the summary
    results.sort(key=lambda r: r['idx'])
    
    run_summary = {
        **initial_run_details, # Start with the initial details
        "status": "completed",
        "run_time_end": datetime.now().isoformat(),
        "execution_time_seconds": round(execution_time, 2),
        "summary": {
            "total_samples": len(dataset),
            "processed_samples": len(results),
            "successful_api_calls": sum(1 for r in results if r["successful_api_call"]),
            "correctly_formatted_answers": sum(1 for r in results if r["right_format"]),
        }
    }

    # Overwrite the initial run_details.json with the final complete summary
    with open(run_details_file, "w") as f:
        json.dump(run_summary, f, indent=4)
        
    # --- Cleanup ---
    if cached_exp_dir_to_delete:
        print(f"\nRun complete. Deleting old cache directory: {cached_exp_dir_to_delete}")
        try:
            shutil.rmtree(cached_exp_dir_to_delete)
            print("Cache directory deleted successfully.")
        except Exception as e:
            print(f"Warning: Could not delete old cache directory. Error: {e}")

    print("\n--- Evaluation Complete ---")
    print(json.dumps(run_summary, indent=4))
    print(f"Results saved to {results_file}")
    print(f"Run summary saved to {run_details_file}")

if __name__ == "__main__":
    evaluate_model(
        dataset_name="medqa",
        model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        api_key="hTQSRchoqsaXBEtFp4tG994VgvCVEaoBDuYTPUZTbYdhMFQ4Rc31xYWoHkRfxTAB",
        max_tokens=15000,
        temperature=0.7,
        use_cache=True
    )