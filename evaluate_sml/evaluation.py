import os
import re
import json
import yaml
import time
import argparse
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from loader import load_med_qa_dataset
from api_handler import get_model_response

def process_sample(sample, idx, model_name, api_key, max_tokens, temperature, format_prompt):
    """
    Processes a single data sample: formats prompt, calls API, parses result.
    This function is designed to be run in a parallel worker.
    """
    with open("evaluate_sml/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    sleep_time = config["sleep_time"]

    time.sleep(sleep_time)
    input_text = sample["question"] + "\n" + str(sample["options"]) + "\n" + format_prompt
    
    error_message = None  # To store potential error messages
    
    try:
        model_output = get_model_response(model_name, api_key, input_text, max_tokens, temperature)
        successful_api_call = True
    except Exception as e:
        # Capture the error message instead of printing it directly
        error_message = f"API call failed for sample {idx}: {e}"
        model_output = None
        successful_api_call = False

    # Initialize parsing-related variables
    right_format = False
    extracted_letter = None

    # Only attempt to parse if the API call was successful
    if successful_api_call:
        # This parsing logic now matches the user's required prompt format
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
        "correct_answer": sample["answer"],
        "error": error_message,
    }

def evaluate_model(
    dataset_name: str, 
    model_name: str, 
    api_key: str, 
    max_tokens: int, 
    temperature: float,
    num_workers: int,
    sleep_time: float,
    use_cache: bool
):
    """
    Main function to run the model evaluation pipeline.
    """
    start_time = time.time()
    run_timestamp = datetime.now()

    with open("evaluate_sml/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_config = config["datasets"][dataset_name]
    format_prompt = dataset_config["format_prompt"]
    
    output_base_dir = dataset_config["output_dir"]
    os.makedirs(output_base_dir, exist_ok=True)
    
    existing_experiments = [d for d in os.listdir(output_base_dir) if d.isdigit()]
    experiment_id = int(max(existing_experiments)) + 1 if existing_experiments else 1
    
    experiment_dir = os.path.join(output_base_dir, str(experiment_id))
    os.makedirs(experiment_dir, exist_ok=True)
    
    results_file = os.path.join(experiment_dir, "results.jsonl")
    run_details_file = os.path.join(experiment_dir, "run_details.json")
    
    print(f"Starting Experiment ID: {experiment_id}")
    print(f"Results will be saved in: {experiment_dir}")

    results = []
    processed_indices = set()
    cached_exp_dir_to_delete = None
    
    if use_cache:
        latest_exp_id = experiment_id - 1
        if latest_exp_id > 0:
            prev_exp_dir = os.path.join(output_base_dir, str(latest_exp_id))
            prev_run_details_file = os.path.join(prev_exp_dir, "run_details.json")
            prev_results_file = os.path.join(prev_exp_dir, "results.jsonl")

            if os.path.exists(prev_run_details_file):
                with open(prev_run_details_file, "r") as f:
                    prev_run_details = json.load(f)
                
                is_same_config = (
                    prev_run_details["config"].get("model_name") == model_name and
                    prev_run_details["config"].get("max_tokens") == max_tokens and
                    prev_run_details["config"].get("temperature") == temperature
                )

                if is_same_config and os.path.exists(prev_results_file):
                    print(f"Found matching cache in experiment {latest_exp_id}. Resuming...")
                    with open(prev_results_file, "r") as f:
                        for line in f:
                            cached_result = json.loads(line)
                            results.append(cached_result)
                            processed_indices.add(cached_result["idx"])
                    cached_exp_dir_to_delete = prev_exp_dir
                    print(f"Loaded {len(results)} results from cache.")
                else:
                    print("Config mismatch or cache invalid. Starting from scratch.")
            else:
                print("No previous run details found. Starting from scratch.")
    else:
        print("`use_cache` is False. Starting from scratch.")

    if dataset_name == "medqa":
        dataset = list(load_med_qa_dataset())
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    unprocessed_samples = [(sample, idx) for idx, sample in enumerate(dataset) if idx not in processed_indices]
    print(f"Total samples: {len(dataset)}. Processed from cache: {len(processed_indices)}. Remaining: {len(unprocessed_samples)}")

    if not unprocessed_samples:
        print("All samples have been processed. Exiting.")
        return

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_sample = {
            executor.submit(process_sample, sample, idx, model_name, api_key, max_tokens, temperature, format_prompt): (sample, idx)
            for sample, idx in unprocessed_samples
        }
        
        # Wrap as_completed with tqdm for a progress bar
        progress_bar = tqdm(as_completed(future_to_sample), total=len(unprocessed_samples), desc="Processing Samples", unit="sample")
        
        # Append new results to the main list as they complete
        for future in progress_bar:
            try:
                result = future.result()
                # If the worker returned an error, log it using tqdm.write
                if result.get("error"):
                    tqdm.write(result["error"])
                results.append(result)
            except Exception as exc:
                sample_info = future_to_sample[future]
                # Use tqdm.write for unhandled exceptions to avoid breaking the bar
                tqdm.write(f'Sample {sample_info[1]} generated an unhandled exception: {exc}')
    
    # Sort all results by index to ensure order before writing
    results.sort(key=lambda r: r['idx'])
    with open(results_file, "w") as f:
        for result in results:
            # Create a copy and remove the temporary 'error' key before writing
            result_to_write = result.copy()
            result_to_write.pop('error', None)
            f.write(json.dumps(result_to_write) + "\n")

    execution_time = time.time() - start_time
    
    run_summary = {
        "experiment_id": experiment_id,
        "run_time": run_timestamp.isoformat(),
        "execution_time_seconds": round(execution_time, 2),
        "config": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_workers": num_workers,
            "sleep_time": sleep_time
        },
        "total_samples": len(results),
        "successful_api_calls": sum(1 for r in results if r["successful_api_call"]),
        "correctly_formatted_answers": sum(1 for r in results if r["right_format"]),
    }

    with open(run_details_file, "w") as f:
        json.dump(run_summary, f, indent=4)
        
    # --- Finalization and Cleanup ---
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
        model_name="Qwen/Qwen3-32B",
        api_key="hTQSRchoqsaXBEtFp4tG994VgvCVEaoBDuYTPUZTbYdhMFQ4Rc31xYWoHkRfxTAB",
        max_tokens=2048,
        temperature=0.7,
        num_workers=4,
        sleep_time=0.5,
        use_cache=True
    )