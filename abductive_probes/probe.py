from ast import parse
from nt import system
import os
import json
import yaml
import time
import traceback
from datetime import datetime
from tqdm import tqdm
from typing import List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from loader import load_dataset
from api_handler import get_model_response
from prompting.metric import AbductiveMetric


def _process_sample(
    sample,
    sample_id,
    judge_model,
    metric,
    max_tokens,
    temperature,
    sleep_time,
    dataset_name,
):
    """
    Processes a single data sample: formats prompt, calls API, parses result.
    """
    time.sleep(sleep_time)

    system_prompt, user_prompt = metric.construct_prompt(dataset_name, sample)
    error_message = None
    model_output = None
    usage = None
    successful_api_call = False

    try:
        model_output, usage = get_model_response(
            judge_model, system_prompt, user_prompt, max_tokens, temperature
        )
        successful_api_call = True
    except Exception as e:
        error_message = f"API call failed for sample {sample_id}: {e}"

    # Only attempt to parse if we have a non-empty string output from the API
    parsed_response = metric.parse_response(
        dataset_name, model_output
    )
    
    return {
            "idx": sample["id"], 
            "raw_data": sample,
            "successful_api_call": successful_api_call,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "metric": metric,
            "model_output": model_output,
            "parsed_response": parsed_response,
            "error": error_message,
            "token_usage": usage,
        }


def run_single_probing(
    dataset_name: str,
    judge_model: str,
    metric: AbductiveMetric,
    use_cache: bool,
    parallel: bool,
    n_samples: int = -1,
    check_for_existing_ids: bool = False,
):
    """
    Main function to run the model evaluation pipeline sequentially for a single abductive metric.

    Args:
        dataset_name: Name of the dataset to use
        judge_model: Name of the model for llm-as-a-judge
        metric: abductive metric to use
        use_cache: Whether to use cached results
        parallel: Whether to run API calls in parallel.
        n_samples: Number of samples to process (-1 for all/config default, >0 for specific count)
        check_for_existing_ids: If True, find a prior experiment on the same dataset and use its sample IDs.
    """
    start_time = time.time()
    run_timestamp = datetime.now()

    with open("evaluate_faithfulness/configs/models_config.yaml", "r") as f:
        models_config = yaml.safe_load(f)

    with open("evaluate_faithfulness/configs/experiments_config.yaml", "r") as f:
        experiments_config = yaml.safe_load(f)

    with open("evaluate_faithfulness/configs/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    general_prompt = prompts[dataset_name]["general_prompt"]
    defense_prompt = prompts[dataset_name]["defense_prompt"]
    sleep_time = models_config[judge_model]["sleep_time"]
    max_tokens = models_config[judge_model]["max_output_tokens"]
    temperature = models_config[judge_model]["temperature"]
    output_base_dir = experiments_config["output_dir"]

    experiment_name = f"{dataset_name}/{judge_model}/{str(metric)}"

    experiment_dir = os.path.join(output_base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    results_file = os.path.join(experiment_dir, "results.jsonl")
    run_details_file = os.path.join(experiment_dir, "run_details.json")

    print(f"Starting Experiment: {experiment_name}")
    print(f"Results will be saved in: {experiment_dir}")

    initial_run_details = {
        "experiment_name": experiment_name,
        "status": "running",
        "run_time_start": run_timestamp.isoformat(),
        "config": {
            "judge_model": judge_model,
            "dataset_name": dataset_name,
            "general_prompt": general_prompt,
            "defense_prompt": defense_prompt,
            "metric": metric.to_dict(),
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    }
    with open(run_details_file, "w") as f:
        json.dump(initial_run_details, f, indent=4)

    results = []
    processed_indices = set()

    if use_cache:
        if os.path.exists(run_details_file) and os.path.exists(results_file):
            print(f"Found existing run in '{experiment_dir}'. Checking config for resumption...")
            with open(run_details_file, "r") as f:
                prev_run_details = json.load(f)
            is_same_config = (
                prev_run_details["config"].get("judge_model") == judge_model
                and prev_run_details["config"].get("general_prompt") == general_prompt
                and prev_run_details["config"].get("defense_prompt") == defense_prompt
                and prev_run_details["config"].get("metric") == metric.to_dict()
                and prev_run_details["config"].get("max_tokens") == max_tokens
                and prev_run_details["config"].get("temperature") == temperature
            )
            if is_same_config:
                print("Config matches. Resuming by loading cached results...")
                with open(results_file, "r") as f:
                    for line in f:
                        cached_result = json.loads(line)
                        if cached_result.get("successful_api_call") and cached_result.get("right_format"):
                            results.append(cached_result)
                            processed_indices.add(cached_result["idx"])
                print(f"Loaded {len(results)} valid results from cache. Failed/invalid samples will be re-processed.")
            else:
                print("Existing run found but with a different config. Starting from scratch.")
        else:
            print("No valid cache found in the target directory. Starting from scratch.")
    else:
        print("`use_cache` is False. Starting from scratch.")

    # Check for existing IDs from a prior experiment if requested
    existing_ids_to_use: Set[int] | None = None
    if check_for_existing_ids:
        print("`check_for_existing_ids` is True. Searching for prior experiments...")
        dataset_experiments_dir = os.path.join(output_base_dir, dataset_name)
        
        best_results_file = None
        max_lines = -1

        if os.path.isdir(dataset_experiments_dir):
            for root, _, files in os.walk(dataset_experiments_dir):
                if "results.jsonl" in files:
                    results_path = os.path.join(root, "results.jsonl")
                    try:
                        with open(results_path, "r") as f:
                            num_lines = sum(1 for _ in f)
                        print(f"Found prior experiment results at: {results_path} with {num_lines} lines.")
                        if num_lines > max_lines:
                            max_lines = num_lines
                            best_results_file = results_path
                    except IOError as e:
                        print(f"Warning: Could not read file {results_path}. Error: {e}.")

        if best_results_file:
            print(f"Choosing the best results file with the most lines: {best_results_file} ({max_lines} lines)")
            try:
                with open(best_results_file, "r") as f:
                    # Read all non-empty lines and extract 'idx'
                    existing_ids_to_use = {json.loads(line)["idx"] for line in f if line.strip()}
                if existing_ids_to_use:
                    print(f"Found {len(existing_ids_to_use)} existing IDs to reuse.")
                else:
                    print(f"Warning: The best results file '{best_results_file}' contained no valid IDs.")
            except (IOError, json.JSONDecodeError, KeyError) as e:
                print(f"Error: Could not read IDs from the best file {best_results_file}. Error: {e}. Proceeding with normal sampling.")
                existing_ids_to_use = None
        else:
            print("No prior experiments found for this dataset. Proceeding with normal sampling.")


    print(f"Loading dataset: {dataset_name}")
    if n_samples != -1 and not existing_ids_to_use:
        print(f"Using experiment-specified n_samples: {n_samples}")
    
    # Pass the found IDs to the loader; it will handle the filtering.
    dataset = load_dataset(dataset_name, n_samples, specific_ids=existing_ids_to_use)
    print(f"Dataset loaded: {len(dataset)} samples")

    unprocessed_samples = [
        (sample, sample["id"])
        for sample in dataset
        if sample["id"] not in processed_indices
    ]
    print(f"Total samples: {len(dataset)}. Processed from cache: {len(processed_indices)}. Remaining: {len(unprocessed_samples)}")

    if len(dataset) == 0:
        print("Warning: Dataset is empty! Check your dataset configuration.")

    if not unprocessed_samples:
        print("All samples were processed in the cached run.")

    if parallel:
        with ThreadPoolExecutor() as executor:
            future_to_sample = {
                executor.submit(
                    _process_sample,
                    sample,
                    sample_id,
                    judge_model,
                    metric,
                    max_tokens,
                    temperature,
                    sleep_time,
                    dataset_name,
                ): (sample, sample_id)
                for sample, sample_id in unprocessed_samples
            }
            for future in tqdm(
                as_completed(future_to_sample),
                total=len(unprocessed_samples),
                desc="Processing Samples (Parallel)",
            ):
                try:
                    result = future.result()
                    if result.get("error"):
                        tqdm.write(result["error"])
                    results.append(result)
                    with open(results_file, "a") as f:
                        f.write(json.dumps(result) + "\n")
                except Exception as exc:
                    sample, sample_id = future_to_sample[future]
                    last_tb = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
                    location = f"{os.path.basename(last_tb.filename)}:{last_tb.lineno} in {last_tb.name}" if last_tb else "unknown location"
                    tqdm.write(f"Sample {sample_id} exception at {location}: {type(exc).__name__}: {exc}")
                    tqdm.write(traceback.format_exc())
    else:
        progress_bar = tqdm(unprocessed_samples, desc="Processing Samples", unit="sample")
        for sample, sample_id in progress_bar:
            try:
                result = _process_sample(
                    sample,
                    sample_id,
                    judge_model,
                    metric,
                    max_tokens,
                    temperature,
                    sleep_time,
                    dataset_name,
                )
                if result.get("error"):
                    tqdm.write(result["error"])
                results.append(result)
                result_to_write = result.copy()
                result_to_write.pop("error", None)
                with open(results_file, "a") as f:
                    f.write(json.dumps(result_to_write) + "\n")
            except Exception as exc:
                last_tb = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
                location = f"{os.path.basename(last_tb.filename)}:{last_tb.lineno} in {last_tb.name}" if last_tb else "unknown location"
                tqdm.write(f"Sample {sample_id} exception at {location}: {type(exc).__name__}: {exc}")
                tqdm.write(traceback.format_exc())

    execution_time = time.time() - start_time
    results.sort(key=lambda r: r["idx"])

    run_summary = {
        **initial_run_details,
        "status": "completed",
        "run_time_end": datetime.now().isoformat(),
        "execution_time_seconds": round(execution_time, 2),
        "summary": {
            "total_samples": len(dataset),
            "processed_samples": len(results),
            "successful_api_calls": sum(1 for r in results if r["successful_api_call"]),
            "correctly_formatted_answers": sum(1 for r in results if r["right_format"]),
        },
    }

    with open(run_details_file, "w") as f:
        json.dump(run_summary, f, indent=4)

    print("\n--- Evaluation Complete ---")
    print(json.dumps(run_summary, indent=4))
    print(f"Results saved to {results_file}")
    print(f"Run summary saved to {run_details_file}")


def run_probing(
    dataset_name: str,
    judge_model: str,
    metrics: List[AbductiveMetric],
    use_cache: bool,
    parallel: bool,
    n_samples: int = -1,
    check_for_existing_ids: bool = False,
):
    """
    Main function to run probing.

    Args:
        dataset_name: The name of the dataset
        metrics: List of abductive metrics to test
        use_cache: Whether to use cached results
        parallel: Whether to run API calls in parallel.
        n_samples: Number of samples to process (-1 for all/config default, >0 for specific count)
        check_for_existing_ids: If True, all runs will use the same sample IDs from a prior experiment.
    """
    experiment_results = []
    print(f"Starting Probing Experiment Suite for dataset: {dataset_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Number of metrics: {len(metrics)}")
    print(f"Parallel: {parallel}")
    if n_samples != -1:
        print(f"Sample limit: {n_samples}")
    print("=" * 60)

    for i, metric in enumerate(metrics, 1):
        metric_name = f"metric_{metric.metric_type.name}"
        print(f"\nRunning experiment {i}/{len(metrics)}: {metric_name}")
        try:
            run_single_probing(dataset_name, judge_model, metric, use_cache, parallel, n_samples, check_for_existing_ids)
            experiment_results.append({"metric": metric.to_dict(), "status": "completed"})
            print(f"Experiment {i} completed successfully")
        except Exception as e:
            print(f"Experiment {i} failed: {e}")
            experiment_results.append({"metric": metric.to_dict(), "status": "failed", "error": str(e)})

    print("\n" + "=" * 60)
    print("EXPERIMENT SUITE SUMMARY")
    print("=" * 60)
    total_experiments = len(experiment_results)
    completed_experiments = sum(1 for r in experiment_results if r["status"] == "completed")
    failed_experiments = total_experiments - completed_experiments
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed_experiments}")
    print(f"Failed: {failed_experiments}")
    if failed_experiments > 0:
        print(f"\nFailed experiments:")
        for i, result in enumerate(experiment_results):
            if result["status"] == "failed":
                hint_desc = "No-hint" if result["hint"] is None else f"Hint {i}"
                print(f"   - {hint_desc}: {result.get('error', 'Unknown error')}")
    print(f"\nMulti-hint experiment suite completed!")


if __name__ == "__main__":
    pass
