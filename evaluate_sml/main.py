from evaluation import evaluate_model
from analyze import analyze_results, analyze_all_experiments
import yaml
import os

def full_experiment():
    # Find config.yaml relative to this file's location
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    api_key = config["api"]["api_key"]

    for model_name, _ in config["models"].items():
        if model_name == "DeepSeek-V3.1":
            continue
        for dataset_name, dataset_config in config["datasets"].items():
            for prompt in dataset_config["prompts"]:
                prompt_type = prompt["type"]
                print(f"Evaluating {dataset_name} with {model_name} and {prompt_type} ...")
                evaluate_model(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    prompt_type=prompt_type,
                    api_key=api_key,
                    use_cache=False,
                    parallel=True,
                    n_samples=2  # Use all samples by default
                )
    
    # Analyze all experiments using the new directory structure
    print("\n=== Analysis Phase ===")
    analyze_all_experiments("results")


if __name__ == "__main__":
    full_experiment()