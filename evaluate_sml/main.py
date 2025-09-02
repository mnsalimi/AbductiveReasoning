from evaluation import evaluate_model
from analyze import analyze_results
import yaml
import os

def full_experiment():
    with open("evaluate_sml/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    api_key = config["api"]["api_key"]

    for model_name, _ in config["models"].items():
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
                    n_samples=-1  # Use all samples by default
                )
    
    for dataset_name, dataset_config in config["datasets"].items():
        list_exp = os.listdir(dataset_config["output_dir"])
        for exp in list_exp:
            try:
                print(f"Analyzing {dataset_name} experiment {exp} ...")
                analyze_results(
                    dataset_name=dataset_name,
                    exp=int(exp)
                )
            except:
                pass


if __name__ == "__main__":
    full_experiment()