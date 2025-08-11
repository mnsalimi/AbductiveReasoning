import yaml
import json
import os
from sklearn.metrics import classification_report

def analyze_results(dataset_name: str, exp: int):
    """
    Analyzes model evaluation results. For multi-label tasks like 'uniadilr',
    it calculates exact match accuracy and subset relationships. For standard
    classification tasks, it uses classification_report.

    Args:
        dataset_name (str): The name of the dataset.
        exp (int): The experiment number.
    """
    try:
        with open("evaluate_sml/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please ensure 'evaluate_sml/config.yaml' exists.")
        return
    except Exception as e:
        print(f"Error loading or parsing config.yaml: {e}")
        return


    dir_path = os.path.join(config['datasets'][dataset_name]['output_dir'], str(exp))
    results_file = os.path.join(dir_path, "results.jsonl")

    run_details_path = os.path.join(dir_path, "run_details.json")
    run_data = {}
    if os.path.exists(run_details_path):
        with open(run_details_path, 'r') as file:
            run_data = json.load(file)
    else:
        print(f"Warning: run_details.json not found at {run_details_path}")


    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return

    total_samples = 0
    unsuccessful_api_calls = 0
    wrong_format = 0
    y_true = []
    y_pred = []

    with open(results_file, "r") as f:
        for line in f:
            total_samples += 1
            data = json.loads(line)

            if not data.get("successful_api_call", False):
                unsuccessful_api_calls += 1
            
            if not data.get("right_format", False):
                wrong_format += 1
            
            if data.get("successful_api_call", False) and data.get("right_format", False):
                true_answer = data["correct_answer"]
                model_answer = data["model_answer"]

                if dataset_name == "uniadilr":
                    y_true.append(set(true_answer))
                    y_pred.append(set(model_answer))
                elif dataset_name == "medqa" or dataset_name == "medmcqa":
                    y_true.append(true_answer)
                    y_pred.append(model_answer)

    percent_unsuccessful_api = (unsuccessful_api_calls / total_samples) * 100 if total_samples > 0 else 0
    percent_wrong_format = (wrong_format / total_samples) * 100 if total_samples > 0 else 0

    metrics_title = "Evaluation Metrics"
    metrics_report = "Not enough valid samples to generate metrics."
    num_valid_samples = len(y_true)

    if num_valid_samples > 0:
        if dataset_name == "uniadilr":
            exact_matches = 0
            pred_is_proper_subset = 0
            true_is_proper_subset = 0

            for true_set, pred_set in zip(y_true, y_pred):
                if pred_set == true_set:
                    exact_matches += 1
                elif pred_set.issubset(true_set):
                    pred_is_proper_subset += 1
                elif true_set.issubset(pred_set):
                    true_is_proper_subset += 1
            
            accuracy = (exact_matches / num_valid_samples)
            percent_pred_subset = (pred_is_proper_subset / num_valid_samples)
            percent_true_subset = (true_is_proper_subset / num_valid_samples)

            metrics_report = (
                f"1. Simple Accuracy (Exact Match): {accuracy:.2%}\n"
                f"   - The model's answer was exactly correct.\n\n"
                f"2. Prediction is Subset of Truth: {percent_pred_subset:.2%}\n"
                f"   - The model's answer was correct but incomplete (e.g., predicted {{A}} when it should be {{A, B}}).\n\n"
                f"3. Truth is Subset of Prediction: {percent_true_subset:.2%}\n"
                f"   - The model's answer included the correct answer plus extra incorrect items (e.g., predicted {{A, B, C}} when it should be {{A, B}})."
            )

        elif dataset_name == "medqa" or dataset_name == "medmcqa":
            metrics_title = "Classification Metrics"
            metrics_report = classification_report(y_true, y_pred, zero_division=0)

    analysis_content = f"""
    Analysis Report for {dataset_name} - Experiment {exp}
    Config:

    {json.dumps(run_data.get('config', 'Not available'), indent=4)}
    ======================================================

    1. Total Samples: {total_samples}

    2. Unsuccessful API Calls:
       - Count: {unsuccessful_api_calls}
       - Percentage: {percent_unsuccessful_api:.2f}%

    3. Wrong Formatting:
       - Count: {wrong_format}
       - Percentage: {percent_wrong_format:.2f}%

    4. {metrics_title} (for successful and correctly formatted samples):
    ----------------------------------------------------------------------
    {metrics_report}
    """

    report_path = os.path.join(dir_path, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(analysis_content)
    
    print(f"Analysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    analyze_results("uniadilr", 1)