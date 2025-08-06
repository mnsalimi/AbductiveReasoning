import yaml
import json
import os
import json
from sklearn.metrics import classification_report

def analyze_results(dataset_name: str, exp: int):
    """
    Analyzes the results of a model evaluation from a .jsonl file.

    Args:
        dataset_name (str): The name of the dataset.
        exp (int): The experiment number.
    """
    with open("evaluate_sml/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dir_path = os.path.join(config['datasets'][dataset_name]['output_dir'], str(exp))
    results_file = os.path.join(dir_path, "results.jsonl")

    with open(os.path.join(dir_path, "run_details.json"), 'r') as file:
        run_data = json.load(file)

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
                y_true.append(data["correct_answer"])
                y_pred.append(data["model_answer"])

    # Calculate percentages
    percent_unsuccessful_api = (unsuccessful_api_calls / total_samples) * 100 if total_samples > 0 else 0
    percent_wrong_format = (wrong_format / total_samples) * 100 if total_samples > 0 else 0

    # Generate classification report
    report = classification_report(y_true, y_pred, zero_division=0)

    # Prepare the analysis report content
    analysis_content = f"""
    Analysis Report for {dataset_name} - Experiment {exp}
    Config:

    {run_data['config']}
    ======================================================

    1. Total Samples: {total_samples}

    2. Unsuccessful API Calls:
       - Count: {unsuccessful_api_calls}
       - Percentage: {percent_unsuccessful_api:.2f}%

    3. Wrong Formatting:
       - Count: {wrong_format}
       - Percentage: {percent_wrong_format:.2f}%

    4. Classification Metrics (for successful and correctly formatted samples):
    ----------------------------------------------------------------------
    {report}
    """

    # Save the report to a file
    report_path = os.path.join(dir_path, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(analysis_content)
    
    print(report_path)

    print(f"Analysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    analyze_results("medqa", 1)