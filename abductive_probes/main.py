import os
import shutil
from datetime import datetime

def create_experiment_version():
    """Create a versioned copy of the experiments folder before running main."""
    
    # Get current date in YYYY-MM-DD format
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Define paths
    experiments_path = "experiments"
    versions_path = os.path.join(experiments_path, "versions")
    
    # Create versions directory if it doesn't exist
    os.makedirs(versions_path, exist_ok=True)
    
    # Find the next version number for today's date
    version_num = 1
    while True:
        version_name = f"{current_date}_v{version_num}"
        version_path = os.path.join(versions_path, version_name)
        if not os.path.exists(version_path):
            break
        version_num += 1
    
    print(f"Creating experiment version: {version_name}")
    
    # Copy experiments folder (excluding the versions subfolder)
    if os.path.exists(experiments_path):
        # Create the versioned directory
        os.makedirs(version_path, exist_ok=True)
        
        # Copy all items from experiments except versions folder
        for item in os.listdir(experiments_path):
            if item != "versions":  # Skip versions folder to avoid recursion
                source_path = os.path.join(experiments_path, item)
                dest_path = os.path.join(version_path, item)
                
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, dest_path)
                else:
                    shutil.copy2(source_path, dest_path)
        
        print(f"Experiment version created successfully at: {version_path}")
        return version_name
    else:
        print("Experiments folder not found. Skipping versioning.")
        return None

if __name__ == "__main__":
    # Create version backup before running
    version_name = create_experiment_version()
    if version_name:
        print(f"Current experiments backed up as: {version_name}")
    
    from probe import run_probing
    from prompting.metric import AbductiveMetric
    from prompting.metrics.branchiness import Branchiness

    metrics: list[AbductiveMetric] = [Branchiness()]    

    judge_model = "openai/gpt-4o-mini"
    datasets = ["sample"]

    
    # Run experiments
    print("Starting experiments...")
    for dataset in datasets:
        run_probing(
            dataset_name=dataset,
            judge_model=judge_model,
            metrics=metrics,
            use_cache=False,
            parallel=True,
            n_samples=100,
            check_for_existing_ids=True
        )
    
    print("\nExperiments completed.")