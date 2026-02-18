from typing import Dict, Any, List, Set, Optional
import os
import yaml
import random

def load_dataset(dataset_name: str, n_samples: int = -1, specific_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
    """
    Load samples from the dataset of choice.

    Args:
        dataset_name: Name of the dataset
        n_samples: Number of samples to load (-1 for all, 0 for none, >0 for specific count).
                   If -1, will use the value from experiments_config.yaml. This is ignored if specific_ids is provided.
        specific_ids: A set of specific sample IDs to load. If provided, n_samples is ignored.

    Returns:
        List of dictionaries containing the dataset samples
    """
    if n_samples == -1 and not specific_ids:
        try:
            config_path = "configs/experiments_config.yaml"
            experiments_config = None
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    experiments_config = yaml.safe_load(f)
            if experiments_config and "dataset" in experiments_config:
                n_samples = experiments_config["dataset"].get("max_samples", -1)
                print(f"Using n_samples={n_samples} from configuration")
            else:
                print("No dataset configuration found, using all samples")
        except Exception as e:
            print(f"Warning: Could not load configuration for dataset limits: {e}")
            print("Using all samples as fallback")

    if dataset_name == "sample":
        return load_dataset(n_samples, specific_ids)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


def _get_sampling_config() -> tuple[int, bool]:
    """
    Get random sampling configuration from datasets_config.yaml.
    """
    try:
        config_path = "configs/datasets_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                datasets_config = yaml.safe_load(f)
                loading_config = datasets_config["loading"]
                random_seed = loading_config.get("random_seed", 42)
                shuffle_samples = loading_config.get("shuffle_samples", True)
                return random_seed, shuffle_samples
    except Exception as e:
        print(f"Warning: Could not load sampling configuration: {e}")
    return 42, True


def _sample_data(data: List[Dict[str, Any]], n_samples: int) -> List[Dict[str, Any]]:
    """
    Sample data with optional shuffling based on configuration.
    """
    if n_samples == -1:
        return data
    elif n_samples == 0:
        return []
    elif n_samples > 0:
        random_seed, shuffle_samples = _get_sampling_config()
        if shuffle_samples and len(data) > n_samples:
            random.seed(random_seed)
            sampled_data = random.sample(data, min(n_samples, len(data)))
            print(f"Randomly sampled {len(sampled_data)} samples using seed {random_seed}")
            return sampled_data
        else:
            return data[:n_samples]
    else:
        raise ValueError(f"n_samples must be -1 (all), 0 (none), or positive integer, got {n_samples}")


def load_dataset(n_samples: int = -1, specific_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
    data = []
    
    if specific_ids:
        print(f"Filtering dataset to {len(specific_ids)} specific IDs.")
        filtered_data = [item for item in data if item['id'] in specific_ids]
        filtered_data.sort(key=lambda x: x['id']) # Ensure consistent order
        return _sample_data(filtered_data, n_samples)
    
    return _sample_data(data, n_samples)
