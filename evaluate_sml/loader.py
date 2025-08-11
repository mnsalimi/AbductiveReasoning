from datasets import load_dataset
from typing import Dict, Any, List
import json
import yaml

def load_med_qa_dataset(n_samples: int = -1) -> List[Dict[str, Any]]:
    """
    Loads MedQA US 4 options test split from the path specified in the config.
    
    Returns a list of samples (rows).
    """
    with open("evaluate_sml/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    file_path = config["datasets"]["medqa"]["file_path"]
    l = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            l.append(json.loads(line))
    if n_samples == -1:
        return l
    else:
        return l[:n_samples]

def load_med_mcqa_dataset(n_samples: int = -1) -> List[Dict[str, Any]]:
    """
    Loads MedMCQA validation split from the Hugging Face path specified in the config.
    
    Returns a list of samples (rows).
    """
    with open("evaluate_sml/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    hf_path = config["datasets"]["medmcqa"]["hf_path"]

    ds = load_dataset(hf_path, split="validation")

    if n_samples == -1:
        return list(ds)
    else:
        return list(ds)[:n_samples]


def load_uniadilr_hgc_dataset(n_samples: int = -1) -> List[Dict[str, Any]]:
    """
    Loads UniADILR-HGc dataset from the path specified in the config.
    
    Returns a list of samples (rows).
    """
    with open("evaluate_sml/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    file_path = config["datasets"]["uniadilr"]["file_path"]
    l = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            l.append(json.loads(line))
    if n_samples == -1:
        return l
    else:
        return l[:n_samples]

if __name__ == "__main__":
    print(load_uniadilr_hgc_dataset(n_samples=1))