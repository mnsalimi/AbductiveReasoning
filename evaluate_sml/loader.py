from datasets import load_dataset, Dataset
from typing import Iterator, Dict, Any
import json
import yaml

def load_med_qa_dataset() -> Iterator[Dict[str, Any]]:
    """
    Loads MedQA US 4 options test split from the path specified in the config.
    
    Returns an iterator on the samples, which is memory-efficient.
    """
    with open("evaluate_sml/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    file_path = config["datasets"]["medqa"]["file_path"]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

# class DatasetLoader:
#     def __init__(self):
#         self.dataset_name = dataset_name
#         self.split = split
#         try:
#             self.dataset: Dataset = load_dataset(self.dataset_name, split = self.split)
#             print(f"✅ Successfully loaded '{self.dataset_name}' split '{self.split}'.")
#         except Exception as e:
#             print(f"❌ Failed to load dataset: {e}")
#             self.dataset = None

#     def __len__(self) -> int:
#         """Returns the number of samples in the dataset."""
#         return len(self.dataset) if self.dataset else 0

#     def __getitem__(self, index: int) -> Dict[str, Any]:
#         """Allows accessing a sample by its index, e.g., loader[5]."""
#         if self.dataset:
#             return self.dataset[index]
#         raise IndexError("Dataset is not loaded.")

#     def __iter__(self):
#         """Allows iterating over the dataset, e.g., for sample in loader:"""
#         return iter(self.dataset)


# if __name__ == "__main__":
#     loader = DatasetLoader("zou-lab/MedCaseReasoning", "train")
#     print(loader[0])