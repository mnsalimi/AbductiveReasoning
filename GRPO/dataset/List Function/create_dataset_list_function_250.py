#!/usr/bin/env python3
"""
Create List Function 250 train/val JSON files.

Behavior:
- Downloads the list_function_250 repository archive directly from GitHub
- Uses all JSON files in the repository's json/ directory as the source dataset
- Randomly samples exactly N examples from the full dataset
- Converts each source file into the expected task format:
  {
    "idx": <int>,
    "train": [{"input": "[...]", "output": "[...]"}, ...],
    "test":  [{"input": "[...]", "output": "[...]"}, ...]
  }
- Splits the sampled examples into train and val according to VAL_SIZE
- Reproducible via fixed seed

Note:
- In list_function_250, each JSON file contains a single "data" list.
- The raw files inspected from the repository contain 11 input/output pairs per file,
  so this script uses the first TRAIN_DEMO_COUNT pairs as "train" and the remaining
  pairs as "test".

Outputs written next to this script:
- list_function_train.json
- list_function_val.json
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List


# =========================
# User-configurable values
# =========================
N = 200  # Total number of examples to sample
VAL_SIZE = 40  # Number of validation examples
SEED = 42  # Random seed for reproducibility
TRAIN_DEMO_COUNT = 8  # Number of demonstrations stored in each sample's "train"

DATA_URL = "https://github.com/joshrule/list_function_250/archive/refs/heads/main.zip"

OUTPUT_DIR = Path(__file__).resolve().parent
DOWNLOADED_ARCHIVE = OUTPUT_DIR / "list_function_250_main.zip"
EXTRACTED_DIR = OUTPUT_DIR / "list_function_250_main"
TRAIN_OUTPUT = OUTPUT_DIR / "list_function_train.json"
VAL_OUTPUT = OUTPUT_DIR / "list_function_val.json"

REQUIRED_TOP_LEVEL_FIELDS = [
    "id",
    "program",
    "data",
]

REQUIRED_DATA_FIELDS = [
    "i",
    "o",
]


def log(msg: str) -> None:
    """Log messages to the console."""
    print(msg, flush=True)


def validate_config() -> int:
    """Validate user-configurable values and calculate train size."""
    if N <= 0:
        raise ValueError("N must be positive.")
    if VAL_SIZE <= 0:
        raise ValueError("VAL_SIZE must be positive.")
    if VAL_SIZE >= N:
        raise ValueError("VAL_SIZE must be smaller than N.")
    if TRAIN_DEMO_COUNT <= 0:
        raise ValueError("TRAIN_DEMO_COUNT must be positive.")

    return N - VAL_SIZE


def download_file(url: str, output_path: Path) -> None:
    """Download a file from a URL to a local path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise ValueError(f"Failed to download data. HTTP status: {response.status}")

        data = response.read()

    output_path.write_bytes(data)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract a ZIP archive to a local directory."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


def list_json_files(extracted_root: Path) -> List[Path]:
    """List all JSON files in the repository's json directory."""
    repo_root = extracted_root / "list_function_250-main"
    json_dir = repo_root / "json"

    if not json_dir.exists():
        raise ValueError(f"Could not find json directory: {json_dir}")

    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")

    return json_files


def read_json(path: Path) -> Dict:
    """Read a JSON file into a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        try:
            row = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from {path}: {e}") from e

    if not isinstance(row, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    return row


def validate_source_example(example: Dict, path: Path) -> None:
    """Validate that a source example contains the expected fields."""
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in example:
            raise ValueError(f"{path} is missing required field: {field}")

    if not isinstance(example["data"], list):
        raise ValueError(f"Expected 'data' to be a list in {path}.")

    if len(example["data"]) <= TRAIN_DEMO_COUNT:
        raise ValueError(
            f"Expected more than {TRAIN_DEMO_COUNT} data pairs in {path}, "
            f"but found {len(example['data'])}."
        )

    for datum in example["data"]:
        if not isinstance(datum, dict):
            raise ValueError(f"Expected each datum in {path} to be a dictionary.")

        for field in REQUIRED_DATA_FIELDS:
            if field not in datum:
                raise ValueError(f"A datum in {path} is missing required field: {field}")

        if not isinstance(datum["i"], list):
            raise ValueError(f"Expected datum['i'] to be a list in {path}.")
        if not isinstance(datum["o"], list):
            raise ValueError(f"Expected datum['o'] to be a list in {path}.")


def format_io_pair(datum: Dict) -> Dict[str, str]:
    """Convert one source datum into the expected input/output string format."""
    return {
        "input": json.dumps(datum["i"]),
        "output": json.dumps(datum["o"]),
    }


def convert_example(example: Dict, idx: int) -> Dict:
    """Convert one repository JSON file into the target sample format."""
    data = example["data"]

    train_pairs = [format_io_pair(datum) for datum in data[:TRAIN_DEMO_COUNT]]
    test_pairs = [format_io_pair(datum) for datum in data[TRAIN_DEMO_COUNT:]]

    if not train_pairs:
        raise ValueError("Converted example has an empty 'train' split.")
    if not test_pairs:
        raise ValueError("Converted example has an empty 'test' split.")

    return {
        "idx": idx,
        "train": train_pairs,
        "test": test_pairs,
    }


def convert_examples(json_files: List[Path]) -> List[Dict]:
    """Convert source JSON files into the target task format."""
    output: List[Dict] = []

    for idx, path in enumerate(json_files):
        row = read_json(path)
        validate_source_example(row, path)
        output.append(convert_example(row, idx=idx))

    return output


def write_json(path: Path, data: List[Dict]) -> None:
    """Write data to a JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Main function to create train and validation splits."""
    train_size = validate_config()

    log("Configuration:")
    log(f"  DATA_URL = {DATA_URL}")
    log(f"  N = {N}")
    log(f"  VAL_SIZE = {VAL_SIZE}")
    log(f"  TRAIN_SIZE = {train_size}")
    log(f"  TRAIN_DEMO_COUNT = {TRAIN_DEMO_COUNT}")
    log(f"  SEED = {SEED}")
    log("")

    log(f"Downloading data from {DATA_URL} ...")
    download_file(DATA_URL, DOWNLOADED_ARCHIVE)

    log(f"Extracting archive to {EXTRACTED_DIR} ...")
    extract_zip(DOWNLOADED_ARCHIVE, EXTRACTED_DIR)

    log("Collecting JSON files from repository ...")
    json_files = list_json_files(EXTRACTED_DIR)

    if len(json_files) < N:
        raise ValueError(
            f"Requested {N} samples, but only {len(json_files)} JSON files were found."
        )

    log(f"Sampling {N} examples from the full dataset ...")
    rng = random.Random(SEED)
    selected_indices = rng.sample(range(len(json_files)), N)
    sampled_files = [json_files[i] for i in selected_indices]

    log("Converting sampled examples ...")
    sampled_rows = convert_examples(sampled_files)

    log("Shuffling sampled examples ...")
    rng.shuffle(sampled_rows)

    for idx, row in enumerate(sampled_rows):
        row["idx"] = idx

    train_samples = sampled_rows[:train_size]
    val_samples = sampled_rows[train_size:]

    if len(train_samples) != train_size:
        raise RuntimeError("Unexpected train split size.")
    if len(val_samples) != VAL_SIZE:
        raise RuntimeError("Unexpected validation split size.")

    log(f"Train split complete: {len(train_samples)} samples")
    log(f"Val split complete: {len(val_samples)} samples")
    log("")

    log("Writing output files...")
    write_json(TRAIN_OUTPUT, train_samples)
    write_json(VAL_OUTPUT, val_samples)

    log("Done.")
    log(
        f"Created:\n  {DOWNLOADED_ARCHIVE}\n  {EXTRACTED_DIR}\n  {TRAIN_OUTPUT}\n  {VAL_OUTPUT}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr, flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
