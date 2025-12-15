# dataset/build_mixed_splits.py

import json
import os
import random
from typing import Any, Dict, List, Tuple

DATASET_NAMES = [
    "balanced_copa_cause_only",
    "UniADILR",
    "list_function",
    "miniarc",
    "climate_fever",
    "causelogics_level3&4",
]

TRAIN_PER_DATASET = 380
VAL_PER_DATASET = 100
SEED = 42

DATA_DIR = "dataset"
OUT_TRAIN = os.path.join(DATA_DIR, "train_split.json")
OUT_VAL = os.path.join(DATA_DIR, "val_split.json")


def _load_json_array(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def _normalize_entry(entry: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    e = dict(entry)

    # Replace old key for causelogics
    if "dataset_name" in e:
        e.pop("dataset_name")

    # Always add / overwrite
    e["datasetName"] = dataset_name
    return e


def _sample_upto(
    data: List[Dict[str, Any]],
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Sample up to k elements.
    If data has fewer than k items, return all of them.
    """
    if len(data) <= k:
        return list(data)
    return rng.sample(data, k)


def _build_split(split_suffix: str, k_per_dataset: int, rng: random.Random) -> List[Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    counts: List[Tuple[str, int]] = []

    for ds_name in DATASET_NAMES:
        path = os.path.join(DATA_DIR, f"{ds_name}_{split_suffix}.json")
        data = _load_json_array(path)

        picked = _sample_upto(data, k_per_dataset, rng)
        picked = [_normalize_entry(x, ds_name) for x in picked]

        combined.extend(picked)
        counts.append((ds_name, len(picked)))

    rng.shuffle(combined)

    print(f"\nBuilt {split_suffix}: total={len(combined)}")
    for ds_name, c in counts:
        print(f"  - {ds_name}: {c}")

    return combined


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = random.Random(SEED)

    train = _build_split("train", TRAIN_PER_DATASET, rng)
    val = _build_split("val", VAL_PER_DATASET, rng)

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with open(OUT_VAL, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"\n✓ Wrote {OUT_TRAIN} (n={len(train)})")
    print(f"✓ Wrote {OUT_VAL}   (n={len(val)})")

    # Sanity check
    from collections import Counter
    print("\nDatasetName distribution:")
    print("  train:", dict(Counter(x["datasetName"] for x in train)))
    print("  val:  ", dict(Counter(x["datasetName"] for x in val)))


if __name__ == "__main__":
    main()
