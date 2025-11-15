import json
import random

# Configure dataset names here
COPA_DATASET_NAME = "balanced_copa_cause_only"
UNIADILR_DATASET_NAME = "UniADILR"

# -------------------Train Split-----------------------#
print("TEST SPLIT")

# Configure file paths here
COPA_FILE_PATH = 'balanced_copa_cause_only_train.json'
UNIADILR_FILE_PATH = 'UniADILR_train.json'
OUTPUT_FILE_PATH = 'train_split.json'

# Set random seed for reproducibility (optional)
random.seed(42)

# Read COPA JSON file
with open(COPA_FILE_PATH, 'r', encoding='utf-8') as f:
    copa_data = json.load(f)

# Read UniADILR JSON file
with open(UNIADILR_FILE_PATH, 'r', encoding='utf-8') as f:
    uniadilr_data = json.load(f)

# Add datasetName field to each COPA sample
for sample in copa_data:
    sample['datasetName'] = COPA_DATASET_NAME

# Add datasetName field to each UniADILR sample
for sample in uniadilr_data:
    sample['datasetName'] = UNIADILR_DATASET_NAME

# Combine the datasets
combined_data = copa_data + uniadilr_data

# Shuffle the combined data
random.shuffle(combined_data)

# Save the combined and shuffled data to a new file
with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)

# Print statistics
print(f"COPA entries: {len(copa_data)} (labeled as '{COPA_DATASET_NAME}')")
print(f"UniADILR entries: {len(uniadilr_data)} (labeled as '{UNIADILR_DATASET_NAME}')")
print(f"Total combined entries: {len(combined_data)}")
print(f"\nCombined and shuffled data saved to '{OUTPUT_FILE_PATH}'")

# Show a sample entry from each dataset (optional)
print("\n--- Sample COPA entry ---")
copa_sample = next((item for item in combined_data if item['datasetName'] == COPA_DATASET_NAME), None)
if copa_sample:
    print(json.dumps(copa_sample, indent=2, ensure_ascii=False)[:500] + "...")

print("\n--- Sample UniADILR entry ---")
uniadilr_sample = next((item for item in combined_data if item['datasetName'] == UNIADILR_DATASET_NAME), None)
if uniadilr_sample:
    print(json.dumps(uniadilr_sample, indent=2, ensure_ascii=False)[:500] + "...")


# -------------------Val Split-----------------------#
print("VAL SPLIT")

# Configure file paths here
COPA_FILE_PATH = 'balanced_copa_cause_only_val.json'
UNIADILR_FILE_PATH = 'UniADILR_val.json'
OUTPUT_FILE_PATH = 'val_split.json'

# Set random seed for reproducibility (optional)
random.seed(42)

# Read COPA JSON file
with open(COPA_FILE_PATH, 'r', encoding='utf-8') as f:
    copa_data = json.load(f)

# Read UniADILR JSON file
with open(UNIADILR_FILE_PATH, 'r', encoding='utf-8') as f:
    uniadilr_data = json.load(f)

# Add datasetName field to each COPA sample
for sample in copa_data:
    sample['datasetName'] = COPA_DATASET_NAME

# Add datasetName field to each UniADILR sample
for sample in uniadilr_data:
    sample['datasetName'] = UNIADILR_DATASET_NAME

# Combine the datasets
combined_data = copa_data + uniadilr_data

# Shuffle the combined data
random.shuffle(combined_data)

# Save the combined and shuffled data to a new file
with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)

# Print statistics
print(f"COPA entries: {len(copa_data)} (labeled as '{COPA_DATASET_NAME}')")
print(f"UniADILR entries: {len(uniadilr_data)} (labeled as '{UNIADILR_DATASET_NAME}')")
print(f"Total combined entries: {len(combined_data)}")
print(f"\nCombined and shuffled data saved to '{OUTPUT_FILE_PATH}'")

# Show a sample entry from each dataset (optional)
print("\n--- Sample COPA entry ---")
copa_sample = next((item for item in combined_data if item['datasetName'] == COPA_DATASET_NAME), None)
if copa_sample:
    print(json.dumps(copa_sample, indent=2, ensure_ascii=False)[:500] + "...")

print("\n--- Sample UniADILR entry ---")
uniadilr_sample = next((item for item in combined_data if item['datasetName'] == UNIADILR_DATASET_NAME), None)
if uniadilr_sample:
    print(json.dumps(uniadilr_sample, indent=2, ensure_ascii=False)[:500] + "...")
