import json
import time
from sklearn.model_selection import train_test_split

TRAIN_SPLIT = 0.7

# Load pre-split datasets from JSON files
print("\n📊 Data Loading and Preparation")
print("=" * 40)

print(f"Loading datasets from JSONL file...")
start_time = time.time()

# Load datasets from JSONL file
data_path = "./dataset/abduction.jsonl"

all_data = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        all_data.append(json.loads(line.strip()))

print(f"Total examples loaded: {len(all_data):,}")

# Split the raw data (before transformation)
# First split: separate out training data 
train_data, temp_data = train_test_split(
    all_data,
    train_size=TRAIN_SPLIT, 
    random_state=42,
    shuffle=True
)

# Second split: split remaining into validation and test
val_data, test_data = train_test_split(
    temp_data, 
    train_size=0.5,  # 50% of the remaining
    random_state=42,
    shuffle=True
)

# Save raw splits to JSON files
print("\n💾 Saving raw splits to JSON files...")
with open('./dataset/train_split.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open('./dataset/val_split.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

with open('./dataset/test_split.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

load_time = time.time() - start_time
print(f"\n✅ Data loaded, split, and saved in {load_time:.2f} seconds")

total = len(train_data) + len(val_data) + len(test_data)
print(f"\n📈 Split Statistics:")
print(f"   Total samples: {total:,}")
print(f"   Training samples: {len(train_data):,} ({len(train_data)/total*100:.1f}%)")
print(f"   Validation samples: {len(val_data):,} ({len(val_data)/total*100:.1f}%)")
print(f"   Test samples: {len(test_data):,} ({len(test_data)/total*100:.0f}%)")
