from datasets import load_dataset
import json
import os

DIRECTORY = "data/synthetic_data"
OUTPUT_FILE = "data/synthetic_data/sentences.json"
DATASET_HF = "Jotschi/visual_genome-simple-en"
SPLIT = "train"
os.makedirs(DIRECTORY, exist_ok=True)

print(f"Loading dataset from {DATASET_HF}")
ds = load_dataset(DATASET_HF, split=SPLIT)

print("Extracting sentences")
data = []
for index, row in enumerate(ds):
    data.append({
        "id": index,
        "sentence": row["caption"]
    })

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

print(f"Writing to {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=2)
print(f"{OUTPUT_FILE} written to successfully")