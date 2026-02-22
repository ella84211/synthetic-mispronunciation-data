import json
import random

INPUT_FILE = "real_data/real_transcriptions.json"
TRAIN_FILE = "real_data/train_real.json"
VALIDATE_FILE = "real_data/validate_real.json"
TEST_FILE = "real_data/test_real.json"
SEED = 3407

random.seed(SEED)
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

random.shuffle(data)
ten_percent = len(data)/10

train_set = []
validate_set = []
test_set = []
index = 0

for _ in range(round(ten_percent*8)):
    train_set.append(data[index])
    index += 1
for _ in range(round(ten_percent)):
    validate_set.append(data[index])
    index += 1
while index < len(data):
    test_set.append(data[index])
    index += 1

with open(TRAIN_FILE, "w", encoding="utf-8") as file:
    json.dump(train_set, file, ensure_ascii=False, indent=2)
with open(VALIDATE_FILE, "w", encoding="utf-8") as file:
    json.dump(validate_set, file, ensure_ascii=False, indent=2)
with open(TEST_FILE, "w", encoding="utf-8") as file:
    json.dump(test_set, file, ensure_ascii=False, indent=2)
    
print(f"Train: {len(train_set)}\nValidate: {len(validate_set)}\nTest: {len(test_set)}")