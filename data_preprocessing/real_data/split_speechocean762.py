import json
import random

INPUT_FILE = "data/real_data/transcriptions.json"
TRAIN_FILE = "data/real_data/train.json"
VALIDATE_FILE = "data/real_data/validate.json"
TEST_FILE = "data/real_data/test.json"
SEED = 3407

# The error numbers to keep track of, the last bucket will be all numbers not already included
# For example: buckets = 4, the buckets will be 0, 1, 2, and >= 3
BUCKETS = 6

random.seed(SEED)
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

ten_percent = len(data)/10
error_nums = {}

for sample in data:
    num_errors = sum(1 for word in sample["words"] if word["mispronunciations"])
    if num_errors not in error_nums: error_nums[num_errors] = []
    error_nums[num_errors].append(sample)

"""for num in sorted(error_nums.keys()):
    print(num, len(error_nums[num]))"""

buckets = {}
for num in sorted(error_nums.keys()):
    if num < BUCKETS:
        buckets[num] = error_nums[num]
        random.shuffle(buckets[num])
    else:
        buckets[BUCKETS-1] += error_nums[num]
        random.shuffle(buckets[BUCKETS-1])

train_set = []
validate_set = []
test_set = []

train_num = round(ten_percent*8)
validate_num = round(ten_percent)

nums = list(buckets.keys())
for num in nums:
    samples = buckets[num]
    if num == nums[-1]:
        a = train_num-len(train_set)
        b = a + validate_num-len(validate_set)
    else:
        ten_percent = len(samples)/10
        a = round(ten_percent*8)
        b = a+round(ten_percent)
    train_set += samples[:a]
    validate_set += samples[a:b]
    test_set += samples[b:]

random.shuffle(train_set)
random.shuffle(validate_set)
random.shuffle(test_set)

with open(TRAIN_FILE, "w", encoding="utf-8") as file:
    json.dump(train_set, file, ensure_ascii=False, indent=2)
with open(VALIDATE_FILE, "w", encoding="utf-8") as file:
    json.dump(validate_set, file, ensure_ascii=False, indent=2)
with open(TEST_FILE, "w", encoding="utf-8") as file:
    json.dump(test_set, file, ensure_ascii=False, indent=2)
    
print(f"Train: {len(train_set)}\nValidate: {len(validate_set)}\nTest: {len(test_set)}")
