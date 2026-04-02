import json
import os


TEST_FILE = "data/real_data/test.json"
RESULTS_FILE = "baselines/predictions/majority_class_results.json"
output_dir = "baselines/predictions"
os.makedirs(output_dir, exist_ok=True)


with open(TEST_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

predictions = []
for sample in data:
    words = sample["words"]

    # 0 means no error, so this makes the prediction all 0s
    pred_phone_errors = [0 for _ in range(len(words[0]["pronounced"]))]
    for word in words[1:]:
        pred_phone_errors.append("<|>")
        pred_phone_errors += [0 for _ in range(len(word["pronounced"]))]
        
    prediction = {
        "id": sample["id"],
        "predicted sentence errors": 0,
        "predicted word errors": [0 for _ in range(len(sample["words"]))],
        "predicted phone errors": pred_phone_errors,
        "sentence errors": sample["sentence errors"],
        "word errors": sample["word errors"],
        "phone errors": sample["phone errors"],
        "prediction": sample["pronounced"],
        "pronounced": sample["pronounced"],
        "reference": sample["reference"],
    }
    predictions.append(prediction)

with open(RESULTS_FILE, "w", encoding="utf-8") as file:
    json.dump(predictions, file, ensure_ascii=False, indent=2)
