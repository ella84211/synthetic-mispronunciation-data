import json
import random
import os

"""
P_SENTENCE_ERRORS   are the weights for how many errors a sentence has;
                    the default is that a sentence has 0.8 chance of having 0 errors,
                    0.1 chance of having 1 error, 0.05 chance of having 2 errors, etc.
P_DELETION  is the probability that an error is a deletion.
            the default is that once the position for an error is selected,
            there is 0.25 chance that the error is a deletion.

"""

TEST_FILE = "data/real_data/test.json"
RESULTS_FILE = "baselines/predictions/random_bias_80_results.json"
P_SENTENCE_ERRORS = [0.8, 0.1, 0.05, 0.025, 0.025]
PHONES = ["ɡ", "ɔɪ", "aʊ", "ʃ", "ɪ", "dʒ", "ʊ", "ɑ", "b", "i", "æ", "ɝ", "n", "θ", "aɪ", "h", "v", "ɔ", "tʃ", "ɹ", "s", "u", "p", "f", "z", "j", "oʊ", "ʒ", "d", "ð", "ŋ", "k", "w", "m", "l", "ɛ", "eɪ", "t", "ʌ"]
P_DELETION = 0.25
output_dir = "baselines/predictions"
os.makedirs(output_dir, exist_ok=True)


with open(TEST_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

predictions = []
for sample in data:
    words = sample["words"]

    # Use the weights to choose how many errors the sentence has
    num_sentence_errors = random.choices([0, 1, 2, 3, 4], weights=P_SENTENCE_ERRORS, k=1)[0]
    num_phones = sum(len(word["phones"]) for word in words)
    if num_phones < num_sentence_errors:
        num_sentence_errors = num_phones
    
    # Randomly choose the phone error indexes
    phone_error_indexes = set(random.sample(range(0, num_phones), num_sentence_errors))

    pred_phone_errors = []
    pred_word_errors = []
    prediction = []

    phone_index = 0
    for word in words:
        error = False
        for phone in word["pronounced"]:
            if phone_index in phone_error_indexes:
                error = True

                # Now that we know this phone has an error, we use P_DELETION to decide if the error is a deletion
                deletion = random.choices([0, 1], weights=[1-P_DELETION, P_DELETION], k=1)[0]
                replacement = random.choices(PHONES, k=1)[0]
                if deletion:
                    pred_phone_errors.append(0)
                    prediction.append(phone)
                    prediction.append(replacement)
                else:
                    prediction.append(replacement)
                    pred_phone_errors.append(1)
            else:
                pred_phone_errors.append(0)
                prediction.append(phone)
            phone_index += 1
        pred_phone_errors.append("<|>")
        prediction.append("<|>")
        if error: pred_word_errors.append(1)
        else: pred_word_errors.append(0)
    pred_phone_errors = pred_phone_errors[:-1]
    prediction = prediction[:-1]
    predicted_sample = {
        "id": sample["id"],
        "predicted sentence errors": num_sentence_errors,
        "predicted word errors": pred_word_errors,
        "predicted phone errors": pred_phone_errors,
        "sentence errors": sample["sentence errors"],
        "word errors": sample["word errors"],
        "phone errors": sample["phone errors"],
        "prediction": prediction,
        "pronounced": sample["pronounced"],
        "reference": sample["reference"],
    }
    predictions.append(predicted_sample)

with open(RESULTS_FILE, "w", encoding="utf-8") as file:
    json.dump(predictions, file, ensure_ascii=False, indent=2)
