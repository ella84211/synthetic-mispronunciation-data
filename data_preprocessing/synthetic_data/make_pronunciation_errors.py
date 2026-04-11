import json
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from format_data import format_dataset

"""
This script injects the synthetic errors into the transcriptions
The oversample factor increases the probability that a sentence has an error
Usage:
    Default oversample factor: python make_pronunciation_errors.py output_directory
    Custom oversample factor:  python make_pronunciation_errors.py output_directory oversample_factor
"""
oversample_factor = 1
if len(sys.argv) == 1:
    OUTPUT_DIR = sys.argv[1]
else:
    OUTPUT_DIR = "data/synthetic_data"
if len(sys.argv) > 2:
    try:
        oversample_factor = float(sys.argv[2])
    except ValueError:
        sys.exit("Usage: python make_pronunciation_errors.py output_directory oversample_factor")

INPUT_FILE = "data/synthetic_data/transcriptions_set*.json"
OUTPUT_FILE = f"{OUTPUT_DIR}/synthetic_transcriptions_set*.json"
REAL_DATA = "data/real_data/train.json"
SETS = 5
SEED = 3407
SENTENCE_OVERSAMPLE_FACTOR = oversample_factor

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(REAL_DATA, "r", encoding="utf-8") as file:
    data = json.load(file)

sentences_with_errors = 0
sentences = len(data)
num_errors = {}
positions = {}
phone_error_counts = {}
word_error_counts = {}
total_errors = 0
for sample in data:
    words_with_errors = 0
    error = False
    for word in sample["words"]:
        if word["mispronunciations"]:
            n = len(word["mispronunciations"])
            if n not in word_error_counts: word_error_counts[n] = 0
            word_error_counts[n] += 1
            error = True
            words_with_errors += 1
            for error in word["mispronunciations"]:
                if len(word["phones"]) < 2: continue
                total_errors += 1
                position = (error["index"]+1) / len(word["phones"])
                if position not in positions: positions[position] = 0
                positions[position] += 1
                if error["canonical"] not in phone_error_counts: phone_error_counts[error["canonical"]] = 0
                phone_error_counts[error["canonical"]] += 1
    if error:
        sentences_with_errors += 1
        errors = words_with_errors
        if errors not in num_errors: num_errors[errors] = 0
        num_errors[errors] += 1


# get substitution counts
phone_substitutions = {}

for sample in data:
    for word in sample["words"]:
        for phone in word["phones"]:
            if phone not in phone_substitutions: phone_substitutions[phone] = {"occurences": 0, "mispronounced": 0, "replaced by": {}}
            phone_substitutions[phone]["occurences"] += 1
        if word["mispronunciations"]:
            for error in word["mispronunciations"]:
                phone_substitutions[error["canonical"]]["mispronounced"] += 1
                if error["pronounced"] not in phone_substitutions[error["canonical"]]["replaced by"]:
                    phone_substitutions[error["canonical"]]["replaced by"][error["pronounced"]] = 0
                phone_substitutions[error["canonical"]]["replaced by"][error["pronounced"]] += 1


# p_num_words is probabilities of how many words with errors a sentence has (given that it does have at least one error)
p_sentence = sentences_with_errors/sentences*SENTENCE_OVERSAMPLE_FACTOR
sents_with_errors = sum(num_errors.values())
words_with_errors = sum(k*num_errors[k] for k in num_errors)
p_num_words = {k: num_errors[k]/sents_with_errors for k in sorted(num_errors.keys())}
p_phone_error = {k: phone_error_counts[k]/total_errors for k in phone_error_counts}
p_position = {k: positions[k]/total_errors for k in sorted(positions.keys())}
p_errors_per_word = {k: word_error_counts[k]/words_with_errors for k in word_error_counts}

# once we know the words, we need to figure out how many errors each has and where to put them (and what they are)
def get_errors(phones):
    def get_position():
        # p_phones is {index: probability of error}
        p_phones = {}
        for i, phone in enumerate(phones):
            phone_error = 0
            if phone in p_phone_error:
                phone_error = phone_substitutions[phone]["mispronounced"] / phone_substitutions[phone]["occurences"]
            position = (i+1)/len(phones)
            if position in p_position: 
                position_error = p_position[position]
            else:
                prev_key = None
                next_key = None
                position_error = None
                for key in p_position:
                    if key > position:
                        next_key = key
                        break
                    prev_key = key
                if not prev_key:
                    position_error = p_position[key]
                if not next_key:
                    position_error = p_position[prev_key]
                if not position_error:
                    position_error = (position-prev_key)/(next_key-prev_key)*(p_position[next_key]-p_position[prev_key])+p_position[prev_key]
            p_phones[i] = phone_error*position_error
        probs = sum(p_phones.values())
        p_phones = {k: p_phones[k]/probs for k in p_phones}
        return random.choices(list(p_phones.keys()), weights=p_phones.values(), k=1)[0]
    
    num_errors = random.choices(list(p_errors_per_word.keys()), weights=p_errors_per_word.values(), k=1)[0]
    num_errors = min(num_errors, len(phones)-1)
    if num_errors == 0: num_errors = 1

    substitutions = {}
    error_positions = []
    for _ in range(num_errors):
        position = get_position()
        while position in error_positions:
            position = get_position()
        error_positions.append(position)

    for position in error_positions:
        expected = phones[position]
        if expected not in phone_substitutions:
            substitutions[position] = "<UNK>"
            continue
        info = phone_substitutions[expected]
        if info["mispronounced"] == 0:
            substitutions[position] = "<UNK>"
            continue
        weights = [info["replaced by"][phone]/info["mispronounced"] for phone in info["replaced by"].keys()]
        substitution = random.choices(list(info["replaced by"].keys()), weights=weights, k=1)[0]
        substitutions[position] = substitution

    return substitutions

def main():
    for i in range(1, SETS+1):
        input_file = INPUT_FILE.replace("*", str(i))
        output_file = OUTPUT_FILE.replace("*", str(i))

        with open(input_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        synthetic_data = []
        for sample in data:
            error = random.choices([0, 1], weights=[1-p_sentence, p_sentence], k=1)[0]
            for word in sample["words"]:
                word["mispronunciations"] = []
            if not error:
                synthetic_data.append(sample)
                continue
            num_erroneous_words = random.choices(list(p_num_words.keys()), weights=p_num_words.values(), k=1)[0]
            if num_erroneous_words >= len(sample["words"]):
                num_erroneous_words = len(sample["words"])-1
            if num_erroneous_words == 0:
                num_erroneous_words = 1
            erroneous_words = random.sample(range(0, len(sample["words"])), num_erroneous_words)
            for i in erroneous_words:
                errors = get_errors(sample["words"][i]["phones"])
                word = sample["words"][i]
                for error in errors:
                    sample["words"][i]["mispronunciations"].append({"canonical": word["phones"][error], "index": error, "pronounced": errors[error]})
            synthetic_data.append(sample)

        formatted_data = format_dataset(synthetic_data)
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(formatted_data, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()