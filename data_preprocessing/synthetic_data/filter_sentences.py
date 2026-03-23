import json
import random


REAL_DATA_FILE = "data/real_data/real_transcriptions.json"
SYNTHETIC_DATA_FILE = "data/synthetic_data/sentences.json"
OUTPUT_FILE = "data/synthetic_data/filtered_sentences.json"

MIN_WORDS = 4
MAX_WORDS = 7
SETS = 5
NUM_PER_SET = 4000
SEED = 3407

skips = [
    "simplified",
    "becomes",
    "=",
    '"',
    "yummy",
    "pretty",
    "tasty",
    "sit",
    "colored",
    "building",
    "giraffe",
    "rhino",
    "lots",
    "fence",
    "tiny",
    "shiny",
    "clouds",
    "leaves",
    "fluffy",
    "tiny",
    "wear",
    "stick",
    "stripe",
    "sheep",
    "umbrella",
    "yucky",
]

with open(REAL_DATA_FILE, "r", encoding="utf-8") as file:
    real_sentences = json.load(file)
with open(SYNTHETIC_DATA_FILE, "r", encoding="utf-8") as file:
    sentences = json.load(file)

real_sentences = set(["".join(c for c in sentence["text"].upper() if (c.isalnum() or c.isspace() or c == "'")) for sentence in real_sentences])
for sentence in sentences:
    sentence["sentence"] = "".join(c for c in sentence["sentence"].upper() if (c.isalnum() or c.isspace() or c == "'"))
sentences = [sentence for sentence in sentences if sentence["sentence"].upper() not in real_sentences]
random.shuffle(sentences)

filtered_sentences = []
for sentence in sentences:
    if not MIN_WORDS <= len(sentence["sentence"].strip().split(" ")) <= MAX_WORDS:
        continue
    skipping = False
    for word in skips:
        if word.upper() in sentence["sentence"]:
            skipping = True
    if skipping:
        continue
    filtered_sentences.append(sentence)

if len(filtered_sentences) < SETS*NUM_PER_SET:
    exit(f"{SETS*NUM_PER_SET} needed, {len(filtered_sentences)} found; reduce skip words")

print("Sentences without skip words:", len(filtered_sentences))

pools = []
index = 0
for i in range(1, SETS+1):
    old_index = index
    index = round(i/SETS*len(filtered_sentences))+1
    pools.append(filtered_sentences[old_index:index])

pools = [pool[:NUM_PER_SET] for pool in pools]

new_samples = []
for i in range(len(pools)):
    for sample in pools[i]:
        sample["set"] = i+1
        new_samples.append(sample)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(new_samples, file, ensure_ascii=False, indent=2)
