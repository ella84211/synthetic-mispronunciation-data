import json
import nltk
from nltk.corpus import cmudict
from phonecodes import phonecodes
from g2p_en import G2p
from tqdm import tqdm


INPUT_FILE = "data/synthetic_data/filtered_sentences.json"
OUTPUT_FILE = "data/synthetic_data/transcriptions_set*.json"
SETS = 5

nltk.download('cmudict')
nltk.download('averaged_perceptron_tagger_eng')
arpabet = cmudict.dict()
g2p = G2p()

convert_phone = lambda phone: phonecodes.arpabet2ipa(phone.rstrip("012"), "eng")

print(f"Reading data from {INPUT_FILE}")
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Processing data")
samples = [[] for _ in range(SETS)]
for sentence in tqdm(data):
    set = sentence["set"]
    sample = {
        "id": sentence["id"],
        "text": "",
        "words": []
    }
    sentence = sentence["sentence"].upper()
    sentence = "".join(c for c in sentence if (c == "'" or c.isspace() or c.isalnum())).strip()
    sample["text"] = sentence
    for word in sentence.split(" "):
        word_sample = {"text": word, "phones": []}
        if word.startswith("'"):
            word = word[1:]
        if word.endswith("'"):
            word = word[:-1]
        word = word.lower()
        if word in arpabet:
            word = arpabet[word][0]
        else:
            word = g2p(word)
        for phone in word:
            phone = convert_phone(phone)
            if not phone: continue
            word_sample["phones"].append(phone)
        sample["words"].append(word_sample)
    samples[set-1].append(sample)

for i in range(SETS):
    output_file = OUTPUT_FILE.replace("*", str(i+1))
    print(f"Writing data to {output_file}")
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(samples[i], file, ensure_ascii=False, indent=2)
    print(f"{output_file} written to successfully")