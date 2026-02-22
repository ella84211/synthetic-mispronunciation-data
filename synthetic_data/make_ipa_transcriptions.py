import json
import nltk
from nltk.corpus import cmudict
from phonecodes import phonecodes
from g2p_en import G2p
from tqdm import tqdm


INPUT_FILE = "synthetic_data/sentences.json"
OUTPUT_FILE = "synthetic_data/ipa_transcriptions.json"

nltk.download('cmudict')
nltk.download('averaged_perceptron_tagger_eng')
arpabet = cmudict.dict()
g2p = G2p()

convert_phone = lambda phone: phonecodes.arpabet2ipa(phone.rstrip("012"), "eng")

print(f"Reading data from {INPUT_FILE}")
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Processing data")
samples = []
for sentence in tqdm(data):
    sample = {
        "id": sentence["id"],
        "text": "",
        "words": []
    }
    sentence = sentence["sentence"]
    sentence = ''.join(c for c in sentence if c.isalnum() or c.isspace()).upper()
    sample["text"] = sentence
    for word in sentence.split(" "):
        word_sample = {"text": word, "phones": []}
        word = word.lower()
        if word in arpabet:
            word = arpabet[word][0]
        else:
            word = g2p(word)
        for phone in word:
            phone = convert_phone(phone)
            word_sample["phones"].append(phone)
        sample["words"].append(word_sample)
    samples.append(sample)

print(f"Writing data to {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(samples, file, ensure_ascii=False, indent=2)
print(f"{OUTPUT_FILE} written to successfully")