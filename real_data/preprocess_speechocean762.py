import json
from phonecodes import phonecodes


INPUT_FILE = "real_data/scores.json"
OUTPUT_FILE = "real_data/real_transcriptions.json"

print(f"Reading data from {INPUT_FILE}")
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

def convert_phone(phone):
    if phone == "<DEL>": return phone
    if phone == "<unk>": return phone.upper()
    return phonecodes.arpabet2ipa(phone.rstrip("012"), "eng")

print(f"Processing data...")
processed_data = []
for new_id, assigned_id in enumerate(data):
    transcription = data[assigned_id]
    new_transcription = {
        "id": new_id,
        "accuracy": transcription["accuracy"],
        "completeness": transcription["completeness"],
        "text": transcription["text"],
        "words": []
    }
    for word in transcription["words"]:
        new_word = {
            "accuracy": word["accuracy"],
            "text": word["text"],
            "phones": [],
            "phones-accuracy": word["phones-accuracy"]
        }
        for phone in word["phones"]:
            new_word["phones"].append(convert_phone(phone))
        new_transcription["words"].append(new_word)
        new_word["mispronunciations"] = []
        if "mispronunciations" in word:
            for set_error in word["mispronunciations"]:
                new_set = {
                    "canonical": convert_phone(set_error["canonical-phone"]),
                    "index": set_error["index"],
                    "pronounced": convert_phone(set_error["pronounced-phone"])
                }
                new_word["mispronunciations"].append(new_set)
    processed_data.append(new_transcription)

print(f"Writing to {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(processed_data, file, ensure_ascii=False, indent=2)
print(f"{OUTPUT_FILE} written to successfully")