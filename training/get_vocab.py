import json
from collections import defaultdict

DATA_FILES = [
    "data/real_data/train.json",
    "data/real_data/validate.json",
    "data/real_data/test.json"
]

OUTPUT_PATH = "training/vocab.json"

PAD           = "<PAD>"
UNK           = "<UNK>"
WORD_BOUNDARY = "<|>"
GAP           = "<GAP>"
SPECIAL_TOKENS = [PAD, UNK, WORD_BOUNDARY, GAP]

def build_vocab(data_files: list, output_path: str):
    counts: dict = defaultdict(int)

    for path in data_files:
        print(f"  Reading {path} ...")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for sentence in data:
            for tok in sentence.get("pronounced", []):
                counts[tok] += 1
            for tok in sentence.get("reference", []):
                counts[tok] += 1

    token2idx: dict = {}

    # Special tokens first so their indices are always stable
    for tok in SPECIAL_TOKENS:
        token2idx[tok] = len(token2idx)

    # All corpus tokens in sorted order for reproducibility
    for tok in sorted(counts.keys()):
        if tok not in token2idx:
            token2idx[tok] = len(token2idx)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(token2idx, f, ensure_ascii=False, indent=2)

    # Summary
    special = set(SPECIAL_TOKENS)
    ipa_tokens = [t for t in token2idx if t not in special]
    print(f"\nVocabulary built successfully.")
    print(f"  Total tokens  : {len(token2idx)}")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"  IPA tokens    : {len(ipa_tokens)}")
    print(f"  Saved to      : {output_path}")


if __name__ == "__main__":
    print("Building shared vocabulary from:")
    for p in DATA_FILES:
        print(f"  {p}")
    print()
    build_vocab(DATA_FILES, OUTPUT_PATH)