import json
import csv
import argparse


# ----------------------------
# Metrics
# ----------------------------

def prf1(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    if sum(y_true) == 0 and sum(y_pred) == 0:
        return 1.0, 1.0, 1.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return precision, recall, f1


def compute_sample_metrics(sample):
    sid = sample["id"]

    # --------------------
    # PHONE LEVEL
    # --------------------
    gt_phone = [int(x) for x in sample["phone errors"] if x != "<|>"]
    pred_phone = [int(x) for x in sample["predicted phone errors"] if x != "<|>"]

    p_p, r_p, f1_p = prf1(gt_phone, pred_phone)

    # --------------------
    # WORD LEVEL
    # --------------------
    gt_word = [1 if int(x) > 0 else 0 for x in sample["word errors"]]
    pred_word = [1 if int(x) > 0 else 0 for x in sample["predicted word errors"]]

    p_w, r_w, f1_w = prf1(gt_word, pred_word)

    # --------------------
    # COMBINED SCORE
    # --------------------
    combined_f1 = 0.5 * f1_w + 0.5 * f1_p
    difficulty = 1.0 - combined_f1

    return {
        "id": sid,

        # phone
        "phone_precision": p_p,
        "phone_recall": r_p,
        "phone_f1": f1_p,

        # word
        "word_precision": p_w,
        "word_recall": r_w,
        "word_f1": f1_w,

        # combined
        "combined_f1": combined_f1,
        "difficulty": difficulty,

        # diagnostics
        "n_phones": len(gt_phone),
        "n_words": len(gt_word),
        "gt_sentence_errors": sample.get("sentence errors", -1),
        "pred_sentence_errors": sample.get("predicted sentence errors", -1),
        "reference": extract_transcription(sample["reference"]),
        "pronounced": extract_transcription(sample["pronounced"]),
        "prediction": extract_transcription(sample["prediction"]),
        "phone errors": extract_transcription(sample["phone errors"]),
        "predicted phone errors": extract_transcription(sample["predicted phone errors"])
    }


# ----------------------------
# pipeline
# ----------------------------

def run(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = [compute_sample_metrics(s) for s in data]

    fields = [
        "id",

        "phone_f1",
        "word_f1",

        "phone_precision",
        "phone_recall",
        "word_precision",
        "word_recall",

        "combined_f1",
        "difficulty",

        "n_phones",
        "n_words",
        "gt_sentence_errors",
        "pred_sentence_errors",
        "reference",
        "pronounced",
        "prediction",
        "phone errors",
        "predicted phone errors"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} samples to {output_path}")


extract_transcription = lambda transcription: "".join(str(symbol) if symbol != "<|>" else " " for symbol in transcription)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run(args.input, args.output)