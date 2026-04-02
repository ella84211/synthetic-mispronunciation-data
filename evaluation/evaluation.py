"""
Mispronunciation Detection Evaluation Pipeline
===============================================

PREDICTION FILE FORMAT
----------------------
A JSON file containing a list of self-contained prediction objects.
Each object must have the following keys:

  "id"                       : int, sentence identifier

  --- Ground truth ---
  "phone errors"             : list, flat phone-level GT labels aligned to
                               the pronounced sequence. Values are 0, 1, or
                               "<|>" (word boundary). 1 = error.
  "word errors"              : list of ints, one per word. Value is the number
                               of errors in that word. Treated as binary
                               (>0 = error) for classification metrics.
  "sentence errors"          : int, GT count of errors in the sentence.
  "reference"                : list, canonical phone sequence with "<|>"
                               word boundary tokens.
  "pronounced"               : list, what was actually said, with "<|>"
                               word boundary tokens.

  --- Predictions ---
  "predicted phone errors"   : list, same structure as "phone errors".
                               Values are 0, 1, or "<|>". Must match the
                               length and boundary positions of "phone errors".
  "predicted word errors"    : list of ints, one per word. Treated as binary.
  "predicted sentence errors": int, predicted error count for the sentence.
  "prediction"               : list, model's predicted canonical phone
                               sequence with "<|>" word boundary tokens.
                               May differ in length from "pronounced" due to
                               predicted deletions/insertions. Aligned against
                               "reference" using edit distance to compute
                               phoneme error rate (PER).

  --- Optional ---
  "deletion_predictions"     : list of [i, j] index pairs (into the flat
                               non-boundary phone sequence) indicating a
                               predicted deletion between positions i and j.
                               Omit or set to [] if not predicting deletions.

USAGE
-----
Single condition:
    python evaluate.py single --pred predictions.json

Compare multiple conditions:
    python evaluate.py compare \\
        --conditions real_only:pred_real.json \\
                     synthetic_only:pred_synth.json \\
                     real_and_synthetic:pred_both.json
"""

import json
import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    balanced_accuracy_score
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sequence splitting on word boundaries
# ---------------------------------------------------------------------------

def split_on_boundaries(seq: list) -> list:
    """
    Split a flat phone sequence (with '<|>' boundary tokens) into a list of
    per-word phone lists, with boundary tokens removed.

    Example:
        ['h', 'i', '<|>', 'w', 'ʌ', 'z'] -> [['h', 'i'], ['w', 'ʌ', 'z']]
    """
    words = []
    current = []
    for token in seq:
        if token == "<|>":
            words.append(current)
            current = []
        else:
            current.append(token)
    if current:
        words.append(current)
    return words


# ---------------------------------------------------------------------------
# Edit distance alignment (for sequence prediction evaluation)
# ---------------------------------------------------------------------------

def edit_distance_alignment(ref: list, hyp: list) -> tuple:
    """
    Align hyp to ref using Levenshtein edit distance with traceback.
    Returns two equal-length lists representing the alignment, where gaps
    are represented by None:

        aligned_ref: ref tokens, with None where hyp inserted something
        aligned_hyp: hyp tokens, with None where a deletion occurred

    Costs: substitution=1, insertion=1, deletion=1.
    """
    n, m = len(ref), len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i - 1][j],      # deletion (hyp missed ref phone)
                    dp[i][j - 1],      # insertion (hyp added extra phone)
                )

    # Traceback
    aligned_ref = []
    aligned_hyp = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append(hyp[j - 1])
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append(hyp[j - 1])
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Deletion: ref has a phone hyp didn't predict
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append(None)
            i -= 1
        else:
            # Insertion: hyp predicted a phone not in ref
            aligned_ref.append(None)
            aligned_hyp.append(hyp[j - 1])
            j -= 1

    aligned_ref.reverse()
    aligned_hyp.reverse()
    return aligned_ref, aligned_hyp


def sequence_metrics_from_alignment(aligned_ref: list, aligned_hyp: list) -> dict:
    """
    Given an alignment (output of edit_distance_alignment), compute:
      - n_ref     : number of phones in reference
      - n_correct : positions where ref == hyp (match, no gap)
      - n_sub     : substitutions (both non-None, but different)
      - n_del     : deletions (hyp is None)
      - n_ins     : insertions (ref is None)
      - per       : phoneme error rate = (sub + del + ins) / n_ref
    """
    n_ref     = sum(1 for r in aligned_ref if r is not None)
    n_correct = sum(
        1 for r, h in zip(aligned_ref, aligned_hyp)
        if r is not None and h is not None and r == h
    )
    n_sub = sum(
        1 for r, h in zip(aligned_ref, aligned_hyp)
        if r is not None and h is not None and r != h
    )
    n_del = sum(1 for h in aligned_hyp if h is None)
    n_ins = sum(1 for r in aligned_ref if r is None)
    per   = (n_sub + n_del + n_ins) / n_ref if n_ref > 0 else 0.0

    return {
        "n_ref": n_ref, "n_correct": n_correct,
        "n_sub": n_sub, "n_del": n_del,
        "n_ins": n_ins, "per": per,
    }


# ---------------------------------------------------------------------------
# Sequence prediction evaluation (word-level + sentence-level PER)
# ---------------------------------------------------------------------------

def evaluate_sequence_predictions(data: list) -> dict:
    """
    Evaluate the 'prediction' field against 'reference' using edit-distance
    alignment, at both the word level and sentence level.

    Word-level: each word's predicted phones aligned against its reference
    phones independently, then aggregated.

    Sentence-level: full flat sequences (boundaries stripped) aligned and
    scored as a single sequence per sentence, then aggregated.

    Returns a dict with 'word' and 'sentence' sub-dicts, each containing
    macro_per, micro_per, and raw counts.
    """
    word_pers     = []
    sentence_pers = []
    word_totals   = defaultdict(int)
    sent_totals   = defaultdict(int)
    skipped       = 0

    for sentence in data:
        sid = sentence["id"]

        if "prediction" not in sentence or "reference" not in sentence:
            skipped += 1
            continue

        ref_words  = split_on_boundaries(sentence["reference"])
        pred_words = split_on_boundaries(sentence["prediction"])

        if len(ref_words) != len(pred_words):
            print(
                f"[WARNING] Sentence {sid}: word count mismatch between "
                f"'reference' ({len(ref_words)}) and 'prediction' "
                f"({len(pred_words)}). Skipping sequence eval for this sentence."
            )
            skipped += 1
            continue

        sent_ref_flat  = []
        sent_pred_flat = []

        for ref_word, pred_word in zip(ref_words, pred_words):
            sent_ref_flat.extend(ref_word)
            sent_pred_flat.extend(pred_word)

            aligned_ref, aligned_hyp = edit_distance_alignment(ref_word, pred_word)
            m = sequence_metrics_from_alignment(aligned_ref, aligned_hyp)
            word_pers.append(m["per"])
            for k, v in m.items():
                if k != "per":
                    word_totals[k] += v

        aligned_ref, aligned_hyp = edit_distance_alignment(
            sent_ref_flat, sent_pred_flat
        )
        m = sequence_metrics_from_alignment(aligned_ref, aligned_hyp)
        sentence_pers.append(m["per"])
        for k, v in m.items():
            if k != "per":
                sent_totals[k] += v

    def micro_per(t):
        denom = t["n_ref"]
        return (t["n_sub"] + t["n_del"] + t["n_ins"]) / denom if denom > 0 else 0.0

    return {
        "word": {
            "macro_per": float(np.mean(word_pers)) if word_pers else float("nan"),
            "micro_per": micro_per(word_totals),
            "n_ref":     word_totals["n_ref"],
            "n_correct": word_totals["n_correct"],
            "n_sub":     word_totals["n_sub"],
            "n_del":     word_totals["n_del"],
            "n_ins":     word_totals["n_ins"],
        },
        "sentence": {
            "macro_per": float(np.mean(sentence_pers)) if sentence_pers else float("nan"),
            "micro_per": micro_per(sent_totals),
            "n_ref":     sent_totals["n_ref"],
            "n_correct": sent_totals["n_correct"],
            "n_sub":     sent_totals["n_sub"],
            "n_del":     sent_totals["n_del"],
            "n_ins":     sent_totals["n_ins"],
        },
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Deletion localisation evaluation
# ---------------------------------------------------------------------------

def evaluate_deletions(data: list):
    """
    Evaluate deletion localisation predictions against ground truth deletions.

    Ground truth deletions are identified from each word's 'mispronunciations'
    list where pronounced == '<DEL>'. The after-index is the flat non-boundary
    index of the phone immediately following the deletion in the pronounced
    sequence.

    Predicted deletions come from 'deletion_predictions': [[i, j], ...]
    where j is the after-index in the flat non-boundary pronounced sequence.

    Returns None if no deletion GT or predictions are found in the data.
    """
    has_deletion_preds = any(
        "deletion_predictions" in s and s["deletion_predictions"]
        for s in data
    )
    has_deletion_gt = any(
        any(
            mis.get("pronounced") == "<DEL>"
            for word in s.get("words", [])
            for mis in word.get("mispronunciations", [])
        )
        for s in data
    )

    if not has_deletion_gt and not has_deletion_preds:
        return None

    true_set = set()  # (sentence_id, after_index)
    pred_set = set()

    for sentence in data:
        sid      = sentence["id"]
        flat_idx = 0

        for word in sentence.get("words", []):
            pronounced = word.get("pronounced", [])
            for mis in word.get("mispronunciations", []):
                if mis.get("pronounced") == "<DEL>":
                    after_idx = flat_idx + min(
                        mis["index"], len(pronounced) - 1
                    )
                    true_set.add((sid, after_idx))
            flat_idx += len(pronounced)

        for pair in sentence.get("deletion_predictions", []):
            if len(pair) == 2:
                pred_set.add((sid, pair[1]))

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


# ---------------------------------------------------------------------------
# Sentence error count evaluation
# ---------------------------------------------------------------------------

def evaluate_sentence_error_counts(data: list):
    """
    Evaluate predicted sentence error counts against GT.
    Reports MAE, exact-match accuracy, and binary F1 (any errors vs. none).
    """
    if not any("predicted sentence errors" in s for s in data):
        return None

    gt_counts   = []
    pred_counts = []
    gt_binary   = []
    pred_binary = []

    for s in data:
        if "predicted sentence errors" not in s or "sentence errors" not in s:
            continue
        gt   = int(s["sentence errors"])
        pred = int(s["predicted sentence errors"])
        gt_counts.append(gt)
        pred_counts.append(pred)
        gt_binary.append(1 if gt > 0 else 0)
        pred_binary.append(1 if pred > 0 else 0)

    if not gt_counts:
        return None

    mae         = float(np.mean(np.abs(np.array(gt_counts) - np.array(pred_counts))))
    exact_match = float(np.mean(np.array(gt_counts) == np.array(pred_counts)))
    binary_f1   = f1_score(gt_binary, pred_binary, zero_division=0)
    binary_prec = precision_score(gt_binary, pred_binary, zero_division=0)
    binary_rec  = recall_score(gt_binary, pred_binary, zero_division=0)

    return {
        "n_sentences":      len(gt_counts),
        "mae":              mae,
        "exact_match":      exact_match,
        "binary_f1":        binary_f1,
        "binary_precision": binary_prec,
        "binary_recall":    binary_rec,
    }


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------

def check_lengths(sid, gt_phones, pred_phones, gt_words, pred_words) -> bool:
    ok = True
    if len(gt_phones) != len(pred_phones):
        print(
            f"[WARNING] Sentence {sid}: phone label length mismatch "
            f"(gt={len(gt_phones)}, pred={len(pred_phones)}). Skipping."
        )
        ok = False
    if len(gt_words) != len(pred_words):
        print(
            f"[WARNING] Sentence {sid}: word label length mismatch "
            f"(gt={len(gt_words)}, pred={len(pred_words)}). Skipping."
        )
        ok = False
    return ok


def compute_clf_metrics(y_true: list, y_pred: list, level_name: str) -> dict:
    report = classification_report(
        y_true, y_pred,
        target_names=["correct", "error"],
        zero_division=0,
        output_dict=True,
    )
    return {
        "level":        level_name,
        "n_samples":    len(y_true),
        "n_positive":   int(sum(y_true)),
        "accuracy":     accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision":    precision_score(y_true, y_pred, zero_division=0),
        "recall":       recall_score(y_true, y_pred, zero_division=0),
        "f1":           f1_score(y_true, y_pred, zero_division=0),
        "class_report": report,
    }


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(pred_path: str, verbose: bool = True) -> dict:
    """
    Load a self-contained predictions file and compute all metrics.

    Parameters
    ----------
    pred_path : Path to the predictions JSON file.
    verbose   : If True, print a formatted report to stdout.

    Returns
    -------
    Dict with keys "phone", "word", "sequence", and optionally
    "deletion" and "sentence_counts".
    """
    data = load_json(pred_path)

    all_phone_gt   = []
    all_phone_pred = []
    all_word_gt    = []
    all_word_pred  = []
    skipped        = 0

    for sentence in data:
        sid = sentence["id"]

        gt_phone_labels   = [int(x) for x in sentence["phone errors"]           if x != "<|>"]
        pred_phone_labels = [int(x) for x in sentence["predicted phone errors"]  if x != "<|>"]
        gt_word_labels    = [1 if int(e) > 0 else 0 for e in sentence["word errors"]]
        pred_word_labels  = [1 if int(e) > 0 else 0 for e in sentence["predicted word errors"]]

        if not check_lengths(sid, gt_phone_labels, pred_phone_labels,
                                   gt_word_labels,  pred_word_labels):
            skipped += 1
            continue

        all_phone_gt.extend(gt_phone_labels)
        all_phone_pred.extend(pred_phone_labels)
        all_word_gt.extend(gt_word_labels)
        all_word_pred.extend(pred_word_labels)

    if skipped:
        print(f"\n[INFO] Skipped {skipped} sentence(s) due to length mismatches.\n")

    phone_metrics    = compute_clf_metrics(all_phone_gt,  all_phone_pred,  "phone")
    word_metrics     = compute_clf_metrics(all_word_gt,   all_word_pred,   "word")
    sequence_metrics = evaluate_sequence_predictions(data)
    #deletion_metrics = evaluate_deletions(data)
    sentence_metrics = evaluate_sentence_error_counts(data)

    results = {
        "phone":    phone_metrics,
        "word":     word_metrics,
        "sequence": sequence_metrics,
    }
    # if deletion_metrics is not None:
    #     results["deletion"] = deletion_metrics
    if sentence_metrics is not None:
        results["sentence_counts"] = sentence_metrics

    if verbose:
        _print_report(results)

    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "   N/A"


def _print_report(results: dict):
    sep = "=" * 62

    for level in ("phone", "word"):
        m = results[level]
        print(sep)
        print(f"  {m['level'].upper()}-LEVEL CLASSIFICATION")
        print(sep)
        n = max(m["n_samples"], 1)
        print(f"  Samples evaluated : {m['n_samples']}")
        print(f"  Positive (error)  : {m['n_positive']} ({100 * m['n_positive'] / n:.1f}%)")
        print(f"  Accuracy          : {_fmt(m['accuracy'])}")
        print(f"  Balanced Accuracy : {_fmt(m['balanced_accuracy'])}")
        print(f"  Precision (error) : {_fmt(m['precision'])}")
        print(f"  Recall    (error) : {_fmt(m['recall'])}")
        print(f"  F1        (error) : {_fmt(m['f1'])}")
        print()
        cr = m["class_report"]
        print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"  {'-' * 46}")
        for cls in ("correct", "error"):
            r = cr[cls]
            print(f"  {cls:<12} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                  f"{r['f1-score']:>10.4f} {int(r['support']):>10}")
        print()

    sq = results["sequence"]
    print(sep)
    print("  SEQUENCE PREDICTION  (Phoneme Error Rate)")
    print(sep)
    if sq["skipped"]:
        print(f"  Skipped sentences : {sq['skipped']}")
    for granularity in ("word", "sentence"):
        g = sq[granularity]
        print(f"  [{granularity.upper()}]")
        print(f"    Macro PER  : {_fmt(g['macro_per'])}")
        print(f"    Micro PER  : {_fmt(g['micro_per'])}")
        print(f"    Correct: {g['n_correct']}  Sub: {g['n_sub']}  "
              f"Del: {g['n_del']}  Ins: {g['n_ins']}")
        print()

    if "deletion" in results:
        d = results["deletion"]
        print(sep)
        print("  DELETION LOCALISATION")
        print(sep)
        print(f"  True positives  : {d['true_positives']}")
        print(f"  False positives : {d['false_positives']}")
        print(f"  False negatives : {d['false_negatives']}")
        print(f"  Precision       : {_fmt(d['precision'])}")
        print(f"  Recall          : {_fmt(d['recall'])}")
        print(f"  F1              : {_fmt(d['f1'])}")
        print()

    if "sentence_counts" in results:
        sc = results["sentence_counts"]
        print(sep)
        print("  SENTENCE ERROR COUNT")
        print(sep)
        print(f"  Sentences evaluated : {sc['n_sentences']}")
        print(f"  MAE                 : {_fmt(sc['mae'])}")
        print(f"  Exact match         : {_fmt(sc['exact_match'])}")
        print(f"  Binary F1           : {_fmt(sc['binary_f1'])}")
        print(f"  Binary Precision    : {_fmt(sc['binary_precision'])}")
        print(f"  Binary Recall       : {_fmt(sc['binary_recall'])}")
        print()

    print(sep)


# ---------------------------------------------------------------------------
# Multi-condition comparison (ablation table)
# ---------------------------------------------------------------------------

def compare_conditions(conditions: dict, verbose: bool = True) -> dict:
    """
    Evaluate multiple prediction files and print a side-by-side comparison.

    Parameters
    ----------
    conditions : Dict mapping condition name to predictions file path, e.g.:
                 {"real_only":          "pred_real.json",
                  "synthetic_only":     "pred_synth.json",
                  "real_and_synthetic": "pred_both.json"}
    verbose    : Whether to print the comparison table.

    Returns
    -------
    Dict mapping condition name to its full metrics dict.
    """
    all_results = {}
    for name, pred_path in conditions.items():
        all_results[name] = evaluate(pred_path, verbose=False)

    if verbose:
        _print_comparison(all_results)

    return all_results


def _print_comparison(all_results: dict):
    conditions = list(all_results.keys())
    col_w = 14
    sep   = "=" * (22 + col_w * len(conditions))

    def hdr():
        return f"  {'Metric':<20}" + "".join(f"{c:>{col_w}}" for c in conditions)

    def div():
        return f"  {'-' * (20 + col_w * len(conditions))}"

    def row(label, vals):
        return f"  {label:<20}" + "".join(
            f"{v:>{col_w}.4f}" if not (isinstance(v, float) and np.isnan(v))
            else f"{'N/A':>{col_w}}"
            for v in vals
        )

    for level in ("phone", "word"):
        print(sep)
        print(f"  {level.upper()}-LEVEL CLASSIFICATION COMPARISON")
        print(sep)
        print(hdr()); print(div())
        for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
            print(row(metric, [all_results[c][level][metric] for c in conditions]))
        print()

    print(sep)
    print("  SEQUENCE PREDICTION COMPARISON  (PER)")
    print(sep)
    print(hdr()); print(div())
    for granularity in ("word", "sentence"):
        for metric in ("macro_per", "micro_per"):
            print(row(
                f"{granularity}_{metric}",
                [all_results[c]["sequence"][granularity][metric] for c in conditions]
            ))
    print()

    if any("deletion" in r for r in all_results.values()):
        print(sep)
        print("  DELETION LOCALISATION COMPARISON")
        print(sep)
        print(hdr()); print(div())
        for metric in ("precision", "recall", "f1"):
            print(row(metric, [
                all_results[c].get("deletion", {}).get(metric, float("nan"))
                for c in conditions
            ]))
        print()

    if any("sentence_counts" in r for r in all_results.values()):
        print(sep)
        print("  SENTENCE ERROR COUNT COMPARISON")
        print(sep)
        print(hdr()); print(div())
        for metric in ("mae", "exact_match", "binary_precision", "binary_recall", "binary_f1"):
            print(row(metric, [
                all_results[c].get("sentence_counts", {}).get(metric, float("nan"))
                for c in conditions
            ]))
        print()

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate mispronunciation detection predictions."
    )
    subparsers = parser.add_subparsers(dest="command")

    single = subparsers.add_parser("single", help="Evaluate one predictions file.")
    single.add_argument("--pred", required=True, help="Predictions JSON path.")

    compare = subparsers.add_parser(
        "compare", help="Compare multiple prediction files side by side."
    )
    compare.add_argument(
        "--conditions", required=True, nargs="+", metavar="NAME:PATH",
        help="One or more name:path pairs, e.g. real_only:pred_real.json"
    )

    args = parser.parse_args()

    if args.command == "single":
        evaluate(args.pred, verbose=True)
    elif args.command == "compare":
        conditions = {}
        for item in args.conditions:
            name, path = item.split(":", 1)
            conditions[name] = path
        compare_conditions(conditions, verbose=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()