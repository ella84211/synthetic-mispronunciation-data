"""
Mispronunciation Detection Significance Testing
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
  "deletion_predictions"     : list of [i, j] index pairs (into the flat
                               non-boundary phone sequence) indicating a
                               predicted deletion between positions i and j.
                               Omit or set to [] if not predicting deletions.

USAGE
-----
python evaluate.py significance \
    --pred_a runs/real_only/predictions.json \
    --pred_b runs/real_and_synthetic/predictions.json
"""

import json
import argparse
from tqdm import tqdm

from sklearn.metrics import f1_score, precision_score, recall_score


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def _approximate_randomization(
    sents_a: list,
    sents_b: list,
    extract_fn,
    metric_fn,
    n_resamples: int = 10000,
    seed: int = 3407,
) -> tuple:
    """
    Approximate randomization test (permutation test) at the sentence level.
    For each trial, randomly swaps A and B predictions for each sentence
    with probability 0.5, then recomputes the metric difference.
    Returns (value_a, value_b, p_value).
    """
    import numpy as np_local
    rng = np_local.random.default_rng(seed)
    n   = len(sents_a)

    gt_all, pa_all, pb_all = [], [], []
    for sa, sb in zip(sents_a, sents_b):
        gt_a, pred_a = extract_fn(sa)
        _,    pred_b = extract_fn(sb)
        gt_all.extend(gt_a)
        pa_all.extend(pred_a)
        pb_all.extend(pred_b)

    val_a = metric_fn(np_local.array(gt_all), np_local.array(pa_all))
    val_b = metric_fn(np_local.array(gt_all), np_local.array(pb_all))
    observed_diff = val_a - val_b

    # Pre-extract per-sentence labels to avoid redundant calls in the loop
    per_sentence = []
    for sa, sb in zip(sents_a, sents_b):
        gt_i, pred_a_i = extract_fn(sa)
        _,    pred_b_i = extract_fn(sb)
        per_sentence.append((gt_i, pred_a_i, pred_b_i))

    count = 0
    for _ in tqdm(range(n_resamples)):
        # Randomly swap A and B predictions for each sentence
        swaps = rng.integers(0, 2, size=n)
        gt_r, pa_r, pb_r = [], [], []
        for i, swap in enumerate(swaps):
            gt_i, pred_a_i, pred_b_i = per_sentence[i]
            gt_r.extend(gt_i)
            if swap:
                pa_r.extend(pred_b_i)
                pb_r.extend(pred_a_i)
            else:
                pa_r.extend(pred_a_i)
                pb_r.extend(pred_b_i)

        diff = (metric_fn(np_local.array(gt_r), np_local.array(pa_r)) -
                metric_fn(np_local.array(gt_r), np_local.array(pb_r)))
        if abs(diff) >= abs(observed_diff):
            count += 1

    return val_a, val_b, count / n_resamples


def _mcnemar_test(
    sents_a: list,
    sents_b: list,
    extract_fn,
) -> tuple:
    """
    McNemar's test for comparing two classifiers on paired binary data.
    Only considers discordant cases (where A and B disagree), which makes
    it robust to zero-inflation from easy negative examples.

    Uses the mid-p correction for small discordant counts, and the
    asymptotic chi-squared approximation otherwise.

    Returns (n_a_right_b_wrong, n_a_wrong_b_right, p_value).
    """
    from scipy.stats import binom

    n01 = 0  # A correct, B wrong
    n10 = 0  # A wrong,   B correct

    for sa, sb in zip(sents_a, sents_b):
        gt_a,  pred_a = extract_fn(sa)
        gt_b,  pred_b = extract_fn(sb)
        for g, pa, pb in zip(gt_a, pred_a, gt_b if gt_b else pred_b):
            a_correct = (pa == g)
            b_correct = (pb == g)
            if a_correct and not b_correct:
                n01 += 1
            elif not a_correct and b_correct:
                n10 += 1

    total = n01 + n10
    if total == 0:
        return n01, n10, 1.0

    # Mid-p McNemar: more powerful than standard McNemar for small samples
    # p = 2 * P(X <= min(n01,n10)) - P(X == min(n01,n10))  where X ~ Bin(n, 0.5)
    smaller = min(n01, n10)
    p_value = 2 * binom.cdf(smaller, total, 0.5) - binom.pmf(smaller, total, 0.5)
    p_value = min(p_value, 1.0)

    return n01, n10, p_value


def significance_test(
    pred_path_a: str,
    pred_path_b: str,
    n_resamples: int = 10000,
    verbose: bool = True,
) -> dict:
    """
    Significance testing comparing two prediction files.

    Uses two complementary tests:
      - McNemar's test (mid-p) for phone-level and word-level binary
        classification. Operates on discordant cases only, making it
        robust to zero-inflation from easy negative examples.
      - Approximate randomization test for F1, precision, and recall
        as aggregate metrics. Permutes predictions at the sentence level.

    p < 0.05 = significant at 95% confidence level.
    """

    data_a = load_json(pred_path_a)
    data_b = load_json(pred_path_b)

    by_id_a = {s["id"]: s for s in data_a}
    by_id_b = {s["id"]: s for s in data_b}
    shared_ids = sorted(set(by_id_a.keys()) & set(by_id_b.keys()))

    if len(shared_ids) != len(data_a) or len(shared_ids) != len(data_b):
        print(f"[WARNING] Files have different sentence ids. "
              f"Evaluating on {len(shared_ids)} shared sentences only.")

    sents_a = [by_id_a[i] for i in shared_ids]
    sents_b = [by_id_b[i] for i in shared_ids]

    def _f1(yt, yp):   return f1_score(yt, yp, zero_division=0)
    def _prec(yt, yp): return precision_score(yt, yp, zero_division=0)
    def _rec(yt, yp):  return recall_score(yt, yp, zero_division=0)

    results = {}

    # --- McNemar's test for binary classification ---
    for level, extract_fn in [
        ("phone", _extract_phone_labels_sentence),
        ("word",  _extract_word_labels_sentence),
    ]:
        n01, n10, p_val = _mcnemar_test(sents_a, sents_b, extract_fn)
        results[f"{level}_mcnemar"] = {
            "n_a_correct_b_wrong": n01,
            "n_a_wrong_b_correct": n10,
            "p_value":             p_val,
            "significant":         p_val < 0.05,
        }

    # --- Approximate randomization for aggregate metrics ---
    for name, extract_fn, metric_fn in [
        ("phone_f1",        _extract_phone_labels_sentence, _f1),
        ("phone_precision",  _extract_phone_labels_sentence, _prec),
        ("phone_recall",     _extract_phone_labels_sentence, _rec),
        ("word_f1",          _extract_word_labels_sentence,  _f1),
        ("word_precision",   _extract_word_labels_sentence,  _prec),
        ("word_recall",      _extract_word_labels_sentence,  _rec),
    ]:
        val_a, val_b, p_val = _approximate_randomization(
            sents_a, sents_b, extract_fn, metric_fn, n_resamples=n_resamples
        )
        results[name] = {
            "value_a":    val_a,
            "value_b":    val_b,
            "diff":       val_a - val_b,
            "p_value":    p_val,
            "significant": p_val < 0.05,
        }

    if verbose:
        _print_significance(results, pred_path_a, pred_path_b)

    return results


def _print_significance(results: dict, path_a: str, path_b: str):
    import os
    name_a = os.path.basename(os.path.dirname(path_a)) or path_a
    name_b = os.path.basename(os.path.dirname(path_b)) or path_b

    sep = "=" * 72
    print(sep)
    print("  SIGNIFICANCE TESTING")
    print(f"  A: {name_a}")
    print(f"  B: {name_b}")
    print(sep)

    # McNemar results
    print("  McNemar's test — binary classification")
    print(f"  {'Level':<10} {'A correct B wrong':>18} {'A wrong B correct':>18} "
          f"{'p-value':>10}  ")
    print(f"  {'-' * 62}")
    for level in ("phone", "word"):
        r = results[f"{level}_mcnemar"]
        sig = "(*)" if r["significant"] else "   "
        print(f"  {level:<10} {r['n_a_correct_b_wrong']:>18} "
              f"{r['n_a_wrong_b_correct']:>18} {r['p_value']:>10.4f}  {sig}")
    print()

    # Approximate randomization results
    print("  Approximate randomization test — aggregate metrics")
    print(f"  {'Metric':<20} {'A':>8} {'B':>8} {'Diff':>8} {'p-value':>10}  ")
    print(f"  {'-' * 62}")
    for name in ("phone_f1", "phone_precision", "phone_recall",
                 "word_f1", "word_precision", "word_recall"):
        r   = results[name]
        sig = "(*)" if r["significant"] else "   "
        print(f"  {name:<20} {r['value_a']:>8.4f} {r['value_b']:>8.4f} "
              f"{r['diff']:>+8.4f} {r['p_value']:>10.4f}  {sig}")
    print()
    print("  (*) = significant at p < 0.05")
    print(sep)


def _extract_phone_labels_sentence(sentence: dict) -> tuple:
    """Extract GT and predicted phone labels from a single sentence dict."""
    gt, pred = [], []
    for x, p in zip(sentence["phone errors"], sentence["predicted phone errors"]):
        if x != "<|>":
            gt.append(int(x))
            pred.append(int(p))
    return gt, pred


def _extract_word_labels_sentence(sentence: dict) -> tuple:
    """Extract GT and predicted word labels from a single sentence dict."""
    gt   = [1 if int(g) > 0 else 0 for g in sentence["word errors"]]
    pred = [1 if int(p) > 0 else 0 for p in sentence["predicted word errors"]]
    return gt, pred


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate mispronunciation detection predictions."
    )
    
    parser.add_argument("--pred_a", required=True,
                    help="First predictions JSON file.")
    parser.add_argument("--pred_b", required=True,
                    help="Second predictions JSON file.")
    parser.add_argument("--n_resamples", type=int, default=10000,
                    help="Number of approximate randomization resamples (default: 10000).")

    args = parser.parse_args()

    significance_test(args.pred_a, args.pred_b,
                      n_resamples=args.n_resamples, verbose=True)


if __name__ == "__main__":
    main()