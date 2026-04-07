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

USAGE
-----
python significance.py \
    --pred_a runs/real_only/predictions.json \
    --pred_b runs/real_and_synthetic/predictions.json \
    [--n_resamples 10000]

Tests
--------------
McNemar's test (mid-p) and approximate randomization are complementary,
not redundant:

  - McNemar operates on individual sample-level correctness, counting only
    discordant cases (A right/B wrong vs. A wrong/B right). This makes it
    robust to class imbalance — the large number of easy correct-phone
    predictions don't inflate the statistic. It is the most powerful test
    for detecting per-sample classification differences.

  - Approximate randomization permutes predictions at the sentence level
    and tests whether the observed difference in an aggregate metric (F1,
    PER, MAE, etc.) could arise by chance. It generalises to any scalar
    metric, including ones (PER, MAE, exact-match) that have no natural
    McNemar formulation.
"""

import json
import os
import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import binom
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Label extractors  (sentence dict → (gt_list, pred_list))
# ---------------------------------------------------------------------------

def _extract_phone_labels(s: dict) -> tuple:
    gt, pred = [], []
    for x, p in zip(s["phone errors"], s["predicted phone errors"]):
        if x != "<|>":
            gt.append(int(x))
            pred.append(int(p))
    return gt, pred


def _extract_word_labels(s: dict) -> tuple:
    gt   = [1 if int(g) > 0 else 0 for g in s["word errors"]]
    pred = [1 if int(p) > 0 else 0 for p in s["predicted word errors"]]
    return gt, pred


def _extract_sentence_binary(s: dict) -> tuple:
    """Binary: does the sentence contain any error?"""
    gt   = [1 if int(s["sentence errors"]) > 0 else 0]
    pred = [1 if int(s["predicted sentence errors"]) > 0 else 0]
    return gt, pred


def _extract_sentence_exact_match(s: dict) -> tuple:
    """1 if count prediction is exactly right, 0 otherwise — used for MAE/EM."""
    gt   = int(s["sentence errors"])
    pred = int(s["predicted sentence errors"])
    return gt, pred   # scalars, handled specially in MAE/EM helpers


# ---------------------------------------------------------------------------
# PER helpers  (mirrors evaluate.py logic)
# ---------------------------------------------------------------------------

def _split_on_boundaries(seq: list) -> list:
    words, current = [], []
    for token in seq:
        if token == "<|>":
            words.append(current)
            current = []
        else:
            current.append(token)
    if current:
        words.append(current)
    return words


def _edit_distance_alignment(ref: list, hyp: list) -> tuple:
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
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    aligned_ref, aligned_hyp = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            aligned_ref.append(ref[i - 1]); aligned_hyp.append(hyp[j - 1])
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned_ref.append(ref[i - 1]); aligned_hyp.append(hyp[j - 1])
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned_ref.append(ref[i - 1]); aligned_hyp.append(None)
            i -= 1
        else:
            aligned_ref.append(None); aligned_hyp.append(hyp[j - 1])
            j -= 1
    aligned_ref.reverse(); aligned_hyp.reverse()
    return aligned_ref, aligned_hyp


def _per_from_alignment(aligned_ref, aligned_hyp) -> tuple:
    """Returns (per, n_ref, n_errors)."""
    n_ref = sum(1 for r in aligned_ref if r is not None)
    n_sub = sum(1 for r, h in zip(aligned_ref, aligned_hyp)
                if r is not None and h is not None and r != h)
    n_del = sum(1 for h in aligned_hyp if h is None)
    n_ins = sum(1 for r in aligned_ref if r is None)
    n_err = n_sub + n_del + n_ins
    per   = n_err / n_ref if n_ref > 0 else 0.0
    return per, n_ref, n_err


def _sentence_per_pair(s: dict) -> tuple | None:
    """
    Returns (word_pers, sent_per, n_ref_words, n_ref_phones, n_err_words, n_err_phones)
    for one sentence, or None if the sentence must be skipped.
    """
    if "prediction" not in s or "reference" not in s:
        return None

    ref_words  = _split_on_boundaries(s["reference"])
    pred_words = _split_on_boundaries(s["prediction"])

    if len(ref_words) != len(pred_words):
        return None

    word_pers = []
    sent_ref_flat, sent_pred_flat = [], []
    n_ref_words = n_err_words = 0

    for rw, pw in zip(ref_words, pred_words):
        sent_ref_flat.extend(rw)
        sent_pred_flat.extend(pw)
        ar, ah = _edit_distance_alignment(rw, pw)
        per, n_ref, n_err = _per_from_alignment(ar, ah)
        word_pers.append(per)
        n_ref_words += n_ref
        n_err_words += n_err

    ar, ah = _edit_distance_alignment(sent_ref_flat, sent_pred_flat)
    sent_per, n_ref_phones, n_err_phones = _per_from_alignment(ar, ah)

    return word_pers, sent_per, n_ref_words, n_ref_phones, n_err_words, n_err_phones


# ---------------------------------------------------------------------------
# Metric functions operating on pre-extracted per-sentence data
# ---------------------------------------------------------------------------
# For PER the "extracted" representation per sentence is the tuple returned
# by _sentence_per_pair.  We build metric_fn callables that accept a list of
# these tuples and return a scalar.

def _macro_word_per(sentence_data: list) -> float:
    pers = [p for d in sentence_data for p in d[0]]   # d[0] = word_pers list
    return float(np.mean(pers)) if pers else float("nan")


def _micro_word_per(sentence_data: list) -> float:
    n_ref = sum(d[2] for d in sentence_data)           # d[2] = n_ref_words
    n_err = sum(d[4] for d in sentence_data)           # d[4] = n_err_words
    return n_err / n_ref if n_ref > 0 else 0.0


def _macro_sent_per(sentence_data: list) -> float:
    pers = [d[1] for d in sentence_data]               # d[1] = sent_per
    return float(np.mean(pers)) if pers else float("nan")


def _micro_sent_per(sentence_data: list) -> float:
    n_ref = sum(d[3] for d in sentence_data)           # d[3] = n_ref_phones
    n_err = sum(d[5] for d in sentence_data)           # d[5] = n_err_phones
    return n_err / n_ref if n_ref > 0 else 0.0


# ---------------------------------------------------------------------------
# Core statistical tests
# ---------------------------------------------------------------------------

def _mcnemar_test(sents_a: list, sents_b: list, extract_fn) -> tuple:
    """
    Mid-p McNemar's test on paired binary per-sample correctness.
    Returns (n_a_right_b_wrong, n_a_wrong_b_right, p_value).
    """
    n01 = n10 = 0
    for sa, sb in zip(sents_a, sents_b):
        gt_a, pred_a = extract_fn(sa)
        gt_b, pred_b = extract_fn(sb)
        for g, pa, pb in zip(gt_a, pred_a, gt_b):
            a_ok = (pa == g)
            b_ok = (pb == g)
            if     a_ok and not b_ok: n01 += 1
            elif not a_ok and b_ok:   n10 += 1

    total = n01 + n10
    if total == 0:
        return n01, n10, 1.0
    smaller = min(n01, n10)
    p_value = min(2 * binom.cdf(smaller, total, 0.5) - binom.pmf(smaller, total, 0.5), 1.0)
    return n01, n10, p_value


def _approximate_randomization_clf(
    sents_a: list,
    sents_b: list,
    extract_fn,
    metric_fn,
    n_resamples: int = 10_000,
    seed: int = 3407,
) -> tuple:
    """
    Approximate randomization for flat classification metrics (F1, P, R).
    Permutes at the sentence level.
    Returns (value_a, value_b, p_value).
    """
    rng = np.random.default_rng(seed)
    n   = len(sents_a)

    # Pre-extract
    per_sentence = []
    for sa, sb in zip(sents_a, sents_b):
        gt_a, pa = extract_fn(sa)
        _,    pb = extract_fn(sb)
        per_sentence.append((gt_a, pa, pb))

    gt_all = np.array([g for gt, _, __ in per_sentence for g in gt])
    pa_all = np.array([p for _,  pa, __ in per_sentence for p in pa])
    pb_all = np.array([p for _,  __,  pb in per_sentence for p in pb])

    val_a = metric_fn(gt_all, pa_all)
    val_b = metric_fn(gt_all, pb_all)
    observed_diff = val_a - val_b

    count = 0
    for _ in tqdm(range(n_resamples), leave=False):
        swaps = rng.integers(0, 2, size=n)
        gt_r, pa_r, pb_r = [], [], []
        for i, swap in enumerate(swaps):
            gt_i, pa_i, pb_i = per_sentence[i]
            gt_r.extend(gt_i)
            if swap:
                pa_r.extend(pb_i); pb_r.extend(pa_i)
            else:
                pa_r.extend(pa_i); pb_r.extend(pb_i)
        diff = metric_fn(np.array(gt_r), np.array(pa_r)) - metric_fn(np.array(gt_r), np.array(pb_r))
        if abs(diff) >= abs(observed_diff):
            count += 1

    return val_a, val_b, count / n_resamples


def _approximate_randomization_per(
    per_data_a: list,
    per_data_b: list,
    metric_fn,
    n_resamples: int = 10_000,
    seed: int = 3407,
) -> tuple:
    """
    Approximate randomization for PER metrics.
    Each element of per_data_a/b is the tuple from _sentence_per_pair.
    Permutes at the sentence level.
    Returns (value_a, value_b, p_value).
    """
    rng = np.random.default_rng(seed)
    n   = len(per_data_a)

    val_a = metric_fn(per_data_a)
    val_b = metric_fn(per_data_b)
    observed_diff = val_a - val_b

    count = 0
    for _ in tqdm(range(n_resamples), leave=False):
        swaps = rng.integers(0, 2, size=n)
        mixed_a = [per_data_b[i] if swaps[i] else per_data_a[i] for i in range(n)]
        mixed_b = [per_data_a[i] if swaps[i] else per_data_b[i] for i in range(n)]
        diff = metric_fn(mixed_a) - metric_fn(mixed_b)
        if abs(diff) >= abs(observed_diff):
            count += 1

    return val_a, val_b, count / n_resamples


def _approximate_randomization_sentence_counts(
    sents_a: list,
    sents_b: list,
    metric_fn,
    n_resamples: int = 10_000,
    seed: int = 3407,
) -> tuple:
    """
    Approximate randomization for sentence error count metrics (MAE, exact match).
    Each per-sentence datum is (gt_scalar, pred_scalar).
    Returns (value_a, value_b, p_value).
    """
    rng = np.random.default_rng(seed)
    n   = len(sents_a)

    pairs_a = [_extract_sentence_exact_match(s) for s in sents_a]
    pairs_b = [_extract_sentence_exact_match(s) for s in sents_b]

    val_a = metric_fn(pairs_a)
    val_b = metric_fn(pairs_b)
    observed_diff = val_a - val_b

    count = 0
    for _ in tqdm(range(n_resamples), leave=False):
        swaps = rng.integers(0, 2, size=n)
        mixed_a = [pairs_b[i] if swaps[i] else pairs_a[i] for i in range(n)]
        mixed_b = [pairs_a[i] if swaps[i] else pairs_b[i] for i in range(n)]
        diff = metric_fn(mixed_a) - metric_fn(mixed_b)
        if abs(diff) >= abs(observed_diff):
            count += 1

    return val_a, val_b, count / n_resamples


# ---------------------------------------------------------------------------
# Sentence count metric functions  (operate on list of (gt, pred) scalars)
# ---------------------------------------------------------------------------

def _sc_mae(pairs: list) -> float:
    errs = [abs(gt - pred) for gt, pred in pairs]
    # Negate so that "higher is better" holds uniformly (lower MAE = better).
    # The randomization test checks |diff| >= |observed|, so the sign of
    # observed_diff matters for directionality but not for the p-value.
    return -float(np.mean(errs)) if errs else 0.0


def _sc_exact_match(pairs: list) -> float:
    return float(np.mean([gt == pred for gt, pred in pairs])) if pairs else 0.0


def _sc_binary_f1(pairs: list) -> float:
    gt   = [1 if g > 0 else 0 for g, _ in pairs]
    pred = [1 if p > 0 else 0 for _, p in pairs]
    return f1_score(gt, pred, zero_division=0)


def _sc_binary_precision(pairs: list) -> float:
    gt   = [1 if g > 0 else 0 for g, _ in pairs]
    pred = [1 if p > 0 else 0 for _, p in pairs]
    return precision_score(gt, pred, zero_division=0)


def _sc_binary_recall(pairs: list) -> float:
    gt   = [1 if g > 0 else 0 for g, _ in pairs]
    pred = [1 if p > 0 else 0 for _, p in pairs]
    return recall_score(gt, pred, zero_division=0)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def significance_test(
    pred_path_a: str,
    pred_path_b: str,
    n_resamples: int = 10_000,
    verbose: bool = True,
) -> dict:
    """
    Run all significance tests comparing two prediction files.

    Tests performed
    ---------------
    Phone / Word classification:
      - McNemar's test (mid-p)  — sample-level binary correctness
      - Approx. randomization   — F1, precision, recall

    PER (sequence prediction):
      - Approx. randomization   — word macro PER, word micro PER,
                                  sentence macro PER, sentence micro PER

    Sentence error counts:
      - McNemar's test (mid-p)  — binary (any error?) correctness
      - Approx. randomization   — MAE (negated), exact match,
                                  binary F1, precision, recall
    """
    data_a = load_json(pred_path_a)
    data_b = load_json(pred_path_b)

    by_id_a = {s["id"]: s for s in data_a}
    by_id_b = {s["id"]: s for s in data_b}
    shared_ids = sorted(set(by_id_a) & set(by_id_b))

    if len(shared_ids) != len(data_a) or len(shared_ids) != len(data_b):
        print(f"[WARNING] Files have different sentence ids. "
              f"Evaluating on {len(shared_ids)} shared sentences only.")

    sents_a = [by_id_a[i] for i in shared_ids]
    sents_b = [by_id_b[i] for i in shared_ids]

    def _f1(yt, yp):   return f1_score(yt, yp, zero_division=0)
    def _prec(yt, yp): return precision_score(yt, yp, zero_division=0)
    def _rec(yt, yp):  return recall_score(yt, yp, zero_division=0)

    results = {}

    # -----------------------------------------------------------------------
    # 1. Phone-level classification
    # -----------------------------------------------------------------------
    print("Running phone-level tests...")
    results["phone_mcnemar"] = dict(zip(
        ("n_a_correct_b_wrong", "n_a_wrong_b_correct", "p_value"),
        _mcnemar_test(sents_a, sents_b, _extract_phone_labels),
    ))
    results["phone_mcnemar"]["significant"] = results["phone_mcnemar"]["p_value"] < 0.05

    for metric_name, mfn in (("phone_f1", _f1), ("phone_precision", _prec), ("phone_recall", _rec)):
        va, vb, pv = _approximate_randomization_clf(
            sents_a, sents_b, _extract_phone_labels, mfn, n_resamples)
        results[metric_name] = {"value_a": va, "value_b": vb, "diff": va - vb,
                                "p_value": pv, "significant": pv < 0.05}

    # -----------------------------------------------------------------------
    # 2. Word-level classification
    # -----------------------------------------------------------------------
    print("Running word-level tests...")
    results["word_mcnemar"] = dict(zip(
        ("n_a_correct_b_wrong", "n_a_wrong_b_correct", "p_value"),
        _mcnemar_test(sents_a, sents_b, _extract_word_labels),
    ))
    results["word_mcnemar"]["significant"] = results["word_mcnemar"]["p_value"] < 0.05

    for metric_name, mfn in (("word_f1", _f1), ("word_precision", _prec), ("word_recall", _rec)):
        va, vb, pv = _approximate_randomization_clf(
            sents_a, sents_b, _extract_word_labels, mfn, n_resamples)
        results[metric_name] = {"value_a": va, "value_b": vb, "diff": va - vb,
                                "p_value": pv, "significant": pv < 0.05}

    # -----------------------------------------------------------------------
    # 3. PER (sequence prediction)
    # -----------------------------------------------------------------------
    print("Running PER tests...")
    per_data_a = [_sentence_per_pair(s) for s in sents_a]
    per_data_b = [_sentence_per_pair(s) for s in sents_b]

    # Drop sentences that couldn't be aligned (None) — must drop from both
    valid = [(a, b) for a, b in zip(per_data_a, per_data_b) if a is not None and b is not None]
    n_skipped = len(per_data_a) - len(valid)
    if n_skipped:
        print(f"[INFO] Skipped {n_skipped} sentence(s) for PER tests (word count mismatch).")
    per_data_a_valid, per_data_b_valid = zip(*valid) if valid else ([], [])
    per_data_a_valid = list(per_data_a_valid)
    per_data_b_valid = list(per_data_b_valid)

    for metric_name, mfn in (
        ("per_word_macro",    _macro_word_per),
        ("per_word_micro",    _micro_word_per),
        ("per_sentence_macro", _macro_sent_per),
        ("per_sentence_micro", _micro_sent_per),
    ):
        va, vb, pv = _approximate_randomization_per(
            per_data_a_valid, per_data_b_valid, mfn, n_resamples)
        results[metric_name] = {"value_a": va, "value_b": vb, "diff": va - vb,
                                "p_value": pv, "significant": pv < 0.05}

    # -----------------------------------------------------------------------
    # 4. Sentence error counts
    # -----------------------------------------------------------------------
    has_sent_counts = all(
        "predicted sentence errors" in s and "sentence errors" in s
        for s in sents_a + sents_b
    )
    if has_sent_counts:
        print("Running sentence error count tests...")

        # McNemar on binary (any error vs. none)
        results["sentence_count_mcnemar"] = dict(zip(
            ("n_a_correct_b_wrong", "n_a_wrong_b_correct", "p_value"),
            _mcnemar_test(sents_a, sents_b, _extract_sentence_binary),
        ))
        results["sentence_count_mcnemar"]["significant"] = (
            results["sentence_count_mcnemar"]["p_value"] < 0.05
        )

        # Approximate randomization for all five count metrics
        for metric_name, mfn in (
            ("sentence_count_mae",              _sc_mae),
            ("sentence_count_exact_match",      _sc_exact_match),
            ("sentence_count_binary_f1",        _sc_binary_f1),
            ("sentence_count_binary_precision", _sc_binary_precision),
            ("sentence_count_binary_recall",    _sc_binary_recall),
        ):
            va, vb, pv = _approximate_randomization_sentence_counts(
                sents_a, sents_b, mfn, n_resamples)
            # Un-negate MAE values for display
            if metric_name == "sentence_count_mae":
                va, vb = -va, -vb
            results[metric_name] = {"value_a": va, "value_b": vb, "diff": va - vb,
                                    "p_value": pv, "significant": pv < 0.05}
    else:
        print("[INFO] Sentence error count fields missing; skipping those tests.")

    if verbose:
        _print_significance(results, pred_path_a, pred_path_b)

    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_significance(results: dict, path_a: str, path_b: str):
    name_a = os.path.basename(os.path.dirname(path_a)) or path_a
    name_b = os.path.basename(os.path.dirname(path_b)) or path_b

    sep = "=" * 72
    print(sep)
    print("  SIGNIFICANCE TESTING")
    print(f"  A: {name_a}")
    print(f"  B: {name_b}")
    print(sep)

    # --- McNemar ---
    print("  McNemar's test (mid-p) — binary classification")
    print(f"  {'Level':<26} {'A✓B✗':>8} {'A✗B✓':>8} {'p-value':>10}  ")
    print(f"  {'-' * 56}")
    for key, label in (
        ("phone_mcnemar",          "phone"),
        ("word_mcnemar",           "word"),
        ("sentence_count_mcnemar", "sentence counts (binary)"),
    ):
        if key not in results:
            continue
        r   = results[key]
        sig = " (*)" if r["significant"] else "     "
        print(f"  {label:<26} {r['n_a_correct_b_wrong']:>8} "
              f"{r['n_a_wrong_b_correct']:>8} {r['p_value']:>10.4f}{sig}")
    print()

    # --- Approximate randomization: classification ---
    print("  Approximate randomization — phone / word classification")
    print(f"  {'Metric':<28} {'A':>8} {'B':>8} {'Diff':>8} {'p-value':>10}  ")
    print(f"  {'-' * 66}")
    for name in ("phone_f1", "phone_precision", "phone_recall",
                 "word_f1",  "word_precision",  "word_recall"):
        if name not in results:
            continue
        r   = results[name]
        sig = " (*)" if r["significant"] else "     "
        print(f"  {name:<28} {r['value_a']:>8.4f} {r['value_b']:>8.4f} "
              f"{r['diff']:>+8.4f} {r['p_value']:>10.4f}{sig}")
    print()

    # --- Approximate randomization: PER ---
    print("  Approximate randomization — Phoneme Error Rate (PER)")
    print(f"  {'Metric':<28} {'A':>8} {'B':>8} {'Diff':>8} {'p-value':>10}  ")
    print(f"  {'-' * 66}")
    for name in ("per_word_macro", "per_word_micro",
                 "per_sentence_macro", "per_sentence_micro"):
        if name not in results:
            continue
        r   = results[name]
        sig = " (*)" if r["significant"] else "     "
        print(f"  {name:<28} {r['value_a']:>8.4f} {r['value_b']:>8.4f} "
              f"{r['diff']:>+8.4f} {r['p_value']:>10.4f}{sig}")
    print()

    # --- Approximate randomization: sentence counts ---
    sent_keys = [k for k in (
        "sentence_count_mae", "sentence_count_exact_match",
        "sentence_count_binary_f1", "sentence_count_binary_precision",
        "sentence_count_binary_recall",
    ) if k in results]
    if sent_keys:
        print("  Approximate randomization — sentence error counts")
        print(f"  {'Metric':<28} {'A':>8} {'B':>8} {'Diff':>8} {'p-value':>10}  ")
        print(f"  {'-' * 66}")
        for name in sent_keys:
            r   = results[name]
            sig = " (*)" if r["significant"] else "     "
            label = name.replace("sentence_count_", "")
            print(f"  {label:<28} {r['value_a']:>8.4f} {r['value_b']:>8.4f} "
                  f"{r['diff']:>+8.4f} {r['p_value']:>10.4f}{sig}")
        print()
        print("  Note: for MAE, Diff = A − B; negative means A has lower (better) MAE.")
        print()

    print("  (*) = significant at p < 0.05")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Significance testing for mispronunciation detection predictions."
    )
    parser.add_argument("--pred_a",      required=True,
                        help="First predictions JSON file.")
    parser.add_argument("--pred_b",      required=True,
                        help="Second predictions JSON file.")
    parser.add_argument("--n_resamples", type=int, default=10_000,
                        help="Approximate randomization resamples (default: 10000).")

    args = parser.parse_args()
    significance_test(args.pred_a, args.pred_b,
                      n_resamples=args.n_resamples, verbose=True)


if __name__ == "__main__":
    main()