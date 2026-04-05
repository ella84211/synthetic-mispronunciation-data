"""
Mispronunciation Detection — Training & Inference Script
=========================================================

Architecture: small custom Transformer encoder with four prediction heads.

  INPUT REPRESENTATION
  --------------------
  For each sentence, the pronounced phone sequence is expanded by interleaving
  <GAP> tokens between every adjacent phone pair and at the start/end of each
  word. Word boundary <|> tokens are preserved. The full token sequence fed to
  the model is therefore:

    [<GAP>, p0, <GAP>, p1, <GAP>, ..., pN, <GAP>, <|>, <GAP>, p0, ...]

  This gives the model explicit positions to classify deletions into.

  HEADS
  -----
  1. phone_head      — binary classifier on every non-GAP, non-boundary token.
                       0 = correct, 1 = error (substitution).
  2. gap_head        — binary classifier on every <GAP> token.
                       0 = no deletion, 1 = deletion occurred here.
  3. correction_head — multiclass classifier over the full phone vocabulary,
                       run on every phone token predicted as an error (phone_head=1)
                       AND every gap token predicted as a deletion (gap_head=1).
                       Predicts the canonical phone that should have been there.

  SEQUENCE PREDICTION (inference only, no extra training target)
  --------------------------------------------------------------
  At inference time, `prediction` is assembled from head outputs:
    - Phone tokens with phone_head=0  → keep as-is (correct)
    - Phone tokens with phone_head=1  → replace with correction_head output
    - Gap tokens with gap_head=1      → insert correction_head output here
  This produces a predicted reference sequence comparable to the ground truth
  `reference` field and compatible with the evaluation pipeline's PER metric.

  NOTE: If you later want to train an explicit seq2seq decoder to predict the
  reference sequence, replace the inference-time assembly block at the bottom
  of predict() with a cross-attention decoder over the encoder hidden states.

USAGE
-----
  # Train (prompts for training file paths at runtime)
  python train_transformer.py train \\
      --val    data/val.json   \\
      --output runs/experiment_1

  # Evaluate on test set (loads best checkpoint, writes predictions JSON)
  python train_transformer.py predict \\
      --test   data/test.json \\
      --output runs/experiment_1

  # Full pipeline in one command
  python train_transformer.py train --output runs/experiment_1 \\
      --train data/train.json
"""

import json
import math
import os
import random
import argparse
import time
import warnings
from collections import defaultdict
from typing import Optional

warnings.filterwarnings("ignore", message="enable_nested_tensor")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# Random seed — fixed for reproducibility across all ablation conditions
# ---------------------------------------------------------------------------

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

PAD           = "<PAD>"
UNK           = "<UNK>"
WORD_BOUNDARY = "<|>"
GAP           = "<GAP>"
SPECIAL_TOKENS = [PAD, UNK, WORD_BOUNDARY, GAP]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    """Bidirectional mapping between IPA tokens and integer indices."""

    def __init__(self):
        self.token2idx: dict = {}
        self.idx2token: dict = {}

    def build(self, data: list):
        """
        Build vocabulary from a list of sentence dicts.
        Collects all tokens seen in 'pronounced' and 'reference' fields.
        Special tokens are always added first so their indices are stable.
        """
        counts: dict = defaultdict(int)
        for sentence in data:
            for tok in sentence.get("pronounced", []):
                counts[tok] += 1
            for tok in sentence.get("reference", []):
                counts[tok] += 1

        for tok in SPECIAL_TOKENS:
            self._add(tok)
        for tok in sorted(counts.keys()):
            if tok not in self.token2idx:
                self._add(tok)

    def _add(self, token: str):
        idx = len(self.token2idx)
        self.token2idx[token] = idx
        self.idx2token[idx] = token

    def encode(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx[UNK])

    def decode(self, idx: int) -> str:
        return self.idx2token.get(idx, UNK)

    def __len__(self):
        return len(self.token2idx)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        v = cls()
        with open(path, encoding="utf-8") as f:
            v.token2idx = json.load(f)
        v.idx2token = {i: t for t, i in v.token2idx.items()}
        return v


# ---------------------------------------------------------------------------
# Data helpers: expanding sequences with <GAP> tokens
# ---------------------------------------------------------------------------

def expand_with_gaps(pronounced: list) -> tuple:
    """
    Expand a flat pronounced sequence (with <|> boundaries) by interleaving
    <GAP> tokens. Returns:

      expanded      : the expanded token list
      phone_mask    : 1 at positions that are real phones (not GAP or <|>)
      gap_mask      : 1 at positions that are <GAP> tokens

    Example (one word):
      pronounced = ['b', 'ɹ', 'aɪ']
      expanded   = [GAP, 'b', GAP, 'ɹ', GAP, 'aɪ', GAP]

    Word boundaries are kept as-is; no GAPs are inserted around them so the
    model doesn't try to predict deletions at word edges (those are handled
    by the last/first gap of each word).
    """
    expanded   = []
    phone_mask = []
    gap_mask   = []

    words   = []
    current = []
    for tok in pronounced:
        if tok == WORD_BOUNDARY:
            words.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        words.append(current)

    for w_idx, word in enumerate(words):
        if w_idx > 0:
            expanded.append(WORD_BOUNDARY)
            phone_mask.append(0)
            gap_mask.append(0)

        for phone in word:
            expanded.append(GAP)
            phone_mask.append(0)
            gap_mask.append(1)
            expanded.append(phone)
            phone_mask.append(1)
            gap_mask.append(0)

        expanded.append(GAP)
        phone_mask.append(0)
        gap_mask.append(1)

    return expanded, phone_mask, gap_mask


def build_phone_labels(
    expanded: list,
    phone_errors: list,
    phone_mask: list,
) -> list:
    """
    Map the flat `phone errors` label list (aligned to pronounced, with <|>)
    onto the expanded sequence. Returns a label list of len(expanded) where
    non-phone positions are -100 (ignored in loss).
    """
    flat_labels = [int(x) for x in phone_errors if x != WORD_BOUNDARY]
    labels = []
    phone_idx = 0
    for is_phone in phone_mask:
        if is_phone:
            labels.append(flat_labels[phone_idx] if phone_idx < len(flat_labels) else 0)
            phone_idx += 1
        else:
            labels.append(-100)
    return labels


def build_gap_labels(
    expanded: list,
    gap_mask: list,
    words: list,
) -> list:
    """
    Build binary gap labels from the per-word mispronunciations lists.
    A gap at position g (the i-th gap within a word, 0-indexed) is labeled 1
    if a deletion was recorded at canonical index g in that word.

    Returns a label list of len(expanded) where non-gap positions are -100.
    """
    gap_positions = []
    w_idx   = 0
    gap_cnt = 0

    for tok, is_gap in zip(expanded, gap_mask):
        if tok == WORD_BOUNDARY:
            w_idx  += 1
            gap_cnt = 0
        elif is_gap:
            gap_positions.append((w_idx, gap_cnt))
            gap_cnt += 1

    deletion_indices: list = []
    for word in words:
        dels = set()
        for mis in word.get("mispronunciations", []):
            if mis.get("pronounced") == "<DEL>":
                dels.add(mis["index"])
        deletion_indices.append(dels)

    labels  = []
    gap_ptr = 0
    for is_gap in gap_mask:
        if is_gap:
            w_idx, g_pos = gap_positions[gap_ptr]
            gap_ptr += 1
            if w_idx < len(deletion_indices) and g_pos in deletion_indices[w_idx]:
                labels.append(1)
            else:
                labels.append(0)
        else:
            labels.append(-100)

    return labels


def build_correction_labels(
    expanded: list,
    phone_mask: list,
    gap_mask: list,
    phone_labels: list,
    gap_labels: list,
    words: list,
    vocab: Vocab,
) -> list:
    """
    Build correction targets: for positions that are substitution errors
    (phone_label=1) or deletions (gap_label=1), the target is the vocab index
    of the canonical phone. All other positions are -100 (ignored in loss).
    """
    sub_canonicals = []
    for word in words:
        phone_list = word.get("pronounced", [])
        mis_by_idx = {
            m["index"]: m["canonical"]
            for m in word.get("mispronunciations", [])
            if m.get("pronounced") != "<DEL>"
        }
        for i in range(len(phone_list)):
            sub_canonicals.append(mis_by_idx.get(i, None))

    del_canonicals = []
    for word in words:
        n_phones   = len(word.get("pronounced", []))
        del_by_idx = {
            m["index"]: m["canonical"]
            for m in word.get("mispronunciations", [])
            if m.get("pronounced") == "<DEL>"
        }
        for g in range(n_phones + 1):
            del_canonicals.append(del_by_idx.get(g, None))

    labels  = []
    sub_ptr = 0
    del_ptr = 0

    for is_phone, is_gap, ph_lbl, g_lbl in zip(
        phone_mask, gap_mask, phone_labels, gap_labels
    ):
        if is_phone:
            if ph_lbl == 1 and sub_ptr < len(sub_canonicals):
                canonical = sub_canonicals[sub_ptr]
                labels.append(vocab.encode(canonical) if canonical else -100)
            else:
                labels.append(-100)
            sub_ptr += 1
        elif is_gap:
            if g_lbl == 1 and del_ptr < len(del_canonicals):
                canonical = del_canonicals[del_ptr]
                labels.append(vocab.encode(canonical) if canonical else -100)
            else:
                labels.append(-100)
            del_ptr += 1
        else:
            labels.append(-100)

    return labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MispronunciationDataset(Dataset):

    def __init__(self, data: list, vocab: Vocab):
        self.samples = []
        for sentence in data:
            try:
                self._process(sentence, vocab)
            except Exception as e:
                print(f"[WARNING] Skipping sentence {sentence.get('id')}: {e}")

    def _process(self, sentence: dict, vocab: Vocab):
        pronounced      = sentence["pronounced"]
        words           = sentence["words"]
        phone_errors    = sentence["phone errors"]
        word_errors     = sentence["word errors"]
        sentence_errors = int(sentence.get("sentence errors", 0))

        expanded, phone_mask, gap_mask = expand_with_gaps(pronounced)

        token_ids = [vocab.encode(t) for t in expanded]

        phone_labels      = build_phone_labels(expanded, phone_errors, phone_mask)
        gap_labels        = build_gap_labels(expanded, gap_mask, words)
        correction_labels = build_correction_labels(
            expanded, phone_mask, gap_mask,
            phone_labels, gap_labels, words, vocab
        )

        self.samples.append({
            "id":                sentence["id"],
            "token_ids":         token_ids,
            "phone_mask":        phone_mask,
            "gap_mask":          gap_mask,
            "phone_labels":      phone_labels,
            "gap_labels":        gap_labels,
            "correction_labels": correction_labels,
            "sentence_errors":   sentence_errors,
            "pronounced":        pronounced,
            "reference":         sentence.get("reference", []),
            "text":              sentence.get("text", ""),
            # Original GT fields in eval-pipeline format
            "phone errors":      sentence.get("phone errors", []),
            "word errors":       sentence.get("word errors", []),
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list) -> dict:
    """Pad all sequences in a batch to the same length."""
    max_len = max(len(s["token_ids"]) for s in batch)

    def pad(seq, pad_val=0):
        return seq + [pad_val] * (max_len - len(seq))

    def pad_labels(seq):
        return seq + [-100] * (max_len - len(seq))

    token_ids      = torch.tensor([pad(s["token_ids"])                for s in batch], dtype=torch.long)
    attention_mask = torch.tensor(
        [[1] * len(s["token_ids"]) + [0] * (max_len - len(s["token_ids"])) for s in batch],
        dtype=torch.bool
    )
    phone_mask   = torch.tensor([pad(s["phone_mask"])          for s in batch], dtype=torch.bool)
    gap_mask     = torch.tensor([pad(s["gap_mask"])            for s in batch], dtype=torch.bool)
    phone_labels = torch.tensor([pad_labels(s["phone_labels"]) for s in batch], dtype=torch.long)
    gap_labels   = torch.tensor([pad_labels(s["gap_labels"])   for s in batch], dtype=torch.long)
    corr_labels  = torch.tensor([pad_labels(s["correction_labels"]) for s in batch], dtype=torch.long)

    sentence_errors = torch.tensor([s["sentence_errors"] for s in batch], dtype=torch.long)

    return {
        "ids":               [s["id"] for s in batch],
        "token_ids":         token_ids,
        "attention_mask":    attention_mask,
        "phone_mask":        phone_mask,
        "gap_mask":          gap_mask,
        "phone_labels":      phone_labels,
        "gap_labels":        gap_labels,
        "correction_labels": corr_labels,
        "sentence_errors":   sentence_errors,
        "raw":               batch,
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MispronunciationModel(nn.Module):
    """
    Small Transformer encoder with three classification heads.

    Defaults target an A5000 / RTX 3090 with <10k training samples:
      d_model=256, n_heads=8, n_layers=4, d_ff=1024
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 256,
        n_heads:    int   = 8,
        n_layers:   int   = 4,
        d_ff:       int   = 1024,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.phone_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        self.gap_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        self.correction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )

        # Class weights for imbalanced phone/gap labels (updated via set_class_weights)
        self.register_buffer("phone_class_weights", torch.ones(2))
        self.register_buffer("gap_class_weights",   torch.ones(2))

    def set_class_weights(self, phone_weights: torch.Tensor, gap_weights: torch.Tensor):
        """Call before training to apply inverse-frequency class weights."""
        self.phone_class_weights = phone_weights.to(self.phone_class_weights.device)
        self.gap_class_weights   = gap_weights.to(self.gap_class_weights.device)

    def encode(
        self,
        token_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        pad_mask = ~attention_mask
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x

    def forward(
        self,
        token_ids:         torch.Tensor,
        attention_mask:    torch.Tensor,
        phone_mask:        torch.Tensor,
        gap_mask:          torch.Tensor,
        phone_labels:      Optional[torch.Tensor] = None,
        gap_labels:        Optional[torch.Tensor] = None,
        correction_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        hidden = self.encode(token_ids, attention_mask)

        phone_logits = self.phone_head(hidden)
        gap_logits   = self.gap_head(hidden)
        corr_logits  = self.correction_head(hidden)

        out = {
            "phone_logits": phone_logits,
            "gap_logits":   gap_logits,
            "corr_logits":  corr_logits,
            "hidden":       hidden,
        }

        if phone_labels is not None:
            out["loss"] = self._compute_loss(
                phone_logits, gap_logits, corr_logits,
                phone_labels, gap_labels, correction_labels,
            )

        return out

    def _compute_loss(
        self,
        phone_logits, gap_logits, corr_logits,
        phone_labels, gap_labels, correction_labels,
    ) -> torch.Tensor:

        phone_loss = F.cross_entropy(
            phone_logits.view(-1, 2),
            phone_labels.view(-1),
            weight=self.phone_class_weights,
            ignore_index=-100,
        )

        gap_loss = F.cross_entropy(
            gap_logits.view(-1, 2),
            gap_labels.view(-1),
            weight=self.gap_class_weights,
            ignore_index=-100,
        )

        corr_loss = F.cross_entropy(
            corr_logits.view(-1, corr_logits.size(-1)),
            correction_labels.view(-1),
            ignore_index=-100,
        )

        # Phone and gap losses are primary; correction is a bonus so weighted lower
        return phone_loss + gap_loss + 0.5 * corr_loss


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: MispronunciationDataset) -> dict:
    """
    Compute inverse-frequency weights for phone and gap binary labels
    to handle class imbalance (most phones are correct, most gaps have no deletion).
    """
    phone_counts = [0, 0]
    gap_counts   = [0, 0]

    for s in dataset.samples:
        for lbl in s["phone_labels"]:
            if lbl != -100:
                phone_counts[lbl] += 1
        for lbl in s["gap_labels"]:
            if lbl != -100:
                gap_counts[lbl] += 1

    def weights(counts):
        total = sum(counts)
        return torch.tensor(
            [total / (2 * c) if c > 0 else 1.0 for c in counts],
            dtype=torch.float,
        )

    return {"phone": weights(phone_counts), "gap": weights(gap_counts)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model:     MispronunciationModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device:    torch.device,
    log_f,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        out = model(
            token_ids         = batch["token_ids"].to(device),
            attention_mask    = batch["attention_mask"].to(device),
            phone_mask        = batch["phone_mask"].to(device),
            gap_mask          = batch["gap_mask"].to(device),
            phone_labels      = batch["phone_labels"].to(device),
            gap_labels        = batch["gap_labels"].to(device),
            correction_labels = batch["correction_labels"].to(device),
        )
        loss = out["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg = total_loss / max(len(loader), 1)
    log_f.write(f"  train_loss={avg:.4f}\n")
    log_f.flush()
    return avg


@torch.no_grad()
def evaluate_epoch(
    model:  MispronunciationModel,
    loader: DataLoader,
    device: torch.device,
    log_f,
    epoch:  int,
) -> dict:
    model.eval()
    total_loss     = 0.0
    all_phone_gt   = []
    all_phone_pred = []

    for batch in loader:
        out = model(
            token_ids         = batch["token_ids"].to(device),
            attention_mask    = batch["attention_mask"].to(device),
            phone_mask        = batch["phone_mask"].to(device),
            gap_mask          = batch["gap_mask"].to(device),
            phone_labels      = batch["phone_labels"].to(device),
            gap_labels        = batch["gap_labels"].to(device),
            correction_labels = batch["correction_labels"].to(device),
        )
        total_loss += out["loss"].item()

        phone_preds = out["phone_logits"].argmax(-1)
        phone_mask  = batch["phone_mask"].to(device)
        phone_lbls  = batch["phone_labels"].to(device)

        for b in range(phone_preds.size(0)):
            for pos in range(phone_preds.size(1)):
                if phone_lbls[b, pos] != -100:
                    all_phone_gt.append(phone_lbls[b, pos].item())
                    all_phone_pred.append(phone_preds[b, pos].item())

    avg_loss   = total_loss / max(len(loader), 1)
    phone_f1   = f1_score(all_phone_gt,  all_phone_pred,  zero_division=0) if all_phone_gt else 0.0
    phone_prec = precision_score(all_phone_gt, all_phone_pred, zero_division=0) if all_phone_gt else 0.0
    phone_rec  = recall_score(all_phone_gt,  all_phone_pred,  zero_division=0) if all_phone_gt else 0.0

    metrics = {
        "val_loss":   avg_loss,
        "phone_f1":   phone_f1,
        "phone_prec": phone_prec,
        "phone_rec":  phone_rec,
    }

    line = (
        f"  epoch={epoch} val_loss={avg_loss:.4f} "
        f"phone_f1={phone_f1:.4f} phone_prec={phone_prec:.4f} "
        f"phone_rec={phone_rec:.4f}"
    )
    print(line)
    log_f.write(line + "\n")
    log_f.flush()
    return metrics


# ---------------------------------------------------------------------------
# Prediction / inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model:  MispronunciationModel,
    loader: DataLoader,
    vocab:  Vocab,
    device: torch.device,
) -> list:
    """
    Run inference and return a list of prediction dicts compatible with
    evaluate.py's self-contained prediction file format.
    """
    model.eval()
    results     = []
    special_ids = {vocab.encode(t) for t in SPECIAL_TOKENS}

    for batch in loader:
        out = model(
            token_ids      = batch["token_ids"].to(device),
            attention_mask = batch["attention_mask"].to(device),
            phone_mask     = batch["phone_mask"].to(device),
            gap_mask       = batch["gap_mask"].to(device),
        )

        phone_preds = out["phone_logits"].argmax(-1)
        gap_preds   = out["gap_logits"].argmax(-1)
        corr_preds  = out["corr_logits"].argmax(-1)
        hidden      = out["hidden"]
        B           = phone_preds.size(0)

        for b in range(B):
            raw        = batch["raw"][b]
            pm_b       = batch["phone_mask"][b]
            gm_b       = batch["gap_mask"][b]
            ph_p       = phone_preds[b].cpu()
            g_p        = gap_preds[b].cpu()
            c_p        = corr_preds[b].cpu()
            pronounced = raw["pronounced"]

            expanded, phone_mask_list, gap_mask_list = expand_with_gaps(pronounced)

            # --- Phone error labels and word error counts ---
            pred_phone_errors      = []
            pred_word_errors_counts = []
            deletion_predictions   = []
            word_phone_error_count = 0
            flat_phone_idx         = 0
            pos                    = 0

            for tok, is_ph, is_g in zip(expanded, phone_mask_list, gap_mask_list):
                if tok == WORD_BOUNDARY:
                    pred_phone_errors.append(WORD_BOUNDARY)
                    pred_word_errors_counts.append(word_phone_error_count)
                    word_phone_error_count = 0
                    continue

                if is_ph and pos < len(pm_b):
                    err = int(ph_p[pos].item())
                    pred_phone_errors.append(err)
                    if err == 1:
                        word_phone_error_count += 1
                    flat_phone_idx += 1
                    pos += 1

                elif is_g and pos < len(gm_b):
                    del_pred = int(g_p[pos].item())
                    if del_pred == 1:
                        # Record gap but do NOT count toward word_phone_error_count —
                        # word errors are defined over phone positions only
                        deletion_predictions.append(
                            [flat_phone_idx - 1, flat_phone_idx]
                            if flat_phone_idx > 0 else [0, 0]
                        )
                    pos += 1

            pred_word_errors_counts.append(word_phone_error_count)

            # --- Assemble predicted reference sequence ---
            # phone_head=0  → keep phone as-is
            # phone_head=1  → replace with correction_head output
            # gap_head=1    → insert correction_head output (predicted deleted phone)
            # gap_head=0    → skip gap (no deletion)
            # Special token ids are never valid correction outputs; fall back to original phone.
            prediction = []
            pos2       = 0
            for tok, is_ph, is_g in zip(expanded, phone_mask_list, gap_mask_list):
                if tok == WORD_BOUNDARY:
                    prediction.append(WORD_BOUNDARY)
                    continue
                if is_ph and pos2 < len(ph_p):
                    if ph_p[pos2].item() == 0:
                        prediction.append(tok)
                    else:
                        corr_id = int(c_p[pos2].item())
                        prediction.append(
                            vocab.decode(corr_id) if corr_id not in special_ids else tok
                        )
                    pos2 += 1
                elif is_g and pos2 < len(g_p):
                    if g_p[pos2].item() == 1:
                        corr_id = int(c_p[pos2].item())
                        if corr_id not in special_ids:
                            prediction.append(vocab.decode(corr_id))
                    pos2 += 1

            pred_sentence_errors = int(
                sum(1 for w in pred_word_errors_counts if w > 0)
            )

            result = {
                "id":                        raw["id"],
                "phone errors":              raw.get("phone errors", []),
                "word errors":               raw.get("word errors", []),
                "sentence errors":           int(raw["sentence_errors"]),
                "reference":                 raw["reference"],
                "pronounced":                raw["pronounced"],
                "predicted phone errors":    pred_phone_errors,
                "predicted word errors":     pred_word_errors_counts,
                "predicted sentence errors": pred_sentence_errors,
                "prediction":                prediction,
                "deletion_predictions":      deletion_predictions,
            }
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_train(args):
    os.makedirs(args.output, exist_ok=True)

    set_seed(SEED)

    if not args.train:
        train_files = input("Training files (space-separated paths): ").split(" ")
        args.train = train_files

    train_files = args.train

    print("Loading data...")
    train_data = []
    for f in train_files:
        with open(f, encoding="utf-8") as f_in:
            train_data.extend(json.load(f_in))
    val_data = json.load(open(args.val, encoding="utf-8"))

    print("Building vocabulary...")
    if args.vocab:
        vocab = Vocab.load(args.vocab)
        print(f"  Loaded shared vocab from {args.vocab}")
    else:
        vocab      = Vocab()
        vocab_data = train_data + val_data
        for extra in (args.vocab_extra or []):
            vocab_data += json.load(open(extra, encoding="utf-8"))
        vocab.build(vocab_data)
        vocab.save(os.path.join(args.output, "vocab.json"))
        print("  Built vocab from provided data")
    print(f"  Vocab size: {len(vocab)}")

    print("Building datasets...")
    train_ds = MispronunciationDataset(train_data, vocab)
    val_ds   = MispronunciationDataset(val_data,   vocab)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    cw = compute_class_weights(train_ds)
    print(f"  Phone class weights: correct={cw['phone'][0]:.3f}  error={cw['phone'][1]:.3f}")
    print(f"  Gap   class weights: no-del={cw['gap'][0]:.3f}    del={cw['gap'][1]:.3f}")

    model = MispronunciationModel(
        vocab_size = len(vocab),
        d_model    = args.d_model,
        n_heads    = args.n_heads,
        n_layers   = args.n_layers,
        d_ff       = args.d_ff,
        dropout    = args.dropout,
    ).to(device)
    model.set_class_weights(cw["phone"].to(device), cw["gap"].to(device))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler    = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    log_path  = os.path.join(args.output, "training_log.txt")
    best_ckpt = os.path.join(args.output, "best_model.pt")
    best_f1   = -1.0

    with open(log_path, "w") as log_f:
        log_f.write(f"train data from {train_files}\n")
        log_f.write(f"vocab_size={len(vocab)} params={n_params}\n")
        log_f.write(f"train={len(train_ds)} val={len(val_ds)}\n\n")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            log_f.write(f"Epoch {epoch}/{args.epochs}\n")
            print(f"\nEpoch {epoch}/{args.epochs}")

            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, log_f)
            print(f"  train_loss={train_loss:.4f}")

            metrics = evaluate_epoch(model, val_loader, device, log_f, epoch)

            elapsed = time.time() - t0
            log_f.write(f"  elapsed={elapsed:.1f}s\n\n")

            if metrics["phone_f1"] > best_f1:
                best_f1 = metrics["phone_f1"]
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "vocab_size":  len(vocab),
                    "d_model":     args.d_model,
                    "n_heads":     args.n_heads,
                    "n_layers":    args.n_layers,
                    "d_ff":        args.d_ff,
                    "dropout":     args.dropout,
                    "metrics":     metrics,
                }, best_ckpt)
                print(f"  ✓ New best phone_f1={best_f1:.4f} — checkpoint saved.")
                log_f.write(f"  ✓ New best phone_f1={best_f1:.4f}\n")

    print(f"\nTraining complete. Best val phone_f1={best_f1:.4f}")
    print(f"Checkpoint: {best_ckpt}")
    print(f"Log:        {log_path}")

    if args.test:
        _run_predict_from_checkpoint(best_ckpt, args.test, args.output, vocab, device)


def run_predict(args):
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")
    ckpt_path  = os.path.join(output_dir, "best_model.pt")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab      = Vocab.load(vocab_path)
    _run_predict_from_checkpoint(ckpt_path, args.test, output_dir, vocab, device)


def _run_predict_from_checkpoint(
    ckpt_path:  str,
    test_path:  str,
    output_dir: str,
    vocab:      Vocab,
    device:     torch.device,
):
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = MispronunciationModel(
        vocab_size = ckpt["vocab_size"],
        d_model    = ckpt["d_model"],
        n_heads    = ckpt["n_heads"],
        n_layers   = ckpt["n_layers"],
        d_ff       = ckpt["d_ff"],
        dropout    = ckpt.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded from epoch {ckpt['epoch']} "
          f"(val phone_f1={ckpt['metrics']['phone_f1']:.4f})")

    test_data   = json.load(open(test_path, encoding="utf-8"))
    test_ds     = MispronunciationDataset(test_data, vocab)
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"Running inference on {len(test_ds)} test samples...")
    predictions = predict(model, test_loader, vocab, device)

    out_path = os.path.join(output_dir, "predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Predictions written to: {out_path}")
    print(f"  (Pass this file to evaluate.py for full metrics)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a mispronunciation detection model."
    )
    subparsers = parser.add_subparsers(dest="command")

    tr = subparsers.add_parser("train", help="Train the model.")
    tr.add_argument("--val",         type=str, default="data/real_data/validate.json")
    tr.add_argument("--output",      required=True, help="Output directory.")
    tr.add_argument("--test",        type=str, default="data/real_data/test.json")
    tr.add_argument("--train",       type=str, default=None, nargs="+")
    tr.add_argument("--epochs",      type=int,   default=20)
    tr.add_argument("--batch_size",  type=int,   default=32)
    tr.add_argument("--lr",          type=float, default=3e-4)
    tr.add_argument("--d_model",     type=int,   default=256)
    tr.add_argument("--n_heads",     type=int,   default=8)
    tr.add_argument("--n_layers",    type=int,   default=4)
    tr.add_argument("--d_ff",        type=int,   default=1024)
    tr.add_argument("--dropout",     type=float, default=0.1)
    tr.add_argument("--vocab",       default="training/vocab.json",
                    help="Path to a pre-built shared vocab.json. Recommended for "
                         "ablation studies so all conditions use identical vocabularies.")
    tr.add_argument("--vocab_extra", nargs="+", default=None,
                    help="Extra JSON data files whose tokens should be included when "
                         "building vocab (e.g. test.json, synthetic data). "
                         "Only used if --vocab is not set.")

    pr = subparsers.add_parser("predict", help="Run inference with a trained model.")
    pr.add_argument("--test",   required=True, help="Test JSON file.")
    pr.add_argument("--output", required=True, help="Directory with vocab.json and best_model.pt.")

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()