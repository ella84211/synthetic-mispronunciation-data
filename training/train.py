"""
Mispronunciation Detection — BiLSTM Training & Inference Script
===============================================================

Architecture: stacked BiLSTM encoder with three prediction heads.

  INPUT REPRESENTATION
  --------------------
  The pronounced sequence is expanded with <GAP> tokens interleaved between
  every adjacent phone pair and at word edges. This gives the BiLSTM explicit
  gap positions for deletion detection.

  HEADS
  -----
  1. phone_head      — binary classifier on phone positions (correct / error)
  2. gap_head        — binary classifier on gap positions (no deletion / deletion)
  3. correction_head — multiclass over vocab; predicts canonical phone at
                       error and deletion positions

  ARCHITECTURE NOTES
  ------------------
  - Stacked BiLSTM (default: 2 layers, 256 hidden units per direction)
  - Hidden size per position = 2 * hidden_size (concatenation of fwd + bwd)
  - Dropout applied between LSTM layers and before each classification head
  - Class weights applied to phone and gap losses to handle imbalance

USAGE
-----
  # Train
  python train.py train \\
      --val    data/val.json   \\
      --output runs/bilstm_real_only

  # Inference only
  python train.py predict \\
      --test   data/test.json \\
      --output runs/bilstm_real_only

  # With shared vocab
  python train.py train \\
      --val    data/val.json   \\
      --output bilstm_experiments/bilstm_real_only \\
      --vocab  training/shared_vocab.json \\
      --test   data/test.json
"""

import json
import os
import random
import argparse
import time
import warnings
from collections import defaultdict
from typing import Optional

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------

SEED = 3407

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
# Data helpers
# ---------------------------------------------------------------------------

def expand_with_gaps(pronounced: list) -> tuple:
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


def build_phone_labels(expanded: list, phone_errors: list, phone_mask: list) -> list:
    flat_labels = [int(x) for x in phone_errors if x != WORD_BOUNDARY]
    labels      = []
    phone_idx   = 0
    for is_phone in phone_mask:
        if is_phone:
            labels.append(flat_labels[phone_idx] if phone_idx < len(flat_labels) else 0)
            phone_idx += 1
        else:
            labels.append(-100)
    return labels


def build_gap_labels(expanded: list, gap_mask: list, words: list) -> list:
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
            labels.append(
                1 if w_idx < len(deletion_indices) and g_pos in deletion_indices[w_idx]
                else 0
            )
        else:
            labels.append(-100)
    return labels


def build_correction_labels(
    expanded: list, phone_mask: list, gap_mask: list,
    phone_labels: list, gap_labels: list, words: list, vocab: Vocab,
) -> list:
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
        token_ids         = [vocab.encode(t) for t in expanded]
        phone_labels      = build_phone_labels(expanded, phone_errors, phone_mask)
        gap_labels        = build_gap_labels(expanded, gap_mask, words)
        correction_labels = build_correction_labels(
            expanded, phone_mask, gap_mask,
            phone_labels, gap_labels, words, vocab
        )
        word_labels = [1 if int(e) > 0 else 0 for e in word_errors]

        self.samples.append({
            "id":                sentence["id"],
            "token_ids":         token_ids,
            "phone_mask":        phone_mask,
            "gap_mask":          gap_mask,
            "phone_labels":      phone_labels,
            "gap_labels":        gap_labels,
            "correction_labels": correction_labels,
            "word_labels":       word_labels,
            "sentence_errors":   sentence_errors,
            "pronounced":        pronounced,
            "reference":         sentence.get("reference", []),
            "text":              sentence.get("text", ""),
            "phone errors":      sentence.get("phone errors", []),
            "word errors":       sentence.get("word errors", []),
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list) -> dict:
    max_len = max(len(s["token_ids"]) for s in batch)

    def pad(seq, pad_val=0):
        return seq + [pad_val] * (max_len - len(seq))

    def pad_labels(seq):
        return seq + [-100] * (max_len - len(seq))

    token_ids      = torch.tensor([pad(s["token_ids"]) for s in batch], dtype=torch.long)
    attention_mask = torch.tensor(
        [[1] * len(s["token_ids"]) + [0] * (max_len - len(s["token_ids"])) for s in batch],
        dtype=torch.bool,
    )
    phone_mask   = torch.tensor([pad(s["phone_mask"])           for s in batch], dtype=torch.bool)
    gap_mask     = torch.tensor([pad(s["gap_mask"])             for s in batch], dtype=torch.bool)
    phone_labels = torch.tensor([pad_labels(s["phone_labels"])  for s in batch], dtype=torch.long)
    gap_labels   = torch.tensor([pad_labels(s["gap_labels"])    for s in batch], dtype=torch.long)
    corr_labels  = torch.tensor([pad_labels(s["correction_labels"]) for s in batch], dtype=torch.long)

    max_words   = max(len(s["word_labels"]) for s in batch)
    word_labels = torch.tensor(
        [s["word_labels"] + [-100] * (max_words - len(s["word_labels"])) for s in batch],
        dtype=torch.long,
    )
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
        "word_labels":       word_labels,
        "sentence_errors":   sentence_errors,
        "raw":               batch,
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BiLSTMMispronunciationModel(nn.Module):
    """
    Stacked BiLSTM encoder with three classification heads.

    Hidden state at each position is the concatenation of the forward and
    backward LSTM outputs, giving a vector of size 2 * hidden_size.
    All three heads operate on this combined representation.

    Default hyperparameters:
      embedding_dim : 128
      hidden_size   : 256  (output per position = 512 after bidirectional concat)
      n_layers      : 2
      dropout       : 0.1
    """

    def __init__(
        self,
        vocab_size:    int,
        embedding_dim: int   = 128,
        hidden_size:   int   = 256,
        n_layers:      int   = 2,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.output_size   = hidden_size * 2   # bidirectional concatenation

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size    = embedding_dim,
            hidden_size   = hidden_size,
            num_layers    = n_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if n_layers > 1 else 0.0,
        )

        D = self.output_size   # 512 with defaults

        self.phone_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D // 2, 2),
        )

        self.gap_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D // 2, 2),
        )

        self.correction_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D, vocab_size),
        )

        # Class weights for imbalanced labels
        self.register_buffer("phone_class_weights", torch.ones(2))
        self.register_buffer("gap_class_weights",   torch.ones(2))

    def set_class_weights(self, phone_weights: torch.Tensor, gap_weights: torch.Tensor):
        self.phone_class_weights = phone_weights.to(self.phone_class_weights.device)
        self.gap_class_weights   = gap_weights.to(self.gap_class_weights.device)

    def encode(
        self,
        token_ids:      torch.Tensor,   # (B, L)
        attention_mask: torch.Tensor,   # (B, L) bool
    ) -> torch.Tensor:                  # (B, L, 2*hidden_size)
        x = self.dropout(self.embedding(token_ids))   # (B, L, E)

        # Pack padded sequence so the LSTM ignores padding
        lengths = attention_mask.sum(dim=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        hidden, _     = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=token_ids.size(1)
        )
        return hidden   # (B, L, 2*hidden_size)

    def forward(
        self,
        token_ids:         torch.Tensor,
        attention_mask:    torch.Tensor,
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

        return phone_loss + gap_loss + 0.5 * corr_loss


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: MispronunciationDataset) -> dict:
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
    model:     BiLSTMMispronunciationModel,
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
    model:  BiLSTMMispronunciationModel,
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
            phone_labels      = batch["phone_labels"].to(device),
            gap_labels        = batch["gap_labels"].to(device),
            correction_labels = batch["correction_labels"].to(device),
        )
        total_loss += out["loss"].item()

        phone_preds = out["phone_logits"].argmax(-1)
        phone_lbls  = batch["phone_labels"].to(device)

        for b in range(phone_preds.size(0)):
            for pos in range(phone_preds.size(1)):
                if phone_lbls[b, pos] != -100:
                    all_phone_gt.append(phone_lbls[b, pos].item())
                    all_phone_pred.append(phone_preds[b, pos].item())

    avg_loss   = total_loss / max(len(loader), 1)
    phone_f1   = f1_score(all_phone_gt,  all_phone_pred,  zero_division=0) if all_phone_gt  else 0.0
    phone_prec = precision_score(all_phone_gt, all_phone_pred, zero_division=0) if all_phone_gt else 0.0
    phone_rec  = recall_score(all_phone_gt,  all_phone_pred,  zero_division=0) if all_phone_gt  else 0.0

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
    model:  BiLSTMMispronunciationModel,
    loader: DataLoader,
    vocab:  Vocab,
    device: torch.device,
) -> list:
    model.eval()
    results     = []
    special_ids = {vocab.encode(t) for t in SPECIAL_TOKENS}

    for batch in loader:
        out = model(
            token_ids      = batch["token_ids"].to(device),
            attention_mask = batch["attention_mask"].to(device),
        )

        phone_preds = out["phone_logits"].argmax(-1)
        gap_preds   = out["gap_logits"].argmax(-1)
        corr_preds  = out["corr_logits"].argmax(-1)
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

            pred_phone_errors       = []
            pred_word_errors_counts = []
            deletion_predictions    = []
            word_phone_error_count  = 0
            flat_phone_idx          = 0
            pos                     = 0

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
                    if int(g_p[pos].item()) == 1:
                        deletion_predictions.append(
                            [flat_phone_idx - 1, flat_phone_idx]
                            if flat_phone_idx > 0 else [0, 0]
                        )
                    pos += 1

            pred_word_errors_counts.append(word_phone_error_count)

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

            results.append({
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
            })

    return results


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_train(args):
    os.makedirs(args.output, exist_ok=True)
    set_seed(SEED)

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

    model = BiLSTMMispronunciationModel(
        vocab_size    = len(vocab),
        embedding_dim = args.embedding_dim,
        hidden_size   = args.hidden_size,
        n_layers      = args.n_layers,
        dropout       = args.dropout,
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
                    "epoch":         epoch,
                    "model_state":   model.state_dict(),
                    "vocab_size":    len(vocab),
                    "embedding_dim": args.embedding_dim,
                    "hidden_size":   args.hidden_size,
                    "n_layers":      args.n_layers,
                    "dropout":       args.dropout,
                    "metrics":       metrics,
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

    model = BiLSTMMispronunciationModel(
        vocab_size    = ckpt["vocab_size"],
        embedding_dim = ckpt["embedding_dim"],
        hidden_size   = ckpt["hidden_size"],
        n_layers      = ckpt["n_layers"],
        dropout       = ckpt.get("dropout", 0.1),
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
        description="Train and evaluate a BiLSTM mispronunciation detection model."
    )
    subparsers = parser.add_subparsers(dest="command")

    tr = subparsers.add_parser("train", help="Train the model.")
    tr.add_argument("--train", required=True, nargs="+",
                    help="One or more training JSON files.")
    tr.add_argument("--val",    default="data/real_data/validate.json", help="Validation JSON file.")
    tr.add_argument("--output", required=True, help="Output directory.")
    tr.add_argument("--test",   default="data/real_data/test.json",  help="Test JSON (optional; runs inference after training).")
    tr.add_argument("--epochs",        type=int,   default=20)
    tr.add_argument("--batch_size",    type=int,   default=32)
    tr.add_argument("--lr",            type=float, default=3e-4)
    tr.add_argument("--embedding_dim", type=int,   default=128,
                    help="Embedding dimension fed into the LSTM.")
    tr.add_argument("--hidden_size",   type=int,   default=256,
                    help="LSTM hidden size per direction (output is 2x this).")
    tr.add_argument("--n_layers",      type=int,   default=2,
                    help="Number of stacked BiLSTM layers.")
    tr.add_argument("--dropout",       type=float, default=0.1)
    tr.add_argument("--vocab",         default="training/vocab.json",
                    help="Path to a pre-built shared vocab.json.")
    tr.add_argument("--vocab_extra",   nargs="+", default=None,
                    help="Extra JSON files to include when building vocab. "
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