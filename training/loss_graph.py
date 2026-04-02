"""
plot_training_log.py

Usage:
    python plot_training_log.py <log_file>
    python plot_training_log.py <log_file> --output_dir ./plots

Parses a BiLSTM training log and saves two charts:
  1. Loss curves  (train loss + val loss)
  2. Metrics      (phone F1, precision, recall)
"""

import re
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_log(path: str) -> dict:
    epochs, train_loss, val_loss = [], [], []
    phone_f1, phone_prec, phone_rec = [], [], []

    epoch_pat     = re.compile(r"Epoch\s+(\d+)/\d+")
    train_pat     = re.compile(r"train_loss\s*=\s*([\d.]+|nan)", re.I)
    val_pat       = re.compile(r"epoch\s*=\s*(\d+)\s+val_loss\s*=\s*([\d.]+|nan)", re.I)
    phone_f1_pat  = re.compile(r"phone_f1\s*=\s*([\d.]+|nan)", re.I)
    phone_prec_pat= re.compile(r"phone_prec\s*=\s*([\d.]+|nan)", re.I)
    phone_rec_pat = re.compile(r"phone_rec\s*=\s*([\d.]+|nan)", re.I)

    current_epoch      = None
    current_train_loss = None

    with open(path) as f:
        for line in f:
            # Epoch header
            m = epoch_pat.search(line)
            if m:
                current_epoch = int(m.group(1))
                current_train_loss = None
                continue

            # Train loss (appears on the line after the epoch header)
            m = train_pat.search(line)
            if m and current_epoch is not None and current_train_loss is None:
                raw = m.group(1)
                current_train_loss = float("nan") if raw == "nan" else float(raw)
                continue

            # Validation line (contains epoch number + all val metrics)
            m = val_pat.search(line)
            if m:
                ep = int(m.group(1))
                vl = m.group(2)
                vl = float("nan") if vl == "nan" else float(vl)

                pf1  = phone_f1_pat.search(line)
                ppr  = phone_prec_pat.search(line)
                pre  = phone_rec_pat.search(line)

                def safe(match):
                    if match is None:
                        return float("nan")
                    v = match.group(1)
                    return float("nan") if v == "nan" else float(v)

                epochs.append(ep)
                train_loss.append(current_train_loss if current_train_loss is not None else float("nan"))
                val_loss.append(vl)
                phone_f1.append(safe(pf1))
                phone_prec.append(safe(ppr))
                phone_rec.append(safe(pre))

    return dict(
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        phone_f1=phone_f1,
        phone_prec=phone_prec,
        phone_rec=phone_rec,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "train_loss": "#0C4EB7",
    "val_loss":   "#82149D",
    "phone_f1":   "#1DF8F4",
    "phone_prec": "#220BD0",
    "phone_rec":    "#ED3AD5",
}

def _apply_common_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.85)


def plot_loss(data: dict, ax):
    ep = data["epochs"]
    ax.plot(ep, data["train_loss"], label="Train loss",
            color=COLORS["train_loss"], linewidth=2, marker="o", markersize=3)
    ax.plot(ep, data["val_loss"],   label="Val loss",
            color=COLORS["val_loss"],   linewidth=2, marker="s", markersize=3,
            linestyle="--")
    _apply_common_style(ax, "Loss curves", "Epoch", "Loss")


def plot_metrics(data: dict, ax):
    ep = data["epochs"]
    ax.plot(ep, data["phone_f1"],  label="Phone F1",
            color=COLORS["phone_f1"],  linewidth=2, marker="o", markersize=3)
    ax.plot(ep, data["phone_prec"],label="Phone precision",
            color=COLORS["phone_prec"],linewidth=2, marker="^", markersize=3,
            linestyle="--")
    ax.plot(ep, data["phone_rec"], label="Phone recall",
            color=COLORS["phone_rec"], linewidth=2, marker="v", markersize=3,
            linestyle=":")
    ax.set_ylim(0, 1)
    _apply_common_style(ax, "Phone metrics", "Epoch", "Score")


def save_charts(data: dict, output_dir: Path, log_stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1 — Loss curves
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    plot_loss(data, ax1)
    fig1.tight_layout()
    p1 = output_dir / f"{log_stem}_loss.pdf"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"Saved: {p1}")
    plt.close(fig1)

    # Chart 2 — Metrics
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    plot_metrics(data, ax2)
    fig2.tight_layout()
    p2 = output_dir / f"{log_stem}_metrics.pdf"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"Saved: {p2}")
    plt.close(fig2)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot train/val loss and phone metrics from a BiLSTM training log."
    )
    parser.add_argument("log_file", help="Path to the training log file")
    parser.add_argument(
        "--output_dir", default=".",
        help="Directory to write  charts (default: current directory)"
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: file not found — {log_path}", file=sys.stderr)
        sys.exit(1)

    data = parse_log(log_path)
    if not data["epochs"]:
        print("Error: no epoch data found in log file.", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(data['epochs'])} epochs from {log_path.name}")
    save_charts(data, Path(args.output_dir), log_path.stem)


if __name__ == "__main__":
    main()