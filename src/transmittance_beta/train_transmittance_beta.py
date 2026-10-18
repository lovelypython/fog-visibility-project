#!/usr/bin/env python3
"""Train a small CNN to estimate Foggy Cityscapes beta from transmittance maps.

Physical basis: t(x) = exp(-beta * d(x)), hence -log(t) = beta*d.
Because this dataset folder contains transmittance but not depth, beta is
learned as a 3-class image-level label from the filename; depth cannot be
uniquely recovered from transmittance alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BETAS = (0.005, 0.01, 0.02)
SPLITS = ("train", "val", "test")


def beta_from_name(name: str) -> float:
    match = re.search(r"_beta_(0\.005|0\.01|0\.02)\.png$", name)
    if not match:
        raise ValueError(f"Cannot read beta label from filename: {name}")
    return float(match.group(1))


def load_split(root: Path, split: str, size: tuple[int, int]):
    """Decode each source PNG once; reusing in-memory tensors keeps epochs fast."""
    paths = sorted((root / "leftImg8bit_transmittance" / split).rglob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG maps found for split={split} under {root}")
    images = np.empty((len(paths), 1, size[0], size[1]), dtype=np.float32)
    labels = np.empty(len(paths), dtype=np.int64)
    for i, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("L").resize((size[1], size[0]), Image.Resampling.BILINEAR)
            images[i, 0] = np.asarray(image, dtype=np.float32) / 255.0
        labels[i] = BETAS.index(beta_from_name(path.name))
    return TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))


class BetaCNN(nn.Module):
    def __init__(self, classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.1), nn.Linear(24, classes))

    def forward(self, x):
        return self.head(self.features(x))


def evaluate(model, loader, device):
    model.eval()
    confusion = np.zeros((3, 3), dtype=np.int64)
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            pred = logits.argmax(1)
            for actual, guessed in zip(y.cpu().numpy(), pred.cpu().numpy()):
                confusion[actual, guessed] += 1
    n = confusion.sum()
    return loss_sum / n, float(np.trace(confusion) / n), confusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/transmittance_beta"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = (args.height, args.width)
    datasets = {}
    for split in SPLITS:
        print(f"Loading {split} maps into memory...", flush=True)
        datasets[split] = load_split(args.root, split, size)
    loaders = {
        s: DataLoader(datasets[s], batch_size=args.batch_size,
                      shuffle=(s == "train"), num_workers=args.workers,
                      pin_memory=(device.type == "cuda"))
        for s in SPLITS
    }
    args.output.mkdir(parents=True, exist_ok=True)
    model = BetaCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = -1.0
    history = []

    print(f"Device: {device}; samples: " + ", ".join(f"{s}={len(datasets[s])}" for s in SPLITS), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_correct = total = 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        val_loss, val_acc, _ = evaluate(model, loaders["val"], device)
        row = {"epoch": epoch, "train_loss": total_loss / total,
               "train_accuracy": total_correct / total,
               "val_loss": val_loss, "val_accuracy": val_acc}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "betas": BETAS,
                        "image_size": size, "epoch": epoch}, args.output / "best_model.pt")

    with (args.output / "training_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(args.output / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_acc, confusion = evaluate(model, loaders["test"], device)
    results = {
        "target": "filename beta class (inverse meters)", "betas": list(BETAS),
        "best_epoch": checkpoint["epoch"], "best_val_accuracy": best_val,
        "test_loss": test_loss, "test_accuracy": test_acc,
        "test_confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "test_class_counts": {str(beta): int(confusion[i].sum()) for i, beta in enumerate(BETAS)},
        "physical_model": "t = exp(-beta*d); -ln(t) = beta*d",
        "limitation": "Without depth maps, beta and depth are not separately identifiable from transmittance alone.",
    }
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
