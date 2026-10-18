"""Train the CNN + BiLSTM patch-level Marine Debris classifier.

Usage:
  python scripts/train_classifier.py --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ingestion import MaridaDataset
from src.data.splits import load_official_splits
from src.models.cnn_lstm import CNNBiLSTMClassifier, PatchClassDataset


def _run_epoch(model, loader, device, criterion=None, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, n = 0.0, 0, 0
    tp = fp = fn = 0
    with torch.set_grad_enabled(is_train):
        for image, label in loader:
            image, label = image.to(device), label.to(device)
            logits = model(image)
            loss = criterion(logits, label) if criterion is not None else None
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = logits.argmax(dim=1)
            correct += int((preds == label).sum())
            n += int(label.numel())
            tp += int(((preds == 1) & (label == 1)).sum())
            fp += int(((preds == 1) & (label == 0)).sum())
            fn += int(((preds == 0) & (label == 1)).sum())
            total_loss += float(loss.item()) * label.numel() if loss is not None else 0.0
    acc = correct / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"loss": total_loss / max(n, 1), "acc": acc, "precision": prec, "recall": rec, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")
    if args.smoke:
        splits = {k: v[:60] for k, v in splits.items()}

    train_ds = PatchClassDataset(marida, splits["train"])
    val_ds = PatchClassDataset(marida, splits["val"])
    test_ds = PatchClassDataset(marida, splits["test"])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    pos = sum(1 for _, y in train_ds)
    neg = len(train_ds) - pos
    w = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32).to(device)

    model = CNNBiLSTMClassifier().to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    out_dir = ROOT / "artifacts" / "cnn_lstm_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_epoch = -1.0, -1
    history = []
    for epoch in range(1, args.epochs + 1):
        tr = _run_epoch(model, train_loader, device, criterion, optimizer)
        va = _run_epoch(model, val_loader, device)
        score = va["f1"]
        history.append({"epoch": epoch, "train_acc": tr["acc"], "val_acc": va["acc"], "val_f1": va["f1"]})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, out_dir / "best.pth")
        print(f"epoch {epoch:02d}: train_acc={tr['acc']:.3f} val_acc={va['acc']:.3f} val_f1={va['f1']:.3f}")

    from src.utils.checkpoint import safe_load_state_dict
    safe_load_state_dict(model, out_dir / "best.pth", map_location="cpu")
    te = _run_epoch(model, test_loader, device)
    summary = {"accuracy": te["acc"], "precision": te["precision"], "recall": te["recall"], "f1_debris": te["f1"], "best_epoch": best_epoch}
    (out_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"test: acc={te['acc']:.3f} precision={te['precision']:.3f} recall={te['recall']:.3f} f1={te['f1']:.3f}")


if __name__ == "__main__":
    main()