"""Training loop with checkpointing, loss logging and AMP."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp
from src.data.encoding import IGNORE_INDEX
from src.evaluate.metrics import aggregate_metrics, evaluate_predictions
from src.train.losses import CompoundFocalDiceLoss


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    seed: int = 42
    num_workers: int = 0
    amp: bool = True
    ce_weight: float = 0.5
    dice_weight: float = 0.5
    patience: int = 6
    run_dir: Path = Path("artifacts")
    run_name: str = "run"
    num_classes: int = 11
    class_weights: torch.Tensor | None = None
    label_smoothing: float = 0.0
    loss_type: str = "compound_focal_dice"  # options: "compound_focal_dice", "ce_dice", "tversky"
    focal_gamma: float = 2.0
    lambda_dice: float = 1.0
    lambda_lovasz: float = 0.5
    dice_smooth: float = 1.0
    tversky_alpha: float = 0.7  # alpha > beta heavily penalizes false positives (increases precision)
    tversky_beta: float = 0.3

    # Learning rate schedule settings
    scheduler: str = "cosine_warmup"  # "cosine_warmup" | "plateau" | "cosine" | "none"
    warmup_epochs: int = 3            # linear warmup period before cosine decay
    min_lr: float = 1e-6             # cosine annealing floor
    plateau_factor: float = 0.5      # LR reduction factor for ReduceLROnPlateau
    plateau_patience: int = 3        # epochs without improvement before LR drop


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """Trains one segmentation model, logging per-epoch metrics to CSV/JSON."""

    def __init__(self, model: nn.Module, config: TrainConfig, device: torch.device) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.run_dir = Path(config.run_dir) / config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _set_seed(config.seed)
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.ce_weights = None
        if config.class_weights is not None:
            self.ce_weights = config.class_weights.to(device)
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX,
            weight=self.ce_weights,
            label_smoothing=config.label_smoothing,
        )
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)
        self.compound_loss = None
        if config.loss_type == "compound_focal_dice":
            self.compound_loss = CompoundFocalDiceLoss(
                class_weights=self.ce_weights,
                gamma=config.focal_gamma,
                lambda_dice=config.lambda_dice,
                lambda_lovasz=config.lambda_lovasz,
                smooth=config.dice_smooth,
            )
        self.loss_fn = None
        if config.loss_type == "tversky":
            self.loss_fn = smp.losses.TverskyLoss(
                mode="multiclass",
                alpha=config.tversky_alpha,
                beta=config.tversky_beta,
                ignore_index=IGNORE_INDEX,
            )
        self.scaler = torch.amp.GradScaler(enabled=config.amp and device.type == "cuda")
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self):
        cfg = self.config
        if cfg.scheduler == "cosine_warmup":
            # Linear warmup then cosine annealing to min_lr
            def lr_lambda(epoch):
                if epoch < cfg.warmup_epochs:
                    return float(epoch + 1) / float(max(1, cfg.warmup_epochs))
                progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                alpha = cfg.min_lr / cfg.lr
                return alpha + (1.0 - alpha) * cosine_decay
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",            # monitoring val_mean_iou (higher = better)
                factor=cfg.plateau_factor,
                patience=cfg.plateau_patience,
                min_lr=cfg.min_lr,
                verbose=True,
            )
        elif cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.epochs,
                eta_min=cfg.min_lr,
            )
        else:  # "none"
            return None

    def _record(self, epoch: int, train_loss: float, val_metrics: dict) -> None:
        current_lr = self.optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_mean_iou": val_metrics["mean_iou"],
            "f1_class_1_marine_debris": val_metrics.get("f1_class_1_marine_debris"),
            "iou_class_1_marine_debris": val_metrics.get("iou_class_1_marine_debris"),
            "lr": current_lr,
        }
        csv_path = self.run_dir / "history.csv"
        expected_fields = ["epoch", "train_loss", "val_mean_iou",
                           "f1_class_1_marine_debris", "iou_class_1_marine_debris", "lr"]

        # If file exists, check if it has the 'lr' column
        if csv_path.exists():
            with csv_path.open("r", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader, [])
            if "lr" not in header:
                # Rewrite the entire CSV with the new header
                rows = []
                with csv_path.open("r", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for r in reader:
                        # Ensure all expected fields exist
                        new_row = {field: r.get(field, "") for field in expected_fields}
                        rows.append(new_row)
                with csv_path.open("w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=expected_fields)
                    writer.writeheader()
                    writer.writerows(rows)

        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=expected_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        json_metrics = {k: v for k, v in val_metrics.items() if k != "confusion_matrix"}
        (self.run_dir / "history.json").write_text(json.dumps(json_metrics, indent=2))

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        for image, mask in tqdm(loader, desc="train", leave=False):
            image = image.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.config.amp):
                logits = self.model(image)
                if self.compound_loss is not None:
                    loss = self.compound_loss(logits, mask)
                elif self.loss_fn is not None:
                    loss = self.loss_fn(logits, mask)
                else:
                    loss = (
                        self.config.ce_weight * self.ce_loss(logits, mask)
                        + self.config.dice_weight * self.dice_loss(
                            torch.softmax(logits, dim=1).float(), mask.float()
                        )
                    )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += float(loss.item())
            steps += 1
        return total_loss / max(steps, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        metrics_list = []
        for image, mask in tqdm(loader, desc="valid", leave=False):
            image = image.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.config.amp):
                logits = self.model(image)
            metrics_list.append(evaluate_predictions(logits.float(), mask))
        return aggregate_metrics(metrics_list)

    def run(self, train_loader: DataLoader, val_loader: DataLoader, save_best: bool = True) -> dict:
        best_iou = -1.0
        best_epoch = -1
        best_metrics: dict | None = None
        stagnation = 0
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self._record(epoch, train_loss, val_metrics)
            iou = val_metrics["mean_iou"]
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_mean_iou={val_metrics['mean_iou']} "
                f"f1_class_1_marine_debris={val_metrics.get('f1_class_1_marine_debris')} "
                f"iou_class_1_marine_debris={val_metrics.get('iou_class_1_marine_debris')}"
                f" lr={current_lr:.2e}"
            )
            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(iou)
                else:
                    self.scheduler.step()
            if iou > best_iou:
                best_iou = iou
                best_epoch = epoch
                best_metrics = {
                    k: v for k, v in val_metrics.items() if k != "confusion_matrix"
                }
                stagnation = 0
                if save_best:
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.config.__dict__,
                    }
                    if self.scheduler is not None:
                        checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
                    torch.save(checkpoint, self.run_dir / "best.pth")
            else:
                stagnation += 1
                if stagnation >= self.config.patience:
                    break
        return {
            "best_epoch": best_epoch,
            "best_mean_iou": best_iou,
            "best_metrics": best_metrics,
        }