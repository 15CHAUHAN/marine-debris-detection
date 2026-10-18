---
name: experiment-tracking
description: Local-first run manifests, metrics capture, and run comparison under artifacts/ and runs/ so every result is reproducible and defensible in review.
---

# Experiment Tracking

## 1. Principle: Local-First, Offline-Reproducible
Faculty review may happen on a machine with no network. Never make a run's record
depend on a cloud service. Files on disk are the source of truth; MLflow/W&B are
optional, opt-in mirrors (see §7).

## 2. The Run Directory Contract
Every run owns one directory and never mutates a finished one:

```text
artifacts/<run_id>/
├── run.json          # config + environment + provenance (required)
├── metrics.json      # final metrics incl. per-class F1 + mIoU
├── metrics.csv       # per-epoch history (long form, greppable)
├── config.yaml       # verbatim resolved config (CLI + --set merged)
├── checkpoints/
│   ├── best.pth
│   └── last.pth
└── plots/
    ├── loss_curves.png
    └── confusion_matrix.png
```

Existing conventions to respect: `artifacts/` holds pipeline outputs,
`runs/` holds Ultralytics YOLO run directories (`runs/detect/...`), `splits/` holds
manifests, `weights/` holds model weights. Do not renumber or reorganize them.

## 3. Run ID
`<model>_<dataset>_<yyyymmdd-HHMM>_<shortsha>`
e.g. `unet_attention_marida_20261018-1430_a1b2c3d`
- Timestamp is local time, zero-padded, minutes resolution.
- Short SHA is the git commit, or `nogit` when the tree is not a repository yet.
- The ID is the directory name; it is never reused or renamed.

## 4. run.json — Mandatory Contents
Write it before training starts and never overwrite it; add `finished_at` and
`status` at the end.

```json
{
  "schema_version": 1,
  "run_id": "unet_attention_marida_20261018-1430_a1b2c3d",
  "status": "running",
  "started_at": "2026-10-18T14:30:11+05:30",
  "finished_at": null,
  "config": { "...": "resolved config, verbatim" },
  "command": ".venv\\Scripts\\python.exe -m scripts.train_one --config configs/unet_marida.yaml",
  "git": { "commit": null, "dirty": null },
  "environment": {
    "python": "3.12.x",
    "torch": "2.14.0+cu126",
    "cuda": "12.6",
    "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
    "cudnn_benchmark": true,
    "amp": true
  },
  "data": {
    "splits_dir": "splits",
    "split_counts": { "train": 694, "val": 328, "test": 359 },
    "num_classes": 11,
    "patch_size": 256,
    "augmentation": "albumentations/2.0.8",
    "gan_augmented": false
  },
  "seed": 0,
  "requirements_sha256": "<hash of requirements.txt>"
}
```

Rationale: `requirements_sha256` plus `torch`/`cuda`/`gpu` answers "why did 0.61
become 0.58?" without re-running anything. See `references/run-manifest.md` for
the writer.

## 5. Metrics To Always Record
| Metric | Notes |
|---|---|
| mIoU | overall segmentation quality |
| per-class F1 (all 11) | exposes rare-class collapse |
| **Class-1 F1 (marine debris)** | the headline number for this project |
| per-class IoU | pairs with F1; same confusion matrix |
| loss history | train + val, per epoch |
| macro F1 | single comparable scalar |
| timing | s/epoch, samples/s, peak VRAM MB |

Class-1 F1 must appear in `metrics.json` under a stable key
(e.g. `"f1_class_1"`) so comparisons and the dashboard report do not parse plots.

## 6. Comparing Runs
- Produce a table from `run.json` + `metrics.json` rather than hand-copying:
  columns = run_id, model, epochs, mIoU, class-1 F1, peak VRAM, duration.
- Only compare runs with the same split manifests. If `split_counts` differ, the
  comparison is invalid — say so explicitly instead of tabulating it.
- Never delete a run. Mark superseded work in `run.json`
  (`"superseded_by": "<run_id>"`) so the lineage stays auditable.
- Guard against leakage when comparing: GAN-augmented runs must be flagged
  (`gan_augmented: true`) and their val/test metrics must come from real data only.

## 7. Optional Mirrors
If MLflow or W&B is added:
- Add the pinned dependency to `requirements.txt`; it is not there today.
- Make the integration opt-in via `--tracker mlflow|wandb|none`, default `none`.
- Wrap import and init in a try/except so a missing package or offline machine
  degrades to file-based logging rather than crashing the run.
- File artifacts remain authoritative; the tracker is a convenience index.

## 8. Anti-Patterns
- Overwriting `metrics.json` to "fix" a bad run.
- Storing only plots — a PNG cannot be re-aggregated.
- Recording the best epoch's metrics without which epoch it was.
- Silent config drift: the run directory's `config.yaml` must match what ran.
- Committing `artifacts/` or `runs/` to git (13.7 GB + 46 MB; see `git-workflow`).
