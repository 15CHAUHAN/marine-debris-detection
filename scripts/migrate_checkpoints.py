#!/usr/bin/env python
"""Migrate legacy checkpoints saved with weights_only=False to safe format.

This script loads each .pth checkpoint with weights_only=False (legacy),
extracts only the safe tensor keys, and re-saves with torch.save()
which defaults to safe serialization. The migrated files can then be
loaded safely with weights_only=True.

Usage:
    python scripts/migrate_checkpoints.py
    python scripts/migrate_checkpoints.py --dry-run
    python scripts/migrate_checkpoints.py --path artifacts/attention/best.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch


SAFE_KEYS = {
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "epoch",
    "config",
    "best_metrics",
    "best_mean_iou",
    "best_epoch",
}


def _sanitize_config(config: dict) -> dict:
    """Remove non-serializable objects from config dict (e.g., Path objects)."""
    sanitized = {}
    for k, v in config.items():
        if isinstance(v, Path):
            sanitized[k] = str(v)
        elif isinstance(v, dict):
            sanitized[k] = _sanitize_config(v)
        elif isinstance(v, (list, tuple)):
            sanitized[k] = [
                str(item) if isinstance(item, Path) else item
                for item in v
            ]
        else:
            sanitized[k] = v
    return sanitized


def migrate_checkpoint(src: Path, dst: Path | None = None, dry_run: bool = False) -> bool:
    """Migrate a single checkpoint file.

    Returns True if migration was performed, False if skipped (already safe or error).
    """
    if dst is None:
        dst = src

    print(f"Checking {src}...", end=" ")

    # First try loading with weights_only=True to see if already safe
    try:
        torch.load(src, map_location="cpu", weights_only=True)
        print("already safe (weights_only=True works)")
        return False
    except Exception:
        pass  # Not safe, needs migration

    # Try loading with weights_only=False (legacy)
    try:
        state = torch.load(src, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"ERROR: failed to load even with weights_only=False: {e}")
        return False

    # Extract only safe keys
    safe_state = {}
    for key in SAFE_KEYS:
        if key in state:
            if key == "config" and isinstance(state[key], dict):
                safe_state[key] = _sanitize_config(state[key])
            else:
                safe_state[key] = state[key]

    if not safe_state:
        print("ERROR: no safe keys found in checkpoint")
        return False

    print(f"migrating (keys: {list(safe_state.keys())})", end=" ")

    if not dry_run:
        try:
            torch.save(safe_state, dst)
            # Verify the new file loads safely
            torch.load(dst, map_location="cpu", weights_only=True)
            print("-> done and verified")
        except Exception as e:
            print(f"ERROR: failed to save/verify: {e}")
            return False
    else:
        print("-> [dry-run] would migrate")

    return True


def find_checkpoints(root: Path, recursive: bool = True) -> list[Path]:
    """Find all .pth checkpoint files."""
    if recursive:
        return list(root.rglob("*.pth"))
    return list(root.glob("*.pth"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy checkpoints to safe format")
    parser.add_argument("--path", type=Path, help="Specific checkpoint to migrate")
    parser.add_argument("--root", type=Path, default=Path("artifacts"), help="Root directory to search for checkpoints")
    parser.add_argument("--recursive", action="store_true", default=True, help="Search recursively")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Don't search recursively")
    args = parser.parse_args()

    if args.path:
        checkpoints = [args.path]
    else:
        checkpoints = find_checkpoints(args.root, args.recursive)

    if not checkpoints:
        print("No checkpoints found.")
        return 0

    print(f"Found {len(checkpoints)} checkpoint(s)")
    if args.dry_run:
        print("DRY RUN - no files will be modified")
    print()

    migrated = 0
    errors = 0
    for ckpt in checkpoints:
        try:
            if migrate_checkpoint(ckpt, dry_run=args.dry_run):
                migrated += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    print()
    if args.dry_run:
        print(f"Dry run complete. Would migrate {migrated} checkpoint(s).")
    else:
        print(f"Migration complete. {migrated} checkpoint(s) migrated, {errors} error(s).")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())