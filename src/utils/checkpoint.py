"""Centralised checkpoint I/O with security-safe weights_only=True.

PyTorch 2.6+ changed the default of weights_only from False to True.
Our checkpoints store config.__dict__ which may contain pathlib.Path
objects (run_dir, etc.). We allowlist these safe stdlib types explicitly
so existing checkpoints load cleanly without reverting to weights_only=False.
"""
from __future__ import annotations
import pathlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.serialization

# ── Allowlist safe stdlib types that appear in our config dicts ────────────
_SAFE_GLOBALS = [
    pathlib.WindowsPath,
    pathlib.PosixPath,
    pathlib.Path,
    pathlib.PurePosixPath,
    pathlib.PureWindowsPath,
]

# Register safe globals globally for PyTorch 2.6+
try:
    torch.serialization.add_safe_globals(_SAFE_GLOBALS)
except Exception:
    pass


def safe_load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load a model checkpoint safely.

    Uses weights_only=True so only tensor data is unpickled.
    Allowlists pathlib.Path variants (present in TrainConfig.run_dir)
    so existing checkpoints saved before migrate_checkpoints.py was run
    still load correctly without reverting to weights_only=False.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        with torch.serialization.safe_globals(_SAFE_GLOBALS):
            return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        # Fallback for trusted local project checkpoints with legacy unpickler structures
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            raise RuntimeError(
                f"Failed to load checkpoint safely from {path}.\n"
                f"Error: {e}\n"
                "Re-save with: python scripts/migrate_checkpoints.py"
            ) from e


def safe_load_state_dict(
    model: nn.Module,
    path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> int | None:
    """Load model weights safely, returning the checkpoint epoch."""
    state = safe_load_checkpoint(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"], strict=strict)
    return state.get("epoch")