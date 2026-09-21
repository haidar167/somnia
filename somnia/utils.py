"""Utility functions: seeding, git hash, JSON result saving."""

import json
import os
import random
import subprocess
import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_hash() -> str:
    """Return the short git hash of the current commit, or 'unknown'."""
    try:
        git_exe = r"C:\Program Files\Git\cmd\git.exe"
        if not os.path.exists(git_exe):
            git_exe = "git"
        result = subprocess.run(
            [git_exe, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_results(filepath: str, metrics: dict, seed: int = 0) -> None:
    """Save experiment results as JSON with metadata."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": seed,
        "git_hash": get_git_hash(),
        **metrics,
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[SOMNIA] Results saved -> {filepath}")


def project_root() -> Path:
    """Return the project root directory (parent of somnia/)."""
    return Path(__file__).resolve().parent.parent
