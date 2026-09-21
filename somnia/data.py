"""MNIST data loading — fast IDX binary reader with sklearn fallback."""

import gzip
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


# Cached MNIST location from previous projects
_MNIST_CACHE = Path(r"C:\Users\haida\.gemini\antigravity\scratch\proprioception\data\MNIST\raw")

_IDX_FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images":  "t10k-images-idx3-ubyte",
    "test_labels":  "t10k-labels-idx1-ubyte",
}


def _read_idx_images(path: Path) -> np.ndarray:
    """Read IDX image file (plain or gzipped)."""
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        opener = gzip.open(gz, "rb")
    elif path.exists():
        opener = open(path, "rb")
    else:
        raise FileNotFoundError(f"Neither {path} nor {gz} found")

    with opener as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _read_idx_labels(path: Path) -> np.ndarray:
    """Read IDX label file (plain or gzipped)."""
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        opener = gzip.open(gz, "rb")
    elif path.exists():
        opener = open(path, "rb")
    else:
        raise FileNotFoundError(f"Neither {path} nor {gz} found")

    with opener as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def _try_idx_load():
    """Attempt to load MNIST from cached IDX files."""
    try:
        train_x = _read_idx_images(_MNIST_CACHE / _IDX_FILES["train_images"])
        train_y = _read_idx_labels(_MNIST_CACHE / _IDX_FILES["train_labels"])
        test_x = _read_idx_images(_MNIST_CACHE / _IDX_FILES["test_images"])
        test_y = _read_idx_labels(_MNIST_CACHE / _IDX_FILES["test_labels"])
        return train_x, train_y, test_x, test_y
    except FileNotFoundError:
        return None


def _sklearn_fallback():
    """Fallback to sklearn digits (8x8 → upscaled to 784 via repetition)."""
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0  # 8x8 = 64 features
    y = digits.target.astype(np.int64)
    # Tile to 784 dimensions to match MNIST interface
    reps = 784 // 64  # = 12, gives 768, pad remaining 16
    X_big = np.concatenate([np.tile(X, (1, reps)), X[:, :16]], axis=1)
    # Split 80/20
    n = int(0.8 * len(X_big))
    return X_big[:n], y[:n], X_big[n:], y[n:]


def load_mnist():
    """Load MNIST data, return (train_x, train_y, test_x, test_y) as numpy arrays.

    train_x/test_x: float32 in [0,1], shape (N, 784)
    train_y/test_y: int64, shape (N,)
    """
    result = _try_idx_load()
    if result is not None:
        print(f"[SOMNIA] Loaded MNIST from IDX cache ({_MNIST_CACHE})")
        return result
    print("[SOMNIA] IDX cache not found, falling back to sklearn digits")
    return _sklearn_fallback()


def make_dataloaders(batch_size: int = 128, val_split: int = 5000):
    """Create train, val, and test DataLoaders.

    Splits the last `val_split` samples from training set for calibration.
    Returns (train_loader, val_loader, test_loader, val_x_np, val_y_np).
    """
    train_x, train_y, test_x, test_y = load_mnist()

    # Hold out validation set for introspective head calibration
    val_x_np = train_x[-val_split:]
    val_y_np = train_y[-val_split:]
    trn_x_np = train_x[:-val_split]
    trn_y_np = train_y[:-val_split]

    train_ds = TensorDataset(
        torch.from_numpy(trn_x_np), torch.from_numpy(trn_y_np)
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_x_np), torch.from_numpy(val_y_np)
    )
    test_ds = TensorDataset(
        torch.from_numpy(test_x), torch.from_numpy(test_y)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, val_x_np, val_y_np
