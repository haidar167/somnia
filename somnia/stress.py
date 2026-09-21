"""Stress dataset generation for training the v2 introspective head."""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def apply_gaussian_noise(images: np.ndarray, std: float = 0.25, seed: int = 42) -> np.ndarray:
    """Add Gaussian noise and clip to [0, 1]."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, std, size=images.shape).astype(np.float32)
    return np.clip(images + noise, 0.0, 1.0)


def apply_rotation(images: np.ndarray, max_angle: float = 30.0, seed: int = 42) -> np.ndarray:
    """Rotate 28x28 images using simple affine grid or scipy/torch."""
    rng = np.random.RandomState(seed)
    n = len(images)
    angles = rng.uniform(-max_angle, max_angle, size=n)
    # Use torch affine grid for fast vectorized rotation on CPU
    x = torch.from_numpy(images).reshape(-1, 1, 28, 28)
    theta = torch.zeros(n, 2, 3)
    for i, ang in enumerate(angles):
        rad = np.radians(ang)
        theta[i, 0, 0] = np.cos(rad)
        theta[i, 0, 1] = -np.sin(rad)
        theta[i, 1, 0] = np.sin(rad)
        theta[i, 1, 1] = np.cos(rad)
    grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
    rotated = torch.nn.functional.grid_sample(x, grid, align_corners=False)
    return rotated.reshape(-1, 784).numpy()


def apply_permutation(images: np.ndarray, seed: int = 42) -> np.ndarray:
    """Apply fixed pixel permutation (Streaming Permuted-MNIST drift)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(784)
    return images[:, perm]


def build_stress_dataset(x_real: np.ndarray, y_real: np.ndarray, seed: int = 42):
    """Build a comprehensive stress dataset containing:
    1. Clean calibration samples (25%)
    2. Gaussian noise perturbed samples (25%)
    3. Rotated ±30° samples (25%)
    4. Permuted-MNIST samples (25%)

    Target error rate under base classifier: ~15-25%.
    """
    n = len(x_real)
    chunk = n // 4

    clean_x = x_real[:chunk]
    clean_y = y_real[:chunk]

    noise_x = apply_gaussian_noise(x_real[chunk:2*chunk], std=0.30, seed=seed)
    noise_y = y_real[chunk:2*chunk]

    rot_x = apply_rotation(x_real[2*chunk:3*chunk], max_angle=30.0, seed=seed)
    rot_y = y_real[2*chunk:3*chunk]

    perm_x = apply_permutation(x_real[3*chunk:], seed=seed)
    perm_y = y_real[3*chunk:]

    stress_x = np.concatenate([clean_x, noise_x, rot_x, perm_x], axis=0)
    stress_y = np.concatenate([clean_y, noise_y, rot_y, perm_y], axis=0)

    return stress_x, stress_y
