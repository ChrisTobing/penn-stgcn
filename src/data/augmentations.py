"""
Skeleton data augmentation for Penn Action.

Four augmentations applied independently at training time:
    - Random rotation (spatial)
    - Gaussian joint noise (spatial)
    - Time interpolation (temporal)
    - Time warping (temporal)

Reference: "Enhancing Human Action Recognition with 3D Skeleton Data:
A Comprehensive Study of Deep Learning and Data Augmentation,"
Electronics 13(4), 2024.
"""

import numpy as np


# ─── Spatial augmentations ────────────────────────────────────────────────────

def augment_rotate(sequence: np.ndarray, angle_range: tuple = (5.0, 20.0)) -> np.ndarray:
    """
    Rotate all joints by a random angle to simulate viewpoint variation.

    For 2D Penn Action data this is a standard in-plane rotation:
        R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]

    Args:
        sequence    : (T, V, C=2) float32 array
        angle_range : (min_deg, max_deg)

    Returns:
        (T, V, C=2) float32 array
    """
    theta = np.radians(np.random.uniform(angle_range[0], angle_range[1]))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]], dtype=np.float32)
    return (sequence @ R.T).astype(np.float32)


def augment_gaussian_noise(sequence: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """
    Add independent Gaussian noise to every joint coordinate.

    Simulates localisation error in 2D pose estimation. σ=0.01 is the
    value from the reference paper (Section 3.3.1).

    Args:
        sequence : (T, V, C=2) float32 array
        sigma    : standard deviation of the noise distribution

    Returns:
        (T, V, C=2) float32 array
    """
    noise = np.random.normal(0.0, sigma, sequence.shape).astype(np.float32)
    return (sequence + noise).astype(np.float32)


# ─── Temporal augmentations ───────────────────────────────────────────────────

def augment_time_interpolation(sequence: np.ndarray, gamma: int = 2) -> np.ndarray:
    """
    Densify via interpolation, then uniformly stride-sample to original length.

    Builds a T*gamma densified sequence (original frames + interpolated frames),
    then samples T evenly-spaced frames across the full output.

    Args:
        sequence : (T, V, C=2) float32 array
        gamma    : interpolation factor (≥ 2)

    Returns:
        (T, V, C=2) float32 array
    """
    T, V, C = sequence.shape
    T_out = T * gamma
    output = np.empty((T_out, V, C), dtype=np.float32)

    # Region 1: preserve originals
    output[:T] = sequence

    # Region 2: interpolated frames
    for t in range(T, T_out):
        src_lo = int(np.floor(t / gamma))
        src_hi = min(src_lo + 1, T - 1)
        weight = float((gamma * t) % gamma) / gamma
        output[t] = sequence[src_lo] + weight * (sequence[src_hi] - sequence[src_lo])

    # Uniform stride-sample across full densified sequence
    indices = np.linspace(0, T_out - 1, T, dtype=int)
    return output[indices]


def augment_time_warp(
    sequence: np.ndarray,
    num_segments: int = 5,
    t_std: float = 1.5,
) -> np.ndarray:
    """
    Piecewise random time-scaling across independent segments.

    A fresh offset t ~ N(0, t_std) is drawn per segment and clipped to
    ±segment_length/2. frame lookup uses floor() with no interpolation.

    Args:
        sequence     : (T, V, C=2) float32 array
        num_segments : number of independently warped segments
        t_std        : std of the Gaussian offset

    Returns:
        (T, V, C=2) float32 array
    """
    T = sequence.shape[0]
    seg_len = T // num_segments
    warped = []

    for seg in range(num_segments):
        start = seg * seg_len
        end = T if seg == num_segments - 1 else start + seg_len

        half = seg_len / 2.0
        t = float(np.clip(np.random.normal(0.0, t_std), -half, half))

        for j in range(start, end):
            idx = int(np.clip(int(np.floor(j + t)), 0, T - 1))
            warped.append(sequence[idx])

    return np.stack(warped, axis=0).astype(np.float32)


# ─── Composed pipeline ───────────────────────────────────────────────────────

def apply_augmentations(
    sequence: np.ndarray,
    training: bool = True,
    p_rotate: float = 0.5,
    p_noise: float = 0.5,
    p_interp: float = 0.5,
    p_warp: float = 0.5,
) -> np.ndarray:
    """
    Apply the full augmentation pipeline to one skeleton sequence.

    Each augmentation is applied independently at its own probability.
    Val/test sequences are returned unchanged (training=False).

    Args:
        sequence : (T=100, V=13, C=2) float32 array
        training : if False, returns unmodified (strict no-op)

    Returns:
        (T=100, V=13, C=2) float32 array
    """
    if not training:
        return sequence

    if np.random.random() < p_rotate:
        sequence = augment_rotate(sequence)
    if np.random.random() < p_noise:
        sequence = augment_gaussian_noise(sequence)
    if np.random.random() < p_interp:
        sequence = augment_time_interpolation(sequence)
    if np.random.random() < p_warp:
        sequence = augment_time_warp(sequence)

    return sequence
