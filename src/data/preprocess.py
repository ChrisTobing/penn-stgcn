"""
Data preprocessing for Penn Action Dataset.

Handles:
    - .mat file parsing → raw (T, V=13, C=2) arrays
    - Temporal normalisation to T=100 frames
    - Spatial normalisation (hip-centred, torso-scaled)
    - Saving processed joint data as .npy arrays
"""

import os
from pathlib import Path

import numpy as np
import scipy.io as sio


# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    'baseball_pitch', 'baseball_swing', 'bench_press', 'bowl',
    'clean_and_jerk', 'golf_swing', 'jump_rope', 'jumping_jacks',
    'pullup', 'pushup', 'situp', 'squat', 'strum_guitar',
    'tennis_forehand', 'tennis_serve',
]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# Penn Action 13-joint skeleton: 12 directed bone connections.
# Each tuple is (parent, child) — bone vector = joint[child] - joint[parent].
BONE_CONNECTIONS = [
    (0, 1), (0, 2),       # head → shoulders
    (1, 3), (3, 5),       # left arm chain
    (2, 4), (4, 6),       # right arm chain
    (1, 7), (2, 8),       # shoulders → hips
    (7, 8),               # hip cross-bar
    (7, 9), (9, 11),      # left leg chain
    (8, 10), (10, 12),    # right leg chain
]

TARGET_FRAMES = 100


# ─── Raw data extraction ─────────────────────────────────────────────────────

def load_raw_data(mat_directory: str):
    """
    Parse all .mat files in the Penn Action labels directory.

    Returns:
        train_data   : list of (T_i, 13, 2) arrays  (variable-length)
        train_labels : (N_train,) int64 array
        test_data    : list of (T_i, 13, 2) arrays
        test_labels  : (N_test,) int64 array
    """
    train_data, train_labels = [], []
    test_data, test_labels = [], []

    for matfile in sorted(os.listdir(mat_directory)):
        if not matfile.endswith('.mat'):
            continue

        mat = sio.loadmat(os.path.join(mat_directory, matfile))
        pose = np.stack((mat['x'], mat['y']), axis=-1)  # (T, 13, 2)

        action = mat['action']
        if isinstance(action, np.ndarray):
            action = action.flat[0]
        action = str(action)
        if action == 'strumming_guitar':
            action = 'strum_guitar'

        if action not in CLASS_MAP:
            raise ValueError(f"Unknown action: {action} in {matfile}")

        label = CLASS_MAP[action]
        is_train = mat['train'][0] == 1

        if is_train:
            train_data.append(pose)
            train_labels.append(label)
        else:
            test_data.append(pose)
            test_labels.append(label)

    return (
        train_data, np.array(train_labels, dtype=np.int64),
        test_data, np.array(test_labels, dtype=np.int64),
    )


# ─── Temporal normalisation ──────────────────────────────────────────────────

def prepare_sequence(pose: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """
    Centre-crop (if too long) or last-frame-pad (if too short) to target_frames.

    Args:
        pose          : (T_orig, V, C) array
        target_frames : desired temporal length

    Returns:
        (target_frames, V, C) array
    """
    T_orig = pose.shape[0]

    if T_orig == target_frames:
        return pose
    elif T_orig > target_frames:
        start = (T_orig - target_frames) // 2
        return pose[start : start + target_frames]
    else:
        padding = np.tile(pose[-1:], (target_frames - T_orig, 1, 1))
        return np.concatenate([pose, padding], axis=0)


# ─── Spatial normalisation ───────────────────────────────────────────────────

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Hip-centred, torso-length-scaled normalisation.

    Args:
        data : (N, T, V, C) array

    Returns:
        (N, T, V, C) normalised array
    """
    data_norm = data.copy()
    N, T, V, C = data.shape

    for n in range(N):
        for t in range(T):
            skeleton = data[n, t]  # (V, C)
            hip_centre = (skeleton[7] + skeleton[8]) / 2.0
            shoulder_centre = (skeleton[1] + skeleton[2]) / 2.0
            torso_len = np.linalg.norm(shoulder_centre - hip_centre) + 1e-6
            data_norm[n, t] = (skeleton - hip_centre) / torso_len

    return data_norm


# ─── Full preprocessing pipeline ─────────────────────────────────────────────

def preprocess_and_save(mat_directory: str, output_root: str = 'Penn_Action/processed'):
    """
    End-to-end pipeline: parse .mat files → normalise → save joint stream.

    Creates:
        output_root/joint/  {train,test}_{data,label}.npy
    """
    print("Loading raw .mat files...")
    raw_train, train_labels, raw_test, test_labels = load_raw_data(mat_directory)
    print(f"  Train: {len(raw_train)} samples | Test: {len(raw_test)} samples")

    print(f"Temporal normalisation to T={TARGET_FRAMES}...")
    train_data = np.array([prepare_sequence(s) for s in raw_train])
    test_data = np.array([prepare_sequence(s) for s in raw_test])

    print("Spatial normalisation (hip-centred, torso-scaled)...")
    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)

    out_dir = Path(output_root) / 'joint'
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / 'train_data.npy', train_data)
    np.save(out_dir / 'train_label.npy', train_labels)
    np.save(out_dir / 'test_data.npy', test_data)
    np.save(out_dir / 'test_label.npy', test_labels)

    print(f"  Saved joint/ — train {train_data.shape}, test {test_data.shape}")
    print("Done.")
