"""
PyTorch Dataset classes for Penn Action skeleton data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentations import apply_augmentations
from src.data.streams import STREAMS, derive_stream


class PennActionDataset(Dataset):
    """Basic Penn Action dataset without augmentation (used by sweep)."""

    def __init__(self, data_path: str, label_path: str):
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(label_path).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    @property
    def num_classes(self):
        return int(self.labels.max()) + 1

    def __getitem__(self, idx):
        raw = self.data[idx]
        x = np.expand_dims(np.transpose(raw, (2, 0, 1)), axis=-1)  # (C, T, V, M=1)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class PennActionDatasetAug(Dataset):
    """
    Penn Action dataset with optional skeleton augmentation.

    Augmentation is applied per-sample inside __getitem__, so each epoch
    sees a freshly randomised version of every training sequence.

    Args:
        data_path  : path to .npy file of shape (N, T, V, C)
        label_path : path to .npy file of shape (N,)
        training   : True activates augmentation; False is a strict no-op
    """

    def __init__(self, data_path: str, label_path: str, training: bool = True):
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(label_path).astype(np.int64)
        self.training = training

    def __len__(self):
        return len(self.labels)

    @property
    def num_classes(self):
        return int(self.labels.max()) + 1

    def __getitem__(self, idx):
        x = self.data[idx].copy()                       # (T, V, C)
        x = apply_augmentations(x, self.training)        # (T, V, C)
        x = np.transpose(x, (2, 0, 1))                  # (C, T, V)
        x = np.expand_dims(x, axis=-1)                  # (C, T, V, M=1)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class PennActionStreamDataset(Dataset):
    """
    Penn Action dataset emitting one of the four MS-AAGCN input streams
    (joint, bone, joint_motion, bone_motion).

    Per-sample pipeline in __getitem__:
        1. Load raw joints (T, V, C) from the mmap'd .npy file.
        2. If training=True, apply the augmentation pipeline to raw joints.
        3. Derive the requested stream from the (augmented) joints.
        4. Reshape to ST-GCN input format: (C, T, V, M=1).

    Step ordering matters. Streams must be derived *after* augmentation:
    rotating a bone vector is NOT the same as the difference of the rotated
    endpoint joints. Deriving streams from augmented joints keeps the bone
    and motion tensors geometrically consistent with the augmented skeleton.

    Args:
        data_path  : path to .npy file of shape (N, T, V, C) raw joints
        label_path : path to .npy file of shape (N,)
        stream     : which MS-AAGCN modality to emit (one of STREAMS)
        training   : True activates augmentation (applied to raw joints
                     before stream derivation); False is a strict no-op
    """

    VALID_STREAMS = STREAMS

    def __init__(self, data_path: str, label_path: str,
                 stream: str = 'joint', training: bool = True):
        if stream not in self.VALID_STREAMS:
            raise ValueError(
                f"stream must be one of {self.VALID_STREAMS}, got '{stream}'"
            )
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(label_path).astype(np.int64)
        self.stream = stream
        self.training = training

    def __len__(self):
        return len(self.labels)

    @property
    def num_classes(self):
        return int(self.labels.max()) + 1

    def __getitem__(self, idx):
        x = self.data[idx].copy()                       # (T, V, C)
        x = apply_augmentations(x, self.training)        # (T, V, C)
        x = derive_stream(x, self.stream)                # (T, V, C)
        x = np.transpose(x, (2, 0, 1))                  # (C, T, V)
        x = np.expand_dims(x, axis=-1)                  # (C, T, V, M=1)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
