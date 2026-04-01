"""
PyTorch Dataset classes for Penn Action skeleton data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentations import apply_augmentations


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
