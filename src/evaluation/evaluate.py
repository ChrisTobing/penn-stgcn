"""
Evaluation utilities: single-model test evaluation and confusion matrix plotting.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.preprocess import CLASS_NAMES
from src.data.dataset import PennActionDatasetAug
from src.models.adjacency import get_penn_action_adjacency
from src.models.stgcn_light import STGCN_Light


def load_model(checkpoint_path: str, num_classes: int = 15, dropout: float = 0.2,
               adaptive: bool = False, device: str = 'cpu') -> STGCN_Light:
    """Load a trained STGCN_Light model from checkpoint."""
    A = get_penn_action_adjacency()
    model = STGCN_Light(
        num_class=num_classes, in_channels=2, A=A,
        dropout=dropout, adaptive=adaptive,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def evaluate_model(checkpoint_path: str, test_data_path: str, test_label_path: str,
                   batch_size: int = 64, num_classes: int = 15,
                   dropout: float = 0.2, adaptive: bool = False):
    """
    Evaluate a single trained model on the test set.

    Returns:
        test_acc : float, test accuracy percentage
        cm       : (num_classes, num_classes) confusion matrix
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = PennActionDatasetAug(test_data_path, test_label_path, training=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    model = load_model(checkpoint_path, num_classes, dropout, adaptive, device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    test_acc = accuracy_score(all_labels, all_preds) * 100.0
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n  Test Accuracy: {test_acc:.2f}%")
    print(f"\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3)}")

    return test_acc, cm


def plot_confusion_matrix(cm, title: str = "Confusion Matrix",
                          save_path: str = None):
    """Plot a normalised confusion matrix."""
    fig, ax = plt.subplots(figsize=(13, 11))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.4, linecolor="lightgrey", ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved to {save_path}")
    plt.close()
