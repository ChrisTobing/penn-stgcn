"""
Training utilities: epoch runner, early stopping, full train+eval pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from src.data.preprocess import CLASS_NAMES
from src.data.dataset import PennActionDatasetAug
from src.models.adjacency import get_penn_action_adjacency
from src.models.stgcn_light import STGCN_Light


# ─── Reproducibility ─────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    """Pin every source of randomness."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Ensure each DataLoader worker has a deterministic seed."""
    np.random.seed(torch.initial_seed() % 2**32)


# ─── Data helpers ─────────────────────────────────────────────────────────────

def stratified_split(dataset, val_ratio=0.20, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(dataset)), dataset.labels))
    print(f"  Split -> {len(train_idx)} train / {len(val_idx)} val")
    return train_idx, val_idx


def make_loaders(train_idx, val_idx, train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader


# ─── Epoch runner ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None):
    """One training or evaluation epoch. Returns (avg_loss, accuracy%)."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, 100.0 * correct / total


# ─── Early Stopping ──────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when val_loss fails to improve for `patience` epochs.

    Monitors val_loss (lower is better) as it is a smoother signal than accuracy.
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ─── Model builder ────────────────────────────────────────────────────────────

def build_model_and_optimiser(cfg, num_classes, device):
    A = get_penn_action_adjacency()
    model = STGCN_Light(
        num_class=num_classes,
        in_channels=2,
        A=A,
        dropout=cfg["dropout"],
        adaptive=cfg.get("adaptive", False),
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    sched = CosineAnnealingLR(opt, T_max=cfg["num_epochs"], eta_min=1e-6)
    return model, opt, sched


# ─── Full train + eval pipeline ──────────────────────────────────────────────

def run_train_eval(cfg, data_root: str = "Penn_Action/processed",
                   model_save_path: str = "outputs/checkpoints/best_model.pth"):
    """
    Full training run with the given config.
    Saves best-val-acc checkpoint, then evaluates once on the test set.

    Args:
        cfg            : dict with all hyperparameters (from YAML)
        data_root      : root directory containing joint/ subdirectory
        model_save_path: where to save the best checkpoint
    """
    joint_dir = f"{data_root}/joint"
    train_data_path = f"{joint_dir}/train_data.npy"
    train_label_path = f"{joint_dir}/train_label.npy"
    test_data_path = f"{joint_dir}/test_data.npy"
    test_label_path = f"{joint_dir}/test_label.npy"

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  TRAIN MODE — {cfg['num_epochs']} epochs | device: {device}")
    print(f"  Adaptive: {cfg.get('adaptive', False)}")
    print(f"  Augmentation: {cfg.get('augmentation', True)}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    use_aug = cfg.get("augmentation", True)
    train_ds = PennActionDatasetAug(train_data_path, train_label_path, training=use_aug)
    val_ds = PennActionDatasetAug(train_data_path, train_label_path, training=False)
    test_ds = PennActionDatasetAug(test_data_path, test_label_path, training=False)
    num_classes = train_ds.num_classes

    t_idx, v_idx = stratified_split(train_ds, cfg["val_ratio"], cfg["seed"])
    train_loader, val_loader = make_loaders(t_idx, v_idx, train_ds, val_ds, cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model, opt, sched = build_model_and_optimiser(cfg, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params:,}\n")

    # ── W&B run ───────────────────────────────────────────────────────────
    run = wandb.init(
        project=cfg.get("wandb_project", "stgcn-penn-action"),
        entity=cfg.get("wandb_entity"),
        name=f"{'adaptive' if cfg.get('adaptive') else 'fixed'}-train",
        config={**cfg, "num_params": num_params, "mode": "train_eval"},
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = 0
    early_stopper = EarlyStopping(
        patience=cfg["es_patience"],
        min_delta=cfg["es_min_delta"],
    )

    for epoch in range(1, cfg["num_epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, opt)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device)
        sched.step()

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)

        wandb.log({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_acc": vl_acc,
            "best_val_acc": best_val_acc,
            "lr": opt.param_groups[0]["lr"],
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  "
                  f"train {tr_loss:.4f}/{tr_acc:.1f}%  "
                  f"val {vl_loss:.4f}/{vl_acc:.1f}%  "
                  f"(best val {best_val_acc:.1f}% @ ep {best_epoch})  "
                  f"[patience {early_stopper.counter}/{early_stopper.patience}]")

        if early_stopper(vl_loss):
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"\n  Best val acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"  Checkpoint: {model_save_path}\n")

    # ── Test evaluation ───────────────────────────────────────────────────
    print("  Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

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

    wandb.summary["best_val_acc"] = best_val_acc
    wandb.summary["test_accuracy"] = test_acc

    # ── Confusion matrix ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 11))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.4, linecolor="lightgrey", ax=ax)
    ax.set_title(f"Confusion Matrix — Test Set (acc: {test_acc:.2f}%)")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_path = "outputs/figures/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    wandb.log({"confusion_matrix": wandb.Image(cm_path)})
    plt.close()
    print(f"  Confusion matrix saved to {cm_path}")

    run.finish()
    return test_acc, cm
