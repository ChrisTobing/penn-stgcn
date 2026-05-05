"""
Four-stream input fusion

Trains four independent ST-GCN models (joint, bone, joint_motion, bone_motion)
with identical hyperparameters and the same stratified train/val split, then
combines their softmax outputs at test time with weights (2:1:2:1).

Reuses the single-stream training infrastructure from src.training.train —
only the dataset is swapped to PennActionStreamDataset.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from src.data.preprocess import CLASS_NAMES
from src.data.dataset import PennActionStreamDataset
from src.data.streams import STREAMS, FUSION_WEIGHTS
from src.training.train import (
    EarlyStopping,
    build_model_and_optimiser,
    make_loaders,
    run_epoch,
    seed_everything,
    stratified_split,
)


STREAM_CKPT_PATTERN = "best_model_{stream}.pth"


def _stream_ckpt_path(ckpt_dir: str, stream: str) -> str:
    return str(Path(ckpt_dir) / STREAM_CKPT_PATTERN.format(stream=stream))


def train_single_stream(stream: str, cfg: dict, device: torch.device,
                        train_data_path: str, train_label_path: str,
                        test_data_path: str, test_label_path: str,
                        ckpt_dir: str, verbose: bool = True):
    """
    Train one ST-GCN model on a single input stream. Mirrors run_train_eval
    structure (same training loop, early stopping, W&B logging) but configures
    the dataset to emit the requested stream and saves to a stream-specific
    checkpoint path.

    Args:
        stream    : one of STREAMS
        cfg       : config dict (best_config-shaped)
        device    : torch device
        ckpt_dir  : directory where best_model_{stream}.pth will be saved
        verbose   : print per-epoch updates

    Returns:
        (best_val_acc, test_acc_single, ckpt_path)
    """
    ckpt_path = _stream_ckpt_path(ckpt_dir, stream)

    # ── Data: three dataset instances, all on the same stream ───────────────
    # train_ds has training=True (augmentation active); val_ds and test_ds
    # have training=False (clean sequences). Augmentation runs on raw joints,
    # then the stream is derived — keeping bone/motion geometrically valid.
    use_aug = cfg.get("augmentation", True)
    train_ds = PennActionStreamDataset(train_data_path, train_label_path,
                                       stream=stream, training=use_aug)
    val_ds = PennActionStreamDataset(train_data_path, train_label_path,
                                     stream=stream, training=False)
    test_ds = PennActionStreamDataset(test_data_path, test_label_path,
                                      stream=stream, training=False)
    num_classes = train_ds.num_classes

    t_idx, v_idx = stratified_split(train_ds, cfg["val_ratio"], cfg["seed"])
    train_loader, val_loader = make_loaders(t_idx, v_idx, train_ds, val_ds,
                                            cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=2, pin_memory=True)

    # Model 

    model, opt, sched = build_model_and_optimiser(cfg, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"\n  Stream: {stream}  |  params: {num_params:,}  "
              f"|  adaptive: {cfg.get('adaptive', False)}")

    # W&B run (one per stream, grouped for easy comparison) ──────────────
    run = wandb.init(
        project=cfg.get("wandb_project", "stgcn-penn-action"),
        entity=cfg.get("wandb_entity"),
        name=f"fusion-{stream}",
        group="four-stream-fusion",
        config={**cfg, "stream": stream, "num_params": num_params,
                "mode": "single_stream"},
        reinit=True,
    )

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    early_stopper = EarlyStopping(patience=cfg["es_patience"],
                                  min_delta=cfg["es_min_delta"])

    for epoch in range(1, cfg["num_epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, opt)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device)
        sched.step()

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

        wandb.log({
            "epoch":        epoch,
            "train_loss":   tr_loss, "train_acc": tr_acc,
            "val_loss":     vl_loss, "val_acc":   vl_acc,
            "best_val_acc": best_val_acc,
            "lr":           opt.param_groups[0]["lr"],
        })

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"    ep {epoch:3d}  "
                  f"tr {tr_loss:.3f}/{tr_acc:.1f}%  "
                  f"val {vl_loss:.3f}/{vl_acc:.1f}%  "
                  f"(best {best_val_acc:.1f}% @ {best_epoch})")

        if early_stopper(vl_loss):
            if verbose:
                print(f"    early stop @ epoch {epoch}")
            break

    # Single-stream test evaluation
    # Per-stream test accuracy contextualises the fusion benefit and feeds
    # the discussion of which streams dominate (typically joint and joint-motion).
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x.to(device)).argmax(1).cpu()
            correct += (preds == y).sum().item()
            total += y.size(0)
    test_acc_single = 100.0 * correct / total

    wandb.summary["best_val_acc"] = best_val_acc
    wandb.summary["test_accuracy_single"] = test_acc_single
    run.finish()

    if verbose:
        print(f"  ✓ {stream}: best_val={best_val_acc:.2f}%  "
              f"test_single={test_acc_single:.2f}%  → {ckpt_path}")

    return best_val_acc, test_acc_single, ckpt_path


def _collect_stream_softmax(stream: str, cfg: dict, device: torch.device,
                            num_classes: int, test_data_path: str,
                            test_label_path: str, ckpt_dir: str):
    """
    Load a trained stream's checkpoint and return its softmax predictions
    on the held-out test set, alongside the corresponding labels.

    The test DataLoader uses shuffle=False, so label order is deterministic
    and matches across all four streams — enabling element-wise fusion.

    Returns:
        probs  : (N_test, num_classes) numpy array of class probabilities
        labels : (N_test,) numpy array of ground-truth labels
    """
    ckpt_path = _stream_ckpt_path(ckpt_dir, stream)
    test_ds = PennActionStreamDataset(test_data_path, test_label_path,
                                      stream=stream, training=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=2, pin_memory=True)

    model, _, _ = build_model_and_optimiser(cfg, num_classes, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    probs_chunks, labels_chunks = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_chunks.append(probs)
            labels_chunks.append(y.numpy())
    return np.concatenate(probs_chunks, axis=0), np.concatenate(labels_chunks)


def run_four_stream_fusion(cfg: dict,
                           data_root: str = "Penn_Action/processed",
                           ckpt_dir: str = "outputs/checkpoints",
                           figures_dir: str = "outputs/figures"):
    """
    End-to-end runner for Improvement 3 (four-stream fusion).

    Stages:
        1. Train one ST-GCN per stream (saves 4 checkpoints to ckpt_dir).
        2. Reload each checkpoint and collect softmax predictions on the
           held-out test set.
        3. Weighted-average the four probability tensors with (2:1:2:1)
           normalised to sum to 1.
        4. Report fused test accuracy + classification report, save the
           fused confusion matrix to figures_dir/confusion_matrix_fusion.png.

    Returns:
        results: dict with per-stream test accuracies, fused test accuracy,
                 fused confusion matrix, labels, and fused probabilities.
    """
    joint_dir = f"{data_root}/joint"
    train_data_path = f"{joint_dir}/train_data.npy"
    train_label_path = f"{joint_dir}/train_label.npy"
    test_data_path = f"{joint_dir}/test_data.npy"
    test_label_path = f"{joint_dir}/test_label.npy"

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*62}")
    print(f"  FOUR-STREAM FUSION  —  {cfg['num_epochs']} epochs/stream")
    print(f"  Device: {device}  |  Adaptive: {cfg.get('adaptive', False)}")
    print(f"  Streams: {list(STREAMS)}")
    print(f"  Fusion weights (unnormalised): {FUSION_WEIGHTS}")
    print(f"{'='*62}")

    #  Stage 1: train each stream independently 
    per_stream_results = {}
    for stream in STREAMS:
        seed_everything(cfg["seed"])
        best_val, test_single, ckpt = train_single_stream(
            stream, cfg, device,
            train_data_path, train_label_path,
            test_data_path, test_label_path,
            ckpt_dir,
        )
        per_stream_results[stream] = {
            "best_val_acc":    best_val,
            "test_acc_single": test_single,
            "ckpt":            ckpt,
        }

    #  Stage 2: collect softmax predictions from each trained model ──────
    num_classes = PennActionStreamDataset(
        train_data_path, train_label_path, stream='joint', training=False
    ).num_classes

    print(f"\n  Collecting softmax outputs from all four streams...")
    stream_probs = {}
    labels = None
    for stream in STREAMS:
        probs, lbls = _collect_stream_softmax(
            stream, cfg, device, num_classes,
            test_data_path, test_label_path, ckpt_dir,
        )
        stream_probs[stream] = probs
        # Checks again if labels are identical across streams 
        if labels is None:
            labels = lbls
        else:
            assert np.array_equal(labels, lbls), \
                f"Label order mismatch on stream {stream} — fusion is invalid"

    # Stage 3: weighted score-level fusion

    w_total = sum(FUSION_WEIGHTS[s] for s in STREAMS)
    fused_probs = sum(
        (FUSION_WEIGHTS[s] / w_total) * stream_probs[s] for s in STREAMS
    )   # (N_test, num_classes)

    fused_preds = fused_probs.argmax(axis=1)
    fusion_test_acc = accuracy_score(labels, fused_preds) * 100.0
    fusion_cm = confusion_matrix(labels, fused_preds)

    # Stage 4: report + logs 
    print(f"\n{'─'*62}")
    print(f"  FOUR-STREAM FUSION — Test Set Results")
    print(f"{'─'*62}")
    for s in STREAMS:
        r = per_stream_results[s]
        print(f"    {s:14s}  test={r['test_acc_single']:6.2f}%   "
              f"val={r['best_val_acc']:6.2f}%   (weight={FUSION_WEIGHTS[s]})")
    print(f"    {'─'*54}")
    print(f"    {'FUSED (2:1:2:1)':14s}  test={fusion_test_acc:6.2f}%")
    print(f"{'─'*62}")

    report = classification_report(labels, fused_preds,
                                   target_names=CLASS_NAMES, digits=3)
    print("\n" + report)

    run = wandb.init(
        project=cfg.get("wandb_project", "stgcn-penn-action"),
        entity=cfg.get("wandb_entity"),
        name="four-stream-fusion-summary",
        group="four-stream-fusion",
        config={**cfg, "mode": "fusion_eval",
                "fusion_weights": FUSION_WEIGHTS,
                "streams": list(STREAMS)},
        reinit=True,
    )
    for s in STREAMS:
        wandb.summary[f"test_acc_{s}"] = per_stream_results[s]["test_acc_single"]
    wandb.summary["test_acc_fused"] = fusion_test_acc

    # Fused confusion matrix 
    fig, ax = plt.subplots(figsize=(13, 11))
    cm_norm = fusion_cm.astype(float) / fusion_cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.4, linecolor="lightgrey", ax=ax,
    )
    ax.set_title(f"Confusion Matrix — Four-Stream Fusion  "
                 f"(acc: {fusion_test_acc:.2f}%)", fontsize=14)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_path = str(Path(figures_dir) / "confusion_matrix_fusion.png")
    plt.savefig(cm_path, dpi=150)
    wandb.log({"confusion_matrix_fusion": wandb.Image(cm_path)})
    plt.close()
    print(f"  Confusion matrix saved to {cm_path}")

    run.finish()

    return {
        "per_stream":      per_stream_results,
        "fusion_test_acc": fusion_test_acc,
        "fusion_cm":       fusion_cm,
        "labels":          labels,
        "fused_probs":     fused_probs,
    }
