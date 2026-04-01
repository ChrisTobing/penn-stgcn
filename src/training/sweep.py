"""
W&B Bayesian hyperparameter sweep.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from src.data.dataset import PennActionDataset
from src.training.train import (
    stratified_split, make_loaders, run_epoch, build_model_and_optimiser,
)


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "best_val_acc", "goal": "maximize"},
    "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 3},
    "parameters": {
        "lr":           {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "batch_size":   {"values": [32, 64]},
        "dropout":      {"values": [0.2, 0.3]},
        "num_epochs":   {"value": 20},
        "val_ratio":    {"value": 0.20},
        "seed":         {"value": 42},
    },
}


def sweep_train_fn(train_data_path: str, train_label_path: str):
    """Called once per sweep run by wandb.agent."""
    run = wandb.init()
    cfg = {
        "lr":           wandb.config.lr,
        "weight_decay": wandb.config.weight_decay,
        "batch_size":   wandb.config.batch_size,
        "dropout":      wandb.config.dropout,
        "num_epochs":   wandb.config.num_epochs,
        "val_ratio":    wandb.config.val_ratio,
        "seed":         wandb.config.seed,
    }

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = PennActionDataset(train_data_path, train_label_path)
    t_idx, v_idx = stratified_split(full_ds, cfg["val_ratio"], cfg["seed"])
    train_loader, val_loader = make_loaders(t_idx, v_idx, full_ds, full_ds, cfg["batch_size"])

    model, opt, sched = build_model_and_optimiser(cfg, full_ds.num_classes, device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, cfg["num_epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, opt)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device)
        sched.step()

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc

        wandb.log({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_acc": vl_acc,
            "best_val_acc": best_val_acc,
            "lr": opt.param_groups[0]["lr"],
        })

    wandb.summary["best_val_acc"] = best_val_acc
    run.finish()


def run_sweep(train_data_path: str, train_label_path: str,
              project: str = "stgcn-penn-action", count: int = 20):
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=project)
    print(f"Sweep ID: {sweep_id}")
    wandb.agent(
        sweep_id,
        function=lambda: sweep_train_fn(train_data_path, train_label_path),
        count=count,
    )
