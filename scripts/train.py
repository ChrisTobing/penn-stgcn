#!/usr/bin/env python3
"""
Train the ST-GCN model on Penn Action (joint stream).

Usage:
    python scripts/train.py --config configs/best_config.yaml
    python scripts/train.py --config configs/best_config.yaml --adaptive
    python scripts/train.py --config configs/best_config.yaml --no-augmentation
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.train import run_train_eval


def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN on Penn Action")
    parser.add_argument('--config', type=str, default='configs/best_config.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--adaptive', action='store_true', default=None,
                        help='Enable adaptive adjacency (overrides config)')
    parser.add_argument('--no-adaptive', dest='adaptive', action='store_false',
                        help='Disable adaptive adjacency (overrides config)')
    parser.add_argument('--no-augmentation', dest='augmentation',
                        action='store_false', default=None,
                        help='Disable augmentation (overrides config)')
    parser.add_argument('--data_root', type=str, default='Penn_Action/processed',
                        help='Root data directory')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.adaptive is not None:
        cfg['adaptive'] = args.adaptive
    if args.augmentation is not None:
        cfg['augmentation'] = args.augmentation

    ckpt_path = "outputs/checkpoints/best_model.pth"
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    test_acc, cm = run_train_eval(cfg, data_root=args.data_root,
                                  model_save_path=ckpt_path)
    print(f"\nFinal test accuracy: {test_acc:.2f}%")


if __name__ == '__main__':
    main()
