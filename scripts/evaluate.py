#!/usr/bin/env python3
"""
Evaluate a trained ST-GCN checkpoint on the Penn Action test set.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.evaluate import evaluate_model, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate ST-GCN on Penn Action test set")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_root', type=str, default='Penn_Action/processed',
                        help='Root data directory')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--adaptive', action='store_true', default=False,
                        help='Set if model was trained with adaptive adjacency')
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()

    test_data = f"{args.data_root}/joint/test_data.npy"
    test_label = f"{args.data_root}/joint/test_label.npy"

    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    test_acc, cm = evaluate_model(
        checkpoint_path=args.checkpoint,
        test_data_path=test_data,
        test_label_path=test_label,
        batch_size=args.batch_size,
        dropout=args.dropout,
        adaptive=args.adaptive,
    )

    cm_path = "outputs/figures/confusion_matrix.png"
    plot_confusion_matrix(cm, title=f"Confusion Matrix (acc: {test_acc:.2f}%)",
                          save_path=cm_path)


if __name__ == '__main__':
    main()
