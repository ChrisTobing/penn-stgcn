#!/usr/bin/env python3
"""
Train four ST-GCN models (one per MS-AAGCN stream) and evaluate the
weighted score-level fusion on the Penn Action test set.

Usage:
    python scripts/train_fusion.py --config configs/best_config.yaml
    python scripts/train_fusion.py --config configs/best_config.yaml --adaptive
    python scripts/train_fusion.py --config configs/best_config.yaml --no-augmentation
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.fusion import run_four_stream_fusion


def main():
    parser = argparse.ArgumentParser(
        description="Train + evaluate four-stream fusion on Penn Action"
    )
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
    parser.add_argument('--ckpt_dir', type=str, default='outputs/checkpoints',
                        help='Where best_model_{stream}.pth checkpoints are saved')
    parser.add_argument('--figures_dir', type=str, default='outputs/figures',
                        help='Where confusion_matrix_fusion.png is saved')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.adaptive is not None:
        cfg['adaptive'] = args.adaptive
    if args.augmentation is not None:
        cfg['augmentation'] = args.augmentation

    results = run_four_stream_fusion(
        cfg,
        data_root=args.data_root,
        ckpt_dir=args.ckpt_dir,
        figures_dir=args.figures_dir,
    )
    print(f"\nFinal fused test accuracy: {results['fusion_test_acc']:.2f}%")


if __name__ == '__main__':
    main()
