#!/usr/bin/env python3
"""
Preprocess Penn Action .mat files into .npy arrays.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --mat_dir /path/to/Penn_Action/labels
"""

import argparse
import sys
import os

# Allow imports from repo root (for Colab: sys.path.insert(0, '/content/penn-action-stgcn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocess import preprocess_and_save


def main():
    parser = argparse.ArgumentParser(description="Preprocess Penn Action dataset")
    parser.add_argument(
        '--mat_dir',
        type=str,
        default='Penn_Action/labels',
        help='Path to the Penn Action labels/ directory containing .mat files',
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='Penn_Action/processed',
        help='Root output directory for processed .npy files',
    )
    args = parser.parse_args()

    preprocess_and_save(args.mat_dir, args.output_root)


if __name__ == '__main__':
    main()
