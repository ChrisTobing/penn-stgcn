# Lightweight ST-GCN for Exercise-Based Action Recognition

BSc Computer Science Dissertation — University of Leeds, 2025/2026.

A 6-block Spatial-Temporal Graph Convolutional Network (ST-GCN) trained on the
Penn Action Dataset (13 joints, 15 classes) with two incremental improvements:

1. **Skeleton data augmentation** — rotation, Gaussian noise, time interpolation, time warping
2. **Adaptive adjacency matrices** — learnable A + B + C decomposition (Shi et al., CVPR 2019)

## Repository Structure

```
penn-action-stgcn/
├── configs/
│   └── best_config.yaml          # Hyperparameters for training
├── src/
│   ├── data/
│   │   ├── preprocess.py         # .mat parsing, normalisation, save .npy
│   │   ├── dataset.py            # PennActionDataset, PennActionDatasetAug
│   │   └── augmentations.py      # rotate, noise, time_interp, time_warp, pipeline
│   ├── models/
│   │   ├── adjacency.py          # get_penn_action_adjacency, normalize_adjacency
│   │   ├── stgcn_block.py        # STGCNBlock (fixed + adaptive)
│   │   └── stgcn_light.py        # STGCN_Light (6-block network)
│   ├── training/
│   │   ├── train.py              # run_epoch, run_train_eval, EarlyStopping
│   │   └── sweep.py              # W&B Bayesian sweep logic
│   └── evaluation/
│       └── evaluate.py           # Test evaluation, confusion matrix
├── scripts/
│   ├── preprocess.py             # CLI: python scripts/preprocess.py
│   ├── train.py                  # CLI: python scripts/train.py --config configs/best_config.yaml
│   └── evaluate.py               # CLI: python scripts/evaluate.py --checkpoint ...
├── notebooks/
│   └── colab_runner.ipynb        # Colab entry point — imports from src/
└── outputs/                      # .gitignored — checkpoints, figures, W&B logs
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/penn-action-stgcn.git
cd penn-action-stgcn
pip install -r requirements.txt
```

### 2. Obtain the Penn Action Dataset

Download from [Penn Action](https://dreamdragon.github.io/PennAction/) and extract
to `Penn_Action/` at the repository root (this directory is `.gitignored`).

### 3. Preprocess

```bash
python scripts/preprocess.py
```

This parses `.mat` files, normalises skeletons (hip-centred, torso-scaled), and saves
`.npy` arrays under `Penn_Action/processed/joint/`.

### 4. Train

```bash
# Default (augmentation on, fixed adjacency)
python scripts/train.py --config configs/best_config.yaml

# With adaptive adjacency
python scripts/train.py --config configs/best_config.yaml --adaptive

# Without augmentation (baseline)
python scripts/train.py --config configs/best_config.yaml --no-augmentation
```

### 5. Evaluate

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
```

### Google Colab

Open `notebooks/colab_runner.ipynb` in Colab. It clones the repo, adds `src/` to the
Python path, and runs preprocessing + training via imports.

## Key Results

| Configuration                     | Test Acc (%) |
|-----------------------------------|-------------|
| (1) Baseline ST-GCN               | 81.65       |
| (2) + Augmentation                 | 83.80       |
| (3) + Adaptive Adjacency           | 79.68       |

Adaptive adjacency underperforms on Penn Action's small training set (~1,258 samples),
consistent with the technique being designed for larger datasets (NTU RGB+D, 40K+ samples).
This is analysed as a well-motivated negative finding in the dissertation.

## References

- Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition," AAAI 2018.
- Shi et al., "Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition," CVPR 2019.
- Shi et al., "Skeleton-Based Action Recognition with Multi-Stream Adaptive Graph Convolutional Networks," IEEE TIP 2020.
- Zhang et al., "From Actemes to Action," ICCV 2013 (Penn Action Dataset).
