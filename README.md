# Lightweight ST-GCN for Exercise-Based Action Recognition

BSc Computer Science Dissertation — University of Leeds, 2025/2026.

A 6-block Spatial-Temporal Graph Convolutional Network (ST-GCN) trained on the
Penn Action Dataset (13 joints, 15 classes) with three incremental improvements:

1. **Skeleton data augmentation** — rotation, Gaussian noise, time interpolation, time warping
2. **Adaptive adjacency matrices** — learnable A + B + C decomposition (Shi et al., CVPR 2019)
3. **Four-stream input fusion** — joint, bone, joint-motion, bone-motion streams trained
   independently and combined at test time via weighted softmax averaging (MS-AAGCN,
   Shi et al., IEEE TIP 2020)

## Repository Structure

```
penn-action-stgcn/
├── configs/
│   └── best_config.yaml          # Hyperparameters for training
├── src/
│   ├── data/
│   │   ├── preprocess.py         # .mat parsing, normalisation, save .npy
│   │   ├── dataset.py            # PennActionDataset, PennActionDatasetAug, PennActionStreamDataset
│   │   ├── augmentations.py      # rotate, noise, time_interp, time_warp, pipeline
│   │   └── streams.py            # PENN_BONE_PARENT, derive_stream (joint/bone/motion)
│   ├── models/
│   │   ├── adjacency.py          # get_penn_action_adjacency, normalize_adjacency
│   │   ├── stgcn_block.py        # STGCNBlock (fixed + adaptive)
│   │   └── stgcn_light.py        # STGCN_Light (6-block network)
│   ├── training/
│   │   ├── train.py              # run_epoch, run_train_eval, EarlyStopping
│   │   ├── fusion.py             # train_single_stream, run_four_stream_fusion
│   │   └── sweep.py              # W&B Bayesian sweep logic
│   └── evaluation/
│       └── evaluate.py           # Test evaluation, confusion matrix
├── scripts/
│   ├── preprocess.py             # CLI: python scripts/preprocess.py
│   ├── train.py                  # CLI: python scripts/train.py --config configs/best_config.yaml
│   ├── train_fusion.py           # CLI: python scripts/train_fusion.py --config configs/best_config.yaml
│   └── evaluate.py               # CLI: python scripts/evaluate.py --checkpoint ...
├── notebooks/
│   └── PENN_STGCN_FINAL_VERSION.ipynb  # Self-contained Google Colab notebook
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

### 6. Four-stream fusion (Improvement 3)

Train one ST-GCN per MS-AAGCN stream (`joint`, `bone`, `joint_motion`, `bone_motion`)
on the same stratified split, then weighted-average their softmax outputs at
test time with weights (2:1:2:1) normalised to sum to 1.

```bash
# Default (augmentation on, fixed adjacency)
python scripts/train_fusion.py --config configs/best_config.yaml

# With adaptive adjacency
python scripts/train_fusion.py --config configs/best_config.yaml --adaptive

# Without augmentation
python scripts/train_fusion.py --config configs/best_config.yaml --no-augmentation
```

This produces four checkpoints under [`outputs/checkpoints/`](outputs/checkpoints/)
(`best_model_{joint,bone,joint_motion,bone_motion}.pth`), per-stream and fused
test accuracies on the console, and the fused confusion matrix at
[`outputs/figures/confusion_matrix_fusion.png`](outputs/figures/). Each stream
gets its own W&B run (grouped under `four-stream-fusion`) plus a summary run
with all per-stream and fused test accuracies.

Stream derivation lives in [`src/data/streams.py`](src/data/streams.py): bones
are computed against a 12-edge spanning tree rooted at l_hip (joint 7), and
motion streams are zero-padded at `t = T-1` so all four modalities share the
same `(C=2, T=100, V=13, M=1)` shape and reuse `STGCN_Light` unchanged.

### Google Colab (recommended for easy reproduction)

For a zero-setup run, open [`notebooks/PENN_STGCN_FINAL_VERSION.ipynb`](notebooks/PENN_STGCN_FINAL_VERSION.ipynb)
in Google Colab. The notebook is the original self-contained version of this
project — preprocessing, model, training, augmentation, adaptive adjacency,
and four-stream fusion are all defined inline, so no local clone or
`pip install` is required.

**Only dependency**: place the Penn Action `.tar` archive in your Google Drive.
The notebook mounts Drive, extracts the dataset, and runs end-to-end on Colab's
GPU runtime.

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
