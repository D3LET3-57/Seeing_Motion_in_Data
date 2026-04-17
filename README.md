# HAR Image Formation CNN Comparison

This project implements image formation from multiple 1D inertial signals (UCI HAR dataset) and evaluates 4 CNN architectures across 6 gamma operators (24 total experiments).

## Project Contents

- `image_formation_har_all_final.ipynb` : Full end-to-end pipeline
- `dataset/UCI HAR Dataset/` : UCI HAR raw data (already present)
- `outputs/results_all.csv` : Compact experiment summary
- `outputs/results_all_detailed.csv` : Detailed summary with metadata
- `outputs/figures/` : Generated plots and visual analysis
- `report.pdf` : Project report

## What The Notebook Does

1. Loads UCI HAR inertial signals (train/test)
2. Normalizes signals (beta operator)
3. Forms 2D images using 6 gamma arrangements
4. Builds and compares 4 CNN models:
   - LeNet-5
   - VGG-Style
   - ResNet-Style
   - Lightweight (depthwise separable)
5. Runs all 24 model × gamma combinations
6. Saves evaluation CSV files and visualization figures

## Requirements

Recommended environment:

- Python 3.10 or 3.11
- TensorFlow 2.19.x (notebook was run with 2.19.0)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

## Run Instructions

### Option A: VS Code Notebook

1. Open this project folder in VS Code
2. Open `image_formation_har_all_final.ipynb`
3. Select your Python kernel (the same environment where dependencies are installed)
4. Run all cells from top to bottom

### Option B: Jupyter Lab

```bash
source .venv/bin/activate
jupyter lab
```

Then open `image_formation_har_all_final.ipynb` and run all cells.

## Dataset Path Notes

The notebook searches for the dataset in either of these locations:

- `dataset/UCI HAR Dataset`
- `UCI HAR Dataset`

In this repository, the expected path already exists:

- `dataset/UCI HAR Dataset`

If you move files and get a dataset error, restore one of the above paths.

## Expected Outputs

After a successful run, you should see:

### CSV Outputs

- `outputs/results_all.csv`
- `outputs/results_all_detailed.csv`

### Figure Outputs

- `outputs/figures/class_distribution.png`
- `outputs/figures/raw_signals.png`
- `outputs/figures/gamma_arrangements.png`
- `outputs/figures/sample_images.png`
- `outputs/figures/training_curves.png`
- `outputs/figures/accuracy_comparison.png`
- `outputs/figures/confusion_matrices.png`
- `outputs/figures/per_class_heatmap.png`
- `outputs/figures/feature_maps.png`
- `outputs/figures/model_complexity.png`

## Current Result Snapshot

From the checked-in results file:

- Best recorded accuracy: 92.13% (ResNet-Style + gamma3 CentBodyAcc)

## Reproducibility

- Random seed is fixed to 42 in the notebook
- Subject-disjoint train/validation split is used via GroupShuffleSplit
- Hardware (CPU/GPU) can still slightly affect runtime and final decimals

## Troubleshooting

### 1) Dataset not found

If you see a FileNotFoundError for UCI HAR, verify that one of these exists:

- `dataset/UCI HAR Dataset/train`
- `dataset/UCI HAR Dataset/test`

### 2) TensorFlow/GPU warnings

These are usually safe. The notebook runs on CPU as well.

### 3) Long runtime

The full 24-run experiment can take a while depending on hardware.

To speed up:

- reduce epochs in the training cell (currently 50)
- test only one model or one gamma operator first

## Citation

If you use this work, cite:

- G. Liu et al., IEEE Transactions on Industrial Informatics, 2021
- UCI HAR dataset paper (Anguita et al., ESANN 2013)
