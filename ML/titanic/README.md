# Titanic

Titanic survival prediction experiments built around Kaggle's Titanic dataset.

## Installation

Create a Python environment and install the packages used by the scripts in this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn catboost torch
```

## Quick Start

Train the CatBoost cross-validation ensemble and write a Kaggle submission file:

```bash
python train_titanic_catboost_cv.py
```

You should see fold-level accuracy and AUC logs, followed by a saved submission path such as:

```text
saved=submission_catboost_cv.csv
```

Generate a synthetic tabular dataset with the VAE script:

```bash
python generate_titanic_vae.py --num-samples 20000
```

## What Is In This Folder?

This project mixes a few different approaches to the same classic classification problem:

- `train_titanic_catboost_cv.py` trains two CatBoost pipelines, one with simple imputations and one with tree-based imputations, then blends them into a final submission.
- `generate_titanic_vae.py` trains a variational autoencoder on a compact Titanic feature set and samples synthetic passenger rows.
- `support-vector-machine.ipynb` and `unti.ipynb` are notebook-based experiments.
- `train.csv`, `test.csv`, and `gender_submission.csv` are the standard Kaggle input files.

## Why This Project Exists

The interesting part here is not just getting a Titanic score. The CatBoost script does a fair amount of feature work before fitting:

- passenger title extraction from `Name`
- deck and cabin-known flags from `Cabin`
- ticket prefix and ticket group size features
- family-size derived features
- binned age and fare features
- optional tree-based imputation for `Age`, `Fare`, and `Embarked`

The VAE script tackles a different problem: generating synthetic structured rows that still look like the training data distribution.

## Main Scripts

### `train_titanic_catboost_cv.py`

Train the full survival pipeline with 5-fold stratified cross-validation.

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `--train` | `train.csv` | Training CSV with the `Survived` target |
| `--test` | `test.csv` | Kaggle test CSV |
| `--output` | `submission_catboost_cv.csv` | Output submission path |
| `--folds` | `5` | Number of CV folds |
| `--iterations` | `800` | Max CatBoost iterations |
| `--tune` | off | Run random-search tuning before the final fit |
| `--tune-trials` | `16` | Number of sampled tuning configs |
| `--ensemble-weight` | auto | Weight for the tree-imputed model in the final blend |

Example:

```bash
python train_titanic_catboost_cv.py \
  --folds 5 \
  --iterations 1200 \
  --learning-rate 0.05 \
  --depth 6 \
  --tune
```

Outputs:

- `submission_catboost_cv.csv`
- console summaries for fold metrics, OOF metrics, and the chosen ensemble weight

### `generate_titanic_vae.py`

Train a VAE over a compact, mixed-type Titanic feature space and sample synthetic rows.

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `--input` | `train.csv` | Source Titanic training data |
| `--output` | `synthetic_titanic_vae_20000.csv` | Output CSV for sampled rows |
| `--num-samples` | `20000` | Number of synthetic passengers to generate |
| `--epochs` | `800` | Training epochs |
| `--latent-dim` | `4` | Latent dimension size |
| `--beta` | `0.02` | KL-divergence weight |

Example:

```bash
python generate_titanic_vae.py --epochs 400 --latent-dim 8 --num-samples 5000
```

Output:

- `synthetic_titanic_vae_20000.csv` by default, or the custom path passed through `--output`

## Data Files

| File | Purpose |
| --- | --- |
| `train.csv` | Kaggle training data with labels |
| `test.csv` | Kaggle competition test data |
| `gender_submission.csv` | Kaggle baseline submission |
| `submission.csv` | Notebook-generated submission artifact |
| `submission_catboost_cv.csv` | Script-generated submission artifact |
| `synthetic_titanic_vae_20000.csv` | Synthetic rows sampled from the VAE |

## Examples

Run the CatBoost pipeline with a fixed blend weight:

```bash
python train_titanic_catboost_cv.py --ensemble-weight 0.65
```

Write the VAE output to a new file:

```bash
python generate_titanic_vae.py --output synthetic_titanic_vae_5000.csv --num-samples 5000
```

## License

No project-specific license file is included in this folder.
