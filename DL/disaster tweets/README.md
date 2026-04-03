# Disaster Tweets

Custom Transformer classifiers for Kaggle's `Natural Language Processing with Disaster Tweets` competition.

## Installation

Create a Python environment and install the packages used by the training scripts:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn torch tqdm
```

## Quick Start

Train the baseline Transformer classifier:

```bash
python train_hierarchical_transformer.py
```

The script writes checkpoints, class probabilities, and a Kaggle submission file under `outputs/`.

If you want the version that mixes in extra supervised data and synthetic examples:

```bash
python train_transformer_with_extra_data.py
```

## What Is In This Folder?

This project trains a tweet-level disaster classifier without relying on a downloaded pretrained tokenizer.

- `train_hierarchical_transformer.py` builds a vocabulary from the local training data and trains a Transformer encoder with attention pooling.
- `train_transformer_with_extra_data.py` extends the baseline with extra labeled files and synthetic hard examples.
- `train.csv`, `test.csv`, and `sample_submission.csv` are the Kaggle data files.
- `extra_data/` is the place for optional CSV or TSV files used in the extended training workflow.

## Why This Project Exists

The baseline script keeps the training stack self-contained:

- a simple regex tokenizer
- a vocabulary built from the local dataset
- sinusoidal positional encodings
- attention pooling over encoded tokens

That makes it easy to experiment without depending on external model downloads. The second script pushes further by using extra supervision and synthetic examples to make overfitting less of a problem.

## Main Scripts

### `train_hierarchical_transformer.py`

Despite the filename, this is now a plain Transformer classifier rather than a hierarchical model.

Input text is built from:

- `keyword`
- `location`
- `text`

The model pipeline is:

1. regex tokenization
2. local vocabulary construction
3. token embeddings + positional encodings
4. Transformer encoder
5. attention pooling
6. binary classification head

Example runs:

```bash
python train_hierarchical_transformer.py
```

```bash
python train_hierarchical_transformer.py --epochs 10 --batch-size 64 --hidden-size 256 --num-layers 4 --max-tokens 96
```

Typical outputs:

- `outputs/best_transformer.pt`
- `outputs/submission_transformer.csv`
- `outputs/test_probabilities_transformer.csv`

### `train_transformer_with_extra_data.py`

This script builds on the baseline and can:

- load extra labeled `.csv` and `.tsv` files from `extra_data/`
- normalize different label conventions into binary labels
- deduplicate and filter overlapping text
- generate synthetic positive and negative tweet-like samples
- pretrain on extra data before fine-tuning on the Kaggle split

Example:

```bash
python train_transformer_with_extra_data.py \
  --extra-training-mode pretrain \
  --extra-pretrain-epochs 2 \
  --synthetic-positives 1500 \
  --synthetic-negatives 1500
```

Typical outputs:

- `outputs_extra/training_summary.json`
- model checkpoints and submission artifacts from the extended run

## Data Files

| Path | Purpose |
| --- | --- |
| `train.csv` | Kaggle training data |
| `test.csv` | Kaggle test data |
| `sample_submission.csv` | Kaggle submission template |
| `extra_data/` | Optional external supervised data |

## Examples

Train on CPU if needed:

```bash
python train_hierarchical_transformer.py --cpu
```

Run the extended trainer with a lighter synthetic setup:

```bash
python train_transformer_with_extra_data.py --synthetic-positives 500 --synthetic-negatives 500
```

## Notes

- Validation metrics are accuracy and binary F1.
- The default model is intentionally larger than the earliest version of this project.
- Because the tokenizer is local and vocabulary-based, results depend on the text seen during training rather than a frozen pretrained token set.

## License

No project-specific license file is included in this folder.
