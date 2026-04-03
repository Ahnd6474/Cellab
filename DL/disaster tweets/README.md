# Disaster Tweets Transformer Classifier

This folder contains a Transformer encoder training script for the Kaggle `Natural Language Processing with Disaster Tweets` dataset.

## Architecture
- The model uses a standard `token -> tweet` Transformer classifier.
- Each tweet is tokenized into a single sequence with `[CLS]` and `[SEP]`.
- Token embeddings and sinusoidal positional encodings are passed through a Transformer encoder.
- Attention pooling compresses the encoded token sequence into one tweet vector.
- A final linear classifier predicts the disaster label.

The training script file keeps the historical name `train_hierarchical_transformer.py`, but the model is now a plain Transformer classifier rather than a hierarchical one.

## Run
```bash
python train_hierarchical_transformer.py
```

CPU-only run:
```bash
python train_hierarchical_transformer.py --cpu
```

Example with custom hyperparameters:
```bash
python train_hierarchical_transformer.py --epochs 10 --batch-size 64 --hidden-size 256 --num-layers 4 --max-tokens 96
```

## Extra Data Training
To reduce overfitting, use the extra-data trainer:

```bash
python train_transformer_with_extra_data.py
```

This script can:
- load extra labeled `.csv` or `.tsv` files from `extra_data/`
- generate synthetic hard positives and hard negatives
- pretrain on extra data first, then finetune on the Kaggle split

Example:
```bash
python train_transformer_with_extra_data.py --extra-training-mode pretrain --extra-pretrain-epochs 2 --synthetic-positives 1500 --synthetic-negatives 1500
```

## Outputs
- `outputs/best_transformer.pt`
- `outputs/submission_transformer.csv`
- `outputs/test_probabilities_transformer.csv`
- `outputs_extra/training_summary.json`

## Notes
- The script uses a built-in vocabulary tokenizer, so it does not depend on downloading pretrained tokenizers.
- Input text is built from `keyword`, `location`, and `text`.
- The current default model is larger than the earlier baseline: `hidden_size=256`, `ff_dim=1024`, `num_heads=8`, `num_layers=4`.
- Validation metrics are `accuracy` and binary `F1`.
