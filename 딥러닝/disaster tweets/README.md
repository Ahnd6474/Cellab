# Disaster Tweets Hierarchical Transformer

This folder contains a hierarchical transformer training script for the Kaggle `Natural Language Processing with Disaster Tweets` dataset.

## Architecture
- The model uses a `token -> chunk -> tweet` hierarchy.
- Each tweet is tokenized, split into fixed-size chunks, and encoded with a token-level transformer.
- Attention pooling converts each chunk into a single vector.
- A second transformer models interactions between chunk vectors.
- Final attention pooling and a linear classifier predict the disaster label.

Tweets are short, so this adapts the usual `sentence -> document` hierarchy into `chunk -> tweet` while keeping the same hierarchical transformer idea.

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
python train_hierarchical_transformer.py --epochs 10 --batch-size 64 --hidden-size 256 --chunk-size 12 --max-chunks 8
```

## Outputs
- `outputs/best_hierarchical_transformer.pt`
- `outputs/submission_hierarchical_transformer.csv`
- `outputs/test_probabilities_hierarchical_transformer.csv`

## Notes
- The script uses a built-in vocabulary tokenizer, so it does not depend on downloading pretrained tokenizers.
- Input text is built from `keyword`, `location`, and `text`.
- Validation metrics are `accuracy` and binary `F1`.
