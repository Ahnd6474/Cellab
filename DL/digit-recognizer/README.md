# Digit Recognizer

MNIST and Kaggle Digit Recognizer experiments with three training paths: a ViT classifier, a YOLO-based ensemble, and a small Flask demo app.

## Installation

Set up a Python environment and install the dependencies used by the scripts in this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas pillow scikit-learn torch transformers ultralytics flask
```

## Quick Start

Train the ViT classifier from the Kaggle CSV files:

```bash
python train_mnist_yolo_cls.py
```

That run saves a Hugging Face model bundle under `mnist_vit_cls/model` and writes a submission CSV such as:

```text
saved_model=/.../mnist_vit_cls/model
saved_submission=/.../mnist_vit_cls/submission_vit.csv
```

If you want the larger YOLO + CNN soft-voting pipeline instead:

```bash
python train_mnist_ensemble.py
```

## What Is In This Folder?

This project collects a few ways to attack handwritten digit classification:

- `train_mnist_yolo_cls.py` fine-tunes a pretrained image classifier from Hugging Face on the CSV-form Kaggle dataset.
- `train_mnist_ensemble.py` exports IDX files to image folders, trains two Ultralytics classification models plus a CNN, and blends them with soft voting.
- `web/app.py` serves a small browser UI for drawing or uploading digits and running inference with the saved ViT model.
- `digit_recognizer_reader.py` and `mnist_idx_reader.py` load Kaggle CSV or raw IDX-form MNIST files.
- `prepare_yolo_detect_dataset.py` and `ultralytics_train.py` support extra dataset and training workflows.

## Why This Project Exists

The folder is useful if you want to compare dataset formats and modeling styles on the same task:

- CSV input that looks like the Kaggle competition
- raw IDX files from MNIST
- pretrained transformer-style image classification
- Ultralytics classification models
- an ensemble that mixes learned probabilities instead of trusting one model

## Main Scripts

### `train_mnist_yolo_cls.py`

Fine-tune `facebook/deit-small-patch16-224` on `train.csv`, evaluate on a validation split, and create a Kaggle submission from `test.csv`.

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `--train-csv` | `train.csv` | Training CSV with the `label` column |
| `--test-csv` | `test.csv` | Kaggle test CSV |
| `--output-dir` | `mnist_vit_cls` | Output directory for the model and metrics |
| `--model-name` | `facebook/deit-small-patch16-224` | Backbone checkpoint |
| `--epochs` | `12` | Fine-tuning epochs |
| `--batch` | `128` | Batch size |
| `--patience` | `4` | Early stopping patience |
| `--submission-filename` | `submission_vit.csv` | Output submission filename |

Example:

```bash
python train_mnist_yolo_cls.py --epochs 6 --batch 64 --dry-run
```

Outputs:

- `mnist_vit_cls/model/`
- `mnist_vit_cls/submission_vit.csv`
- `mnist_vit_cls/metrics.json`

### `train_mnist_ensemble.py`

Build a folder-based dataset from IDX files, train two Ultralytics classifiers plus a CNN, evaluate on the IDX test split, and optionally create a Kaggle submission.

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `--output-dir` | `mnist_ensemble` | Root directory for dataset exports, runs, and submissions |
| `--yolo-models` | `yolo26x-cls.pt yolo11x-cls.pt` | Two YOLO classification backbones |
| `--yolo-epochs` | `30` | Epochs per YOLO model |
| `--cnn-epochs` | `15` | CNN epochs |
| `--prepare-only` | off | Export the dataset and stop |
| `--overwrite` | off | Rebuild the exported dataset |
| `--ensemble-weights` | `1 1 1` | Soft-voting weights for YOLO26x, YOLO11x, and CNN |

Example:

```bash
python train_mnist_ensemble.py --prepare-only --train-limit 5000 --test-limit 1000
```

Outputs land under the chosen output directory:

- `dataset/`
- `dataset_metadata.json`
- `split_indices.npz`
- `runs/`
- submission CSV generated after training

### `web/app.py`

Run the local Flask app after training the ViT model.

```bash
python web/app.py
```

Then open `http://127.0.0.1:5000`.

The app expects the saved model bundle here:

```text
mnist_vit_cls/model
```

## Data Layout

| File or Folder | Purpose |
| --- | --- |
| `train.csv` / `test.csv` | Kaggle Digit Recognizer tabular inputs |
| `sample_submission.csv` | Kaggle submission template |
| `train-images.idx3-ubyte` etc. | Raw MNIST IDX files |
| `mnist_yolo_cls/` | Existing experiment artifacts and submissions |
| `web/templates/index.html` | Browser UI template |

## Examples

Train a smaller ViT run for a quick smoke test:

```bash
python train_mnist_yolo_cls.py --train-limit 5000 --epochs 2 --batch 32
```

Train the ensemble with custom weights:

```bash
python train_mnist_ensemble.py --ensemble-weights 1.5 1.0 0.75
```

## License

No project-specific license file is included in this folder.
