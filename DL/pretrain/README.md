# Pretrain

Image classification workflow for `cat`, `dog`, and `unknown`, plus a FastAPI app for data collection and inference.

## Installation

This project already includes a `requirements.txt` file. Create an environment and install it:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Train the classifier:

```bash
python train.py
```

At the end of training the script saves a Hugging Face image-classification bundle under `classifier_model/` and writes the class list to `classes.json`.

Start the collection and inference server:

```bash
python server.py
```

By default the web app runs on `http://127.0.0.1:8000`.

## What Is In This Folder?

This folder combines three related pieces:

- `train.py` trains a DeiT image classifier on local image folders.
- the same script can generate extra `unknown` images with a ViT-MAE reconstruction pass before classifier training.
- `server.py` serves a small FastAPI backend and static frontend for uploading labeled images and running predictions with the trained model.

## Why This Project Exists

The setup is aimed at a practical image-collection loop:

1. gather images into class folders
2. augment the underrepresented `unknown` class
3. fine-tune a lightweight pretrained classifier
4. serve it behind a browser-based labeling and prediction UI

That makes the folder useful for small, local experiments where the dataset is still changing.

## Training Script

### `train.py`

The training script expects this folder structure:

```text
dataset/
  train/
    cat/
    dog/
    unknown/
```

Important implementation details:

- classifier backbone: `facebook/deit-tiny-patch16-224`
- MAE augmentation backbone: `facebook/vit-mae-base`
- supported classes: `cat`, `dog`, `unknown`
- MAE augmentation target: 5,000 generated images for `unknown`
- loss: focal loss with class-frequency-derived alpha weights

Outputs:

- `classifier_model/`
- `classes.json`
- generated images under `dataset/augmented_train/unknown/`

If a class folder is empty, training stops with a validation error instead of silently continuing.

## Server API

### `server.py`

The server loads `classifier_model/` on startup if it exists.

Routes:

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | `GET` | Serve the static frontend |
| `/api/classes` | `GET` | Return the supported class names |
| `/api/upload` | `POST` | Save uploaded images into class folders |
| `/api/predict` | `POST` | Run classifier inference on one image |

Example inference payload:

```json
{
  "dataUrl": "data:image/png;base64,..."
}
```

Example upload payload:

```json
{
  "images": [
    {
      "className": "cat",
      "type": "original",
      "dataUrl": "data:image/jpeg;base64,..."
    }
  ]
}
```

## Files and Folders

| Path | Purpose |
| --- | --- |
| `requirements.txt` | Python dependency list |
| `dataset/` | Source and augmented training images |
| `classifier_model/` | Saved processor and model weights |
| `static/index.html` | Web frontend served by FastAPI |
| `collect_unknown_coco.py` | Utility script related to unknown-image collection |
| `train_log.txt` | Existing training log artifact |

## Examples

Train after adding more `unknown` images:

```bash
python train.py
```

Run the server once a model exists:

```bash
python server.py
```

## License

No project-specific license file is included in this folder.
