# IM

YOLO-based image detection practice project built around tomato disease detection.

## Installation

This folder already includes a `requirements.txt` file:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer is a reasonable baseline. A GPU helps a lot once image size and batch size grow.

## Quick Start

The main workflow in this folder lives in `tomatodisease.ipynb`. The notebook installs dependencies, loads a pretrained YOLO checkpoint, and trains on a dataset YAML file.

Core training cell:

```python
from ultralytics import YOLO

YAML_PATH = "/path/to/data.yaml"
model = YOLO("yolo11s.pt")
model.train(
    data=str(YAML_PATH),
    task="detect",
    imgsz=1024,
    epochs=200,
    batch=16,
    device=[0, 1],
)
```

## What Is In This Folder?

This project is a practical object-detection exercise rather than a packaged Python module.

- `tomatodisease.ipynb` is the main training and inference notebook.
- `requirements.txt` lists the notebook dependencies.
- `yolo/yolo_11` and `yolo/yolo_26` contain saved experiment outputs from earlier runs.
- `LICENSE` provides the project-specific license for this folder.

## Why This Project Exists

The folder is set up for a straightforward transfer-learning workflow:

- start from a pretrained YOLO detector
- point it at a YOLO-format tomato disease dataset
- train with larger images for fine-grained visual patterns
- inspect curves, confusion matrices, and saved weights afterward

It is a good fit for classroom or notebook-driven experiments where you want fast iteration more than a full training package.

## Training Workflow

The notebook assumes you already have a YOLO dataset YAML file. In practice, that means:

1. your images and labels are in YOLO format
2. the dataset YAML points to the train and validation splits
3. class names are defined in the YAML file

The training call uses Ultralytics directly, so you can change model size, image size, epoch count, and devices in one place.

Device examples:

- one GPU: `device=0`
- multiple GPUs: `device=[0, 1]`
- CPU only: `device="cpu"`

## Expected Outputs

Training runs usually create a directory like this:

```text
yolo/
  yolo_XX/
    args.yaml
    results.csv
    results.png
    confusion_matrix.png
    confusion_matrix_normalized.png
    weights/
      best.pt
      last.pt
```

What the main artifacts mean:

| File | Purpose |
| --- | --- |
| `weights/best.pt` | Best checkpoint by validation performance |
| `weights/last.pt` | Final epoch checkpoint |
| `results.csv` | Epoch-by-epoch metrics |
| `results.png` | Training curves |
| `confusion_matrix*.png` | Class-level error patterns |

## Inference Example

Use a saved checkpoint for prediction:

```python
from ultralytics import YOLO

model = YOLO("yolo/yolo_26/weights/best.pt")
results = model.predict(source="path/to/image.jpg", imgsz=1024, conf=0.25)
```

If you raise `conf`, false positives usually drop, but recall can fall with it.

## File Guide

| Path | Purpose |
| --- | --- |
| `tomatodisease.ipynb` | Main notebook for training and inference |
| `requirements.txt` | Dependencies for the notebook |
| `yolo/yolo_11` | Saved run artifacts |
| `yolo/yolo_26` | Saved run artifacts |

## License

This folder includes its own [LICENSE](LICENSE) file.
