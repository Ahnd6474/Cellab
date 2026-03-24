from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

from digit_recognizer_reader import load_digit_images
from mnist_idx_reader import MnistDataloader, default_split_paths


DEFAULT_YOLO_MODELS = ("yolo26x-cls.pt", "yolo11x-cls.pt")
DEFAULT_YOLO_EPOCHS = 30
DEFAULT_YOLO_BATCH = 64
DEFAULT_YOLO_IMGSZ = 32
DEFAULT_YOLO_PATIENCE = 10
DEFAULT_CNN_EPOCHS = 15
DEFAULT_CNN_BATCH = 256
DEFAULT_CNN_PATIENCE = 5
DEFAULT_CNN_LR = 1e-3
DEFAULT_CNN_WEIGHT_DECAY = 1e-4
DEFAULT_PREDICT_BATCH = 1024


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "mnist_ensemble"


def default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(4, cpu_count)


def default_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return ",".join(str(index) for index in range(torch.cuda.device_count()))


def resolve_model(model_name_or_path: str) -> str:
    model_path = Path(model_name_or_path).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    return model_name_or_path


def primary_torch_device(device_arg: str) -> torch.device:
    normalized = device_arg.strip().lower()
    if normalized == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        return torch.device(normalized)
    first_token = normalized.split(",")[0].strip()
    if first_token.isdigit():
        return torch.device(f"cuda:{first_token}")
    return torch.device("cuda:0")


def sanitize_run_name(model_name_or_path: str) -> str:
    path = Path(model_name_or_path)
    candidate = path.stem if path.suffix else path.name
    return candidate.replace(" ", "-")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class EnsembleConfig:
    train_images: Path
    train_labels: Path
    test_images: Path
    test_labels: Path
    test_csv: Path
    sample_submission: Path
    output_dir: Path
    dataset_dir: Path
    project_dir: Path
    yolo_models: tuple[str, ...]
    yolo_epochs: int
    yolo_batch: int
    yolo_imgsz: int
    yolo_patience: int
    cnn_epochs: int
    cnn_batch: int
    cnn_patience: int
    cnn_learning_rate: float
    cnn_weight_decay: float
    val_size: float
    seed: int
    train_limit: int | None
    test_limit: int | None
    workers: int
    device: str
    cache: bool
    amp: bool
    overwrite: bool
    prepare_only: bool
    dry_run: bool
    rgb: bool
    test_after_train: bool
    submission_after_train: bool
    predict_batch: int
    submission_filename: str
    ensemble_weights: tuple[float, float, float]


@dataclass(frozen=True)
class PreparedDataset:
    dataset_dir: Path
    summary: dict[str, Any]
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    kind: str
    checkpoint_path: Path


class DigitTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    defaults = default_split_paths()
    parser = argparse.ArgumentParser(
        description="Train a digit-recognizer ensemble with YOLO26x, YOLO11x, and a CNN using soft voting."
    )
    parser.add_argument("--train-images", type=Path, default=defaults.training_images_filepath, help="Path to train-images.idx3-ubyte.")
    parser.add_argument("--train-labels", type=Path, default=defaults.training_labels_filepath, help="Path to train-labels.idx1-ubyte.")
    parser.add_argument("--test-images", type=Path, default=defaults.test_images_filepath, help="Path to test-images.idx3-ubyte.")
    parser.add_argument("--test-labels", type=Path, default=defaults.test_labels_filepath, help="Path to test-labels.idx1-ubyte.")
    parser.add_argument("--test-csv", type=Path, default=Path(__file__).resolve().parent / "test.csv", help="Path to Kaggle test.csv used for submission generation.")
    parser.add_argument("--sample-submission", type=Path, default=Path(__file__).resolve().parent / "sample_submission.csv", help="Path to sample_submission.csv.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Output root containing exported data, runs, and submissions.")
    parser.add_argument(
        "--yolo-models",
        nargs=2,
        default=list(DEFAULT_YOLO_MODELS),
        metavar=("YOLO26X", "YOLO11X"),
        help="Two Ultralytics classification models. Defaults to yolo26x-cls.pt and yolo11x-cls.pt.",
    )
    parser.add_argument("--yolo-epochs", type=int, default=DEFAULT_YOLO_EPOCHS, help="Training epochs for each YOLO model.")
    parser.add_argument("--yolo-batch", type=int, default=DEFAULT_YOLO_BATCH, help="Training batch size for each YOLO model.")
    parser.add_argument("--yolo-imgsz", type=int, default=DEFAULT_YOLO_IMGSZ, help="Training image size for each YOLO model.")
    parser.add_argument("--yolo-patience", type=int, default=DEFAULT_YOLO_PATIENCE, help="Early stopping patience for YOLO models.")
    parser.add_argument("--cnn-epochs", type=int, default=DEFAULT_CNN_EPOCHS, help="Training epochs for the CNN.")
    parser.add_argument("--cnn-batch", type=int, default=DEFAULT_CNN_BATCH, help="Training batch size for the CNN.")
    parser.add_argument("--cnn-patience", type=int, default=DEFAULT_CNN_PATIENCE, help="Early stopping patience for the CNN.")
    parser.add_argument("--cnn-learning-rate", type=float, default=DEFAULT_CNN_LR, help="Learning rate for the CNN.")
    parser.add_argument("--cnn-weight-decay", type=float, default=DEFAULT_CNN_WEIGHT_DECAY, help="Weight decay for the CNN optimizer.")
    parser.add_argument(
        "--ensemble-weights",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        metavar=("YOLO26X_W", "YOLO11X_W", "CNN_W"),
        help="Soft-voting weights for YOLO26x, YOLO11x, and CNN in that order.",
    )
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction of the training split reserved for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for train/val splitting and CNN training.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional row limit for the training split.")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional row limit for the test split.")
    parser.add_argument("--workers", type=int, default=default_workers(), help="Worker count for YOLO and CNN dataloaders.")
    parser.add_argument(
        "--device",
        default=default_device(),
        help='Training device for YOLO. Examples: "cpu", "0", "0,1". CNN uses the first visible device.',
    )
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False, help="Cache the YOLO dataset in memory.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable AMP where supported.")
    parser.add_argument("--rgb", action=argparse.BooleanOptionalAction, default=True, help="Save exported PNGs as RGB for YOLO classification.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Rebuild the exported dataset and allow overwriting existing runs.")
    parser.add_argument("--prepare-only", action="store_true", help="Only export and index the dataset for the ensemble, then exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved configuration and exit without training.")
    parser.add_argument("--test-after-train", action=argparse.BooleanOptionalAction, default=True, help="Evaluate each model and the soft-voting ensemble on the IDX test split.")
    parser.add_argument("--submission-after-train", action=argparse.BooleanOptionalAction, default=True, help="Generate a Kaggle submission from test.csv with soft voting.")
    parser.add_argument("--predict-batch", type=int, default=DEFAULT_PREDICT_BATCH, help="Batch size used when predicting the IDX test split or test.csv.")
    parser.add_argument("--submission-filename", default="submission_ensemble.csv", help="Filename written under --output-dir for Kaggle predictions.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EnsembleConfig:
    if not 0.0 < args.val_size < 1.0:
        raise ValueError(f"--val-size must be between 0 and 1, got {args.val_size}")
    if args.train_limit is not None and args.train_limit <= 0:
        raise ValueError("--train-limit must be positive when provided.")
    if args.test_limit is not None and args.test_limit <= 0:
        raise ValueError("--test-limit must be positive when provided.")
    if args.predict_batch <= 0:
        raise ValueError("--predict-batch must be positive.")
    if args.cnn_epochs <= 0 or args.yolo_epochs <= 0:
        raise ValueError("Epoch values must be positive.")
    if args.cnn_batch <= 0 or args.yolo_batch <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.cnn_learning_rate <= 0:
        raise ValueError("--cnn-learning-rate must be positive.")
    if args.cnn_weight_decay < 0:
        raise ValueError("--cnn-weight-decay must be zero or positive.")
    if args.cnn_patience < 0 or args.yolo_patience < 0:
        raise ValueError("Patience values must be zero or positive.")
    if len(args.ensemble_weights) != 3:
        raise ValueError("--ensemble-weights must contain exactly three values.")
    if sum(args.ensemble_weights) <= 0:
        raise ValueError("--ensemble-weights must sum to a positive value.")

    output_dir = args.output_dir.resolve()
    dataset_dir = output_dir / "dataset"
    project_dir = output_dir / "runs"

    config = EnsembleConfig(
        train_images=args.train_images.resolve(),
        train_labels=args.train_labels.resolve(),
        test_images=args.test_images.resolve(),
        test_labels=args.test_labels.resolve(),
        test_csv=args.test_csv.resolve(),
        sample_submission=args.sample_submission.resolve(),
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        project_dir=project_dir,
        yolo_models=tuple(resolve_model(model) for model in args.yolo_models),
        yolo_epochs=args.yolo_epochs,
        yolo_batch=args.yolo_batch,
        yolo_imgsz=args.yolo_imgsz,
        yolo_patience=args.yolo_patience,
        cnn_epochs=args.cnn_epochs,
        cnn_batch=args.cnn_batch,
        cnn_patience=args.cnn_patience,
        cnn_learning_rate=args.cnn_learning_rate,
        cnn_weight_decay=args.cnn_weight_decay,
        val_size=args.val_size,
        seed=args.seed,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        workers=args.workers,
        device=args.device,
        cache=args.cache,
        amp=args.amp,
        overwrite=args.overwrite,
        prepare_only=args.prepare_only,
        dry_run=args.dry_run,
        rgb=args.rgb,
        test_after_train=args.test_after_train,
        submission_after_train=args.submission_after_train,
        predict_batch=args.predict_batch,
        submission_filename=args.submission_filename,
        ensemble_weights=tuple(float(weight) for weight in args.ensemble_weights),
    )

    for input_path in [config.train_images, config.train_labels, config.test_images, config.test_labels]:
        if not input_path.exists():
            raise FileNotFoundError(f"MNIST IDX file not found: {input_path}")
    if config.submission_after_train:
        for input_path in [config.test_csv, config.sample_submission]:
            if not input_path.exists():
                raise FileNotFoundError(f"Submission input file not found: {input_path}")

    return config


def print_config(config: EnsembleConfig) -> None:
    payload: dict[str, Any] = asdict(config)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def save_png(image_array: np.ndarray, output_path: Path, rgb: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(image_array, mode="L")
    if rgb:
        image = image.convert("RGB")
    image.save(output_path, format="PNG", compress_level=0)


def export_split(
    split_name: str,
    dataset_dir: Path,
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    rgb: bool,
) -> dict[str, int]:
    split_root = dataset_dir / split_name
    class_counts = {str(label): 0 for label in range(10)}

    for original_index in indices:
        label = int(labels[original_index])
        image = images[original_index]
        output_path = split_root / str(label) / f"{split_name}_{int(original_index):05d}.png"
        save_png(image, output_path, rgb=rgb)
        class_counts[str(label)] += 1

    return class_counts


def load_split_indices(indices_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(indices_path)
    return payload["train_indices"], payload["val_indices"], payload["test_indices"]


def dataset_metadata_path(output_dir: Path) -> Path:
    return output_dir / "dataset_metadata.json"


def dataset_indices_path(output_dir: Path) -> Path:
    return output_dir / "split_indices.npz"


def validate_reused_dataset(config: EnsembleConfig, metadata: dict[str, Any]) -> None:
    expected = {
        "train_limit": config.train_limit,
        "test_limit": config.test_limit,
        "val_size": config.val_size,
        "seed": config.seed,
        "rgb": config.rgb,
    }
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise FileExistsError(
            f"Existing dataset at {config.dataset_dir} was prepared with different parameters: {actual}. "
            f"Use --overwrite to rebuild it."
        )


def prepare_dataset(config: EnsembleConfig) -> PreparedDataset:
    loader = MnistDataloader(
        training_images_filepath=config.train_images,
        training_labels_filepath=config.train_labels,
        test_images_filepath=config.test_images,
        test_labels_filepath=config.test_labels,
    )
    (x_train, y_train), (x_test, y_test) = loader.load_data(
        train_limit=config.train_limit,
        test_limit=config.test_limit,
    )

    metadata_path = dataset_metadata_path(config.output_dir)
    indices_path = dataset_indices_path(config.output_dir)

    if config.dataset_dir.exists() and not config.overwrite:
        if not metadata_path.exists() or not indices_path.exists():
            raise FileExistsError(
                f"Dataset directory already exists at {config.dataset_dir}, but metadata files are missing. "
                f"Use --overwrite to rebuild it."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_reused_dataset(config=config, metadata=metadata)
        train_indices, val_indices, test_indices = load_split_indices(indices_path)
        print(f"dataset_dir={config.dataset_dir}")
        print("dataset_reused=true")
        return PreparedDataset(
            dataset_dir=config.dataset_dir,
            summary=metadata["summary"],
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

    if config.dataset_dir.exists():
        shutil.rmtree(config.dataset_dir)
    config.dataset_dir.mkdir(parents=True, exist_ok=True)

    all_train_indices = np.arange(len(y_train))
    train_indices, val_indices = train_test_split(
        all_train_indices,
        test_size=config.val_size,
        random_state=config.seed,
        stratify=y_train,
    )
    test_indices = np.arange(len(y_test))

    summary: dict[str, Any] = {
        "train": export_split("train", config.dataset_dir, x_train, y_train, train_indices, rgb=config.rgb),
        "val": export_split("val", config.dataset_dir, x_train, y_train, val_indices, rgb=config.rgb),
        "test": export_split("test", config.dataset_dir, x_test, y_test, test_indices, rgb=config.rgb),
        "train_rows": int(len(train_indices)),
        "val_rows": int(len(val_indices)),
        "test_rows": int(len(test_indices)),
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
    }
    metadata = {
        "train_limit": config.train_limit,
        "test_limit": config.test_limit,
        "val_size": config.val_size,
        "seed": config.seed,
        "rgb": config.rgb,
        "summary": summary,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(indices_path, train_indices=train_indices, val_indices=val_indices, test_indices=test_indices)

    print(f"dataset_dir={config.dataset_dir}")
    print("dataset_summary=")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return PreparedDataset(
        dataset_dir=config.dataset_dir,
        summary=summary,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


def resolve_best_model_path(save_dir: str | os.PathLike[str] | None) -> Path:
    if save_dir is None:
        raise FileNotFoundError("Ultralytics save_dir was not set, so best.pt could not be located.")
    best_model_path = Path(save_dir) / "weights" / "best.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_model_path}")
    return best_model_path


def render_prediction_image(image_array: np.ndarray, rgb: bool) -> Image.Image:
    image = Image.fromarray(image_array, mode="L")
    return image.convert("RGB") if rgb else image


def train_yolo_model(config: EnsembleConfig, dataset_dir: Path, model_name_or_path: str) -> ModelArtifact:
    run_name = sanitize_run_name(model_name_or_path)
    print(f"training_yolo_model={model_name_or_path}")
    model = YOLO(model_name_or_path, task="classify")
    results = model.train(
        data=str(dataset_dir),
        epochs=config.yolo_epochs,
        batch=config.yolo_batch,
        imgsz=config.yolo_imgsz,
        patience=config.yolo_patience,
        device=config.device,
        workers=config.workers,
        project=str(config.project_dir),
        name=run_name,
        seed=config.seed,
        cache=config.cache,
        amp=config.amp,
        verbose=True,
        plots=True,
        exist_ok=config.overwrite,
    )

    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is not None:
        print(f"save_dir={save_dir}")

    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict) and results_dict:
        print("train_metrics=")
        print(json.dumps(results_dict, ensure_ascii=False, indent=2))

    best_model_path = resolve_best_model_path(save_dir)
    return ModelArtifact(
        name=run_name,
        kind="yolo",
        checkpoint_path=best_model_path,
    )


def predict_yolo_probabilities(
    checkpoint_path: Path,
    images: np.ndarray,
    batch_size: int,
    imgsz: int,
    device: str,
    rgb: bool,
) -> np.ndarray:
    model = YOLO(str(checkpoint_path), task="classify")
    probabilities: list[np.ndarray] = []

    for start in range(0, len(images), batch_size):
        batch_images = [
            render_prediction_image(image_array, rgb=rgb)
            for image_array in images[start : start + batch_size]
        ]
        batch_results = model.predict(
            source=batch_images,
            imgsz=imgsz,
            batch=len(batch_images),
            device=device,
            verbose=False,
        )
        for result in batch_results:
            if result.probs is None:
                raise ValueError(f"Prediction result from {checkpoint_path} did not include class probabilities.")
            probabilities.append(result.probs.data.detach().cpu().numpy())

    return np.stack(probabilities, axis=0)


def build_cnn_dataloader(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = DigitTensorDataset(images=images, labels=labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def evaluate_cnn_model(
    model: DigitCNN,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = functional.cross_entropy(logits, labels)

            total_examples += int(labels.size(0))
            total_loss += float(loss.item()) * int(labels.size(0))
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def train_cnn_model(config: EnsembleConfig, prepared: PreparedDataset) -> ModelArtifact:
    device = primary_torch_device(config.device)
    pin_memory = device.type == "cuda"
    train_loader = build_cnn_dataloader(
        images=prepared.x_train[prepared.train_indices],
        labels=prepared.y_train[prepared.train_indices],
        batch_size=config.cnn_batch,
        workers=config.workers,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = build_cnn_dataloader(
        images=prepared.x_train[prepared.val_indices],
        labels=prepared.y_train[prepared.val_indices],
        batch_size=config.cnn_batch,
        workers=config.workers,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = DigitCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.cnn_learning_rate,
        weight_decay=config.cnn_weight_decay,
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    cnn_dir = config.project_dir / "cnn"
    if cnn_dir.exists():
        if config.overwrite:
            shutil.rmtree(cnn_dir)
        else:
            raise FileExistsError(f"CNN run directory already exists: {cnn_dir}. Re-run with --overwrite to replace it.")
    cnn_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = cnn_dir / "best.pt"
    history_path = cnn_dir / "history.json"

    history: list[dict[str, float | int]] = []
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, config.cnn_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = functional.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_examples += int(labels.size(0))
            running_loss += float(loss.item()) * int(labels.size(0))
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())

        train_loss = running_loss / max(running_examples, 1)
        train_accuracy = running_correct / max(running_examples, 1)
        val_metrics = evaluate_cnn_model(model=model, dataloader=val_loader, device=device)

        history_entry: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(history_entry)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        print("cnn_epoch=" + json.dumps(history_entry, ensure_ascii=False))

        improved = (
            val_metrics["accuracy"] > best_accuracy
            or (
                np.isclose(val_metrics["accuracy"], best_accuracy)
                and val_metrics["loss"] < best_loss
            )
        )
        if improved:
            best_accuracy = val_metrics["accuracy"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_accuracy": best_accuracy,
                    "best_val_loss": best_loss,
                    "best_epoch": best_epoch,
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if config.cnn_patience and epochs_without_improvement >= config.cnn_patience:
            print(f"cnn_early_stop_epoch={epoch}")
            break

    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"CNN checkpoint was not created: {best_checkpoint_path}")

    print(f"cnn_best_epoch={best_epoch}")
    print(f"cnn_checkpoint={best_checkpoint_path}")
    return ModelArtifact(
        name="cnn",
        kind="cnn",
        checkpoint_path=best_checkpoint_path,
    )


def load_cnn_checkpoint(checkpoint_path: Path, device: torch.device) -> DigitCNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DigitCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_cnn_probabilities(
    checkpoint_path: Path,
    images: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = load_cnn_checkpoint(checkpoint_path=checkpoint_path, device=device)
    dataset = DigitTensorDataset(images=images, labels=np.zeros(len(images), dtype=np.int64))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for images_batch, _ in dataloader:
            images_batch = images_batch.to(device, non_blocking=True)
            logits = model(images_batch)
            batch_probabilities = functional.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(batch_probabilities)

    return np.concatenate(probabilities, axis=0)


def soft_vote(probabilities: list[np.ndarray], weights: tuple[float, float, float]) -> np.ndarray:
    if len(probabilities) != len(weights):
        raise ValueError(f"Probability count ({len(probabilities)}) must match weight count ({len(weights)}).")
    stacked = np.stack(probabilities, axis=0)
    weighted = stacked * np.asarray(weights, dtype=np.float32).reshape(-1, 1, 1)
    return weighted.sum(axis=0) / float(sum(weights))


def probabilities_to_predictions(probabilities: np.ndarray) -> np.ndarray:
    return probabilities.argmax(axis=1).astype(np.int64)


def accuracy_from_predictions(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float((predictions == labels).mean())


def save_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_models(
    config: EnsembleConfig,
    prepared: PreparedDataset,
    yolo_artifacts: list[ModelArtifact],
    cnn_artifact: ModelArtifact,
) -> tuple[dict[str, Any], np.ndarray]:
    yolo_probabilities = [
        predict_yolo_probabilities(
            checkpoint_path=artifact.checkpoint_path,
            images=prepared.x_test,
            batch_size=config.predict_batch,
            imgsz=config.yolo_imgsz,
            device=config.device,
            rgb=config.rgb,
        )
        for artifact in yolo_artifacts
    ]
    cnn_probabilities = predict_cnn_probabilities(
        checkpoint_path=cnn_artifact.checkpoint_path,
        images=prepared.x_test,
        batch_size=config.predict_batch,
        device=primary_torch_device(config.device),
    )
    all_probabilities = [*yolo_probabilities, cnn_probabilities]
    ensemble_probabilities = soft_vote(all_probabilities, weights=config.ensemble_weights)

    model_summaries: list[dict[str, Any]] = []
    for artifact, probabilities in zip([*yolo_artifacts, cnn_artifact], all_probabilities):
        predictions = probabilities_to_predictions(probabilities)
        model_summaries.append(
            {
                "name": artifact.name,
                "kind": artifact.kind,
                "checkpoint_path": str(artifact.checkpoint_path),
                "accuracy": accuracy_from_predictions(predictions, prepared.y_test),
            }
        )

    ensemble_predictions = probabilities_to_predictions(ensemble_probabilities)
    summary = {
        "models": model_summaries,
        "ensemble": {
            "weights": list(config.ensemble_weights),
            "accuracy": accuracy_from_predictions(ensemble_predictions, prepared.y_test),
        },
    }
    return summary, ensemble_probabilities


def generate_submission(
    config: EnsembleConfig,
    yolo_artifacts: list[ModelArtifact],
    cnn_artifact: ModelArtifact,
) -> Path:
    test_images, labels = load_digit_images(config.test_csv)
    if labels is not None:
        raise ValueError(f"{config.test_csv} should not contain a label column when generating a submission.")

    yolo_probabilities = [
        predict_yolo_probabilities(
            checkpoint_path=artifact.checkpoint_path,
            images=test_images,
            batch_size=config.predict_batch,
            imgsz=config.yolo_imgsz,
            device=config.device,
            rgb=config.rgb,
        )
        for artifact in yolo_artifacts
    ]
    cnn_probabilities = predict_cnn_probabilities(
        checkpoint_path=cnn_artifact.checkpoint_path,
        images=test_images,
        batch_size=config.predict_batch,
        device=primary_torch_device(config.device),
    )
    ensemble_probabilities = soft_vote([*yolo_probabilities, cnn_probabilities], weights=config.ensemble_weights)
    predictions = probabilities_to_predictions(ensemble_probabilities)

    submission = pd.read_csv(config.sample_submission)
    if len(submission) != len(predictions):
        raise ValueError(
            f"sample submission row count ({len(submission)}) does not match predictions ({len(predictions)})."
        )

    submission[submission.columns[-1]] = predictions
    submission_path = config.output_dir / config.submission_filename
    submission.to_csv(submission_path, index=False)
    print(f"submission_path={submission_path}")
    return submission_path


def main() -> None:
    config = build_config(parse_args())
    if config.dry_run:
        print("dry_run=true")
        print_config(config)
        return

    set_seed(config.seed)
    prepared = prepare_dataset(config)
    if config.prepare_only:
        print("prepare_only=true")
        return

    config.project_dir.mkdir(parents=True, exist_ok=True)
    print("ensemble_training_config=")
    print_config(config)

    yolo_artifacts = [
        train_yolo_model(config=config, dataset_dir=prepared.dataset_dir, model_name_or_path=model_name)
        for model_name in config.yolo_models
    ]
    cnn_artifact = train_cnn_model(config=config, prepared=prepared)

    summary: dict[str, Any] = {
        "config": {
            "output_dir": str(config.output_dir),
            "project_dir": str(config.project_dir),
            "dataset_dir": str(prepared.dataset_dir),
        },
        "dataset": prepared.summary,
        "artifacts": {
            "yolo": [str(artifact.checkpoint_path) for artifact in yolo_artifacts],
            "cnn": str(cnn_artifact.checkpoint_path),
        },
    }

    if config.test_after_train:
        evaluation_summary, _ = evaluate_models(
            config=config,
            prepared=prepared,
            yolo_artifacts=yolo_artifacts,
            cnn_artifact=cnn_artifact,
        )
        summary["evaluation"] = evaluation_summary
        print("evaluation_summary=")
        print(json.dumps(evaluation_summary, ensure_ascii=False, indent=2))

    if config.submission_after_train:
        submission_path = generate_submission(
            config=config,
            yolo_artifacts=yolo_artifacts,
            cnn_artifact=cnn_artifact,
        )
        summary["submission_path"] = str(submission_path)

    save_json(config.output_dir / "ensemble_summary.json", summary)


if __name__ == "__main__":
    main()
