from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

from digit_recognizer_reader import load_digit_images
from mnist_idx_reader import MnistDataloader, default_split_paths


DEFAULT_EPOCHS = 30
DEFAULT_BATCH = 128
DEFAULT_IMGSZ = 32
DEFAULT_PATIENCE = 10
DEFAULT_PREDICT_BATCH = 4096
DEFAULT_MODEL = "yolo26n-cls.pt"
DEFAULT_SEEDS = (42, 52, 62)


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "mnist_yolo_cls"


def default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(4, cpu_count)


def default_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return ",".join(str(index) for index in range(torch.cuda.device_count()))


@dataclass(frozen=True)
class TrainConfig:
    train_images: Path
    train_labels: Path
    test_images: Path
    test_labels: Path
    test_csv: Path
    sample_submission: Path
    output_dir: Path
    dataset_dir: Path
    project_dir: Path
    model: str
    run_name: str
    seeds: tuple[int, ...]
    epochs: int
    batch: int
    imgsz: int
    patience: int
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


def parse_args() -> argparse.Namespace:
    defaults = default_split_paths()
    parser = argparse.ArgumentParser(
        description="Read MNIST IDX files, export an Ultralytics classification dataset, and train YOLO."
    )
    parser.add_argument("--train-images", type=Path, default=defaults.training_images_filepath, help="Path to train-images.idx3-ubyte.")
    parser.add_argument("--train-labels", type=Path, default=defaults.training_labels_filepath, help="Path to train-labels.idx1-ubyte.")
    parser.add_argument("--test-images", type=Path, default=defaults.test_images_filepath, help="Path to test-images.idx3-ubyte.")
    parser.add_argument("--test-labels", type=Path, default=defaults.test_labels_filepath, help="Path to test-labels.idx1-ubyte.")
    parser.add_argument("--test-csv", type=Path, default=Path(__file__).resolve().parent / "test.csv", help="Path to the Kaggle test.csv used for submission generation.")
    parser.add_argument("--sample-submission", type=Path, default=Path(__file__).resolve().parent / "sample_submission.csv", help="Path to the Kaggle sample submission template.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Output root containing the dataset export and training runs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics classification model or local checkpoint path.")
    parser.add_argument("--name", default="mnist-yolo26n-ensemble", help="Ultralytics run name prefix.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="One or more training seeds. Each seed trains one model and predictions are averaged.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Training batch size.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Square training image size passed to Ultralytics.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction of the training split reserved for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for train/val splitting.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional row limit for the training split.")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional row limit for the test split.")
    parser.add_argument("--workers", type=int, default=default_workers(), help="Ultralytics dataloader worker count.")
    parser.add_argument(
        "--device",
        default=default_device(),
        help='Training device. Examples: "cpu", "0", "0,1". Defaults to all visible CUDA devices or CPU.',
    )
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False, help="Cache dataset images in memory.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable automatic mixed precision.")
    parser.add_argument("--rgb", action=argparse.BooleanOptionalAction, default=True, help="Save exported PNGs as RGB instead of grayscale.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Delete the existing exported dataset before regenerating it.")
    parser.add_argument("--prepare-only", action="store_true", help="Only export the Ultralytics classification dataset and skip training.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved configuration and exit before writing files.")
    parser.add_argument("--test-after-train", action=argparse.BooleanOptionalAction, default=True, help="Evaluate the best checkpoint on the test split after training.")
    parser.add_argument("--submission-after-train", action=argparse.BooleanOptionalAction, default=True, help="Generate Kaggle submission.csv from test.csv after training.")
    parser.add_argument("--predict-batch", type=int, default=DEFAULT_PREDICT_BATCH, help="Batch size used when predicting Kaggle test.csv.")
    parser.add_argument(
        "--submission-filename",
        default="submission_yolo26n_ensemble.csv",
        help="Filename written under --output-dir for Kaggle predictions.",
    )
    return parser.parse_args()


def resolve_model(model_name_or_path: str) -> str:
    model_path = Path(model_name_or_path).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    return model_name_or_path


def build_config(args: argparse.Namespace) -> TrainConfig:
    if not 0.0 < args.val_size < 1.0:
        raise ValueError(f"--val-size must be between 0 and 1, got {args.val_size}")

    if args.train_limit is not None and args.train_limit <= 0:
        raise ValueError("--train-limit must be positive when provided.")
    if args.test_limit is not None and args.test_limit <= 0:
        raise ValueError("--test-limit must be positive when provided.")
    if args.predict_batch <= 0:
        raise ValueError("--predict-batch must be positive.")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one integer seed.")

    output_dir = args.output_dir.resolve()
    dataset_dir = output_dir / "dataset"
    project_dir = output_dir / "runs"

    config = TrainConfig(
        train_images=args.train_images.resolve(),
        train_labels=args.train_labels.resolve(),
        test_images=args.test_images.resolve(),
        test_labels=args.test_labels.resolve(),
        test_csv=args.test_csv.resolve(),
        sample_submission=args.sample_submission.resolve(),
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        project_dir=project_dir,
        model=resolve_model(args.model),
        run_name=args.name,
        seeds=tuple(int(seed) for seed in args.seeds),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
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
    )

    for input_path in [config.train_images, config.train_labels, config.test_images, config.test_labels]:
        if not input_path.exists():
            raise FileNotFoundError(f"MNIST IDX file not found: {input_path}")
    if config.submission_after_train:
        for input_path in [config.test_csv, config.sample_submission]:
            if not input_path.exists():
                raise FileNotFoundError(f"Submission input file not found: {input_path}")

    return config


def print_config(config: TrainConfig) -> None:
    payload: dict[str, Any] = asdict(config)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def prepare_dataset_root(dataset_dir: Path, overwrite: bool) -> None:
    if dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset directory already exists: {dataset_dir}. Re-run with --overwrite to rebuild it.")
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)


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


def prepare_dataset(config: TrainConfig) -> tuple[Path, dict[str, Any]]:
    prepare_dataset_root(config.dataset_dir, overwrite=config.overwrite)

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

    all_train_indices = np.arange(len(y_train))
    train_indices, val_indices = train_test_split(
        all_train_indices,
        test_size=config.val_size,
        random_state=config.seed,
        stratify=y_train,
    )
    test_indices = np.arange(len(y_test))

    export_summary: dict[str, Any] = {
        "train": export_split("train", config.dataset_dir, x_train, y_train, train_indices, rgb=config.rgb),
        "val": export_split("val", config.dataset_dir, x_train, y_train, val_indices, rgb=config.rgb),
        "test": export_split("test", config.dataset_dir, x_test, y_test, test_indices, rgb=config.rgb),
        "train_rows": int(len(train_indices)),
        "val_rows": int(len(val_indices)),
        "test_rows": int(len(test_indices)),
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
    }

    summary_path = config.output_dir / "dataset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(export_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dataset_dir={config.dataset_dir}")
    print("dataset_summary=")
    print(json.dumps(export_summary, ensure_ascii=False, indent=2))
    return config.dataset_dir, export_summary


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


def predict_probabilities(
    config: TrainConfig,
    checkpoint_path: Path,
    images: np.ndarray,
) -> np.ndarray:
    model = YOLO(str(checkpoint_path), task="classify")
    probabilities: list[np.ndarray] = []

    for start in range(0, len(images), config.predict_batch):
        batch_arrays = images[start : start + config.predict_batch]
        batch_images = [render_prediction_image(image_array, rgb=config.rgb) for image_array in batch_arrays]
        batch_results = model.predict(
            source=batch_images,
            imgsz=config.imgsz,
            batch=len(batch_images),
            device=config.device,
            verbose=False,
        )
        probabilities.extend(result.probs.data.detach().cpu().numpy().astype(np.float32) for result in batch_results)

    if len(probabilities) != len(images):
        raise ValueError(
            f"Prediction count mismatch for {checkpoint_path}: images={len(images)} probabilities={len(probabilities)}."
        )
    return np.stack(probabilities, axis=0)


def ensemble_probabilities(
    config: TrainConfig,
    checkpoint_paths: list[Path],
    images: np.ndarray,
) -> np.ndarray:
    ensemble = np.zeros((len(images), 10), dtype=np.float32)

    for checkpoint_path in checkpoint_paths:
        print(f"predict_checkpoint={checkpoint_path}")
        ensemble += predict_probabilities(config, checkpoint_path=checkpoint_path, images=images)

    ensemble /= float(len(checkpoint_paths))
    return ensemble


def generate_submission(config: TrainConfig, checkpoint_paths: list[Path]) -> Path:
    test_images, labels = load_digit_images(config.test_csv)
    if labels is not None:
        raise ValueError(f"{config.test_csv} should not contain a label column when generating a submission.")

    probabilities = ensemble_probabilities(config, checkpoint_paths=checkpoint_paths, images=test_images)
    predictions = probabilities.argmax(axis=1).astype(np.int64)

    submission = pd.read_csv(config.sample_submission)
    if len(submission) != len(predictions):
        raise ValueError(
            f"sample submission row count ({len(submission)}) does not match predictions ({len(predictions)})."
        )

    target_column = submission.columns[-1]
    submission[target_column] = predictions

    submission_path = config.output_dir / config.submission_filename
    submission.to_csv(submission_path, index=False)
    print(f"submission_path={submission_path}")
    return submission_path


def load_idx_test_split(config: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    loader = MnistDataloader(
        training_images_filepath=config.train_images,
        training_labels_filepath=config.train_labels,
        test_images_filepath=config.test_images,
        test_labels_filepath=config.test_labels,
    )
    (_, _), (x_test, y_test) = loader.load_data(
        train_limit=config.train_limit,
        test_limit=config.test_limit,
    )
    return x_test, y_test


def train_single_model(config: TrainConfig, dataset_dir: Path, seed: int) -> Path:
    run_name = f"{config.run_name}-seed{seed}"
    print(f"training_seed={seed}")
    model = YOLO(config.model, task="classify")
    results = model.train(
        data=str(dataset_dir),
        epochs=config.epochs,
        batch=config.batch,
        imgsz=config.imgsz,
        patience=config.patience,
        device=config.device,
        workers=config.workers,
        project=str(config.project_dir),
        name=run_name,
        seed=seed,
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
        print(f"train_metrics_seed_{seed}=")
        print(json.dumps(results_dict, ensure_ascii=False, indent=2))

    best_model_path = resolve_best_model_path(save_dir)
    print(f"best_model_path_seed_{seed}={best_model_path}")
    return best_model_path


def evaluate_ensemble(config: TrainConfig, checkpoint_paths: list[Path]) -> None:
    x_test, y_test = load_idx_test_split(config)
    probabilities = ensemble_probabilities(config, checkpoint_paths=checkpoint_paths, images=x_test)
    predictions = probabilities.argmax(axis=1)
    accuracy = float((predictions == y_test).mean())

    print("ensemble_test_metrics=")
    print(
        json.dumps(
            {
                "num_models": len(checkpoint_paths),
                "seeds": list(config.seeds),
                "accuracy": accuracy,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def train(config: TrainConfig, dataset_dir: Path) -> None:
    print("training_config=")
    print_config(config)

    config.project_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = [train_single_model(config, dataset_dir=dataset_dir, seed=seed) for seed in config.seeds]

    if config.test_after_train:
        evaluate_ensemble(config, checkpoint_paths=checkpoint_paths)

    if config.submission_after_train:
        generate_submission(config, checkpoint_paths=checkpoint_paths)


def main() -> None:
    config = build_config(parse_args())
    if config.dry_run:
        print("dry_run=true")
        print_config(config)
        return

    dataset_dir, _ = prepare_dataset(config)
    if config.prepare_only:
        print("prepare_only=true")
        return

    train(config, dataset_dir=dataset_dir)


if __name__ == "__main__":
    main()
