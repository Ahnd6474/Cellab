from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


IMAGE_SIZE = 28
NUM_PIXELS = IMAGE_SIZE * IMAGE_SIZE
PIXEL_COLUMNS = [f"pixel{i}" for i in range(NUM_PIXELS)]


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    data_dir = default_data_dir()
    parser = argparse.ArgumentParser(
        description="Convert Kaggle Digit Recognizer CSV files into a YOLO detect dataset."
    )
    parser.add_argument("--train-csv", type=Path, default=data_dir / "train.csv", help="Path to train.csv.")
    parser.add_argument("--test-csv", type=Path, default=data_dir / "test.csv", help="Path to test.csv.")
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir / "yolo_detect",
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of labeled training data to place into the validation split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for the train/val split.")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip images and labels that already exist.",
    )
    return parser.parse_args()


def ensure_digit_columns(frame: pd.DataFrame) -> None:
    missing_columns = [column for column in PIXEL_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing {len(missing_columns)} pixel columns.")


def save_png(image_array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array, mode="L").save(output_path, format="PNG", compress_level=0)


def digit_bbox(image_array: np.ndarray) -> tuple[float, float, float, float]:
    foreground = image_array > 0
    if not foreground.any():
        return 0.5, 0.5, 1.0, 1.0

    ys, xs = np.where(foreground)
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    width = (x_max - x_min + 1) / IMAGE_SIZE
    height = (y_max - y_min + 1) / IMAGE_SIZE
    x_center = (x_min + x_max + 1) / 2 / IMAGE_SIZE
    y_center = (y_min + y_max + 1) / 2 / IMAGE_SIZE
    return x_center, y_center, width, height


def write_label(output_path: Path, label: int, bbox: tuple[float, float, float, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_center, y_center, width, height = bbox
    output_path.write_text(
        f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
        encoding="utf-8",
    )


def build_split_lookup(train_csv: Path, val_size: float, seed: int) -> np.ndarray:
    label_frame = pd.read_csv(train_csv, usecols=["label"])
    labels = label_frame["label"].to_numpy(dtype=np.int64)
    indices = np.arange(labels.shape[0])

    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )

    split_lookup = np.empty(labels.shape[0], dtype=object)
    split_lookup[train_indices] = "train"
    split_lookup[val_indices] = "val"
    return split_lookup


def write_data_yaml(dataset_root: Path) -> Path:
    yaml_path = dataset_root / "data.yaml"
    names = "\n".join(f"  {label}: {label}" for label in range(10))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def convert_train_split(
    train_csv: Path,
    dataset_root: Path,
    split_lookup: np.ndarray,
    skip_existing: bool,
) -> tuple[int, int]:
    image_root = dataset_root / "images"
    label_root = dataset_root / "labels"
    written = 0
    skipped = 0
    start_index = 0

    for chunk in pd.read_csv(train_csv, chunksize=1000):
        ensure_digit_columns(chunk)
        labels = chunk["label"].to_numpy(dtype=np.int64)
        images = chunk[PIXEL_COLUMNS].to_numpy(dtype=np.uint8).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

        for offset, (label, image_array) in enumerate(zip(labels, images)):
            global_index = start_index + offset
            split_name = split_lookup[global_index]
            image_path = image_root / split_name / f"train_{global_index:05d}.png"
            label_path = label_root / split_name / f"train_{global_index:05d}.txt"

            if skip_existing and image_path.exists() and label_path.exists():
                skipped += 1
                continue

            save_png(image_array, image_path)
            write_label(label_path, int(label), digit_bbox(image_array))
            written += 1

        start_index += len(chunk)
        print(f"train_rows={start_index} train_written={written} train_skipped={skipped}", flush=True)

    return written, skipped


def convert_test_split(test_csv: Path, dataset_root: Path, skip_existing: bool) -> tuple[int, int]:
    image_root = dataset_root / "images" / "test"
    written = 0
    skipped = 0
    start_index = 0

    for chunk in pd.read_csv(test_csv, chunksize=1000):
        ensure_digit_columns(chunk)
        images = chunk[PIXEL_COLUMNS].to_numpy(dtype=np.uint8).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

        for offset, image_array in enumerate(images):
            global_index = start_index + offset
            image_path = image_root / f"test_{global_index:05d}.png"

            if skip_existing and image_path.exists():
                skipped += 1
                continue

            save_png(image_array, image_path)
            written += 1

        start_index += len(chunk)
        print(f"test_rows={start_index} test_written={written} test_skipped={skipped}", flush=True)

    return written, skipped


def main() -> None:
    args = parse_args()
    dataset_root = args.output.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    split_lookup = build_split_lookup(args.train_csv, args.val_size, args.seed)
    train_written, train_skipped = convert_train_split(
        train_csv=args.train_csv,
        dataset_root=dataset_root,
        split_lookup=split_lookup,
        skip_existing=args.skip_existing,
    )
    test_written, test_skipped = convert_test_split(
        test_csv=args.test_csv,
        dataset_root=dataset_root,
        skip_existing=args.skip_existing,
    )
    yaml_path = write_data_yaml(dataset_root)

    print(f"dataset_root={dataset_root}")
    print(f"data_yaml={yaml_path}")
    print(f"train_written={train_written} train_skipped={train_skipped}")
    print(f"test_written={test_written} test_skipped={test_skipped}")


if __name__ == "__main__":
    main()
