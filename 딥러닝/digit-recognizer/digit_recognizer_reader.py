from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH
LABEL_COLUMN = "label"
PIXEL_COLUMNS = [f"pixel{i}" for i in range(NUM_PIXELS)]


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent


def read_digit_csv(csv_path: Path, limit: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, nrows=limit)
    missing_columns = [column for column in PIXEL_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{csv_path} is missing {len(missing_columns)} pixel columns.")
    return frame


def dataframe_to_images(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    images = frame[PIXEL_COLUMNS].to_numpy(dtype=np.uint8).reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH)
    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int64) if LABEL_COLUMN in frame.columns else None
    return images, labels


def row_to_image(row: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(row, pd.Series):
        if set(PIXEL_COLUMNS).issubset(row.index):
            flat_pixels = row.loc[PIXEL_COLUMNS].to_numpy(dtype=np.uint8)
        else:
            flat_pixels = row.to_numpy(dtype=np.uint8)
    else:
        flat_pixels = np.asarray(row, dtype=np.uint8).reshape(-1)

    if flat_pixels.size != NUM_PIXELS:
        raise ValueError(f"Expected {NUM_PIXELS} pixel values, got {flat_pixels.size}.")
    return flat_pixels.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)


def load_digit_images(csv_path: Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    return dataframe_to_images(read_digit_csv(csv_path, limit=limit))


def load_digit_sample(csv_path: Path, index: int) -> tuple[np.ndarray, int | None]:
    frame = read_digit_csv(csv_path)
    if index < 0 or index >= len(frame):
        raise IndexError(f"Sample index {index} is out of range for {csv_path} with {len(frame)} rows.")

    row = frame.iloc[index]
    label = int(row[LABEL_COLUMN]) if LABEL_COLUMN in row.index else None
    pixel_row = row.drop(labels=[LABEL_COLUMN]) if LABEL_COLUMN in row.index else row
    return row_to_image(pixel_row), label


def render_digit(
    image: np.ndarray,
    label: int | None = None,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(image, cmap="gray_r", vmin=0, vmax=255)
    axis.set_title(f"label={label}" if label is not None else "digit sample")
    axis.axis("off")
    figure.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved={save_path}")

    if show:
        plt.show()

    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Kaggle Digit Recognizer CSV rows as 28x28 grayscale images."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_data_dir() / "train.csv",
        help="Path to train.csv or test.csv.",
    )
    parser.add_argument("--index", type=int, default=0, help="Row index to render.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to load. Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional image path. If provided, the rendered digit is saved there.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip opening a matplotlib window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images, labels = load_digit_images(args.csv, limit=args.limit)
    if args.index < 0 or args.index >= len(images):
        raise IndexError(f"Sample index {args.index} is out of range for {args.csv} with {len(images)} rows.")

    print(f"csv={args.csv}")
    print(f"images_shape={images.shape}")
    print(f"dtype={images.dtype} min={int(images.min())} max={int(images.max())}")

    if labels is None:
        print("labels=None")
        label = None
    else:
        unique_labels = np.unique(labels)
        print(f"labels_shape={labels.shape} unique_labels={unique_labels.tolist()}")
        label = int(labels[args.index])

    render_digit(
        images[args.index],
        label=label,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
