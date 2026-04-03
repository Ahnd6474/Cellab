from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LABEL_MAGIC = 2049
IMAGE_MAGIC = 2051


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent


@dataclass(frozen=True)
class MnistSplitPaths:
    training_images_filepath: Path
    training_labels_filepath: Path
    test_images_filepath: Path
    test_labels_filepath: Path


def default_split_paths(data_dir: Path | None = None) -> MnistSplitPaths:
    root = data_dir if data_dir is not None else default_data_dir()
    return MnistSplitPaths(
        training_images_filepath=root / "train-images.idx3-ubyte",
        training_labels_filepath=root / "train-labels.idx1-ubyte",
        test_images_filepath=root / "test-images.idx3-ubyte",
        test_labels_filepath=root / "test-labels.idx1-ubyte",
    )


def read_idx_labels(labels_path: Path, limit: int | None = None) -> np.ndarray:
    with labels_path.open("rb") as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != LABEL_MAGIC:
            raise ValueError(f"Magic number mismatch for labels. Expected {LABEL_MAGIC}, got {magic}.")
        labels = np.frombuffer(file.read(), dtype=np.uint8)

    if labels.size != size:
        raise ValueError(f"Label count mismatch in {labels_path}: header={size}, actual={labels.size}")

    if limit is not None:
        labels = labels[:limit]
    return labels.copy()


def read_idx_images(images_path: Path, limit: int | None = None) -> np.ndarray:
    with images_path.open("rb") as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != IMAGE_MAGIC:
            raise ValueError(f"Magic number mismatch for images. Expected {IMAGE_MAGIC}, got {magic}.")
        image_data = np.frombuffer(file.read(), dtype=np.uint8)

    expected_size = size * rows * cols
    if image_data.size != expected_size:
        raise ValueError(f"Image count mismatch in {images_path}: header={expected_size}, actual={image_data.size}")

    images = image_data.reshape(size, rows, cols)
    if limit is not None:
        images = images[:limit]
    return images.copy()


class MnistDataloader:
    def __init__(
        self,
        training_images_filepath: Path,
        training_labels_filepath: Path,
        test_images_filepath: Path,
        test_labels_filepath: Path,
    ) -> None:
        self.training_images_filepath = Path(training_images_filepath)
        self.training_labels_filepath = Path(training_labels_filepath)
        self.test_images_filepath = Path(test_images_filepath)
        self.test_labels_filepath = Path(test_labels_filepath)

    def read_images_labels(
        self,
        images_filepath: Path,
        labels_filepath: Path,
        limit: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        images = read_idx_images(Path(images_filepath), limit=limit)
        labels = read_idx_labels(Path(labels_filepath), limit=limit)

        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Image/label row mismatch: images={images.shape[0]} labels={labels.shape[0]} "
                f"for {images_filepath} and {labels_filepath}"
            )
        return images, labels

    def load_data(
        self,
        train_limit: int | None = None,
        test_limit: int | None = None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath,
            self.training_labels_filepath,
            limit=train_limit,
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath,
            self.test_labels_filepath,
            limit=test_limit,
        )
        return (x_train, y_train), (x_test, y_test)


def load_default_mnist(
    data_dir: Path | None = None,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    paths = default_split_paths(data_dir=data_dir)
    loader = MnistDataloader(
        training_images_filepath=paths.training_images_filepath,
        training_labels_filepath=paths.training_labels_filepath,
        test_images_filepath=paths.test_images_filepath,
        test_labels_filepath=paths.test_labels_filepath,
    )
    return loader.load_data(train_limit=train_limit, test_limit=test_limit)
