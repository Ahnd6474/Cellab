from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from transformers import AutoImageProcessor, AutoModelForImageClassification

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "facebook/deit-small-patch16-224"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mnist_vit_cls"
DEFAULT_EPOCHS = 12
DEFAULT_BATCH = 64
DEFAULT_BACKBONE_LR = 1e-5
DEFAULT_HEAD_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 4
DEFAULT_VAL_SIZE = 0.1
DEFAULT_SEED = 42
TRAIN_DTYPE = torch.float32
NUM_CLASSES = 10
IMAGE_SIZE = 28
PIXEL_COLUMNS = [f"pixel{i}" for i in range(IMAGE_SIZE * IMAGE_SIZE)]


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TrainConfig:
    train_csv: Path
    test_csv: Path
    sample_submission: Path
    output_dir: Path
    model_name: str
    epochs: int
    batch: int
    backbone_lr: float
    head_lr: float
    weight_decay: float
    patience: int
    val_size: float
    seed: int
    train_limit: int | None
    test_limit: int | None
    workers: int
    device: str
    submission_filename: str
    dry_run: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained ViT model for Kaggle Digit Recognizer.")
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "train.csv")
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "test.csv")
    parser.add_argument("--sample-submission", type=Path, default=PROJECT_ROOT / "sample_submission.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--backbone-lr", type=float, default=DEFAULT_BACKBONE_LR)
    parser.add_argument("--head-lr", type=float, default=DEFAULT_HEAD_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--submission-filename", default="submission_vit.csv")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    if not 0.0 < args.val_size < 1.0:
        raise ValueError(f"--val-size must be between 0 and 1, got {args.val_size}")
    if args.epochs <= 0 or args.batch <= 0:
        raise ValueError("--epochs and --batch must be positive")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative")

    config = TrainConfig(
        train_csv=args.train_csv.resolve(),
        test_csv=args.test_csv.resolve(),
        sample_submission=args.sample_submission.resolve(),
        output_dir=args.output_dir.resolve(),
        model_name=args.model_name,
        epochs=args.epochs,
        batch=args.batch,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        val_size=args.val_size,
        seed=args.seed,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        workers=args.workers,
        device=args.device,
        submission_filename=args.submission_filename,
        dry_run=args.dry_run,
    )

    for path in [config.train_csv, config.test_csv, config.sample_submission]:
        if not path.exists():
            raise FileNotFoundError(path)
    return config


def print_config(config: TrainConfig) -> None:
    payload: dict[str, Any] = asdict(config)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    payload["train_dtype"] = str(TRAIN_DTYPE)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


class DigitCsvDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, processor, has_labels: bool):
        self.frame = frame.reset_index(drop=True)
        self.processor = processor
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        pixels = row[PIXEL_COLUMNS].to_numpy(dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE)
        image = Image.fromarray(pixels, mode="L").convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        pixel_values = pixel_values.to(dtype=TRAIN_DTYPE)
        if self.has_labels:
            label = int(row["label"])
            return pixel_values, label
        return pixel_values


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device, dtype=logits.dtype)
            loss = alpha[targets] * loss
        return loss.mean()


def build_alpha_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    weights = counts.sum() / (NUM_CLASSES * counts)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=TRAIN_DTYPE)


def evaluate(model, dataloader, criterion, device: torch.device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for pixel_values, labels in dataloader:
            pixel_values = pixel_values.to(device=device, dtype=TRAIN_DTYPE)
            labels = labels.to(device)
            logits = model(pixel_values=pixel_values).logits
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels).item()
            total += batch_size

    return running_loss / total, running_corrects / total


def predict(model, dataloader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for pixel_values in dataloader:
            pixel_values = pixel_values.to(device=device, dtype=TRAIN_DTYPE)
            logits = model(pixel_values=pixel_values).logits
            outputs.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    config = build_config(parse_args())
    print_config(config)
    if config.dry_run:
        return

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)

    train_frame = pd.read_csv(config.train_csv, nrows=config.train_limit)
    test_frame = pd.read_csv(config.test_csv, nrows=config.test_limit)

    train_split, val_split = train_test_split(
        train_frame,
        test_size=config.val_size,
        random_state=config.seed,
        stratify=train_frame["label"],
    )

    processor = AutoImageProcessor.from_pretrained(config.model_name, use_fast=True)

    train_dataset = DigitCsvDataset(train_split, processor, has_labels=True)
    val_dataset = DigitCsvDataset(val_split, processor, has_labels=True)
    test_dataset = DigitCsvDataset(test_frame, processor, has_labels=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True, num_workers=config.workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch, shuffle=False, num_workers=config.workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch, shuffle=False, num_workers=config.workers)

    alpha = build_alpha_weights(train_split["label"].to_numpy())
    print(f"focal_alpha={[round(x, 4) for x in alpha.tolist()]}")

    id2label = {idx: str(idx) for idx in range(NUM_CLASSES)}
    label2id = {label: idx for idx, label in id2label.items()}
    model = AutoModelForImageClassification.from_pretrained(
        config.model_name,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device=device, dtype=TRAIN_DTYPE)

    classifier = getattr(model, "classifier")
    head_params = list(classifier.parameters())
    head_param_ids = {id(param) for param in head_params}
    backbone_params = [param for param in model.parameters() if id(param) not in head_param_ids]

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": config.backbone_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    criterion = FocalLoss(gamma=2.0, alpha=alpha)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.to(device=device, dtype=TRAIN_DTYPE)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(pixel_values=pixel_values).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels).item()
            total += batch_size

        train_loss = running_loss / total
        train_acc = running_corrects / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch={epoch + 1}/{config.epochs} train_loss={train_loss:.5f} train_acc={train_acc:.5f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if config.patience and patience_counter >= config.patience:
                print(f"early_stop epoch={epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device=device, dtype=TRAIN_DTYPE)

    output_dir = config.output_dir
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)

    predictions = predict(model, test_loader, device)
    submission = pd.read_csv(config.sample_submission)
    submission["Label"] = predictions
    submission_path = output_dir / config.submission_filename
    submission.to_csv(submission_path, index=False)

    metrics = {
        "model_name": config.model_name,
        "train_dtype": str(TRAIN_DTYPE),
        "best_val_loss": best_val_loss,
        "submission_path": str(submission_path),
        "model_dir": str(model_dir),
        "num_train": len(train_split),
        "num_val": len(val_split),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(f"saved_model={model_dir}")
    print(f"saved_submission={submission_path}")


if __name__ == "__main__":
    main()
