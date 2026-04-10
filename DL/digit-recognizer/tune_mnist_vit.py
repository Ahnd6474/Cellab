from __future__ import annotations

import argparse
import gc
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

from train_mnist_vit import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_VAL_SIZE,
    DigitCsvDataset,
    FocalLoss,
    NUM_CLASSES,
    PROJECT_ROOT,
    TRAIN_DTYPE,
    build_alpha_weights,
    default_device,
    evaluate,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mnist_vit_tuning"
DEFAULT_EPOCH_OPTIONS = (4, 6)
DEFAULT_BATCH_OPTIONS = (32, 64)
DEFAULT_BACKBONE_LR_OPTIONS = (1e-5, 2e-5)
DEFAULT_HEAD_LR_OPTIONS = (5e-5, 1e-4)
DEFAULT_WEIGHT_DECAY_OPTIONS = (1e-4, 5e-4)


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    model_name: str
    epochs: int
    batch: int
    backbone_lr: float
    head_lr: float
    weight_decay: float


@dataclass(frozen=True)
class TuneConfig:
    train_csv: Path
    output_dir: Path
    model_names: tuple[str, ...]
    epoch_options: tuple[int, ...]
    batch_options: tuple[int, ...]
    backbone_lr_options: tuple[float, ...]
    head_lr_options: tuple[float, ...]
    weight_decay_options: tuple[float, ...]
    search: str
    max_trials: int | None
    metric: str
    patience: int
    val_size: float
    seed: int
    train_limit: int | None
    workers: int
    device: str
    local_files_only: bool
    fail_fast: bool
    dry_run: bool


@dataclass(frozen=True)
class TrialResult:
    trial_id: int
    status: str
    model_name: str
    epochs: int
    batch: int
    backbone_lr: float
    head_lr: float
    weight_decay: float
    best_epoch: int | None
    best_val_loss: float | None
    best_val_acc: float | None
    elapsed_sec: float
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for a ViT model on MNIST CSV data.")
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "train.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-names", nargs="+", default=[DEFAULT_MODEL_NAME])
    parser.add_argument("--epoch-options", nargs="+", type=int, default=list(DEFAULT_EPOCH_OPTIONS))
    parser.add_argument("--batch-options", nargs="+", type=int, default=list(DEFAULT_BATCH_OPTIONS))
    parser.add_argument("--backbone-lr-options", nargs="+", type=float, default=list(DEFAULT_BACKBONE_LR_OPTIONS))
    parser.add_argument("--head-lr-options", nargs="+", type=float, default=list(DEFAULT_HEAD_LR_OPTIONS))
    parser.add_argument("--weight-decay-options", nargs="+", type=float, default=list(DEFAULT_WEIGHT_DECAY_OPTIONS))
    parser.add_argument("--search", choices=["grid", "random"], default="grid")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--metric", choices=["val_acc", "val_loss"], default="val_acc")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TuneConfig:
    if not args.model_names:
        raise ValueError("--model-names must contain at least one model")
    for name, values in {
        "--epoch-options": args.epoch_options,
        "--batch-options": args.batch_options,
        "--backbone-lr-options": args.backbone_lr_options,
        "--head-lr-options": args.head_lr_options,
        "--weight-decay-options": args.weight_decay_options,
    }.items():
        if not values:
            raise ValueError(f"{name} must contain at least one value")

    if not 0.0 < args.val_size < 1.0:
        raise ValueError(f"--val-size must be between 0 and 1, got {args.val_size}")
    if any(value <= 0 for value in args.epoch_options):
        raise ValueError("--epoch-options must contain positive integers")
    if any(value <= 0 for value in args.batch_options):
        raise ValueError("--batch-options must contain positive integers")
    if any(value <= 0 for value in args.backbone_lr_options + args.head_lr_options + args.weight_decay_options):
        raise ValueError("learning rates and weight decay must be positive")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative")
    if args.max_trials is not None and args.max_trials <= 0:
        raise ValueError("--max-trials must be positive")
    if not args.train_csv.exists():
        raise FileNotFoundError(args.train_csv)

    return TuneConfig(
        train_csv=args.train_csv.resolve(),
        output_dir=args.output_dir.resolve(),
        model_names=tuple(args.model_names),
        epoch_options=tuple(args.epoch_options),
        batch_options=tuple(args.batch_options),
        backbone_lr_options=tuple(args.backbone_lr_options),
        head_lr_options=tuple(args.head_lr_options),
        weight_decay_options=tuple(args.weight_decay_options),
        search=args.search,
        max_trials=args.max_trials,
        metric=args.metric,
        patience=args.patience,
        val_size=args.val_size,
        seed=args.seed,
        train_limit=args.train_limit,
        workers=args.workers,
        device=args.device,
        local_files_only=args.local_files_only,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
    )


def total_search_space(config: TuneConfig) -> int:
    return (
        len(config.model_names)
        * len(config.epoch_options)
        * len(config.batch_options)
        * len(config.backbone_lr_options)
        * len(config.head_lr_options)
        * len(config.weight_decay_options)
    )


def build_trials(config: TuneConfig) -> list[TrialSpec]:
    combinations = list(
        itertools.product(
            config.model_names,
            config.epoch_options,
            config.batch_options,
            config.backbone_lr_options,
            config.head_lr_options,
            config.weight_decay_options,
        )
    )

    if config.search == "random":
        rng = random.Random(config.seed)
        rng.shuffle(combinations)

    if config.max_trials is not None:
        combinations = combinations[: config.max_trials]

    trials = []
    for trial_id, values in enumerate(combinations, start=1):
        model_name, epochs, batch, backbone_lr, head_lr, weight_decay = values
        trials.append(
            TrialSpec(
                trial_id=trial_id,
                model_name=model_name,
                epochs=epochs,
                batch=batch,
                backbone_lr=backbone_lr,
                head_lr=head_lr,
                weight_decay=weight_decay,
            )
        )
    return trials


def print_config(config: TuneConfig, trials: list[TrialSpec]) -> None:
    payload: dict[str, Any] = asdict(config)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    payload["train_dtype"] = str(TRAIN_DTYPE)
    payload["search_space_size"] = total_search_space(config)
    payload["scheduled_trials"] = len(trials)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    processor,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = DigitCsvDataset(train_frame, processor, has_labels=True)
    val_dataset = DigitCsvDataset(val_frame, processor, has_labels=True)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def build_model(model_name: str, device: torch.device, local_files_only: bool):
    id2label = {idx: str(idx) for idx in range(NUM_CLASSES)}
    label2id = {label: idx for idx, label in id2label.items()}
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )
    return model.to(device=device, dtype=TRAIN_DTYPE)


def train_single_trial(
    trial: TrialSpec,
    config: TuneConfig,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    alpha: torch.Tensor,
    processor_cache: dict[str, Any],
) -> tuple[TrialResult, Any | None, Any | None]:
    start_time = time.perf_counter()
    device = torch.device(config.device)
    model = None
    processor = None

    try:
        seed_everything(config.seed)

        processor = processor_cache.get(trial.model_name)
        if processor is None:
            processor = AutoImageProcessor.from_pretrained(
                trial.model_name,
                use_fast=True,
                local_files_only=config.local_files_only,
            )
            processor_cache[trial.model_name] = processor

        train_loader, val_loader = build_dataloaders(
            train_frame=train_frame,
            val_frame=val_frame,
            processor=processor,
            batch_size=trial.batch,
            workers=config.workers,
            seed=config.seed,
            device=device,
        )

        model = build_model(trial.model_name, device=device, local_files_only=config.local_files_only)
        classifier = getattr(model, "classifier")
        head_params = list(classifier.parameters())
        head_param_ids = {id(param) for param in head_params}
        backbone_params = [param for param in model.parameters() if id(param) not in head_param_ids]

        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": trial.backbone_lr},
                {"params": head_params, "lr": trial.head_lr},
            ],
            weight_decay=trial.weight_decay,
        )
        criterion = FocalLoss(gamma=2.0, alpha=alpha)

        best_state = None
        best_epoch = 0
        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(trial.epochs):
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
                f"trial={trial.trial_id} epoch={epoch + 1}/{trial.epochs} train_loss={train_loss:.5f} "
                f"train_acc={train_acc:.5f} val_loss={val_loss:.5f} val_acc={val_acc:.5f}"
            )

            if val_loss < best_val_loss:
                best_epoch = epoch + 1
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if config.patience and patience_counter >= config.patience:
                    print(f"trial={trial.trial_id} early_stop epoch={epoch + 1}")
                    break

        if best_state is None:
            raise RuntimeError(f"trial {trial.trial_id} did not produce a checkpoint")

        model.load_state_dict(best_state)
        elapsed_sec = time.perf_counter() - start_time
        result = TrialResult(
            trial_id=trial.trial_id,
            status="completed",
            model_name=trial.model_name,
            epochs=trial.epochs,
            batch=trial.batch,
            backbone_lr=trial.backbone_lr,
            head_lr=trial.head_lr,
            weight_decay=trial.weight_decay,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_acc=best_val_acc,
            elapsed_sec=elapsed_sec,
            error=None,
        )
        return result, model, processor
    except Exception as error:
        elapsed_sec = time.perf_counter() - start_time
        result = TrialResult(
            trial_id=trial.trial_id,
            status="failed",
            model_name=trial.model_name,
            epochs=trial.epochs,
            batch=trial.batch,
            backbone_lr=trial.backbone_lr,
            head_lr=trial.head_lr,
            weight_decay=trial.weight_decay,
            best_epoch=None,
            best_val_loss=None,
            best_val_acc=None,
            elapsed_sec=elapsed_sec,
            error=str(error),
        )
        return result, None, processor


def is_better(candidate: TrialResult, current_best: TrialResult | None, metric: str) -> bool:
    if candidate.status != "completed":
        return False
    if current_best is None:
        return True

    candidate_val_acc = candidate.best_val_acc if candidate.best_val_acc is not None else 0.0
    current_val_acc = current_best.best_val_acc if current_best.best_val_acc is not None else 0.0
    candidate_val_loss = candidate.best_val_loss if candidate.best_val_loss is not None else float("inf")
    current_val_loss = current_best.best_val_loss if current_best.best_val_loss is not None else float("inf")

    if metric == "val_loss":
        candidate_key = (candidate_val_loss, -candidate_val_acc)
        current_key = (current_val_loss, -current_val_acc)
        return candidate_key < current_key

    candidate_key = (candidate_val_acc, -candidate_val_loss)
    current_key = (current_val_acc, -current_val_loss)
    return candidate_key > current_key


def save_results(output_dir: Path, config: TuneConfig, results: list[TrialResult], best_result: TrialResult | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame([asdict(result) for result in results])
    results_frame.to_csv(output_dir / "trial_results.csv", index=False)

    manifest: dict[str, Any] = asdict(config)
    for key, value in manifest.items():
        if isinstance(value, Path):
            manifest[key] = str(value)
    manifest["train_dtype"] = str(TRAIN_DTYPE)
    manifest["num_trials"] = len(results)
    manifest["num_completed"] = sum(result.status == "completed" for result in results)
    manifest["best_result"] = asdict(best_result) if best_result is not None else None
    (output_dir / "search_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    if best_result is not None:
        (output_dir / "best_config.json").write_text(json.dumps(asdict(best_result), ensure_ascii=False, indent=2) + "\n")


def cleanup_trial_state(*objects: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    config = build_config(parse_args())
    trials = build_trials(config)
    print_config(config, trials)
    if config.dry_run:
        return
    if not trials:
        raise RuntimeError("No trials were scheduled")

    seed_everything(config.seed)
    train_frame = pd.read_csv(config.train_csv, nrows=config.train_limit)
    train_split, val_split = train_test_split(
        train_frame,
        test_size=config.val_size,
        random_state=config.seed,
        stratify=train_frame["label"],
    )
    alpha = build_alpha_weights(train_split["label"].to_numpy())
    print(f"focal_alpha={[round(x, 4) for x in alpha.tolist()]}")

    processor_cache: dict[str, Any] = {}
    results: list[TrialResult] = []
    best_result: TrialResult | None = None

    for trial in trials:
        print(
            f"trial={trial.trial_id}/{len(trials)} model={trial.model_name} epochs={trial.epochs} batch={trial.batch} "
            f"backbone_lr={trial.backbone_lr} head_lr={trial.head_lr} weight_decay={trial.weight_decay}"
        )
        result, model, processor = train_single_trial(
            trial=trial,
            config=config,
            train_frame=train_split,
            val_frame=val_split,
            alpha=alpha,
            processor_cache=processor_cache,
        )
        results.append(result)
        print(
            f"trial={result.trial_id} status={result.status} best_val_loss={result.best_val_loss} "
            f"best_val_acc={result.best_val_acc} elapsed_sec={result.elapsed_sec:.2f}"
        )

        if is_better(result, best_result, config.metric):
            best_result = result
            best_model_dir = config.output_dir / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            if model is not None and processor is not None:
                model.save_pretrained(best_model_dir)
                processor.save_pretrained(best_model_dir)
            print(f"updated_best_trial={result.trial_id} metric={config.metric}")

        save_results(config.output_dir, config, results, best_result)
        model = None
        cleanup_trial_state(model)

        if result.status == "failed" and config.fail_fast:
            raise RuntimeError(result.error)

    if best_result is None:
        raise RuntimeError("All trials failed. Check trial_results.csv for error messages.")

    print(
        f"best_trial={best_result.trial_id} metric={config.metric} val_loss={best_result.best_val_loss:.5f} "
        f"val_acc={best_result.best_val_acc:.5f}"
    )
    print(f"saved_results={config.output_dir / 'trial_results.csv'}")
    print(f"saved_best_config={config.output_dir / 'best_config.json'}")


if __name__ == "__main__":
    main()
