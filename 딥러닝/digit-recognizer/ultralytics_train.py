from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


DEFAULT_EPOCHS = 200
DEFAULT_BATCH = 16
DEFAULT_IMGSZ = 1024
DEFAULT_PATIENCE = 100
DEFAULT_WORKERS = 8


def default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent / "딥러닝" / "이미지 분석 AI"


def default_output_dir() -> Path:
    return default_project_root() / "yolo"


def default_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return ",".join(str(index) for index in range(torch.cuda.device_count()))


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def resolve_model(model_name_or_path: str) -> str:
    model_path = Path(model_name_or_path).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    return model_name_or_path


@dataclass(frozen=True)
class TrainConfig:
    task: str
    model: str
    data: Path
    epochs: int
    batch: int
    imgsz: int
    patience: int
    device: str
    workers: int
    project: Path
    name: str
    optimizer: str
    seed: int
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    close_mosaic: int
    save_period: int
    fraction: float
    pretrained: bool
    amp: bool
    cache: bool
    cos_lr: bool
    rect: bool
    resume: bool
    deterministic: bool
    single_cls: bool
    plots: bool
    val: bool
    verbose: bool
    exist_ok: bool

    def to_train_kwargs(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "data": str(self.data),
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "patience": self.patience,
            "device": self.device,
            "workers": self.workers,
            "project": str(self.project),
            "name": self.name,
            "optimizer": self.optimizer,
            "seed": self.seed,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "close_mosaic": self.close_mosaic,
            "save_period": self.save_period,
            "fraction": self.fraction,
            "pretrained": self.pretrained,
            "amp": self.amp,
            "cache": self.cache,
            "cos_lr": self.cos_lr,
            "rect": self.rect,
            "resume": self.resume,
            "deterministic": self.deterministic,
            "single_cls": self.single_cls,
            "plots": self.plots,
            "val": self.val,
            "verbose": self.verbose,
            "exist_ok": self.exist_ok,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Ultralytics model from a local data.yaml file.")
    parser.add_argument("--data", required=True, help="Path to the dataset YAML file.")
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Ultralytics model name or local checkpoint path. Examples: yolo11s.pt, yolo26s.pt, best.pt",
    )
    parser.add_argument(
        "--task",
        choices=["detect", "segment", "classify", "pose", "obb"],
        default="detect",
        help="Ultralytics task type.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Square input image size.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience.")
    parser.add_argument(
        "--device",
        default=default_device(),
        help='Training device. Examples: "cpu", "0", "0,1". Defaults to all visible CUDA devices or CPU.',
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Data loader worker count.")
    parser.add_argument(
        "--project",
        default=str(default_output_dir()),
        help="Output directory root used by Ultralytics.",
    )
    parser.add_argument("--name", default="train", help="Run name under the project directory.")
    parser.add_argument("--optimizer", default="auto", help="Optimizer name or auto.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning-rate ratio.")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum or Adam beta1.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Optimizer weight decay.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Epochs before disabling mosaic augmentation.")
    parser.add_argument("--save-period", type=int, default=-1, help="Checkpoint save interval. -1 disables periodic saves.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the dataset to use for training.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved training config and exit.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False, help="Resume a previous run.")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained model weights when available.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable automatic mixed precision.")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False, help="Cache dataset images.")
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=False, help="Use cosine LR scheduling.")
    parser.add_argument("--rect", action=argparse.BooleanOptionalAction, default=False, help="Use rectangular training batches.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request deterministic behavior where possible.",
    )
    parser.add_argument("--single-cls", action=argparse.BooleanOptionalAction, default=False, help="Train as a single-class dataset.")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Save training plots.")
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True, help="Run validation during training.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Enable verbose Ultralytics logs.")
    parser.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow reusing an existing project/name directory.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    data_path = resolve_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")
    if data_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Expected a YAML file for --data, got: {data_path}")

    project_path = resolve_path(args.project)

    return TrainConfig(
        task=args.task,
        model=resolve_model(args.model),
        data=data_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=project_path,
        name=args.name,
        optimizer=args.optimizer,
        seed=args.seed,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        close_mosaic=args.close_mosaic,
        save_period=args.save_period,
        fraction=args.fraction,
        pretrained=args.pretrained,
        amp=args.amp,
        cache=args.cache,
        cos_lr=args.cos_lr,
        rect=args.rect,
        resume=args.resume,
        deterministic=args.deterministic,
        single_cls=args.single_cls,
        plots=args.plots,
        val=args.val,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
    )


def print_config(config: TrainConfig) -> None:
    payload = asdict(config)
    payload["data"] = str(config.data)
    payload["project"] = str(config.project)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def train(config: TrainConfig) -> None:
    print("training_config=")
    print_config(config)

    model = YOLO(config.model, task=config.task)
    results = model.train(**config.to_train_kwargs())

    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is not None:
        print(f"save_dir={save_dir}")

    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict) and results_dict:
        print("metrics=")
        print(json.dumps(results_dict, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    config = build_config(args)
    if args.dry_run:
        print("dry_run=true")
        print_config(config)
        return
    train(config)


if __name__ == "__main__":
    main()
