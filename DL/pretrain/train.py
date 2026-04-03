import json
import os
import random
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        ViTImageProcessor,
        ViTMAEForPreTraining,
    )
except ImportError:
    AutoImageProcessor = None
    AutoModelForImageClassification = None
    ViTImageProcessor = None
    ViTMAEForPreTraining = None

# ==========================================
# 0. 기본 설정
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "train"
AUGMENTED_DATA_DIR = BASE_DIR / "dataset" / "augmented_train"
MODEL_DIR = BASE_DIR / "classifier_model"
CLASSES_PATH = BASE_DIR / "classes.json"
SUPPORTED_CLASSES = ["cat", "dog", "unknown"]
CLASSIFIER_MODEL_NAME = "facebook/deit-tiny-patch16-224"
MAE_MODEL_NAME = "facebook/vit-mae-base"
MAE_AUGMENTATION_TARGETS = {"unknown": 5000}
MAE_BLEND = 0.35
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 32


def require_transformers() -> None:
    if AutoImageProcessor is None or AutoModelForImageClassification is None:
        raise ImportError("학습과 추론에는 transformers 패키지가 필요합니다.")
    if ViTImageProcessor is None or ViTMAEForPreTraining is None:
        raise ImportError("ViT-MAE 증강에는 transformers 패키지가 필요합니다.")


def iter_image_files(root: Path):
    if not root.exists():
        return
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        dirpath = Path(dirpath)
        for filename in filenames:
            path = dirpath / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def collect_samples(root: Path, class_names: list[str]):
    samples = []
    class_counts = {name: 0 for name in class_names}

    for label_idx, class_name in enumerate(class_names):
        class_dir = root / class_name
        for file_path in iter_image_files(class_dir):
            samples.append((file_path, label_idx))
            class_counts[class_name] += 1

    return samples, class_counts


def validate_class_counts(class_counts: dict[str, int]) -> None:
    missing = [name for name, count in class_counts.items() if count == 0]
    if missing:
        raise ValueError(f"이미지가 없는 클래스가 있습니다: {missing}")


def build_alpha_weights(class_names: list[str], class_counts: dict[str, int]) -> torch.Tensor:
    total_count = sum(class_counts.values())
    num_classes = len(class_names)
    weights = []

    for class_name in class_names:
        count = class_counts[class_name]
        weights.append(total_count / (num_classes * count))

    alpha = torch.tensor(weights, dtype=torch.float32)
    return alpha / alpha.sum() * num_classes


def denormalize_tensor(tensor: torch.Tensor, mean, std) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    return tensor * std_tensor + mean_tensor


def unpatchify(patches: torch.Tensor, patch_size: int, image_size: int, channels: int) -> torch.Tensor:
    grid_size = image_size // patch_size
    batch_size = patches.shape[0]
    patches = patches.reshape(batch_size, grid_size, grid_size, patch_size, patch_size, channels)
    patches = patches.permute(0, 5, 1, 3, 2, 4)
    return patches.reshape(batch_size, channels, image_size, image_size)


def reconstruct_with_vitmae(image: Image.Image, processor, model, device) -> Image.Image:
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    image_size = model.config.image_size
    patch_size = model.config.patch_size
    channels = model.config.num_channels
    reconstruction = unpatchify(outputs.logits, patch_size, image_size, channels)

    if getattr(processor, "do_normalize", False):
        reconstruction = denormalize_tensor(reconstruction, processor.image_mean, processor.image_std)
        original = denormalize_tensor(pixel_values, processor.image_mean, processor.image_std)
    else:
        original = pixel_values

    mixed = (1 - MAE_BLEND) * original + MAE_BLEND * reconstruction
    mixed = mixed.clamp(0, 1).cpu()[0]
    mixed = mixed.permute(1, 2, 0).numpy()
    return Image.fromarray((mixed * 255).astype("uint8"))


def prepare_vitmae_augmentations(class_names: list[str], device) -> None:
    if not MAE_AUGMENTATION_TARGETS:
        return

    processor = ViTImageProcessor.from_pretrained(MAE_MODEL_NAME)
    mae_model = ViTMAEForPreTraining.from_pretrained(MAE_MODEL_NAME).to(device)
    mae_model.eval()

    for class_name in class_names:
        target_count = MAE_AUGMENTATION_TARGETS.get(class_name, 0)
        if target_count <= 0:
            continue

        source_dir = DATA_DIR / class_name
        source_files = list(iter_image_files(source_dir))
        if not source_files:
            continue

        output_dir = AUGMENTED_DATA_DIR / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        existing_files = sorted(output_dir.glob("mae_*.jpg"))
        if len(existing_files) >= target_count:
            print(f"ViT-MAE 증강이 이미 준비되어 있습니다: {class_name} ({len(existing_files)}장)")
            continue

        print(f"ViT-MAE 증강 생성 시작: {class_name} -> {target_count}장")
        for index in range(len(existing_files), target_count):
            source_path = random.choice(source_files)
            with Image.open(source_path) as img:
                augmented = reconstruct_with_vitmae(img.convert("RGB"), processor, mae_model, device)
            output_path = output_dir / f"mae_{index:05d}.jpg"
            augmented.save(output_path, "JPEG", quality=95)

            if (index + 1) % 100 == 0 or index + 1 == target_count:
                print(f"  {class_name}: {index + 1}/{target_count}")


class ClassificationImageDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, label


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            loss = alpha[targets] * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def main():
    require_transformers()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 기기: {device}")
    print(f"원본 데이터 경로: {DATA_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"학습 데이터 경로가 없습니다: {DATA_DIR}")

    prepare_vitmae_augmentations(SUPPORTED_CLASSES, device)

    classifier_processor = AutoImageProcessor.from_pretrained(CLASSIFIER_MODEL_NAME)
    original_samples, original_counts = collect_samples(DATA_DIR, SUPPORTED_CLASSES)
    augmented_samples, augmented_counts = collect_samples(AUGMENTED_DATA_DIR, SUPPORTED_CLASSES)

    combined_samples = original_samples + augmented_samples
    class_counts = {
        class_name: original_counts[class_name] + augmented_counts[class_name]
        for class_name in SUPPORTED_CLASSES
    }
    validate_class_counts(class_counts)

    dataset = ClassificationImageDataset(combined_samples, classifier_processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    alpha_weights = build_alpha_weights(SUPPORTED_CLASSES, class_counts)

    print(f"원본 클래스별 이미지 수: {original_counts}")
    print(f"증강 클래스별 이미지 수: {augmented_counts}")
    print(f"학습 클래스별 총 이미지 수: {class_counts}")
    print(f"학습 데이터 총 개수: {len(combined_samples)}")
    print(f"Focal Loss alpha 가중치: {dict(zip(SUPPORTED_CLASSES, [round(x, 4) for x in alpha_weights.tolist()]))}")

    id2label = {index: name for index, name in enumerate(SUPPORTED_CLASSES)}
    label2id = {name: index for index, name in id2label.items()}
    model = AutoModelForImageClassification.from_pretrained(
        CLASSIFIER_MODEL_NAME,
        num_labels=len(SUPPORTED_CLASSES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    for param in model.parameters():
        param.requires_grad = True

    head_module = getattr(model, "classifier")
    head_params = list(head_module.parameters())
    head_param_ids = {id(param) for param in head_params}
    backbone_params = [param for param in model.parameters() if id(param) not in head_param_ids]

    criterion = FocalLoss(gamma=2.0, alpha=alpha_weights)
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for pixel_values, labels in dataloader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            _, preds = torch.max(logits, 1)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * pixel_values.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(combined_samples)
        epoch_acc = running_corrects.double() / len(combined_samples)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\\n")

    print("학습 완료!")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    classifier_processor.save_pretrained(MODEL_DIR)
    CLASSES_PATH.write_text(json.dumps(SUPPORTED_CLASSES, ensure_ascii=False, indent=2) + "\\n")
    print(f"분류 모델이 저장되었습니다: {MODEL_DIR}")
    print(f"클래스 정보가 저장되었습니다: {CLASSES_PATH}")


if __name__ == "__main__":
    main()
