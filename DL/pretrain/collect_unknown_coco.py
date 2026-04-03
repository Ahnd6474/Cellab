import argparse
import json
import random
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/train2017"
TARGET_CLASS = "unknown"
EXCLUDED_CATEGORIES = {"cat", "dog"}
DEFAULT_COUNT = 10_000
DEFAULT_SEED = 42
DEFAULT_WORKERS = 16


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as f:
        f.write(response.read())


def ensure_annotations(base_dir: Path) -> Path:
    annotations_zip = base_dir / "annotations_trainval2017.zip"
    annotations_json = base_dir / "annotations" / "instances_train2017.json"

    if annotations_json.exists():
        return annotations_json

    print(f"Downloading annotations: {ANNOTATIONS_URL}")
    download_file(ANNOTATIONS_URL, annotations_zip)

    print(f"Extracting annotations to: {base_dir / 'annotations'}")
    with zipfile.ZipFile(annotations_zip) as zf:
        zf.extract("annotations/instances_train2017.json", path=base_dir)

    return annotations_json


def load_eligible_images(annotations_json: Path) -> list[dict]:
    data = json.loads(annotations_json.read_text())
    categories = {item["id"]: item["name"].strip().lower() for item in data["categories"]}
    excluded_category_ids = {cat_id for cat_id, name in categories.items() if name in EXCLUDED_CATEGORIES}

    blocked_image_ids = {
        annotation["image_id"]
        for annotation in data["annotations"]
        if annotation["category_id"] in excluded_category_ids
    }

    eligible_images = [
        image
        for image in data["images"]
        if image["id"] not in blocked_image_ids
    ]
    return eligible_images


def download_image(image: dict, target_dir: Path) -> tuple[str, bool, str]:
    filename = image["file_name"]
    destination = target_dir / filename
    if destination.exists():
        return filename, False, "exists"

    url = f"{IMAGE_BASE_URL}/{filename}"
    try:
        download_file(url, destination)
        return filename, True, "downloaded"
    except urllib.error.URLError as e:
        return filename, False, f"url_error: {e}"
    except Exception as e:
        return filename, False, f"error: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download random non-cat/dog COCO images for the unknown class")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of images to download")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel download workers")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Pretrain project directory",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    unknown_dir = base_dir / "dataset" / "train" / TARGET_CLASS
    unknown_dir.mkdir(parents=True, exist_ok=True)

    annotations_json = ensure_annotations(base_dir)
    eligible_images = load_eligible_images(annotations_json)

    if len(eligible_images) < args.count:
        raise ValueError(f"Eligible images ({len(eligible_images)}) are fewer than requested count ({args.count})")

    rng = random.Random(args.seed)
    selected_images = rng.sample(eligible_images, args.count)

    print(f"Eligible COCO train2017 images without cat/dog annotations: {len(eligible_images)}")
    print(f"Selected random unknown images: {len(selected_images)}")
    print(f"Target directory: {unknown_dir}")

    downloaded = 0
    skipped = 0
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_image, image, unknown_dir) for image in selected_images]
        for index, future in enumerate(as_completed(futures), start=1):
            filename, changed, status = future.result()
            if changed:
                downloaded += 1
            elif status == "exists":
                skipped += 1
            else:
                failures.append((filename, status))

            if index % 100 == 0 or index == len(futures):
                print(
                    f"Progress {index}/{len(futures)} | downloaded={downloaded} skipped={skipped} failures={len(failures)}"
                )

    manifest = {
        "source": "COCO train2017",
        "annotations": str(annotations_json),
        "excluded_categories": sorted(EXCLUDED_CATEGORIES),
        "requested_count": args.count,
        "seed": args.seed,
        "workers": args.workers,
        "downloaded": downloaded,
        "skipped": skipped,
        "failures": failures,
    }
    manifest_path = unknown_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Manifest written to: {manifest_path}")

    if failures:
        print("Some downloads failed. Re-run the same command to retry missing files.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
