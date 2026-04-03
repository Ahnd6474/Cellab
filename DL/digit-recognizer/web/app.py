from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
RUNS_DIR = PROJECT_ROOT / "mnist_yolo_cls" / "runs"
DEFAULT_RUN_PREFIX = "mnist-yolo26n-ensemble-seed"
DEFAULT_IMGSZ = 32


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(APP_DIR / "templates"))

    model_paths = discover_model_paths()
    models = [YOLO(str(model_path), task="classify") for model_path in model_paths]

    app.config["ENSEMBLE_MODELS"] = models
    app.config["ENSEMBLE_MODEL_PATHS"] = [str(path) for path in model_paths]

    @app.route("/")
    def index() -> str:
        return render_template("index.html", model_paths=app.config["ENSEMBLE_MODEL_PATHS"])

    @app.route("/health", methods=["GET"])
    def health() -> tuple[dict[str, object], int]:
        return {
            "status": "ok",
            "num_models": len(app.config["ENSEMBLE_MODELS"]),
            "model_paths": app.config["ENSEMBLE_MODEL_PATHS"],
        }, 200

    @app.route("/predict", methods=["POST"])
    def predict() -> tuple[object, int]:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")
        if not image_data:
            return jsonify({"error": "image is required"}), 400

        try:
            image = preprocess_image(image_data)
            probabilities = predict_ensemble(
                models=app.config["ENSEMBLE_MODELS"],
                image=image,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        prediction = int(np.argmax(probabilities))
        return jsonify(
            {
                "result": prediction,
                "probabilities": probabilities.tolist(),
                "num_models": len(app.config["ENSEMBLE_MODELS"]),
            }
        ), 200

    return app


def discover_model_paths() -> list[Path]:
    configured_paths = os.environ.get("DIGIT_ENSEMBLE_MODELS", "").strip()
    if configured_paths:
        paths = [Path(item).expanduser().resolve() for item in configured_paths.split(os.pathsep) if item.strip()]
    else:
        paths = sorted(RUNS_DIR.glob(f"{DEFAULT_RUN_PREFIX}*/weights/best.pt"))

    if not paths:
        raise FileNotFoundError(
            "No ensemble checkpoints found. Train the ensemble first or set DIGIT_ENSEMBLE_MODELS."
        )

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ensemble checkpoints: {missing}")
    return paths


def preprocess_image(image_data: str) -> Image.Image:
    if "," not in image_data:
        raise ValueError("Expected a data URL payload.")

    encoded = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    return image.convert("RGB")


def predict_ensemble(models: list[YOLO], image: Image.Image) -> np.ndarray:
    if not models:
        raise ValueError("No models are loaded.")

    ensemble = np.zeros(10, dtype=np.float32)
    for model in models:
        results = model.predict(source=[image], imgsz=DEFAULT_IMGSZ, batch=1, device="cpu", verbose=False)
        probs = results[0].probs.data.detach().cpu().numpy().astype(np.float32)
        ensemble += probs
    ensemble /= float(len(models))
    return ensemble


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
