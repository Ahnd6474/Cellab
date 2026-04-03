from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_DIR = PROJECT_ROOT / "mnist_vit_cls" / "model"
TRAIN_DTYPE = torch.float64


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(APP_DIR / "templates"))

    processor, model = load_model_bundle(MODEL_DIR)
    app.config["MODEL_DIR"] = str(MODEL_DIR)
    app.config["PROCESSOR"] = processor
    app.config["MODEL"] = model

    @app.route("/")
    def index() -> str:
        return render_template("index.html", model_path=app.config["MODEL_DIR"])

    @app.route("/health", methods=["GET"])
    def health() -> tuple[dict[str, object], int]:
        return {
            "status": "ok",
            "model_dir": app.config["MODEL_DIR"],
            "dtype": str(TRAIN_DTYPE),
        }, 200

    @app.route("/predict", methods=["POST"])
    def predict() -> tuple[object, int]:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")
        if not image_data:
            return jsonify({"error": "image is required"}), 400

        try:
            image = preprocess_image(image_data)
            probabilities = predict_probabilities(
                processor=app.config["PROCESSOR"],
                model=app.config["MODEL"],
                image=image,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        prediction = int(np.argmax(probabilities))
        return jsonify(
            {
                "result": prediction,
                "probabilities": probabilities.tolist(),
                "model_dir": app.config["MODEL_DIR"],
            }
        ), 200

    return app


def load_model_bundle(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(f"No ViT checkpoint found: {model_dir}")

    processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model = model.to(dtype=TRAIN_DTYPE)
    model.eval()
    return processor, model


def preprocess_image(image_data: str) -> Image.Image:
    if "," not in image_data:
        raise ValueError("Expected a data URL payload.")

    encoded = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    return image.convert("RGB")


def predict_probabilities(processor, model, image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(dtype=TRAIN_DTYPE)
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=1)
    return probs[0].cpu().numpy().astype(np.float64)


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
