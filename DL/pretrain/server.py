import base64
import json
import re
import threading
import uuid
import webbrowser
from io import BytesIO
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import torch

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except ImportError:
    AutoImageProcessor = None
    AutoModelForImageClassification = None

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
BASE_DATA_DIR = BASE_DIR / "dataset" / "train"
MODEL_DIR = BASE_DIR / "classifier_model"
CLASSES_PATH = BASE_DIR / "classes.json"
DEFAULT_CLASSES = ["cat", "dog", "unknown"]
CLASSIFIER_MODEL_NAME = "facebook/deit-tiny-patch16-224"


def require_transformers() -> None:
    if AutoImageProcessor is None or AutoModelForImageClassification is None:
        raise ImportError("서버 추론에는 transformers 패키지가 필요합니다.")


def load_classes() -> list[str]:
    if CLASSES_PATH.exists():
        classes = json.loads(CLASSES_PATH.read_text())
        if isinstance(classes, list) and classes:
            return [str(name).strip().lower() for name in classes]
    return DEFAULT_CLASSES.copy()


require_transformers()
app = FastAPI(title="이미지 수집 및 예측 백엔드 서버")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
CLASSES = load_classes()
for class_name in CLASSES:
    (BASE_DATA_DIR / class_name).mkdir(parents=True, exist_ok=True)


class ImageData(BaseModel):
    className: str
    type: str = "원본"
    dataUrl: str


class ImageUploadRequest(BaseModel):
    images: List[ImageData]


@app.post("/api/upload")
async def upload_images(request: ImageUploadRequest):
    saved_count = 0
    saved_paths = []

    for img_data in request.images:
        try:
            class_name = img_data.className.strip().lower()
            if class_name not in CLASSES:
                raise HTTPException(status_code=400, detail=f"지원하지 않는 클래스입니다: {img_data.className}")

            class_dir = BASE_DATA_DIR / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            _, encoded = img_data.dataUrl.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            img = Image.open(BytesIO(image_bytes)).convert("RGB")

            image_type = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", img_data.type).strip("_") or "image"
            filename = f"{uuid.uuid4().hex[:8]}_{image_type}.jpg"
            filepath = class_dir / filename
            img.save(filepath, "JPEG")
            saved_count += 1
            saved_paths.append(str(filepath))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"이미지 저장 실패: {str(e)}")

    return {"message": f"성공적으로 {saved_count}장의 이미지를 서버에 저장했습니다.", "saved_paths": saved_paths}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = None
model = None
MODEL_READY = False
MODEL_ERROR = None

if MODEL_DIR.exists():
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
        model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).to(device)
        model.eval()
        MODEL_READY = True
        print(f"학습된 분류 모델을 불러왔습니다: {MODEL_DIR}")
    except Exception as e:
        MODEL_ERROR = f"모델 로딩 실패: {e}"
        print(f"경고: {MODEL_ERROR}")
else:
    MODEL_ERROR = f"학습된 분류 모델 디렉터리가 없습니다: {MODEL_DIR}"
    print(f"경고: {MODEL_ERROR}")


class PredictRequest(BaseModel):
    dataUrl: str


@app.post("/api/predict")
async def predict_image(request: PredictRequest):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail=MODEL_ERROR or "학습된 모델이 준비되지 않았습니다.")

    try:
        _, encoded = request.dataUrl.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        predicted_class = CLASSES[predicted_idx.item()]
        confidence_score = confidence.item() * 100
        return {"class": predicted_class, "confidence": f"{confidence_score:.2f}"}
    except Exception as e:
        print(f"예측 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/classes")
def get_classes():
    return {"classes": CLASSES}


@app.get("/")
def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": f"웹앱 파일이 없습니다: {index_path}"}


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    print("서버와 웹앱을 시작합니다.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
