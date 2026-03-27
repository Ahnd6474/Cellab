import os
import base64
import io
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps


# ==========================================
# 1. 모델 아키텍처 정의 (학습 시와 동일해야 함)
# ==========================================
class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x): return self.classifier(self.features(x))


# ==========================================
# 2. Flask 앱 및 모델 초기화
# ==========================================
app = Flask(__name__)

# 모델 로드
device = torch.device("cpu")  # 웹 서버에서는 일반적으로 CPU 사용
model = AdvancedCNN()
MODEL_PATH = 'mnist_model.pth'

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")
else:
    print(f"Warning: {MODEL_PATH} not found. Please ensure the model file is in the same directory.")


# ==========================================
# 3. 비즈니스 로직 및 API
# ==========================================
def preprocess_image(image_data):
    """Base64 이미지를 28x28 PyTorch 텐서로 변환"""
    # 1. Base64 디코딩
    img_bytes = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_bytes)).convert('L')  # 그레이스케일 변환

    # 2. 전처리 (28x28 리사이징 및 정규화)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0

    # 3. 텐서 변환 (Batch, Channel, H, W)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        input_tensor = preprocess_image(data)

        with torch.no_grad():
            output = model(input_tensor)
            # 확률값 계산 (Softmax)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            prediction = torch.argmax(probabilities).item()

        return jsonify({
            'result': prediction,
            'probabilities': probabilities.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # 0.0.0.0으로 설정 시 같은 네트워크 내 다른 기기에서도 접속 가능
    app.run(host='0.0.0.0', port=5000, debug=True)





