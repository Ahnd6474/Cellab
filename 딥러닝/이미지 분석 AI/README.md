# TomatatoDisease

토마토 질병 탐지를 위한 YOLO 기반 객체 탐지 실험 저장소입니다.  
현재 저장소에는 학습 노트북(`tomatodisease.ipynb`)과 학습 결과 산출물(`yolo/runs/detect/train`)이 포함되어 있습니다.

## 프로젝트 개요
- **목표**: 토마토 잎/과실 이미지에서 질병 증상을 객체 탐지(Object Detection)로 식별
- **모델**: Ultralytics YOLO (`yolo11s.pt` 기반 학습)
- **학습 방식**: 노트북에서 `model.train(...)` 실행
- **모델 학습 결과**는 `runs/detect/train/`를 참고, best/last 가중치, confusion matrix, PR/F1/Precision/Recall curve, 학습 로그가 정리되어 있음

## 저장소 구조
```text
.
├── README.md
├── requirements.txt
├── tomatodisease.ipynb
└── yolo/
    ├── yolo11s.pt
    ├── yolo26n.pt
    └── runs/detect/train/
        ├── args.yaml
        ├── results.csv
        ├── results.png
        ├── confusion_matrix.png
        ├── confusion_matrix_normalized.png
        ├── BoxPR_curve.png
        ├── BoxF1_curve.png
        ├── BoxP_curve.png
        ├── BoxR_curve.png
        └── weights/
            ├── best.pt
            └── last.pt
```

## 환경 설정
아래 명령으로 기본 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

`requirements.txt`에는 다음 패키지가 포함되어 있습니다.
- numpy
- scipy
- torch
- ultralytics

> 참고: 노트북 내부에서는 `opencv-python`을 추가 설치하도록 작성되어 있습니다.

## 학습 방법
노트북(`tomatodisease.ipynb`) 기준 학습 절차는 다음과 같습니다.

1. Ultralytics 및 필수 패키지 설치
2. `YOLO("yolo11s.pt")`로 사전학습 가중치 로드
3. 데이터셋 YAML 경로 지정 후 학습 실행

노트북에 포함된 핵심 코드:

```python
from ultralytics import YOLO

YAML_PATH = '/kaggle/input/datasets/dannyahn1/tomato-disease/tomato-village-diseases.v1i.yolov11/data.yaml'
model = YOLO("yolo11s.pt")
model.train(data=str(YAML_PATH), task="detect", imgsz=1024, epochs=200, batch=16, device=[0, 1])
```

## 추론(Inference) 예시
학습 완료 후 `best.pt`를 이용한 추론 예시는 다음과 같습니다.

```python
from ultralytics import YOLO

model = YOLO("yolo/runs/detect/train/weights/best.pt")
results = model.predict(source="path/to/image.jpg", imgsz=1024, conf=0.25)
```

## 결과 확인 포인트
- `yolo/runs/detect/train/results.csv`: epoch별 지표 로그
- `yolo/runs/detect/train/results.png`: 학습 추이 시각화
- `yolo/runs/detect/train/confusion_matrix*.png`: 클래스별 오분류 분석
- `yolo/runs/detect/train/weights/best.pt`: 최적 성능 가중치

## 주의 사항
- 현재 노트북의 데이터 경로는 Kaggle 환경 경로를 사용합니다.
- 로컬/서버 환경에서 재학습할 경우 `YAML_PATH`를 사용자 환경에 맞게 변경해야 합니다.
- 멀티 GPU(`device=[0, 1]`) 설정은 사용 환경에 맞게 조정하세요.

## 라이선스
이 프로젝트는 `LICENSE` 파일의 정책을 따릅니다.
