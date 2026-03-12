# 이미지 분석 AI 실습 가이드 (YOLO 객체 탐지)

이 폴더는 **YOLO 기반 객체 탐지(Object Detection)** 교육용 프로젝트입니다.
예시 주제는 토마토 질병 탐지이며, 사전학습 모델을 활용해 적은 데이터로도 성능을 확보하는 실습 흐름을 제공합니다.

## 학습 목표
- YOLO 사전학습 모델 구조와 전이학습 개념 이해
- 데이터셋 YAML 기반 학습 파이프라인 실행
- 학습 로그/곡선/혼동행렬(confusion matrix) 해석
- `best.pt` 가중치를 이용한 추론(inference) 수행

---

## 환경 설정
Python 3.10+ 권장. (GPU 환경 강력 권장)

```bash
pip install numpy scipy torch ultralytics
```

필요 시 버전 고정 예시:
```bash
pip install "torch>=2.1" "ultralytics>=8.3"
```

---

## YOLO 모델 선택 가이드
YOLO는 버전(예: 11, 26)과 크기(n/s/m/l/x)에 따라 성능과 속도가 달라집니다.

- `n`(nano): 가장 빠르지만 정확도 낮을 수 있음
- `s`(small): 입문/실습에 가장 무난
- `m/l/x`: 정확도 향상 가능하지만 GPU 메모리 요구량 증가

처음에는 `yolo11s.pt`로 시작하고, 이후 데이터와 장비에 맞춰 확장하세요.

---

## 학습 방법
1. 데이터셋을 YOLO 형식으로 준비하고 `data.yaml` 경로 확인
2. 사전학습 모델 로드
3. `model.train(...)`으로 학습 실행

```python
from ultralytics import YOLO

YAML_PATH = '/kaggle/input/datasets/dannyahn1/tomato-disease/tomato-village-diseases.v1i.yolov11/data.yaml'
model = YOLO("yolo11s.pt")
model.train(
    data=str(YAML_PATH),
    task="detect",
    imgsz=1024,
    epochs=200,
    batch=16,
    device=[0, 1]   # GPU 2개 예시
)
```

### 디바이스 설정 팁
- GPU 1개: `device=0` 또는 `device="0"`
- GPU 2개 이상: `device=[0,1]` 등
- CPU만 사용: `device="cpu"` (학습 속도 매우 느릴 수 있음)

---

## 학습 결과 해석
학습이 완료되면 보통 아래와 같은 산출물이 생성됩니다.

```text
.
└── yolo/
    └── yolo_XX/
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

- `best.pt`: 검증 성능 기준 최적 가중치 (실전 추론에 보통 사용)
- `last.pt`: 마지막 에폭 가중치
- `results.png/csv`: epoch별 정밀도·재현율·mAP·loss 추이
- `confusion_matrix*`: 클래스별 오분류 패턴 분석

---

## 추론(학습한 모델 활용)
```python
from ultralytics import YOLO

model = YOLO("yolo/yolo_26/weights/best.pt")
results = model.predict(source="path/to/image.jpg", imgsz=1024, conf=0.25)
```

> `conf` 임계값을 높이면 오탐(False Positive)이 줄고, 너무 높이면 미탐(False Negative)이 늘 수 있습니다.

---

## 폴더 내 파일 안내
- `tomatodisease.ipynb`: 학습/추론 실습 노트북
- `yolo/yolo_11`, `yolo/yolo_26`: 서로 다른 실험 실행 결과 폴더
- `requirements.txt`: 실행 시 참고할 의존성 목록

## 라이선스
이 프로젝트는 `LICENSE` 파일 정책을 따릅니다.
