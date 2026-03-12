# 이미지 분석 AI 

YOLO 기반 객체 탐지 예시이다. 
YOLO는 사전 학습된 이미지 분석 AI로 사전학습 된 AI 모델이다. 
사전 학습 된 AI 모델은 추가학습 진행시(우리가 하는 것) 적은 데이터와 학습 시간으로 높은 성능을 내도록 해준다. 

## 환경 설정
아래 명령으로 기본 의존성을 설치합니다. 이 코드가 구동하기 위해 필요한 요구사항을 설치한다. 
-pip는 파이썬을 깔 때 같이 깔림. 파이썬은 3.12 나 3.13 사용을 권장함. 
```bash
pip install numpy scipy torch ultralytics
```

## 학습 방법

1. Ultralytics 및 필수 패키지 설치
2. `YOLO("yolo11s.pt")`로 사전학습 가중치 로드
- yolo는 여러 가지 버전이 있지만 11, 26이 제일 좋다.
- yolo는 각 버전마다 nano(n), small(s), medium(m), larage(l), XLarge(x) 총 5개의 버전이 있다.(11의 s면 "yolo11s.pt"임)
- 여러 크기를 시도해보고 적당한 것 고르면 됨
3. 데이터셋 YAML 경로 지정 후 학습 실행
- 이미지 데이터를 yolo 형식으로 다운받으면 Yaml이 같이 옴, 그 경로를 복사해서 넣어주면 됨

학습 코드:

```python
from ultralytics import YOLO

YAML_PATH = '/kaggle/input/datasets/dannyahn1/tomato-disease/tomato-village-diseases.v1i.yolov11/data.yaml'
model = YOLO("yolo11s.pt")
model.train(data=str(YAML_PATH), task="detect", imgsz=1024, epochs=200, batch=16, device=[0, 1])
```
여기서 device=[0.1]은 gpu 2개라 그렇게 함. 1개면 device=1로 해야 함. 없으면 못 돌림

### 학습 결과 확인
weights의 .pt 파일이 학습된 모델로, best.pt를 사용하면 됨.
.png는 학습 결과를 그림으로 표현한것, 보고서에 복붙하자. 
```text
.
|── yolo/
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

## 추론(학습한 모델 써먹기) 예시
학습 완료 후 `best.pt`를 이용해서  추론 예시는 다음과 같습니다.

```python
from ultralytics import YOLO

model = YOLO("yolo/runs/detect/train/weights/best.pt")
results = model.predict(source="path/to/image.jpg", imgsz=1024, conf=0.25)
```

## 라이선스
이 프로젝트는 `LICENSE` 파일의 정책을 따릅니다.
