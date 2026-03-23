# Cellab

Cellab은 동아리의 생명 정보 관련 내용을 다루기 위한 저장소 입니다.
실제로 수업·스터디에서 바로 써볼 수 있도록, 주제별 예제 코드/노트북/결과물을 폴더 단위로 정리해 두었습니다.

## 이 저장소에서 다루는 것
- 전통적인 머신러닝(예: 선형 회귀) 기본 개념과 실습 흐름
- 딥러닝 기반 이미지 분석(YOLO) 프로젝트 진행 방법
- 딥러닝 기반 언어 모델(RoBERTa) 파인튜닝 및 평가 방법
- 실험 결과(그래프, confusion matrix, 가중치 파일)를 해석하는 방법

## 폴더 구조
```text
Cellab/
├── README.md                     # 저장소 전체 안내 (현재 문서)
├── 머신러닝/
│   ├── Readme.md                 # 기초 ML 실습 안내
│   └── ml_map.svg                # 문제 유형별 알고리즘 선택 참고도
└── 딥러닝/
    ├── disaster tweets/
    │   ├── train_hierarchical_transformer.py # Disaster Tweets 계층형 Transformer 학습 진입점
    │   └── README.md                         # 데이터셋/실행/출력 설명
    ├── digit-recognizer/
    │   ├── train_mnist_yolo_cls.py   # MNIST 기반 digit recognizer 분류 학습 진입점
    │   ├── ultralytics_train.py      # 준비된 YOLO 데이터셋으로 범용 학습 실행
    │   └── digit_recognizer_reader.py # CSV 샘플 시각화/점검용
    ├── 이미지 분석 AI/
    │   ├── README.md             # YOLO 기반 객체 탐지 실습
    │   └── yolo/                 # 학습 결과물(가중치/곡선/행렬 등)
    └── 언어 모델/
        ├── README.md             # RoBERTa 분류 실습
        ├── *.ipynb / *.Rmd       # 실습 노트북/보고서
        └── images/               # 학습/평가 시각화 결과
```

## 학습 추천 순서
1. `머신러닝/Readme.md`로 기본 개념 및 모델 선택 감각 익히기
2. `딥러닝/이미지 분석 AI/README.md`로 비전 태스크 학습 흐름 익히기
3. `딥러닝/언어 모델/README.md`로 텍스트 분류 파인튜닝 실습하기

## Digit Recognizer 학습 파일
- `딥러닝/digit-recognizer/train_mnist_yolo_cls.py`를 실행하면 MNIST IDX 파일을 읽어서 분류용 데이터셋을 만든 뒤, YOLO 분류 모델 학습까지 한 번에 진행합니다.
- 가장 직접적인 학습 진입점은 아래 명령입니다.

```bash
python "딥러닝/digit-recognizer/train_mnist_yolo_cls.py"
```

- 무거운 학습을 바로 돌리고 싶지 않다면 설정만 확인하는 `--dry-run`, 데이터셋만 만들고 학습은 건너뛰는 `--prepare-only` 옵션을 사용할 수 있습니다.
- 참고로 `딥러닝/digit-recognizer/digit_recognizer_reader.py`는 샘플 확인용이며 학습 스크립트가 아닙니다.
- `딥러닝/digit-recognizer/prepare_yolo_detect_dataset.py`는 CSV를 YOLO 탐지용 데이터셋으로 변환만 하고, 실제 학습은 `딥러닝/digit-recognizer/ultralytics_train.py`에서 수행합니다.

## Disaster Tweets 학습 파일
- `딥러닝/disaster tweets/train_hierarchical_transformer.py`를 실행하면 같은 폴더의 `train.csv`, `test.csv`, `sample_submission.csv`를 읽어서 계층형 Transformer 기반 분류 모델을 학습합니다.
- 직접적인 학습 진입점은 아래 명령입니다.

```bash
python "딥러닝/disaster tweets/train_hierarchical_transformer.py"
```

- CPU 강제 실행은 `--cpu`, 하이퍼파라미터 조정은 `--epochs`, `--batch-size`, `--hidden-size` 같은 옵션으로 할 수 있습니다.
- 학습 결과물은 기본적으로 `딥러닝/disaster tweets/outputs` 아래에 체크포인트와 제출 파일로 저장됩니다.

## 사용 전 준비
- Python 3.10+ 권장 (일부 예제는 GPU 환경 권장)
- 각 폴더 README에 있는 의존성 설치 후 실행
- 노트북(`.ipynb`)은 JupyterLab/VS Code Notebook에서 실행 권장

## 주의사항
- 본 저장소는 **교육 목적**에 맞춰 작성되어 있어, 운영 환경(프로덕션) 코드 품질 기준과는 다를 수 있습니다.
- 데이터셋 라이선스/저작권은 각 원본 데이터 제공처 정책을 반드시 확인하세요.

## 기여 가이드(간단)
- 설명이 부족한 부분은 README 보강 중심으로 개선해 주세요.
- 실험 결과를 추가할 때는 **실행 환경(버전/GPU/에폭)** 정보를 함께 남겨 재현 가능성을 높여 주세요.
