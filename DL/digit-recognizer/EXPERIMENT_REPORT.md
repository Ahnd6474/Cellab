# MNIST 데이터셋 실험 보고서

MNIST 손글씨 숫자 분류 문제를 대상으로 진행한 실험을 정리한 문서다.  
관심사는 어떤 코드를 만들었는지보다, 어떤 접근이 실제로 어느 정도 성능을 보였고 그 차이를 어떻게 해석할 수 있는지에 있다.

## 목차

- [실험 목적](#실험-목적)
- [데이터셋과 평가 조건](#데이터셋과-평가-조건)
  - [데이터 출처](#데이터-출처)
  - [ViT 실험](#vit-실험)
  - [YOLO classification 실험](#yolo-classification-실험)
- [이론적 배경](#이론적-배경)
  - [Vision Transformer](#vision-transformer)
  - [YOLO classification](#yolo-classification)
- [실험별 설정](#실험별-설정)
  - [1. ViT 전이학습](#1-vit-전이학습)
  - [2. YOLO classification](#2-yolo-classification)
- [실험 결과](#실험-결과)
  - [요약 결과](#요약-결과)
  - [YOLO classification 세부 결과](#yolo-classification-세부-결과)
  - [ViT 결과](#vit-결과)
- [결과 해석](#결과-해석)
- [한계](#한계)
- [참고 문헌 및 그림 출처](#참고-문헌-및-그림-출처)
- [결론](#결론)

## 실험 목적

이 실험의 목표는 손글씨 숫자 분류 문제에서 서로 다른 모델링 접근을 비교하는 것이다.  
저장소 안에는 크게 두 가지 흐름이 있다.

- ViT 기반 전이학습
- YOLO classification 기반 분류

이 보고서는 현재 저장소 기준으로 직접 비교 가능한 ViT 일부 결과와 YOLO classification 반복 실험에 초점을 맞춘다.

## 데이터셋과 평가 조건

이 실험들은 모두 같은 MNIST 손글씨 숫자 분류 문제를 다룬다.  
다만 저장소에 남아 있는 실행 경로는 데이터 포맷과 공개 split이 완전히 같지는 않다.

### 데이터 출처

본 보고서에서 참조한 공개 데이터 출처는 다음과 같다.

1. [Kaggle Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer)
2. [MNIST Dataset (Kaggle mirror)](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

ViT 실험은 첫 번째 출처의 `train.csv` 형식을 사용했고, YOLO classification 실험은 두 번째 출처와 동일한 MNIST IDX 형식을 사용했다.

![MNIST sample digit](./sample_digit.png)

그림 2. MNIST 입력 샘플 예시. 본 실험은 저해상도 grayscale 숫자 영상을 0부터 9까지의 클래스로 분류하는 문제를 다룬다.

### ViT 실험

- 입력 데이터: Kaggle Digit Recognizer 형식의 `train.csv`
- 학습/검증 분할: `37,800 / 4,200`
- 원본 CSV 크기: `42,000` labeled train rows, `28,000` unlabeled test rows

### YOLO classification 실험

- 입력 데이터: raw MNIST IDX
- 학습/검증/테스트 분할: `54,000 / 6,000 / 10,000`
- 총 사용 라벨 데이터: `60,000 train + 10,000 test`

정리하면 두 계열은 모두 같은 MNIST 숫자 분류 실험이라고 볼 수 있다.  
다만 현재 저장소 기준으로는 ViT가 Kaggle CSV 버전을, YOLO가 IDX 버전을 사용했고 공개 split도 다르다. 그래서 여기서의 비교는 "같은 문제에 대한 결과 비교"에는 해당하지만, 완전히 같은 조건에서 측정된 벤치마크라고 보긴 어렵다.

## 이론적 배경

### Vision Transformer

Vision Transformer(ViT)는 입력 영상을 고정 크기 patch로 분할한 뒤, 각 patch를 토큰처럼 다루는 분류 모델이다. 각 patch는 선형 투영을 통해 embedding으로 변환되고, 위치 정보는 positional embedding으로 더해진다. 이후 encoder-only Transformer가 self-attention으로 patch 사이의 관계를 학습하며, 최종적으로 분류 토큰 또는 pooling 출력을 이용해 클래스 확률을 계산한다. 본 실험의 ViT 경로는 이 계열의 `facebook/deit-small-patch16-224` 모델을 MNIST에 맞게 미세조정하는 방식이다.

![Vision Transformer architecture](./figures/vision_transformer_architecture.svg)

그림 1. Vision Transformer의 기본 구조. 입력 이미지를 patch 단위로 분할하고, patch embedding과 positional embedding을 더한 뒤 Transformer encoder를 거쳐 최종 분류를 수행한다.

ViT는 전역 문맥을 한 번에 볼 수 있다는 장점이 있지만, CNN보다 inductive bias가 약하므로 충분한 사전학습이나 적절한 regularization이 중요하다. 특히 MNIST처럼 입력 구조가 단순한 문제에서는 높은 표현력을 얻는 대신 계산 비용이 커질 수 있다.

### YOLO classification

본 실험에서 사용한 `yolo26n-cls.pt`는 Ultralytics의 classification 전용 분기다. 공식 문서 기준으로 classification 모델은 `-cls` suffix를 사용하며, 이미지 전체에 대해 단일 class label과 confidence score를 출력한다. MNIST처럼 이미지 하나에 숫자 하나가 중심에 놓인 문제에서는 이런 분류 설정이 구현과 반복 실험 측면에서 단순하고 직접적이다.

## 실험별 설정

### 1. ViT 전이학습

ViT 경로는 `facebook/deit-small-patch16-224`를 MNIST 분류에 맞게 파인튜닝하는 방식이다.

주요 특징:

- 28x28 grayscale 이미지를 RGB로 변환
- Hugging Face `AutoImageProcessor`, `AutoModelForImageClassification` 사용
- backbone과 classifier head에 서로 다른 learning rate 적용
- focal loss와 클래스 가중치 사용
- validation loss 기준 early stopping 적용

이 접근은 사전학습된 표현을 가져와 작은 숫자 이미지 분류에 맞게 미세조정하는 방향이다. 구조는 탄탄하지만, MNIST처럼 단순한 데이터에 224 해상도 기반 Transformer를 쓰는 만큼 계산량은 상대적으로 크다.

### 2. YOLO classification

YOLO 쪽에서 실제 결과가 남아 있는 실험은 detect가 아니라 classification이다.  
반복 실행된 run은 다음 네 개다.

- `mnist-yolo26n-ensemble-seed42`
- `mnist-yolo26n-ensemble-seed52`
- `mnist-yolo26n-ensemble-seed62`
- `mnist-yolo26n-ensemble-seed72`

공통 설정:

- 모델: `yolo26n-cls.pt`
- epochs: `30`
- batch: `128`
- image size: `32`
- patience: `10`
- pretrained: `true`

이 실험은 여러 seed로 반복되어 있어서, 단일 최고점보다 결과의 안정성을 함께 볼 수 있다는 장점이 있다.

## 실험 결과

### 요약 결과

표 1은 현재 저장소에서 직접 확인 가능한 핵심 결과를 모델링 접근별로 정리한 것이다.

표 1. 모델링 접근별 현재 확인 가능한 핵심 결과 요약.

| 모델링 접근 | 현재 확인 가능한 핵심 결과 | 해석 |
| --- | --- | --- |
| ViT 전이학습 | `best_val_loss = 0.0071516539`, `best_val_acc = 0.99548` | 단일 run 기준 성능은 강하지만 반복 검증은 아직 부족 |
| YOLO classification | 최고 top-1 `0.99017`, 평균 `0.989585` | 현재 가장 안정적으로 검증된 결과 |

### YOLO classification 세부 결과

표 2는 YOLO classification 반복 실험의 seed별 성능을 비교한 결과다.

표 2. YOLO classification 반복 실험의 seed별 성능 비교.

| Seed | 최고 Top-1 Accuracy | 최소 Validation Loss | 최적 Epoch |
| --- | --- | --- | --- |
| 42 | `0.98917` | `0.03729` | `24` |
| 52 | `0.98967` | `0.03206` | `28` |
| 62 | `0.98933` | `0.03287` | `29` |
| 72 | `0.99017` | `0.03183` | `30` |

요약 통계:

- 평균 top-1 accuracy: `0.989585`
- top-1 accuracy 표준편차: `0.000383`
- 평균 validation loss: `0.0335125`

이 수치를 보면 YOLO classification은 한 번 우연히 잘 나온 실험이 아니라, 여러 seed에서도 비슷한 수준을 유지하는 안정적인 기준선이라고 볼 수 있다.

![YOLO seed72 training curves](./yolo/mnist_yolo_cls/runs/mnist-yolo26n-ensemble-seed72/results.png)

그림 3. YOLO classification `seed72` 실험의 학습 곡선. Epoch가 진행됨에 따라 train loss와 validation loss가 함께 감소하며, top-1 accuracy는 약 0.99 수준까지 수렴한다.

![YOLO seed72 normalized confusion matrix](./yolo/mnist_yolo_cls/runs/mnist-yolo26n-ensemble-seed72/confusion_matrix_normalized.png)

그림 4. YOLO classification `seed72` 실험의 정규화 confusion matrix. 대부분 클래스에서 예측이 대각선에 집중되어 있으며, 일부 유사 숫자 쌍에서만 제한적인 혼동이 관찰된다.

### ViT 결과

ViT 쪽에서 현재 확인 가능한 대표 수치는 다음과 같다.

- 모델: `facebook/deit-small-patch16-224`
- best validation loss: `0.0071516539`
- best validation accuracy: `0.99548`
- best epoch: `11 / 12`
- train/val split: `37,800 / 4,200`

주어진 학습 기록 기준으로 보면 ViT는 단일 run에서 상당히 강한 성능을 보였다. 특히 epoch 11에서 validation loss와 validation accuracy가 함께 가장 좋았고, 이후 epoch 12에서는 다시 성능이 내려가서 최적 체크포인트가 분명하게 구분된다.

표 3. ViT epoch별 학습 기록.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| --- | --- | --- | --- | --- |
| 1 | `0.12462` | `0.95021` | `0.01961` | `0.98667` |
| 2 | `0.01231` | `0.99196` | `0.01189` | `0.99357` |
| 3 | `0.00569` | `0.99603` | `0.01147` | `0.99238` |
| 4 | `0.00363` | `0.99728` | `0.01875` | `0.98881` |
| 5 | `0.00321` | `0.99746` | `0.01235` | `0.99262` |
| 6 | `0.00181` | `0.99857` | `0.01248` | `0.99333` |
| 7 | `0.00054` | `0.99960` | `0.01015` | `0.99381` |
| 8 | `0.00046` | `0.99958` | `0.01201` | `0.99405` |
| 9 | `0.00175` | `0.99868` | `0.02113` | `0.98786` |
| 10 | `0.00322` | `0.99730` | `0.01250` | `0.99333` |
| 11 | `0.00053` | `0.99937` | `0.00715` | `0.99548` |
| 12 | `0.00021` | `0.99979` | `0.01629` | `0.99024` |

이 기록을 보면 ViT는 초반부터 빠르게 수렴하고, epoch 11에서 가장 좋은 검증 성능에 도달한다. 다만 여기서 바로 "ViT가 YOLO보다 낫다"거나 반대로 "YOLO가 더 좋다"라고 단정하긴 어렵다. 이유는 간단하다.

- YOLO와 데이터 포맷과 저장된 분할 조건이 다르다.
- ViT는 현재 단일 run 기준 정보만 남아 있다.
- ViT confusion matrix나 오분류 예시 산출물은 저장소에 남아 있지 않다.

그래서 현재 ViT는 "성능이 부족한 모델"이라기보다, 단일 run 성과는 강하지만 비교 실험의 형식이 아직 충분히 맞춰지지 않은 상태라고 보는 편이 더 정확하다.

## 결과 해석

### 1. 현재 기준 가장 안정적으로 검증된 모델은 YOLO classification

이 문서에서 가장 분명한 결론은 이 부분이다.  
현재 저장된 결과만 보면, YOLO classification이 가장 설득력 있는 기준 모델이다.

이유는 세 가지다.

- 정확도가 높다.
- seed를 바꿔도 흔들림이 작다.
- MNIST 문제 구조와 잘 맞는다.

MNIST는 작은 이미지 안에 숫자 하나가 들어 있는 전형적인 단일 객체 분류 문제다. 이럴 때는 detect보다 classify가 자연스럽고, 무거운 ViT보다 가볍고 반복 실험이 쉬운 YOLO 분류기가 실용적으로 강할 수 있다.

### 2. ViT는 단일 run 기준으로는 강하지만 비교는 아직 덜 끝났다

ViT는 설계 자체는 꽤 공들여져 있다.

- 사전학습 백본 사용
- backbone/head learning rate 분리
- focal loss 적용
- best checkpoint 기준으로 `val_acc 0.99548` 확인

지금은 이전보다 해석 근거가 분명하다. ViT는 단일 run 기준으로는 YOLO classification 최고 seed보다 높은 validation accuracy를 기록했다. 특히 MNIST처럼 단순한 데이터에서도 pretrained Transformer가 충분히 잘 맞아들어간다는 점은 확인됐다. 다만 28x28 이미지를 224x224로 키워 넣는 구조가 계산량 대비 항상 유리하다고 보긴 어렵고, 현재 저장소만으로는 반복 실험까지 포함한 안정성을 판단할 수 없다.

그래서 현재의 ViT 평가는 이렇게 정리하는 것이 맞다.

- 단일 run 성능은 충분히 경쟁력이 있다.
- 같은 split과 seed 반복으로 재검증이 필요하다.
- confusion matrix와 오분류 예시까지 남겨야 YOLO와 같은 밀도로 해석할 수 있다.

## 한계

이 보고서는 저장소 안에 남아 있는 결과를 기준으로 정리했다. 그래서 아래 한계가 있다.

- ViT와 YOLO가 같은 MNIST 문제를 다루지만, 저장된 결과는 같은 공개 split 기준으로 남아 있지 않다.
- ViT는 confusion matrix나 오분류 예시 같은 후속 해석 산출물이 저장돼 있지 않다.

즉, 이 문서는 완전한 리더보드라기보다 "현재까지 확보된 결과를 바탕으로 한 중간 정리"에 가깝다.

## 참고 문헌 및 그림 출처

- [Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929)
- [Ultralytics Docs, *Image Classification*](https://docs.ultralytics.com/tasks/classify/)
- [Wikipedia, *Vision transformer*](https://en.wikipedia.org/wiki/Vision_transformer)

## 결론

현재까지의 실험을 성능 중심으로 요약하면 다음과 같다.

- 가장 안정적으로 검증된 기준 모델은 `YOLO classification`
- `ViT`는 단일 run 기준으로 `val_acc 0.99548`까지 도달했지만, 같은 split 반복 검증은 아직 부족하다

실험 보고서 관점에서 지금 가장 자연스러운 다음 단계는 두 가지다.

1. ViT를 같은 split 기준으로 seed 반복 실험해서 YOLO와 같은 형식의 비교표를 만들기
2. ViT confusion matrix와 오분류 예시를 산출물로 저장해서 해석 근거를 보강하기

지금 단계에서 대표 실험 결과로 내세우기 가장 좋은 것은 YOLO classification이고, ViT는 그 다음 후보라고 보는 것이 무리 없는 정리다.
