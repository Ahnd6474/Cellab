# 언어 모델 실습 가이드 (RoBERTa 파인튜닝)

이 폴더는 **텍스트 분류 문제를 딥러닝 언어 모델로 해결하는 교육용 예제**를 담고 있습니다.
주요 실습 시나리오는 `ai_press_releases.csv` 데이터를 사용해, 사람이 쓴 보도자료와 생성형 AI가 쓴 보도자료를 구분하는 것입니다.

## 학습 목표
- 텍스트 데이터를 학습 가능한 형태로 전처리하기
- 학습/검증/테스트 데이터 분할 원칙 익히기
- 사전학습 모델(`roberta-base`) 파인튜닝하기
- 정확도(Accuracy), F1 점수로 성능을 평가하고 해석하기

---

## 환경 설정
```bash
pip install transformers datasets torch scikit-learn evaluate pandas tqdm
```

> GPU가 있으면 학습 속도가 크게 개선됩니다.

---

## 전체 실습 흐름
1. CSV 로드 및 결측치 제거
2. 문장 단위로 텍스트 분해 후 라벨 생성
3. Train/Validation/Test 분할
4. 토크나이저/모델 준비
5. 에폭 단위 학습 + 검증/테스트 평가
6. 체크포인트 저장 및 결과 시각화

---

## 1) 데이터 불러오기 및 분할
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('ai_press_releases.csv')
df = df.dropna()

human = df['non_chat_gpt_press_release']
ai = df['chat_gpt_generated_release']

hu, a = [], []
for i in human:
    hu.extend(i.split('. '))
for i in ai:
    a.extend(i.split('. '))

ap = a.copy()
a.extend(hu)
texts = a
labels = [0 if i < len(ap) else 1 for i in range(len(texts))]

texts_train_val, texts_test, labels_train_val, labels_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts_train_val, labels_train_val, test_size=0.25, random_state=42, stratify=labels_train_val
)
```

### 핵심 포인트
- `stratify`를 사용해 라벨 비율이 train/val/test에 고르게 유지되도록 합니다.
- 문장 단위 분할은 데이터 양을 늘리지만, 문맥 손실이 생길 수 있으므로 태스크에 따라 단락 단위 실험도 고려하세요.

---

## 2) 모델/토크나이저 준비 및 DataLoader 구성
```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import Dataset


tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def collate_fn(batch):
    enc = tokenizer(
        [x["text"] for x in batch],
        padding="longest",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    enc["labels"] = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    return enc

train_ds = Dataset.from_dict({"text": texts_train, "label": labels_train})
val_ds   = Dataset.from_dict({"text": texts_val,   "label": labels_val})
test_ds  = Dataset.from_dict({"text": texts_test,  "label": labels_test})

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
optim  = AdamW(model.parameters(), lr=2e-5)
```

---

## 3) 학습/검증/테스트 루프
(기존 노트북 코드와 동일한 구조로 진행)

```python
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

num_epochs = 8
for epoch in range(1, num_epochs+1):
    model.train()
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [TRAIN]")
    for batch in train_loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            all_preds += preds
            all_labels += labels
    print("VAL ACC/F1:", accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="weighted"))
```

### 체크포인트 저장 팁
- 매 에폭 저장 시 디스크 사용량이 커질 수 있으므로, 보통은 `best` 성능 에폭만 저장합니다.
- 저장 경로는 로컬/Colab/Kaggle 환경에 맞게 수정하세요.

---

## 폴더 내 참고 자료
- `roberta-finetuning.ipynb`: 실습 중심 노트북
- `multi-attention-layer-training.ipynb`: 추가 실험용 노트북
- `Final project.Rmd`, `Final-project.pdf`: 보고서 자료
- `images/`: 학습 곡선, ROC, confusion matrix, heatmap 등 시각화 결과

---

## 자주 발생하는 문제
- **CUDA OOM**: `batch_size`를 줄이거나 `max_length`를 축소하세요.
- **성능 정체**: learning rate(예: `1e-5`, `3e-5`)와 epoch를 조정하세요.
- **데이터 누수 의심**: train/val/test 분할 시 동일 원문이 중복 포함되지 않는지 확인하세요.
