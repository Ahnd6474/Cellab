# 머신러닝 실습 가이드

이 폴더는 **머신러닝 입문 학습자**를 위한 기초 실습 자료입니다.
핵심 목표는 다음 3가지입니다.

1. 선형 회귀(Linear Regression)의 동작 원리 이해
2. 최소한의 코드로 학습 → 예측 → 시각화 흐름 경험
3. 문제 유형에 따라 어떤 알고리즘을 고를지 감각 익히기

---

## 1) 선형 회귀 예제

### 환경 설정
```bash
pip install numpy scikit-learn matplotlib
```

### 실습 코드
```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예제 데이터 (X: 입력, y: 정답)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 예측
y_pred = model.predict(X)

# 결과 출력
print("기울기 (coef):", model.coef_[0])
print("절편 (intercept):", model.intercept_)
print("예측값:", y_pred)

# 시각화
plt.scatter(X, y, label="data")
plt.plot(X, y_pred, label="linear regression")
plt.legend()
plt.show()
```

### 코드 해설
- `model.fit(X, y)`: 데이터로 직선을 학습합니다.
- `coef_` / `intercept_`: 학습된 직선의 기울기/절편입니다.
- `predict(X)`: 학습한 직선으로 입력값의 출력을 예측합니다.
- 산점도와 직선을 함께 보면 모델이 데이터를 얼마나 잘 설명하는지 직관적으로 확인할 수 있습니다.

---

## 2) 어떤 머신러닝 기법을 선택해야 할까?

실무/프로젝트에서는 모델 성능만이 아니라 데이터 양, 해석 가능성, 학습 시간도 함께 고려해야 합니다.

![적합한 머신 러닝 기법을 고르자.](ml_map.svg)

> 팁: 처음에는 단순한 모델(선형/로지스틱/트리)로 베이스라인을 만들고, 필요할 때 복잡한 모델로 확장하는 것이 학습과 프로젝트 모두에 유리합니다.

---

## 3) 전기영동 겔 분석 참고

전기영동 결과 이미지를 분석할 때 사용할 수 있는 외부 도구:
- [GelAnalyzer 다운로드 링크](https://www.gelanalyzer.com/?i=1)

수업/실험 환경에 따라 UI가 낯설 수 있으므로, 처음에는 샘플 이미지로 연습한 뒤 실제 데이터에 적용하는 것을 권장합니다.
