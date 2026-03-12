# 머신러닝

## 선형 회귀

### 환경 설정
```bash
pip install numpy scikit-learn matplotlib
```

### 실행 코드
```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예제 데이터
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
## 다른 머신 러닝에 관하여

자신의 데이터와 상황에 적합한 머신러닝 모델을 골라야 한다. 

![적합한 머신 러닝 기법을 고르자.](ml_map.svg)

## 전기영동 겔 분석
프로그램이 UI나 이런 게 쓰기 어려울 수 있음. 
[프로그램 다운로드 링크](https://www.gelanalyzer.com/?i=1)
