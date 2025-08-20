import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. 모델 불러오기
# ==============================
model = load_model("saved_model.h5")   # 학습 시 저장한 모델 경로

# ==============================
# 2. 테스트 데이터 로드
# ==============================
# 예시: data shape = (N, height, width, channels), labels shape = (N,)
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Test data shape:", X_test.shape)
print("Test label shape:", y_test.shape)

# ==============================
# 3. 추론 (Prediction)
# ==============================
y_pred_proba = model.predict(X_test)                # 클래스별 확률
y_pred = np.argmax(y_pred_proba, axis=1)            # 가장 확률 높은 클래스 선택

# ==============================
# 4. 성능 평가
# ==============================
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, digits=4))

print("\n[Confusion Matrix]")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Confusion Matrix 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# 5. 개별 샘플 추론 예시
# ==============================
idx = 0  # 첫 번째 샘플
sample = X_test[idx:idx+1]   # 배치 차원 유지
pred_class = np.argmax(model.predict(sample), axis=1)[0]

print(f"\n샘플 {idx} → 예측 클래스: {pred_class}, 실제 라벨: {y_test[idx]}")
