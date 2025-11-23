import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 로드
X = np.load("preprocessed_data/X.npy")        # (N, 100, 3)
y = np.load("preprocessed_data/y.npy")        # (N,)
q = np.load("preprocessed_data/quality.npy")  # (N,) 0=1st, 1=2nd

# 2. (N, 100, 3) -> (N, 300) 평탄화해서 traditional ML에 넣기
X_flat = X.reshape(X.shape[0], -1)

# 3. 1st로 학습, 2nd로 테스트
train_mask = (q == 0)   # 1st(고품질)
test_mask = (q == 1)    # 2nd(노이즈)

X_train, y_train = X_flat[train_mask], y[train_mask]
X_test, y_test = X_flat[test_mask], y[test_mask]

print("Train size:", X_train.shape[0], " Test size:", X_test.shape[0])

# 4-1. SVM baseline
svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0),
)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

print("\n=== SVM (1st train → 2nd test) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# 4-2. RandomForest baseline
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=0,
)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("\n=== RandomForest (1st train → 2nd test) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
