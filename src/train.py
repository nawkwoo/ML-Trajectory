"""
Train RandomForest baseline model on split trajectory data
+ Confusion Matrix PNG 저장
+ Feature Importance PNG 저장
+ Classification Report TXT 저장

기본 파이프라인
---------------
1) project_root/data/split 에서 분할된 데이터(X_train, y_train, X_test, y_test) 로드
2) 궤적 (N, T, C) 데이터를 flatten → (N, T*C)
3) RandomForestClassifier 학습
4) test 성능 출력 (accuracy, classification_report)
5) train 내부에서 5-fold 교차검증 실행
6) 모델 저장: project_root/models/ml_model_rf.pkl
7) Confusion Matrix 저장: project_root/models/confusion_matrix.png
8) Feature Importance plot 저장: project_root/models/feature_importance.png
9) Classification Report 저장: project_root/models/rf_report.txt
"""

import os

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold


# --------------------------------------------------------
# 데이터 로더
# --------------------------------------------------------
def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    project_root/data/split 에서 X_train, y_train, X_test, y_test를 로드한다.

    Returns
    -------
    X_train : np.ndarray
        shape (N_train, T, C)의 학습용 궤적 데이터
    y_train : np.ndarray
        shape (N_train,)의 학습용 정수 라벨
    X_test : np.ndarray
        shape (N_test, T, C)의 테스트용 궤적 데이터
    y_test : np.ndarray
        shape (N_test,)의 테스트용 정수 라벨
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../src
    project_root = os.path.dirname(current_dir)
    split_dir = os.path.join(project_root, "data", "split")

    x_train_path = os.path.join(split_dir, "X_train.npy")
    y_train_path = os.path.join(split_dir, "y_train.npy")
    x_test_path = os.path.join(split_dir, "X_test.npy")
    y_test_path = os.path.join(split_dir, "y_test.npy")

    if not (
        os.path.exists(x_train_path)
        and os.path.exists(y_train_path)
        and os.path.exists(x_test_path)
        and os.path.exists(y_test_path)
    ):
        raise FileNotFoundError(
            f"Split data not found in '{split_dir}'. "
            "Run split_data.py first."
        )

    print(f"Using split data from '{split_dir}'")

    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    return X_train, y_train, X_test, y_test


# --------------------------------------------------------
# 메인 학습 파이프라인
# --------------------------------------------------------
def main() -> None:
    """
    RandomForest baseline 모델을 학습하고,
    성능 지표 및 시각화 결과를 저장한다.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../src
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1) 데이터 로드 + flatten
    # --------------------------------------------------------
    X_train, y_train, X_test, y_test = load_data()

    # (N, T, C) -> (N, T*C) 로 평탄화
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print("Train size:", X_train.shape[0], " Test size:", X_test.shape[0])
    print("Flattened feature dim:", X_train_flat.shape[1])

    # --------------------------------------------------------
    # 2) RandomForest 학습 (train 전체로 학습 후, test 평가)
    # --------------------------------------------------------
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=0,
    )
    rf_clf.fit(X_train_flat, y_train)
    y_pred_rf = rf_clf.predict(X_test_flat)

    print("\n=== RandomForest (train → test) ===")
    acc = accuracy_score(y_test, y_pred_rf)
    print("Accuracy:", acc)

    report = classification_report(y_test, y_pred_rf)
    print(report)

    # Classification report를 txt 파일로도 저장 (보고서용)
    report_path = os.path.join(model_dir, "rf_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write(report)
    print(f"Saved classification report → '{report_path}'")

    # --------------------------------------------------------
    # 3) 5-fold 교차검증 (train 내부 검증)
    # --------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    rf_accs: list[float] = []

    for fold_idx, (tr, va) in enumerate(skf.split(X_train_flat, y_train), start=1):
        X_tr, y_tr = X_train_flat[tr], y_train[tr]
        X_va, y_va = X_train_flat[va], y_train[va]

        rf_cv = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=0,
        )
        rf_cv.fit(X_tr, y_tr)
        acc_cv = rf_cv.score(X_va, y_va)
        rf_accs.append(acc_cv)

        print(f"[Fold {fold_idx}] RF acc={acc_cv:.3f}")

    print("\n=== 5-fold CV (on train) ===")
    print(f"RF mean acc={np.mean(rf_accs):.3f} ± {np.std(rf_accs):.3f}")

    # --------------------------------------------------------
    # 4) Confusion Matrix PNG 저장
    # --------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred_rf)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    cm_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.title("Confusion Matrix (RandomForest)")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix → '{cm_path}'")

    # --------------------------------------------------------
    # 5) Feature Importance PNG 저장
    # --------------------------------------------------------
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]  # 중요도 내림차순 인덱스
    top_k = 20  # 상위 20개만 시각화

    plt.figure(figsize=(10, 6))
    plt.bar(range(top_k), importances[indices[:top_k]], align="center")
    plt.xticks(range(top_k), indices[:top_k], rotation=45)
    plt.xlabel("Feature Index (Flattened)")
    plt.ylabel("Importance")
    plt.title("RandomForest Feature Importance (Top 20)")
    fi_path = os.path.join(model_dir, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(fi_path, dpi=200)
    plt.close()
    print(f"Saved feature importance plot → '{fi_path}'")

    # --------------------------------------------------------
    # 6) 모델 저장 (시연용 / 재사용용)
    # --------------------------------------------------------
    model_path = os.path.join(model_dir, "ml_model_rf.pkl")
    joblib.dump(rf_clf, model_path)
    print(f"Saved RandomForest model → '{model_path}'")


if __name__ == "__main__":
    main()
