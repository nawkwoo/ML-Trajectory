import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_data():
    data_dir = "augmented_data"
    if not os.path.exists(os.path.join(data_dir, "X.npy")):
        raise FileNotFoundError("augmented_data/X.npy not found. Run `python augment_data.py` first.")
    print(f"Using data from '{data_dir}'")
    X = np.load(os.path.join(data_dir, "X.npy"))        # (N, 100, 3)
    y = np.load(os.path.join(data_dir, "y.npy"))        # (N,)
    q = np.load(os.path.join(data_dir, "quality.npy"))  # (N,) 0=1st, 1=2nd
    return X, y, q


def main():
    # 1. 데이터 로드
    X, y, q = load_data()

    # 2. (N, 100, 3) -> (N, 300) 평탄화해 traditional ML에 투입
    X_flat = X.reshape(X.shape[0], -1)

    # 3-A. 1st로 학습, 2nd로 테스트 (분포 외 일반화 평가)
    train_mask = q == 0  # 1st(고품질)
    test_mask = q == 1   # 2nd(노이즈)

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

    print("\n=== SVM (1st train -> 2nd test) ===")
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

    print("\n=== RandomForest (1st train -> 2nd test) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

    # 3-B. 동일 분포 5-fold 교차검증 (내부 품질 체크)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    svm_accs, rf_accs = [], []
    for fold_idx, (tr, va) in enumerate(skf.split(X_flat, y), start=1):
        X_tr, y_tr = X_flat[tr], y[tr]
        X_va, y_va = X_flat[va], y[va]

        svm_cv = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0),
        )
        svm_cv.fit(X_tr, y_tr)
        svm_acc = svm_cv.score(X_va, y_va)
        svm_accs.append(svm_acc)

        rf_cv = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=0,
        )
        rf_cv.fit(X_tr, y_tr)
        rf_acc = rf_cv.score(X_va, y_va)
        rf_accs.append(rf_acc)

        print(f"[Fold {fold_idx}] SVM acc={svm_acc:.3f} | RF acc={rf_acc:.3f}")

    print(f"\n=== 5-fold (same distribution) ===")
    print(f"SVM mean acc={np.mean(svm_accs):.3f} ± {np.std(svm_accs):.3f}")
    print(f"RF  mean acc={np.mean(rf_accs):.3f} ± {np.std(rf_accs):.3f}")

    # 4. 저장 (1st로 학습한 모델)
    os.makedirs("models", exist_ok=True)
    ml_bundle = {"svm": svm_clf, "rf": rf_clf}
    joblib.dump(ml_bundle, os.path.join("models", "ml_model.pkl"))
    print("Saved models/ml_model.pkl")


if __name__ == "__main__":
    main()
