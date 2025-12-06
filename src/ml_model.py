import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold


def load_data():
    split_dir = os.path.join("Data", "results", "split_data")
    if not os.path.exists(os.path.join(split_dir, "X_train.npy")):
        raise FileNotFoundError("Data/results/split_data/X_train.npy not found. Run `python src/data_split.py` first.")
    print(f"Using split data from '{split_dir}'")
    X_train = np.load(os.path.join(split_dir, "X_train.npy"))
    y_train = np.load(os.path.join(split_dir, "y_train.npy"))
    X_test = np.load(os.path.join(split_dir, "X_test.npy"))
    y_test = np.load(os.path.join(split_dir, "y_test.npy"))
    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = load_data()

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print("Train size:", X_train.shape[0], " Test size:", X_test.shape[0])

    # RandomForest
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=0,
    )
    rf_clf.fit(X_train_flat, y_train)
    y_pred_rf = rf_clf.predict(X_test_flat)

    print("\n=== RandomForest (train -> test) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

    # 5-fold CV on train set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    rf_accs = []
    for fold_idx, (tr, va) in enumerate(skf.split(X_train_flat, y_train), start=1):
        X_tr, y_tr = X_train_flat[tr], y_train[tr]
        X_va, y_va = X_train_flat[va], y_train[va]

        rf_cv = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=0,
        )
        rf_cv.fit(X_tr, y_tr)
        rf_accs.append(rf_cv.score(X_va, y_va))

        print(f"[Fold {fold_idx}] RF acc={rf_accs[-1]:.3f}")

    print("\n=== 5-fold (train set) ===")
    print(f"RF  mean acc={np.mean(rf_accs):.3f} ± {np.std(rf_accs):.3f}")

    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_clf, os.path.join("models", "ml_modle.pkl"))
    print("Saved models/ml_modle.pkl (RF only)")


if __name__ == "__main__":
    main()
