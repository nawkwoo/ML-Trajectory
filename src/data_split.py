"""
Split augmented data (Data/results/augmented_data) into train/test (stratified 8:2).
Saves to Data/results/split_data: X_train, y_train, quality_train, X_test, y_test, quality_test.
"""

import os

import numpy as np
from sklearn.model_selection import train_test_split


def main():
    data_dir = os.path.join("Data", "results", "augmented_data")
    if not os.path.exists(os.path.join(data_dir, "X.npy")):
        raise FileNotFoundError("Data/results/augmented_data/X.npy not found. Run `python src/augment_data.py` first.")

    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    q = np.load(os.path.join(data_dir, "quality.npy"))

    X_train, X_test, y_train, y_test, q_train, q_test = train_test_split(
        X, y, q, test_size=0.2, random_state=0, stratify=y
    )

    out_dir = os.path.join("Data", "results", "split_data")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "quality_train.npy"), q_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "quality_test.npy"), q_test)

    print(f"Saved split data to '{out_dir}'")
    print("Train:", X_train.shape, "Test:", X_test.shape)


if __name__ == "__main__":
    main()
