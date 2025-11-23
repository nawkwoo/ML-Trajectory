"""
Train and save ML/DL models with clear filenames:
 - models/ml_svm.pkl     : SVM trained on 1st (quality=0)
 - models/ml_rf.pkl      : RandomForest trained on 1st (quality=0)
 - models/dl1_gru_first.pt : GRU trained on 1st (quality=0)
 - models/dl2_gru_aug.pt   : GRU trained on augmented_data (if present) else preprocessed_data

Usage (from repo root):
    python save_models.py

Dependencies: numpy, torch, scikit-learn, joblib
"""

import os
import numpy as np
import joblib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


LABELS = ["circle", "diagonal_left", "diagonal_right", "horizontal", "vertical"]


# ------------------------
# Dataset utilities
# ------------------------

class TrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]  # (batch, hidden)
        return self.fc(last_hidden)


def load_base_data():
    X = np.load(os.path.join("preprocessed_data", "X.npy"))
    y = np.load(os.path.join("preprocessed_data", "y.npy"))
    q = np.load(os.path.join("preprocessed_data", "quality.npy"))
    return X, y, q


def load_aug_data():
    X = np.load(os.path.join("augmented_data", "X.npy"))
    y = np.load(os.path.join("augmented_data", "y.npy"))
    q = np.load(os.path.join("augmented_data", "quality.npy"))
    return X, y, q


# ------------------------
# ML models (SVM / RF)
# ------------------------

def train_ml_models():
    X, y, q = load_base_data()
    X_flat = X.reshape(X.shape[0], -1)

    train_mask = q == 0  # 1st (clean) only
    X_train, y_train = X_flat[train_mask], y[train_mask]

    print(f"[ML] Train size: {X_train.shape[0]}")

    svm_clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0),
    )
    svm_clf.fit(X_train, y_train)

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=0,
    )
    rf_clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(svm_clf, os.path.join("models", "ml_svm.pkl"))
    joblib.dump(rf_clf, os.path.join("models", "ml_rf.pkl"))
    print("[ML] Saved models/ml_svm.pkl and models/ml_rf.pkl")


# ------------------------
# DL models (GRU)
# ------------------------

def train_gru(
    X: np.ndarray,
    y: np.ndarray,
    model_path: str,
    num_epochs: int = 80,
    batch_size: int = 16,
    lr: float = 1e-3,
):
    # channel-wise normalization
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X_norm = (X - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TrajectoryDataset(X_norm, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = int(y.max()) + 1
    model = GRUClassifier(input_size=3, hidden_size=32, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[DL] Epoch {epoch:03d} | loss={epoch_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[DL] Saved {model_path}")


def train_dl1():
    # Train GRU on 1st (quality=0) only
    X, y, q = load_base_data()
    train_mask = q == 0
    X_train, y_train = X[train_mask], y[train_mask]
    print(f"[DL1] Train size (1st only): {X_train.shape[0]}")
    train_gru(X_train, y_train, model_path=os.path.join("models", "dl1_gru_first.pt"))


def train_dl2():
    # Train GRU on augmented_data if exists, else preprocessed_data (all)
    if os.path.exists(os.path.join("augmented_data", "X.npy")):
        X, y, _ = load_aug_data()
        source = "augmented_data"
    else:
        X, y, _ = load_base_data()
        source = "preprocessed_data"
    print(f"[DL2] Train size (all data from {source}): {X.shape[0]}")
    train_gru(X, y, model_path=os.path.join("models", "dl2_gru_aug.pt"))


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    train_ml_models()
    train_dl1()
    train_dl2()
    print("Done. Models saved in 'models/'")


if __name__ == "__main__":
    main()

