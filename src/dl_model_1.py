import os

import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


def load_data():
    data_dir = os.path.join("results", "augmented_data")
    if not os.path.exists(os.path.join(data_dir, "X.npy")):
        raise FileNotFoundError("results/augmented_data/X.npy not found. Run `python src/augment_data.py` first.")
    print(f"Using data from '{data_dir}'")
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    q = np.load(os.path.join(data_dir, "quality.npy"))
    return X, y, q


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 80,
    lr: float = 1e-3,
    log_prefix: str = "",
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)

            acc = correct / total if total > 0 else 0.0
            print(f"{log_prefix}Epoch {epoch:03d} | loss={epoch_loss:.4f} | val_acc={acc:.3f}")

    # final validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0


def main():
    X, y, q = load_data()  # X: (N, 100, 3)

    # 채널 정규화
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X_norm = (X - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 동일 분포 5-fold 교차검증 (내부 품질 체크)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_accs: list[float] = []
    num_classes = int(y.max()) + 1
    for fold_idx, (tr, va) in enumerate(kf.split(X_norm), start=1):
        X_tr, y_tr = X_norm[tr], y[tr]
        X_va, y_va = X_norm[va], y[va]

        train_dataset = TrajectoryDataset(X_tr, y_tr)
        val_dataset = TrajectoryDataset(X_va, y_va)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        model = GRUClassifier(input_size=3, hidden_size=32, num_classes=num_classes)
        acc = train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            num_epochs=40,
            lr=1e-3,
            log_prefix=f"[Fold {fold_idx}] ",
        )
        print(f"[Fold {fold_idx}] final val_acc={acc:.3f}")
        fold_accs.append(acc)

    print("\n=== 5-fold (same distribution) ===")
    print("Fold accuracies:", [round(a, 3) for a in fold_accs])
    print(f"Mean acc={np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")

    # 2) 1st train -> 2nd test (분포 외 일반화 평가)
    train_mask = q == 0
    test_mask = q == 1
    X_train, y_train = X_norm[train_mask], y[train_mask]
    X_test, y_test = X_norm[test_mask], y[test_mask]

    print("\n=== 1st train -> 2nd test ===")
    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

    train_dataset = TrajectoryDataset(X_train, y_train)
    test_dataset = TrajectoryDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = GRUClassifier(input_size=3, hidden_size=32, num_classes=num_classes)
    acc = train_model(
        model,
        train_loader,
        test_loader,
        device=device,
        num_epochs=80,
        lr=1e-3,
        log_prefix="",
    )
    print(f"Final test_acc={acc:.3f}")

    # save checkpoint (1st로 학습한 모델)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", "dl_model_1.pt"))
    print("Saved models/dl_model_1.pt")


if __name__ == "__main__":
    main()
