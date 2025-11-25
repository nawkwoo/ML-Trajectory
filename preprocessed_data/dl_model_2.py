import os
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, T, 3), y: (N,)
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
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]  # (batch, hidden)
        logits = self.fc(last_hidden)
        return logits


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 60,
    lr: float = 1e-3,
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
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
            print(f"  Epoch {epoch:03d} | val_acc={acc:.3f}")

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


def train_full_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_epochs: int = 60,
    lr: float = 1e-3,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"[FULL] Epoch {epoch:03d} done")


def train_eval_split(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 80,
    lr: float = 1e-3,
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            acc = correct / total if total > 0 else 0.0
            print(f"[Holdout] Epoch {epoch:03d} | test_acc={acc:.3f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0


def main():
    # Seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Load dataset: augmented_data만 사용 (없으면 오류)
    data_dir = "augmented_data"
    if not os.path.exists(os.path.join(data_dir, "X.npy")):
        raise FileNotFoundError("augmented_data/X.npy not found. Run `python augment_data.py` first.")

    print(f"Using data from '{data_dir}'")
    X = np.load(os.path.join(data_dir, "X.npy"))  # (N, 100, 3)
    y = np.load(os.path.join(data_dir, "y.npy"))  # (N,)
    q = np.load(os.path.join(data_dir, "quality.npy"))  # (N,)

    # 채널 단위 표준화 (전체 데이터 기준)
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X_norm = (X - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_accuracies: list[float] = []

    num_classes = int(y.max()) + 1

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_norm), start=1):
        print(f"\n=== Fold {fold_idx} ===")
        X_train, y_train = X_norm[train_idx], y[train_idx]
        X_val, y_val = X_norm[val_idx], y[val_idx]

        train_dataset = TrajectoryDataset(X_train, y_train)
        val_dataset = TrajectoryDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        model = GRUClassifier(input_size=3, hidden_size=32, num_classes=num_classes)

        acc = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=60,
            lr=1e-3,
        )
        print(f"Fold {fold_idx} final val_acc={acc:.3f}")
        fold_accuracies.append(acc)

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    print("\n=== 5-fold cross-validation result ===")
    print("Fold accuracies:", [round(a, 3) for a in fold_accuracies])
    print(f"Mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

    # 1st train -> 2nd test (분포 외 일반화 평가)
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

    model_holdout = GRUClassifier(input_size=3, hidden_size=32, num_classes=num_classes)
    holdout_acc = train_eval_split(
        model_holdout,
        train_loader,
        test_loader,
        device=device,
        num_epochs=80,
        lr=1e-3,
    )
    print(f"Final 1st->2nd test_acc={holdout_acc:.3f}")

    # Train on full normalized dataset and save checkpoint
    full_dataset = TrajectoryDataset(X_norm, y)
    full_loader = DataLoader(full_dataset, batch_size=8, shuffle=True)
    model_full = GRUClassifier(input_size=3, hidden_size=32, num_classes=num_classes)
    train_full_model(model_full, full_loader, device=device, num_epochs=60, lr=1e-3)

    os.makedirs("models", exist_ok=True)
    torch.save(model_full.state_dict(), os.path.join("models", "dl_model_2.pt"))
    print("Saved models/dl_model_2.pt (trained on full dataset)")


if __name__ == "__main__":
    main()
