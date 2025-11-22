import numpy as np
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


def main():
    # Load preprocessed arrays
    X = np.load("preprocessed_data/X.npy")  # (N, 100, 3)
    y = np.load("preprocessed_data/y.npy")  # (N,)
    q = np.load("preprocessed_data/quality.npy")  # (N,) 0=first, 1=second

    # Train on first (clean) set, test on second (noisy) set
    train_mask = q == 0
    test_mask = q == 1

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

    train_dataset = TrajectoryDataset(X_train, y_train)
    test_dataset = TrajectoryDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUClassifier(input_size=3, hidden_size=32, num_classes=len(np.unique(y)))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 80

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

        epoch_loss = running_loss / len(train_dataset)

        if epoch % 10 == 0 or epoch == 1:
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
            print(f"Epoch {epoch:03d} | loss={epoch_loss:.4f} | test_acc={acc:.3f}")


if __name__ == "__main__":
    main()

