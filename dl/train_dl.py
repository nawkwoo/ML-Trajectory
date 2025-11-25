"""PyTorch 기반 DL 베이스라인 (MPS 지원)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError as exc:  # pragma: no cover - 의존성 안내용
    sys.stderr.write("PyTorch, scikit-learn이 필요합니다. pip install torch scikit-learn\n")
    raise

from preprocess.augment import augment_sequence, apply_noise_profile
from preprocess.loader import compute_noise_profile, load_dataset

CoordSeq = np.ndarray


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        base_dir: Path,
        target_len: int,
        norm_mode: str,
        augment: bool = False,
        noise_profile: Dict[str, float] | None = None,
        synthetic_from_original: int = 1,
        noise_strength: float = 1.0,
    ) -> None:
        orig_dir = base_dir / "Data" / "Machine Learing(Original)"
        noise_dir = base_dir / "Data" / "Machine Learning(noise)"
        original = load_dataset(orig_dir, target_len=target_len, norm_mode=norm_mode)
        noisy = load_dataset(noise_dir, target_len=target_len, norm_mode=norm_mode)

        self.labels = sorted({label for _, label in original + noisy})
        self.label_to_idx = {lab: i for i, lab in enumerate(self.labels)}
        self.augment = augment
        self.noise_profile = noise_profile
        self.synthetic_from_original = synthetic_from_original
        self.noise_strength = noise_strength
        self.target_len = target_len

        data: List[Tuple[CoordSeq, int]] = []
        for seq, lab in original:
            data.append((seq, self.label_to_idx[lab]))
        for seq, lab in noisy:
            data.append((seq, self.label_to_idx[lab]))
        if noise_profile and synthetic_from_original > 0:
            for seq, lab in original:
                for _ in range(synthetic_from_original):
                    augmented = apply_noise_profile(seq, profile=noise_profile, strength=noise_strength)
                    data.append((augmented, self.label_to_idx[lab]))
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq, label = self.data[idx]
        if self.augment and self.noise_profile:
            seq = augment_sequence(
                seq,
                rotation_deg=5.0,
                stretch_range=(0.95, 1.05),
                noise_profile=self.noise_profile,
                noise_strength=self.noise_strength,
                mask_ratio=0.05,
                target_len=self.target_len,
                noise_sigma=2.0,
            )
        return torch.from_numpy(seq).float(), label


class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, conv_channels: Tuple[int, int] = (32, 64), num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_channels[0], kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        self.lstm = nn.LSTM(
            input_size=conv_channels[1],
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # (B, T, C)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        out = self.dropout(out)
        return self.fc(out)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="DL 베이스라인 학습 (CNN+BiLSTM)")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1], help="프로젝트 루트 경로")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "dl",
        help="모델/리포트를 저장할 경로",
    )
    parser.add_argument("--target-len", type=int, default=256, help="리샘플 길이")
    parser.add_argument("--norm", type=str, default="instance", choices=["instance", "global"], help="정규화 모드")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--synthetic-from-original", type=int, default=1)
    parser.add_argument("--noise-strength", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    ensure_dir(args.save_dir)

    profile = compute_noise_profile(
        original_dir=args.data_root / "Data" / "Machine Learing(Original)",
        noise_dir=args.data_root / "Data" / "Machine Learning(noise)",
        target_len=args.target_len,
        norm_mode=args.norm,
    )

    dataset = TrajectoryDataset(
        base_dir=args.data_root,
        target_len=args.target_len,
        norm_mode=args.norm,
        augment=True,
        noise_profile=profile,
        synthetic_from_original=args.synthetic_from_original,
        noise_strength=args.noise_strength,
    )

    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = CNNBiLSTM(input_dim=3, num_classes=len(dataset.labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    best_state = None
    patience = 10
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"[{epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # 저장
    if best_state is None:
        best_state = model.state_dict()
    ckpt_path = args.save_dir / "cnn_bilstm.pt"
    torch.save(best_state, ckpt_path)

    # 베스트 모델로 검증 세트 평가(지표 저장용)
    model.load_state_dict(best_state)
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x).cpu().numpy()
            all_logits.append(logits)
            all_labels.extend(y.numpy().tolist())
    all_logits_arr = np.concatenate(all_logits, axis=0)
    all_probs = torch.softmax(torch.from_numpy(all_logits_arr), dim=1).numpy()
    all_preds = all_probs.argmax(axis=1).tolist()

    report_dict = classification_report(all_labels, all_preds, target_names=dataset.labels, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    meta = {
        "labels": dataset.labels,
        "target_len": args.target_len,
        "norm": args.norm,
        "synthetic_from_original": args.synthetic_from_original,
        "noise_strength": args.noise_strength,
        "val_ratio": args.val_ratio,
        "epochs_trained": epoch,
        "best_val_loss": best_val,
        "val_classification_report": report_dict,
        "val_confusion_matrix": cm,
        "y_true": all_labels,
        "y_pred": all_preds,
        "y_proba": all_probs.tolist(),
        "device": str(device),
        "model": "CNN+BiLSTM",
    }
    with (args.save_dir / "dl_report.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"모델 저장: {ckpt_path}")
    print(f"리포트 저장: {args.save_dir / 'dl_report.json'}")


if __name__ == "__main__":
    main()
