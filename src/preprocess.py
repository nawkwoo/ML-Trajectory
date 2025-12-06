"""
Label-aware preprocessing with per-class axis scaling/selection.

기본 흐름:
1) Data/{label}/*.txt에서 7번째 컬럼(X/Y/Z)을 파싱해 (T,3) 궤적 로드
2) 원점 이동 → 스케일 정규화(최대거리=1) → 길이 100 선형 보간
3) 라벨별로 축 가중/제거 적용 (제안 비율: max=1, 전체 강도는 scale로 조절)
4) Data/results/preprocessed_data/X.npy, y.npy, quality.npy 저장 (quality=0 통일)

라벨별 축 규칙:
- circle        : X=0.2,  Y=1.0, Z=0.7
- diagonal_left : X=0.55, Y=1.0, Z=0.9
- diagonal_right: dr-mode A) X=0.16, Y=0.86, Z=1.0 / dr-mode B) XY만 사용, Z 삭제
- horizontal    : X=0.5,  Y=1.0, Z=0.0
- vertical      : X=0.4,  Y=0.15, Z=1.0
"""

import glob
import os
from typing import Tuple
import numpy as np

LABELS = ["circle", "diagonal_left", "diagonal_right", "horizontal", "vertical"]


def load_trajectory(file_path: str) -> np.ndarray:
    xs, ys, zs = [], [], []
    with open(file_path, "r") as f:
        for line in f:
            cols = line.strip().split(",")
            if len(cols) <= 6:
                continue
            col = cols[6].strip()
            if not col or col[0] in ("s", "S", "#"):
                continue
            try:
                x, y, z = map(float, col.split("/"))
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
            zs.append(z)
    if not xs:
        raise ValueError(f"No valid trajectory data found in {file_path}")
    return np.stack([xs, ys, zs], axis=1)  # (T,3)


def normalize_origin(traj: np.ndarray) -> np.ndarray:
    return traj - traj[0]


def normalize_scale(traj: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    dists = np.linalg.norm(traj, axis=1)
    max_dist = np.max(dists)
    return traj / (max_dist + eps)


def resample_trajectory(traj: np.ndarray, target_len: int = 128) -> np.ndarray:
    T = len(traj)
    if T == target_len:
        return traj.copy()
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_len)
    out = np.zeros((target_len, 3), dtype=np.float32)
    for dim in range(3):
        out[:, dim] = np.interp(new_idx, old_idx, traj[:, dim])
    return out


def apply_label_weights(traj: np.ndarray, label: str, dr_mode: str = "A", scale: float = 1.0) -> np.ndarray:
    """
    라벨별 축 가중/제거 적용 (비율은 max=1 기준, 강도는 scale로 조절).
      circle        : X=0.2,  Y=1.0, Z=0.7
      diagonal_left : X=0.55, Y=1.0, Z=0.9
      diagonal_right: mode A -> X=0.16, Y=0.86, Z=1.0 (Z 유지)
                      mode B -> XY만 사용, Z 제거
      horizontal    : X=0.5,  Y=1.0, Z=0.0 (Z 무시)
      vertical      : X=0.4,  Y=0.15, Z=1.0
    """
    base = {
        "circle": (0.2, 1.0, 0.7),
        "diagonal_left": (0.55, 1.0, 0.9),
        "diagonal_right": (0.16, 0.86, 1.0),
        "horizontal": (0.5, 1.0, 0.0),
        "vertical": (0.4, 0.15, 1.0),
    }

    if label not in base:
        return traj.astype(np.float32)

    weights = np.asarray(base[label], dtype=np.float32) * float(scale)

    if label == "diagonal_right" and dr_mode.upper() == "B":
        # Z 제거, XY 비율만 적용
        return (traj[:, :2] * weights[:2]).astype(np.float32)

    return (traj * weights).astype(np.float32)


def build_dataset(
    data_root: str,
    save_dir: str,
    dr_mode: str = "A",
    target_len: int = 128,
    weight_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    q_list: list[int] = []

    label_to_idx = {label: i for i, label in enumerate(LABELS)}

    for label in LABELS:
        folder = os.path.join(data_root, label)
        paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
        for p in paths:
            traj = load_trajectory(p)
            traj = normalize_origin(traj)
            traj = normalize_scale(traj)
            traj = resample_trajectory(traj, target_len=target_len)
            traj = apply_label_weights(traj, label, dr_mode=dr_mode, scale=weight_scale)
            X_list.append(traj)
            y_list.append(label_to_idx[label])
            q_list.append(0)

    if not X_list:
        raise ValueError(f"No trajectories built from {data_root}")

    # Pad to 3 channels if some trajectories became 2D (diagonal_right mode B)
    max_channels = max(t.shape[1] for t in X_list)
    padded = []
    for t in X_list:
        if t.shape[1] == max_channels:
            padded.append(t)
        else:
            pad = np.zeros((t.shape[0], max_channels - t.shape[1]), dtype=t.dtype)
            padded.append(np.concatenate([t, pad], axis=1))

    X = np.stack(padded, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    quality = np.array(q_list, dtype=np.int64)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)
    np.save(os.path.join(save_dir, "quality.npy"), quality)

    return X, y, quality


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Label-aware preprocessing with per-class axis weights",
    )
    parser.add_argument("--data-root", default="Data", help="Input data root (default: Data)")
    parser.add_argument(
        "--save-dir",
        default=os.path.join("Data", "results", "preprocessed_data"),
        help="Output dir for npy files (default: Data/results/preprocessed_data)",
    )
    parser.add_argument("--dr-mode", choices=["A", "B"], default="A",
                        help="Diagonal_right mode: A=XY weighted, Z kept / B=XY only (drop Z)")
    parser.add_argument("--weight-scale", type=float, default=1.0,
                        help="Global multiplier for axis weights (default: 1.0)")
    args = parser.parse_args()

    X, y, q = build_dataset(
        data_root=args.data_root,
        save_dir=args.save_dir,
        dr_mode=args.dr_mode,
        target_len=128,
        weight_scale=args.weight_scale,
    )
    print(f"Saved preprocessed dataset to '{args.save_dir}'")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("quality shape:", q.shape)


if __name__ == "__main__":
    main()
