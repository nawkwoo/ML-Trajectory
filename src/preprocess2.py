"""
Label-aware preprocessing with per-class axis scaling/selection.

기본 흐름:
1) Data/{label}/*.txt에서 7번째 컬럼(X/Y/Z)을 파싱해 (T,3) 궤적 로드
2) 원점 이동 → 스케일 정규화(최대거리=1) → 길이 100 선형 보간
3) 라벨별로 축 가중/제거 적용
4) results/preprocessed_data2/X.npy, y.npy, quality.npy 저장

주의: diagonal_right는 --dr-mode로 방법 A/B 선택
 - A: X*0.2, Y*0.7, Z=0
 - B: XY만 사용, Z 제거
"""

import glob
import os
from typing import Dict, Tuple

import numpy as np

LABELS = ["circle", "diagonal_left", "diagonal_right", "horizontal", "vertical"]

# (first_start, first_end, second_start, second_end)
SPLIT_RANGES: Dict[str, Tuple[int, int, int, int]] = {
    "circle": (1, 8, 9, 16),
    "diagonal_left": (1, 7, 8, 12),
    "diagonal_right": (1, 7, 8, 12),
    "horizontal": (1, 6, 7, 11),
    "vertical": (1, 6, 7, 11),
}


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


def resample_trajectory(traj: np.ndarray, target_len: int = 100) -> np.ndarray:
    T = len(traj)
    if T == target_len:
        return traj.copy()
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_len)
    out = np.zeros((target_len, 3), dtype=np.float32)
    for dim in range(3):
        out[:, dim] = np.interp(new_idx, old_idx, traj[:, dim])
    return out


def _parse_index(path: str) -> int | None:
    try:
        return int(os.path.splitext(os.path.basename(path))[0])
    except ValueError:
        return None


def _quality(label: str, idx: int) -> int | None:
    ranges = SPLIT_RANGES.get(label)
    if not ranges:
        return None
    f_s, f_e, s_s, s_e = ranges
    if f_s <= idx <= f_e:
        return 0
    if s_s <= idx <= s_e:
        return 1
    return None


def apply_label_weights(traj: np.ndarray, label: str, dr_mode: str = "A") -> np.ndarray:
    """
    라벨별 축 가중/제거 적용.
      circle        : X=1, Y=1, Z=0.5
      diagonal_left : X=0.5, Y=1.0, Z=0.7
      diagonal_right: mode A -> X=0.2, Y=0.7, Z=0
                      mode B -> XY만 사용, Z 제거
      horizontal    : X=1.0, Y=0.3, Z=0.3
      vertical      : X=0.4, Y=0.0, Z=1.0
    """
    t = traj.copy()
    if label == "circle":
        t[:, 2] *= 0.5
    elif label == "diagonal_left":
        t[:, 0] *= 0.5
        t[:, 2] *= 0.7
    elif label == "diagonal_right":
        if dr_mode.upper() == "A":
            t[:, 0] *= 0.2
            t[:, 1] *= 0.7
            t[:, 2] = 0.0
        else:  # mode B
            t = t[:, :2]  # drop Z
    elif label == "horizontal":
        t[:, 1] *= 0.3
        t[:, 2] *= 0.3
    elif label == "vertical":
        t[:, 0] *= 0.4
        t[:, 1] *= 0.0
    return t.astype(np.float32)


def build_dataset(
    data_root: str,
    save_dir: str,
    dr_mode: str = "A",
    target_len: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    q_list: list[int] = []

    label_to_idx = {label: i for i, label in enumerate(LABELS)}

    for label in LABELS:
        folder = os.path.join(data_root, label)
        paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
        for p in paths:
            idx = _parse_index(p)
            if idx is None:
                continue
            q = _quality(label, idx)
            if q is None:
                continue
            traj = load_trajectory(p)
            traj = normalize_origin(traj)
            traj = normalize_scale(traj)
            traj = resample_trajectory(traj, target_len=target_len)
            traj = apply_label_weights(traj, label, dr_mode=dr_mode)
            X_list.append(traj)
            y_list.append(label_to_idx[label])
            q_list.append(q)

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
    parser.add_argument("--save-dir", default=os.path.join("results", "preprocessed_data2"),
                        help="Output dir for npy files (default: results/preprocessed_data2)")
    parser.add_argument("--dr-mode", choices=["A", "B"], default="A",
                        help="Diagonal_right mode: A=XY weighted, Z=0 / B=XY only (drop Z)")
    args = parser.parse_args()

    X, y, q = build_dataset(
        data_root=args.data_root,
        save_dir=args.save_dir,
        dr_mode=args.dr_mode,
        target_len=100,
    )
    print(f"Saved preprocessed dataset to '{args.save_dir}'")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("quality shape:", q.shape)


if __name__ == "__main__":
    main()
