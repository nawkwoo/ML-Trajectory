import os
import random
from typing import Tuple

import numpy as np


def add_noise(x: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Gaussian noise injection."""
    noise = np.random.normal(0.0, sigma, size=x.shape)
    return x + noise


def time_shift(x: np.ndarray, max_shift: int = 10) -> np.ndarray:
    """Temporal shift within a fixed-length sequence."""
    T = x.shape[0]
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return x.copy()

    out = np.empty_like(x)
    if shift > 0:
        out[shift:] = x[:-shift]
        out[:shift] = x[0]
    else:
        shift = -shift
        out[:-shift] = x[shift:]
        out[-shift:] = x[-1]
    return out


def _resample(traj: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a trajectory (T, 3) to target_len using linear interpolation."""
    T = traj.shape[0]
    if T == target_len:
        return traj.copy()

    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_len)

    out = np.zeros((target_len, traj.shape[1]), dtype=np.float32)
    for dim in range(traj.shape[1]):
        out[:, dim] = np.interp(new_idx, old_idx, traj[:, dim])
    return out


def random_crop(x: np.ndarray, min_ratio: float = 0.7) -> np.ndarray:
    """
    Random crop + resample back to original length.

    Crops a contiguous subsequence (>= min_ratio * T) and rescales to length T.
    """
    T = x.shape[0]
    min_len = int(T * min_ratio)
    crop_len = np.random.randint(min_len, T + 1)
    start = np.random.randint(0, T - crop_len + 1)
    window = x[start : start + crop_len]
    return _resample(window, T)


def random_mask(x: np.ndarray, max_ratio: float = 0.2) -> np.ndarray:
    """Mask a contiguous segment with zeros."""
    T = x.shape[0]
    max_len = max(1, int(T * max_ratio))
    mask_len = np.random.randint(1, max_len + 1)
    start = np.random.randint(0, T - mask_len + 1)

    out = x.copy()
    out[start : start + mask_len] = 0.0
    return out


def random_rotate_xy(x: np.ndarray, max_angle_deg: float = 25.0) -> np.ndarray:
    """Small rotation in the XY-plane."""
    angle = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    out = x.copy()
    xy = out[:, :2] @ R.T
    out[:, :2] = xy
    return out


def mixup_same_class(
    x1: np.ndarray, x2: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    Mixup between two trajectories of the same class.

    Label은 그대로 유지하기 위해 같은 클래스끼리만 섞고,
    간단히 고정 비율(0.5) 또는 Beta 분포에서 샘플한 비율을 사용한다.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5
    return lam * x1 + (1.0 - lam) * x2


def build_augmented_dataset(
    X: np.ndarray,
    y: np.ndarray,
    quality: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    원본 전처리 데이터(X, y, quality)를 입력으로 받아,
    여러 증강 기법을 적용한 샘플을 추가한 새로운 데이터셋을 생성한다.

    적용되는 증강:
      - Noise injection
      - Temporal shift
      - Crop + resample
      - Masking
      - XY-plane rotation
      - Mixup (동일 클래스 샘플끼리)
    """
    X_list = [x.copy() for x in X]
    y_list = [int(label) for label in y]
    q_list = [int(q) for q in quality]

    # 클래스별 인덱스 모음 (mixup용)
    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(y):
        class_to_indices.setdefault(int(label), []).append(idx)

    num_samples = X.shape[0]

    for i in range(num_samples):
        x = X[i]
        label = int(y[i])
        q = int(quality[i])

        # 1) Noise
        X_list.append(add_noise(x))
        y_list.append(label)
        q_list.append(q)

        # 2) Temporal shift
        X_list.append(time_shift(x))
        y_list.append(label)
        q_list.append(q)

        # 3) Crop + resample
        X_list.append(random_crop(x))
        y_list.append(label)
        q_list.append(q)

        # 4) Masking
        X_list.append(random_mask(x))
        y_list.append(label)
        q_list.append(q)

        # 5) Small rotation
        X_list.append(random_rotate_xy(x))
        y_list.append(label)
        q_list.append(q)

        # 6) Mixup with another sample of same class (if possible)
        same_class_indices = class_to_indices.get(label, [])
        if len(same_class_indices) > 1:
            j = i
            # pick a different index with the same label
            while j == i:
                j = random.choice(same_class_indices)
            x2 = X[j]
            X_list.append(mixup_same_class(x, x2))
            y_list.append(label)
            q_list.append(q)

    X_aug = np.stack(X_list, axis=0).astype(np.float32)
    y_aug = np.array(y_list, dtype=np.int64)
    q_aug = np.array(q_list, dtype=np.int64)

    return X_aug, y_aug, q_aug


def main() -> None:
    np.random.seed(0)
    random.seed(0)

    # Load preprocessed base dataset
    X = np.load(os.path.join("preprocessed_data", "X.npy"))
    y = np.load(os.path.join("preprocessed_data", "y.npy"))
    quality = np.load(os.path.join("preprocessed_data", "quality.npy"))

    print("Original dataset:", X.shape, y.shape, quality.shape)

    X_aug, y_aug, q_aug = build_augmented_dataset(X, y, quality)

    print("Augmented dataset:", X_aug.shape, y_aug.shape, q_aug.shape)

    out_dir = "augmented_data"
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X.npy"), X_aug)
    np.save(os.path.join(out_dir, "y.npy"), y_aug)
    np.save(os.path.join(out_dir, "quality.npy"), q_aug)

    print(f"Saved augmented dataset to '{out_dir}'")


if __name__ == "__main__":
    main()

