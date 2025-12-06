"""
Augment trajectories from Data/results/preprocessed_data and save to Data/results/augmented_data.
Basic methods: noise, time shift, crop+resample, masking, XY rotation, same-class mixup.
Default: all qualities(quality is 0 only here) are augmented; originals are kept.
"""

import argparse
import os
import random
from typing import Optional, Tuple

import numpy as np


def add_noise(x: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, size=x.shape)
    return x + noise


def time_shift(x: np.ndarray, max_shift: int = 10) -> np.ndarray:
    T = x.shape[0]
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return x.copy()
    out = np.empty_like(x)
    if shift > 0:
        out[shift:] = x[:-shift]
        out[:shift] = x[0]
    else:
        s = -shift
        out[:-s] = x[s:]
        out[-s:] = x[-1]
    return out


def _resample(traj: np.ndarray, target_len: int) -> np.ndarray:
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
    T = x.shape[0]
    min_len = int(T * min_ratio)
    crop_len = np.random.randint(min_len, T + 1)
    start = np.random.randint(0, T - crop_len + 1)
    window = x[start : start + crop_len]
    return _resample(window, T)


def random_mask(x: np.ndarray, max_ratio: float = 0.1) -> np.ndarray:
    T = x.shape[0]
    max_len = max(1, int(T * max_ratio))
    mask_len = np.random.randint(1, max_len + 1)
    start = np.random.randint(0, T - mask_len + 1)
    out = x.copy()
    out[start : start + mask_len] = 0.0
    return out


def random_rotate_xy(x: np.ndarray, max_angle_deg: float = 20.0) -> np.ndarray:
    if x.shape[1] < 2:
        return x.copy()
    angle = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    out = x.copy()
    out_xy = out[:, :2] @ R.T
    out[:, :2] = out_xy
    return out


def mixup_same_class(x1: np.ndarray, x2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5
    return lam * x1 + (1.0 - lam) * x2


def build_augmented_dataset(
    X: np.ndarray,
    y: np.ndarray,
    quality: np.ndarray,
    quality_filter: Optional[int] = None,
    noise_sigma: float = 0.01,
    mask_max_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list = [x.copy() for x in X]  # originals
    y_list = [int(label) for label in y]
    q_list = [int(q) for q in quality]

    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(y):
        if quality_filter is not None and int(quality[idx]) != quality_filter:
            continue
        class_to_indices.setdefault(int(label), []).append(idx)

    eligible = [
        i for i in range(X.shape[0])
        if quality_filter is None or int(quality[i]) == quality_filter
    ]

    for i in eligible:
        x = X[i]
        label = int(y[i])
        q = int(quality[i])

        X_list.append(add_noise(x, sigma=noise_sigma)); y_list.append(label); q_list.append(q)
        X_list.append(time_shift(x));                y_list.append(label); q_list.append(q)
        X_list.append(random_crop(x));               y_list.append(label); q_list.append(q)
        X_list.append(random_mask(x, max_ratio=mask_max_ratio)); y_list.append(label); q_list.append(q)
        X_list.append(random_rotate_xy(x));          y_list.append(label); q_list.append(q)

        same_class_idx = class_to_indices.get(label, [])
        if len(same_class_idx) > 1:
            j = i
            while j == i:
                j = random.choice(same_class_idx)
            x2 = X[j]
            X_list.append(mixup_same_class(x, x2)); y_list.append(label); q_list.append(q)

    X_aug = np.stack(X_list, axis=0).astype(np.float32)
    y_aug = np.array(y_list, dtype=np.int64)
    q_aug = np.array(q_list, dtype=np.int64)
    return X_aug, y_aug, q_aug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment trajectories from Data/results/preprocessed_data",
    )
    parser.add_argument(
        "--quality-filter",
        choices=["0", "1", "all"],
        default="all",
        help="Which quality to augment: 0=1st only, 1=2nd only, all=both (default)",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.01,
        help="Stddev for Gaussian noise (default: 0.01)",
    )
    parser.add_argument(
        "--mask-max-ratio",
        type=float,
        default=0.1,
        help="Max masked segment ratio (default: 0.1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality_filter: Optional[int]
    if args.quality_filter == "all":
        quality_filter = None
    else:
        quality_filter = int(args.quality_filter)

    np.random.seed(0)
    random.seed(0)

    base_dir = os.path.join("Data", "results", "preprocessed_data")
    if not os.path.exists(os.path.join(base_dir, "X.npy")):
        raise FileNotFoundError("Data/results/preprocessed_data/X.npy not found. Run `python src/preprocess.py` first.")

    X = np.load(os.path.join(base_dir, "X.npy"))
    y = np.load(os.path.join(base_dir, "y.npy"))
    quality = np.load(os.path.join(base_dir, "quality.npy"))

    print("Original dataset:", X.shape, y.shape, quality.shape)
    target_desc = "all (1st + 2nd)" if quality_filter is None else f"quality={quality_filter}"
    print(f"Augmenting targets: {target_desc}")
    print(f"noise_sigma={args.noise_sigma}, mask_max_ratio={args.mask_max_ratio}")

    X_aug, y_aug, q_aug = build_augmented_dataset(
        X,
        y,
        quality,
        quality_filter=quality_filter,
        noise_sigma=args.noise_sigma,
        mask_max_ratio=args.mask_max_ratio,
    )

    print("Augmented dataset:", X_aug.shape, y_aug.shape, q_aug.shape)

    out_dir = os.path.join("Data", "results", "augmented_data")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X.npy"), X_aug)
    np.save(os.path.join(out_dir, "y.npy"), y_aug)
    np.save(os.path.join(out_dir, "quality.npy"), q_aug)

    print(f"Saved augmented dataset to '{out_dir}'")


if __name__ == "__main__":
    main()
