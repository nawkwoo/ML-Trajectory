import os
import numpy as np


def load_trajectory(file_path: str) -> np.ndarray:
    """
    Parse the 7th column (index 6) containing 'X/Y/Z' into a (T, 3) numpy array.
    """
    xs, ys, zs = [], [], []

    with open(file_path, "r") as f:
        for line in f:
            cols = line.strip().split(",")

            # Skip if not enough columns
            if len(cols) <= 6:
                continue

            col = cols[6].strip()

            # Skip separators / comments / empty
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

    traj = np.stack([xs, ys, zs], axis=1)  # (T, 3)
    return traj


def normalize_origin(traj: np.ndarray) -> np.ndarray:
    """
    Shift trajectory so that the first point becomes the origin (0, 0, 0).
    """
    return traj - traj[0]


def normalize_scale(traj: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize by the maximum distance from origin → roughly within [-1, 1].
    """
    dists = np.linalg.norm(traj, axis=1)
    max_dist = np.max(dists)
    return traj / (max_dist + eps)


def resample_trajectory(traj: np.ndarray, target_len: int = 100) -> np.ndarray:
    """
    Linearly resample trajectory to length target_len.
    """
    T = len(traj)
    if T == target_len:
        return traj.copy()

    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_len)

    resampled = np.zeros((target_len, 3), dtype=np.float32)
    for dim in range(3):
        resampled[:, dim] = np.interp(new_idx, old_idx, traj[:, dim])
    return resampled


def preprocess(file_path: str, target_len: int = 100) -> np.ndarray:
    """
    Full preprocessing pipeline:
    load → origin normalize → scale normalize → resample.
    """
    traj = load_trajectory(file_path)
    traj = normalize_origin(traj)
    traj = normalize_scale(traj)
    traj = resample_trajectory(traj, target_len)
    return traj.astype(np.float32)


if __name__ == "__main__":
    # Example usage for quick sanity check
    example_path = os.path.join("Data", "circle", "1.txt")
    if os.path.exists(example_path):
        processed = preprocess(example_path, target_len=100)
        print(processed.shape)
        print(processed[:5])
    else:
        print("Example file not found:", example_path)

