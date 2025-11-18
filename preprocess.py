import os
import glob
import numpy as np


LABELS = ["circle", "diagonal_left", "diagonal_right", "horizontal", "vertical"]

# (first_start, first_end, second_start, second_end)
SPLIT_RANGES: dict[str, tuple[int, int, int, int]] = {
    "circle": (1, 8, 9, 16),
    "diagonal_left": (1, 7, 8, 12),
    "diagonal_right": (1, 7, 8, 12),
    "horizontal": (1, 6, 7, 11),
    "vertical": (1, 6, 7, 11),
}


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


def _parse_index_from_path(path: str) -> int | None:
    """
    Extract integer index from a filename like '3.txt'.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(name)
    except ValueError:
        return None


def _quality_from_index(label: str, index: int) -> int | None:
    """
    Map file index to quality:
        0 -> first (clean)
        1 -> second (noisy)
    """
    ranges = SPLIT_RANGES.get(label)
    if ranges is None:
        return None

    first_start, first_end, second_start, second_end = ranges
    if first_start <= index <= first_end:
        return 0
    if second_start <= index <= second_end:
        return 1
    return None


def build_dataset(
    data_root: str = "Data",
    labels: list[str] | None = None,
    target_len: int = 100,
    save_dir: str | None = None,
    split: str = "both",
    include_quality: bool = False,
):
    """
    Build dataset from folder structure like:
        Data/{label}/*.txt

    Parameters
    ----------
    split : {"first", "second", "both"}
        - "first"  : only clean (1st) recordings
        - "second" : only noisy (2nd) recordings
        - "both"   : use both and keep their quality labels.

    include_quality : bool
        If True, also return quality array (0 for first, 1 for second).

    Returns
    -------
    X : (N, target_len, 3)
    y : (N,) integer class labels
    quality (optional) : (N,) integer quality labels (0/1)

    If save_dir is given, saves X.npy and y.npy (and quality.npy if requested).
    """
    if labels is None:
        labels = LABELS

    if split not in ("first", "second", "both"):
        raise ValueError("split must be one of: 'first', 'second', 'both'")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    q_list: list[int] = []

    label_to_idx = {label: i for i, label in enumerate(labels)}

    for label in labels:
        folder = os.path.join(data_root, label)
        paths = sorted(glob.glob(os.path.join(folder, "*.txt")))

        for path in paths:
            idx = _parse_index_from_path(path)
            if idx is None:
                continue

            quality = _quality_from_index(label, idx)
            if quality is None:
                continue

            if split == "first" and quality != 0:
                continue
            if split == "second" and quality != 1:
                continue

            traj = preprocess(path, target_len=target_len)
            X_list.append(traj)
            y_list.append(label_to_idx[label])
            q_list.append(quality)

    if not X_list:
        raise ValueError(f"No trajectory files found under {data_root} for split={split}")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    quality_arr = np.array(q_list, dtype=np.int64)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X.npy"), X)
        np.save(os.path.join(save_dir, "y.npy"), y)
        if include_quality:
            np.save(os.path.join(save_dir, "quality.npy"), quality_arr)

    if include_quality:
        return X, y, quality_arr
    return X, y


if __name__ == "__main__":
    # When run as a script, build the full dataset (both qualities)
    X, y, q = build_dataset(
        data_root="Data",
        labels=LABELS,
        target_len=100,
        save_dir="preprocessed_data",
        split="both",
        include_quality=True,
    )
    print("Saved preprocessed dataset to 'preprocessed_data'")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("quality shape:", q.shape)
