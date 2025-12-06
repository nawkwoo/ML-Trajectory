"""
Inference script using trained RF model (models/ml_modle.pkl).
Steps:
 1) Load raw TXT (comma-separated, 7th column 'X/Y/Z')
 2) Origin shift -> scale normalize -> resample to 128
 3) Apply label-specific axis weights (same as preprocess.py, dr-mode A/B)
 4) Flatten and predict with RF

Usage:
    python src/predict.py --input path/to/file_or_dir --dr-mode A --weight-scale 1.0
"""

import argparse
import glob
import os
from typing import List

import joblib
import numpy as np

LABELS = ["circle", "diagonal_left", "diagonal_right", "horizontal", "vertical"]


def load_trajectory(path: str) -> np.ndarray:
    xs, ys, zs = [], [], []
    with open(path, "r") as f:
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
            xs.append(x); ys.append(y); zs.append(z)
    if not xs:
        raise ValueError(f"No valid trajectory data in {path}")
    return np.stack([xs, ys, zs], axis=1)


def normalize_origin(traj: np.ndarray) -> np.ndarray:
    return traj - traj[0]


def normalize_scale(traj: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    dists = np.linalg.norm(traj, axis=1)
    max_dist = np.max(dists)
    return traj / (max_dist + eps)


def resample(traj: np.ndarray, target_len: int = 128) -> np.ndarray:
    T = len(traj)
    if T == target_len:
        return traj.copy()
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_len)
    out = np.zeros((target_len, traj.shape[1]), dtype=np.float32)
    for dim in range(3):
        out[:, dim] = np.interp(new_idx, old_idx, traj[:, dim])
    return out


def apply_label_weights(traj: np.ndarray, label: str, dr_mode: str = "A", scale: float = 1.0) -> np.ndarray:
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
        return (traj[:, :2] * weights[:2]).astype(np.float32)
    return (traj * weights).astype(np.float32)


def collect_files(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, "*.txt")))
    return [input_path]


def main():
    parser = argparse.ArgumentParser(description="Predict labels for raw TXT trajectories using RF model.")
    parser.add_argument("--input", required=True, help="Input TXT file or directory of TXT files")
    parser.add_argument("--model", default=os.path.join("models", "ml_modle.pkl"),
                        help="Path to trained RF model (default: models/ml_modle.pkl)")
    parser.add_argument("--dr-mode", choices=["A", "B"], default="A",
                        help="Diagonal_right mode: A=XY weighted, Z kept / B=XY only (drop Z)")
    parser.add_argument("--weight-scale", type=float, default=1.0,
                        help="Global multiplier for axis weights (default: 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    rf = joblib.load(args.model)

    files = collect_files(args.input)
    if not files:
        raise FileNotFoundError(f"No TXT files found at {args.input}")

    results = []
    for path in files:
        traj = load_trajectory(path)
        traj = normalize_origin(traj)
        traj = normalize_scale(traj)
        traj = resample(traj, target_len=128)

        # If diagonal_right, apply that rule; otherwise apply its own rule.
        # For inference without label, apply all rules and keep channel size consistent by using the class weights as "views".
        # Here, we choose to apply each class rule and build a stacked feature vector so RF can use all views.
        feats = []
        for lbl in LABELS:
            t = apply_label_weights(traj, lbl, dr_mode=args.dr_mode, scale=args.weight_scale)
            # pad to 3 channels if dropped Z
            if t.shape[1] < 3:
                pad = np.zeros((t.shape[0], 3 - t.shape[1]), dtype=t.dtype)
                t = np.concatenate([t, pad], axis=1)
            feats.append(t.reshape(-1))
        feat_vec = np.concatenate(feats)  # stacked per-label view

        pred = rf.predict(feat_vec.reshape(1, -1))[0]
        prob = rf.predict_proba(feat_vec.reshape(1, -1))[0]
        results.append((os.path.basename(path), LABELS[pred], prob[pred]))

    print("Predictions (file, label, prob):")
    for fname, lbl, p in results:
        print(f"{fname}\t{lbl}\t{p:.3f}")


if __name__ == "__main__":
    main()
