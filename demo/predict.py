"""
Inference script for ML-Trajectory (RF baseline, GOAL version).

- 모든 출력 PNG 및 로그 파일이 demo/goal/ 아래로 저장되도록 수정됨.
- 입력(raw TXT) → origin shift → scale normalize → resample → flatten → RF prediction
- 모델의 feature 차원(n_features_in_)을 통해 target_len 자동 추론
- 저장 파일:
    goal/prediction_summary.png
    goal/confidence_hist.png
    goal/predictions.txt
    goal/plots/*.png (각 trajectory 시각화)
"""

import argparse
import glob
import os
from typing import List

import joblib
import numpy as np
import matplotlib.pyplot as plt

LABELS = ["circle", "diagonal_left", "diagonal_right", "horizontal", "vertical"]


# ───────────────────────────────
# Utility functions
# ───────────────────────────────
def load_trajectory(path: str) -> np.ndarray:
    """TXT 파일에서 'X/Y/Z' 궤적 데이터를 읽어 (T,3) numpy array 반환."""
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
            xs.append(x)
            ys.append(y)
            zs.append(z)

    if not xs:
        raise ValueError(f"No valid trajectory in {path}")

    return np.stack([xs, ys, zs], axis=1)


def normalize_origin(traj: np.ndarray) -> np.ndarray:
    """시작점을 원점(0,0,0)으로 이동."""
    return traj - traj[0]


def normalize_scale(traj: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """원점으로부터 가장 먼 점이 거리 1이 되도록 정규화."""
    dists = np.linalg.norm(traj, axis=1)
    max_dist = np.max(dists)
    return traj / (max_dist + eps)


def resample(traj: np.ndarray, target_len: int) -> np.ndarray:
    """선형 보간으로 궤적을 target_len 길이로 리샘플."""
    T = len(traj)
    if T == target_len:
        return traj.copy()

    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_len)
    out = np.zeros((target_len, traj.shape[1]), dtype=np.float32)

    for i in range(traj.shape[1]):
        out[:, i] = np.interp(new_idx, old_idx, traj[:, i])

    return out


def collect_files(path: str) -> List[str]:
    """디렉토리 → *.txt 전부 수집, 파일 → 단일 리스트."""
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.txt")))
    return [path]


# ───────────────────────────────
# Main
# ───────────────────────────────
def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))     # demo/
    project_root = os.path.dirname(script_dir)                  # root/

    default_input = os.path.join(script_dir, "data")
    default_model = os.path.join(project_root, "models", "ml_model_rf.pkl")

    parser = argparse.ArgumentParser(
        description="Predict labels for raw TXT trajectories using trained RF model."
    )
    parser.add_argument("--input", default=default_input, help="TXT 파일 또는 디렉토리 경로")
    parser.add_argument("--model", default=default_model, help="학습된 RF 모델 경로")
    args = parser.parse_args()

    # ───────────────────────────────
    # Load model
    # ───────────────────────────────
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    rf = joblib.load(args.model)

    # 모델이 기대하는 feature dimension → 길이 자동 추론
    n_features = rf.n_features_in_
    n_channels = 3
    if n_features % n_channels != 0:
        raise ValueError(f"Model expects {n_features} features, not divisible by 3")

    target_len = n_features // n_channels
    print(f"[INFO] Model expects {n_features} features → target_len={target_len}")

    # ───────────────────────────────
    # Prepare GOAL dirs
    # ───────────────────────────────
    goal_dir = os.path.join(script_dir, "goal")
    plots_dir = os.path.join(goal_dir, "plots")

    os.makedirs(goal_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ───────────────────────────────
    # Collect TXT files
    # ───────────────────────────────
    files = collect_files(args.input)
    if not files:
        raise FileNotFoundError(f"No TXT files in {args.input}")

    print(f"[INFO] Found {len(files)} TXT files in '{args.input}'")

    pred_counts = {lbl: 0 for lbl in LABELS}
    confidences = []
    results = []

    # ───────────────────────────────
    # Prediction loop
    # ───────────────────────────────
    for path in files:
        traj = load_trajectory(path)
        traj = normalize_origin(traj)
        traj = normalize_scale(traj)
        traj = resample(traj, target_len)

        feat_vec = traj.reshape(1, -1)

        pred = rf.predict(feat_vec)[0]
        prob = rf.predict_proba(feat_vec)[0][pred]

        fname = os.path.basename(path)
        pred_label = LABELS[pred]

        results.append((fname, pred_label, float(prob)))
        pred_counts[pred_label] += 1
        confidences.append(float(prob))

        # ───────────────────────────────
        # Save trajectory plot
        # ───────────────────────────────
        plot_path = os.path.join(plots_dir, f"{fname}_plot.png")

        plt.figure(figsize=(4,3))
        plt.plot(traj[:,0], traj[:,1], label="XY path")
        plt.scatter(traj[0,0], traj[0,1], c="green", label="Start")
        plt.scatter(traj[-1,0], traj[-1,1], c="red", label="End")
        plt.title(f"{fname} → {pred_label}")
        plt.xlabel("X"); plt.ylabel("Y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

    # ───────────────────────────────
    # Print results in console
    # ───────────────────────────────
    print("\nPredictions:")
    for fname, lbl, p in results:
        print(f"{fname}\t→ {lbl} ({p:.3f})")

    # ───────────────────────────────
    # Save prediction log
    # ───────────────────────────────
    log_path = os.path.join(goal_dir, "predictions.txt")
    with open(log_path, "w") as f:
        for fname, lbl, p in results:
            f.write(f"{fname}\t→ {lbl}\t({p:.4f})\n")
    print(f"Saved prediction log → {log_path}")

    # ───────────────────────────────
    # Summary PNG
    # ───────────────────────────────
    out_summary = os.path.join(goal_dir, "prediction_summary.png")

    plt.figure(figsize=(6,4))
    plt.bar(pred_counts.keys(), pred_counts.values())
    plt.title("Prediction Summary")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_summary, dpi=200)
    plt.close()

    print(f"Saved summary PNG → {out_summary}")

    # ───────────────────────────────
    # Confidence histogram
    # ───────────────────────────────
    out_hist = os.path.join(goal_dir, "confidence_hist.png")

    plt.figure(figsize=(6,4))
    plt.hist(confidences, bins=10, range=(0.97,1.0))
    plt.title("Prediction Confidence Histogram")
    plt.xlabel("Max Probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_hist, dpi=200)
    plt.close()

    print(f"Saved confidence histogram → {out_hist}")


if __name__ == "__main__":
    main()
