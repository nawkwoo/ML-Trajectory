"""
Inference script for ML-Trajectory (RF baseline).

- --input 은 OPTIONAL (기본: demo/data/)
- 모델의 입력 차원에 맞춰 자동으로 시퀀스 길이(target_len)를 결정
- Train 시점: (T, 3) → flatten 해서 RandomForest 학습
- Inference 시점: raw TXT → normalize → resample(target_len) → flatten 후 예측

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
    """
    하나의 TXT 파일에서 end-effector 궤적을 읽어 (T, 3) 배열로 반환한다.

    - 각 줄은 콤마(,)로 구분된 여러 컬럼으로 구성되어 있으며, 7번째 컬럼(인덱스 6)에 'X/Y/Z' 형태의 문자열이 있다고 가정한다.
    - s, S, # 로 시작하는 줄은 헤더/코멘트로 간주하고 무시한다.
    """
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

    return np.stack([xs, ys, zs], axis=1)  # (T, 3)


def normalize_origin(traj: np.ndarray) -> np.ndarray:
    """궤적의 시작점을 원점(0, 0, 0)으로 이동한다."""
    return traj - traj[0]


def normalize_scale(traj: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    궤적을 '원점으로부터의 최대 거리' 기준으로 스케일 정규화한다.
    가장 먼 점의 거리가 1에 가깝도록 전체를 (max_dist + eps)로 나눈다.
    """
    dists = np.linalg.norm(traj, axis=1)
    max_dist = np.max(dists)
    return traj / (max_dist + eps)


def resample(traj: np.ndarray, target_len: int) -> np.ndarray:
    """
    궤적을 선형 보간하여 고정 길이(target_len)로 리샘플링한다.

    Parameters
    ----------
    traj : np.ndarray
        shape (T, 3)의 궤적
    target_len : int
        리샘플링 후 길이

    Returns
    -------
    np.ndarray
        shape (target_len, 3)의 리샘플링된 궤적
    """
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
    """
    디렉토리일 경우 *.txt 파일을 모두 모으고,
    파일일 경우 해당 경로만 리스트로 반환한다.
    """
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.txt")))
    return [path]


# ───────────────────────────────
# Main
# ───────────────────────────────
def main() -> None:
    """
    학습된 RandomForest 모델(models/ml_model_rf.pkl)을 사용해
    TXT 궤적 파일들을 분류하고, 결과 요약 PNG를 저장한다.

    - 모델의 n_features_in_을 읽어 feature 차원에 맞게 리샘플 길이(target_len)를 자동으로 결정한다.
    """
    # demo/predict.py 기준 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    default_input = os.path.join(script_dir, "data")  # demo/data/
    default_model = os.path.join(project_root, "models", "ml_model_rf.pkl")

    parser = argparse.ArgumentParser(
        description="Predict labels for raw TXT trajectories using trained RF model."
    )
    parser.add_argument(
        "--input",
        default=default_input,
        help=f"TXT 파일 또는 디렉토리 경로 (기본값: {default_input})",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"RF 모델 경로 (기본값: {default_model})",
    )
    args = parser.parse_args()

    # 모델 로드
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    rf = joblib.load(args.model)

    # 모델이 기대하는 feature 차원으로부터 target_len 추론
    n_features = rf.n_features_in_
    n_channels = 3  # (x, y, z)
    if n_features % n_channels != 0:
        raise ValueError(
            f"Model expects {n_features} features, which is not divisible by {n_channels} channels."
        )
    target_len = n_features // n_channels
    print(f"[INFO] Model expects {n_features} features → target_len={target_len}, channels={n_channels}")

    # 입력 파일 수집
    files = collect_files(args.input)
    if not files:
        raise FileNotFoundError(f"No TXT files in {args.input}")

    print(f"[INFO] Loaded model: {args.model}")
    print(f"[INFO] Found {len(files)} TXT files in {args.input}")

    pred_counts = {lbl: 0 for lbl in LABELS}
    results: list[tuple[str, str, float]] = []

    for path in files:
        traj = load_trajectory(path)
        traj = normalize_origin(traj)
        traj = normalize_scale(traj)
        traj = resample(traj, target_len=target_len)  # 길이 모델에 맞춤

        # 여기서는 axis weighting 미적용: (target_len, 3) → 1D 벡터
        feat_vec = traj.reshape(1, -1)

        pred = rf.predict(feat_vec)[0]
        prob = rf.predict_proba(feat_vec)[0][pred]

        fname = os.path.basename(path)
        pred_label = LABELS[pred]
        pred_prob = float(prob)

        results.append((fname, pred_label, pred_prob))
        pred_counts[pred_label] += 1

    # 콘솔 출력
    print("\nPredictions:")
    for fname, lbl, p in results:
        print(f"{fname}\t→ {lbl} ({p:.3f})")

    # 요약 PNG 저장 (클래스별 예측 개수)
    out_png = os.path.join(script_dir, "prediction_summary.png")
    plt.figure(figsize=(6, 4))
    plt.bar(pred_counts.keys(), pred_counts.values())
    plt.title("Prediction Summary")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"\nSaved summary PNG → {out_png}")


if __name__ == "__main__":
    main()
