import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ----- Config -----
# diagonal_right 클래스 라벨 값을 여기에 입력하세요 (예: 2)
DIAGONAL_RIGHT_LABEL = 2  # TODO: set the numeric label for diagonal_right

# 데이터 경로 (이미 전처리된 결과 사용)
PREPRO_DIR = os.path.join("results", "preprocessed_data")
X_PATH = os.path.join(PREPRO_DIR, "X.npy")  # shape: (N, T, 3), 채널 순서: (x, y, z)
Y_PATH = os.path.join(PREPRO_DIR, "y.npy")  # shape: (N,)


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed trajectories and labels."""
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"Not found: {X_PATH}")
    if not os.path.exists(Y_PATH):
        raise FileNotFoundError(f"Not found: {Y_PATH}")
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    return X, y


def filter_diagonal_right(X: np.ndarray, y: np.ndarray, label_value: int) -> np.ndarray:
    """Filter trajectories for the diagonal_right class."""
    mask = y == label_value
    X_dr = X[mask]
    if X_dr.size == 0:
        raise ValueError(f"No samples found for label={label_value}")
    return X_dr


def run_pca(X_dr: np.ndarray) -> PCA:
    """
    Run PCA on diagonal_right samples.
    X_dr: (n_dr, T, 3)
    Reshape to (n_dr * T, 3) treating all points as one cloud.
    """
    n_dr, T, C = X_dr.shape
    assert C == 3, "Expected channel order (x, y, z)"
    X_flat = X_dr.reshape(n_dr * T, C)
    pca = PCA(n_components=3, random_state=0)
    pca.fit(X_flat)
    return pca


def explain_components(pca: PCA):
    """Print explained variance ratios and component loadings with simple interpretations."""
    var_ratio = pca.explained_variance_ratio_
    comps = pca.components_

    print("Explained variance ratio:", np.round(var_ratio, 4))
    print("Components (rows=PC1..PC3, cols=x,y,z):")
    print(np.round(comps, 4))

    axis_names = ["x", "y", "z"]
    for i, (pc, vr) in enumerate(zip(comps, var_ratio), start=1):
        abs_pc = np.abs(pc)
        top_axis = axis_names[int(np.argmax(abs_pc))]
        min_axis = axis_names[int(np.argmin(abs_pc))]
        print(f"\nPC{i}: x={pc[0]:+.4f}, y={pc[1]:+.4f}, z={pc[2]:+.4f} | var_ratio={vr:.3f}")
        print(f" - |coef| max axis: {top_axis} (주된 변화 축)")
        print(f" - |coef| min axis: {min_axis} (기여 낮음)")
        if top_axis == "y":
            print("   → PC{i}: Y축 기반 변화가 크다.")
        elif top_axis == "x":
            print("   → PC{i}: X축 기반 변화가 크다.")
        else:
            print("   → PC{i}: Z축 기반 변화가 크다.")
        if vr < 0.1:
            print("   → 분산 기여도가 작아 정보가 적은 축일 수 있음.")


def plot_scatter(X_dr: np.ndarray, pca: PCA, save_dir: str = None):
    """
    Plot multiple 2D projections to visually compare 축별 정보량.
      - Original: XY, XZ, YZ
      - PCA: PC1-PC2, PC1-PC3, PC2-PC3
    """
    # Original projections
    x = X_dr[..., 0].reshape(-1)
    y = X_dr[..., 1].reshape(-1)
    z = X_dr[..., 2].reshape(-1)

    # PCA transform (all points)
    X_flat = X_dr.reshape(-1, 3)
    X_pca = pca.transform(X_flat)
    pc1, pc2, pc3 = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    ax = axes.flat

    # Original XY, XZ, YZ
    ax[0].scatter(x, y, s=2, alpha=0.4, color="tab:blue")
    ax[0].set_title("Original XY")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].axis("equal")

    ax[1].scatter(x, z, s=2, alpha=0.4, color="tab:green")
    ax[1].set_title("Original XZ")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Z")
    ax[1].axis("equal")

    ax[2].scatter(y, z, s=2, alpha=0.4, color="tab:purple")
    ax[2].set_title("Original YZ")
    ax[2].set_xlabel("Y")
    ax[2].set_ylabel("Z")
    ax[2].axis("equal")

    # PCA PC1-PC2, PC1-PC3, PC2-PC3
    ax[3].scatter(pc1, pc2, s=2, alpha=0.4, color="tab:orange")
    ax[3].set_title("PCA: PC1 vs PC2")
    ax[3].set_xlabel("PC1")
    ax[3].set_ylabel("PC2")
    ax[3].axis("equal")

    ax[4].scatter(pc1, pc3, s=2, alpha=0.4, color="tab:red")
    ax[4].set_title("PCA: PC1 vs PC3")
    ax[4].set_xlabel("PC1")
    ax[4].set_ylabel("PC3")
    ax[4].axis("equal")

    ax[5].scatter(pc2, pc3, s=2, alpha=0.4, color="tab:brown")
    ax[5].set_title("PCA: PC2 vs PC3")
    ax[5].set_xlabel("PC2")
    ax[5].set_ylabel("PC3")
    ax[5].axis("equal")

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "pca_diagonal_right.png")
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main():
    if DIAGONAL_RIGHT_LABEL == "?":
        raise ValueError("Set DIAGONAL_RIGHT_LABEL to the numeric class ID for diagonal_right.")

    X, y = load_data()
    X_dr = filter_diagonal_right(X, y, DIAGONAL_RIGHT_LABEL)
    print(f"Filtered diagonal_right samples: {X_dr.shape[0]} trajectories")

    pca = run_pca(X_dr)
    explain_components(pca)
    plot_scatter(X_dr, pca, save_dir=os.path.join("results"))


if __name__ == "__main__":
    main()
