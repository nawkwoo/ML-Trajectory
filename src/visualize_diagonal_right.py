"""
Visualize diagonal_right trajectories in 3D and projected 2D planes (XY, XZ, YZ).
Uses Matplotlib + NumPy only. Expects raw .txt trajectories in Data/diagonal_right/*.txt.
"""

import glob
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# Config
DATA_GLOB = os.path.join("Data", "diagonal_right", "*.txt")
NORMALIZE = True  # subtract first point to set origin
SAVE_PATH = None  # e.g., "results/diagonal_right_trajs.png" or None to just show


def load_trajectory(path: str, normalize: bool = True) -> np.ndarray:
    """
    Load a single trajectory (N,3) from a txt file with comma-separated columns.
    Assumes 7번째 컬럼(index 6)에 "X/Y/Z" 형태로 좌표가 들어있음
    (preprocess.py와 동일 규칙).
    """
    pts: List[List[float]] = []
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
                pts.append([x, y, z])
            except ValueError:
                continue
    arr = np.array(pts, dtype=float)
    if arr.size == 0:
        return arr
    if normalize:
        arr = arr - arr[0]  # first point -> origin
    return arr


def main():
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        raise FileNotFoundError(f"No files matched {DATA_GLOB}")

    trajs = [load_trajectory(p, normalize=NORMALIZE) for p in paths]
    trajs = [t for t in trajs if t.size > 0]
    if not trajs:
        raise ValueError("No valid trajectories loaded.")

    print(f"Loaded {len(trajs)} trajectories")

    # Prepare colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(trajs)))

    # Flatten for projections
    x_all = np.concatenate([t[:, 0] for t in trajs])
    y_all = np.concatenate([t[:, 1] for t in trajs])
    z_all = np.concatenate([t[:, 2] for t in trajs])

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    # 3D plot spanning top row
    ax3d = fig.add_subplot(gs[0, :], projection="3d")
    for traj, c in zip(trajs, colors):
        ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1.0, color=c, alpha=0.8)
    ax3d.set_title("Diagonal Right - 3D Trajectories")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.view_init(elev=30, azim=-60)  # emphasize XY shape

    # XY projection
    ax_xy = fig.add_subplot(gs[1, 0])
    for traj, c in zip(trajs, colors):
        ax_xy.scatter(traj[:, 0], traj[:, 1], s=3, color=c, alpha=0.7)
    ax_xy.set_title("XY Projection (shape cue)")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.axis("equal")

    # XZ projection
    ax_xz = fig.add_subplot(gs[1, 1])
    for traj, c in zip(trajs, colors):
        ax_xz.scatter(traj[:, 0], traj[:, 2], s=2, color=c, alpha=0.5)
    ax_xz.set_title("XZ Projection (Z noise check)")
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.axis("equal")

    # Separate YZ projection for clarity
    fig_yz, ax_yz = plt.subplots(figsize=(6, 5))
    for traj, c in zip(trajs, colors):
        ax_yz.scatter(traj[:, 1], traj[:, 2], s=2, color=c, alpha=0.5)
    ax_yz.set_title("YZ Projection (Z noise check)")
    ax_yz.set_xlabel("Y")
    ax_yz.set_ylabel("Z")
    ax_yz.axis("equal")

    plt.tight_layout()
    if SAVE_PATH:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        fig.savefig(SAVE_PATH, dpi=200)
        fig_yz.savefig(SAVE_PATH.replace(".png", "_yz.png"), dpi=200)
        print(f"Saved figures to {SAVE_PATH} and *_yz.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
