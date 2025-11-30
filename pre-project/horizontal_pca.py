"""PCA-aligned horizontal trajectories and feature analysis."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from preprocess.loader import load_sequence

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 10


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _resolve_data_root() -> Path:
    candidates = [PROJECT_ROOT / "Data", REPO_ROOT / "Data"]
    for root in candidates:
        if root.exists():
            return root
    raise FileNotFoundError("Data root not found. Expected Data/ under pre-project or repo root.")


DATA_ROOT = _resolve_data_root()
ORIGINAL_DIR = DATA_ROOT / "Machine Learing(Original)" / "horizontal"
NOISE_DIR = DATA_ROOT / "Machine Learning(Noise)" / "horizontal"

OUT_ROOT = PROJECT_ROOT / "outputs" / "horizontal"
OUT_NORM_INDIVIDUAL = OUT_ROOT / "normalized_individual"
OUT_NORM_GROUP = OUT_ROOT / "normalized_group"
OUT_NORM_FEATURES = OUT_ROOT / "normalized_features"


def ensure_output_dirs() -> None:
    for p in [OUT_NORM_INDIVIDUAL, OUT_NORM_GROUP, OUT_NORM_FEATURES]:
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    seq: np.ndarray  # (T, 3)
    file_id: int
    group: str  # "original" | "noise"
    path: Path


def _numeric_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    try:
        return int(stem), stem
    except ValueError:
        return math.inf, stem


def load_all() -> List[Trajectory]:
    trajectories: List[Trajectory] = []
    for group, base in (("original", ORIGINAL_DIR), ("noise", NOISE_DIR)):
        if not base.exists():
            raise FileNotFoundError(f"{base} not found. Place horizontal data under {DATA_ROOT}.")
        for path in sorted(base.glob("*.txt"), key=_numeric_key):
            seq = load_sequence(path)
            try:
                fid = int(path.stem)
            except ValueError:
                fid = len(trajectories) + 1
            trajectories.append(Trajectory(seq=seq, file_id=fid, group=group, path=path))
    if not trajectories:
        raise RuntimeError("No trajectories found under horizontal folders.")
    return trajectories


# ---------------------------------------------------------------------------
# Feature + PCA
# ---------------------------------------------------------------------------
def extract_features(seq: np.ndarray) -> Dict[str, float]:
    coords = np.asarray(seq, dtype=float)
    cx, cy, cz = coords.mean(axis=0)

    cov = np.cov(coords.T) if len(coords) > 1 else np.zeros((3, 3))
    var_x, var_y, var_z = cov[0, 0], cov[1, 1], cov[2, 2]
    cov_xy, cov_xz, cov_yz = cov[0, 1], cov[0, 2], cov[1, 2]

    x_range = np.ptp(coords[:, 0])
    y_range = np.ptp(coords[:, 1])
    z_range = np.ptp(coords[:, 2])
    bbox_volume = x_range * y_range * z_range

    if len(coords) >= 2:
        diffs = np.diff(coords, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        path_length = float(segment_lengths.sum())
    else:
        path_length = 0.0

    centered = coords - np.array([cx, cy, cz])
    radii = np.linalg.norm(centered, axis=1)
    radius_mean = float(radii.mean()) if len(radii) else 0.0
    radius_std = float(radii.std(ddof=1)) if len(radii) > 1 else 0.0

    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "center_z": float(cz),
        "var_x": float(var_x),
        "var_y": float(var_y),
        "var_z": float(var_z),
        "cov_xy": float(cov_xy),
        "cov_xz": float(cov_xz),
        "cov_yz": float(cov_yz),
        "x_range": float(x_range),
        "y_range": float(y_range),
        "z_range": float(z_range),
        "bbox_volume": float(bbox_volume),
        "path_length": path_length,
        "radius_mean": radius_mean,
        "radius_std": radius_std,
    }


def normalize_and_pca(seq: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    coords = np.asarray(seq, dtype=float)
    center = coords.mean(axis=0, keepdims=True)
    coords_centered = coords - center

    radii = np.linalg.norm(coords_centered, axis=1)
    scale = float(radii.mean())
    if scale < eps:
        scale = 1.0
    coords_scaled = coords_centered / scale

    cov = np.cov(coords_scaled, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    aligned = coords_scaled @ eigvecs
    return aligned.astype(np.float32), center.reshape(-1), scale, eigvecs.astype(np.float32)


def save_feature_tables(trajs: Sequence[Trajectory], out_dir: Path, prefix: str) -> pd.DataFrame:
    feat_dict: Dict[str, Dict[str, float]] = {}
    group_labels: Dict[str, str] = {}
    for traj in trajs:
        key = f"{traj.group}_{traj.file_id:02d}"
        feat_dict[key] = extract_features(traj.seq)
        group_labels[key] = traj.group

    df = pd.DataFrame(feat_dict)
    df.index.name = "feature"
    df.columns.name = "file_id"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{prefix}_features_table.csv")

    df_T = df.T
    df_T = df_T.assign(group=pd.Series(group_labels))
    group_mean = df_T.groupby("group").mean(numeric_only=True)
    group_std = df_T.groupby("group").std(numeric_only=True)
    group_mean.to_csv(out_dir / f"{prefix}_features_group_mean.csv")
    group_std.to_csv(out_dir / f"{prefix}_features_group_std.csv")
    return df_T


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _limits(seq: np.ndarray, margin: float) -> Tuple[float, float, float, float, float, float]:
    xmin, ymin, zmin = seq.min(axis=0)
    xmax, ymax, zmax = seq.max(axis=0)
    return xmin - margin, xmax + margin, ymin - margin, ymax + margin, zmin - margin, zmax + margin


def plot_3d(seq: np.ndarray, title: str, save_path: Path, margin: float = 10.0) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    xmin, xmax, ymin, ymax, zmin, zmax = _limits(seq, margin)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.plot(seq[:, 0], seq[:, 1], seq[:, 2], "-o", markersize=2)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_projection(seq: np.ndarray, axes: Tuple[int, int], labels: Tuple[str, str], title: str, save_path: Path, margin: float = 10.0) -> None:
    u, v = seq[:, axes[0]], seq[:, axes[1]]
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    umin, umax = u.min(), u.max()
    vmin, vmax = v.min(), v.max()
    ax.set_xlim(umin - margin, umax + margin)
    ax.set_ylim(vmin - margin, vmax + margin)
    ax.plot(u, v, "-o", markersize=2)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_group_overlay(trajs: Sequence[Trajectory], title: str, save_path: Path, margin: float = 10.0, axes: Tuple[int, int] | None = None) -> None:
    fig = plt.figure()
    colors = {"original": "#1f77b4", "noise": "#d62728"}
    seen_labels = set()

    if axes is None:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        all_coords = np.concatenate([t.seq for t in trajs], axis=0)
        xmin, xmax, ymin, ymax, zmin, zmax = _limits(all_coords, margin)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
    else:
        ax = fig.add_subplot(111)
        u_idx, v_idx = axes
        label_map = {(0, 1): ("X", "Y"), (1, 2): ("Y", "Z"), (0, 2): ("X", "Z")}
        xlabel, ylabel = label_map.get(axes, ("axis0", "axis1"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        all_coords = np.concatenate([t.seq[:, [u_idx, v_idx]] for t in trajs], axis=0)
        (umin, vmin), (umax, vmax) = all_coords.min(axis=0), all_coords.max(axis=0)
        ax.set_xlim(umin - margin, umax + margin)
        ax.set_ylim(vmin - margin, vmax + margin)
        ax.set_aspect("equal", adjustable="box")

    ax.set_title(title)

    for traj in trajs:
        label = None
        if traj.group not in seen_labels:
            label = traj.group
            seen_labels.add(traj.group)

        color = colors.get(traj.group)
        if axes is None:
            ax.plot(traj.seq[:, 0], traj.seq[:, 1], traj.seq[:, 2], linewidth=1.0, alpha=0.7, color=color, label=label)
        else:
            u_idx, v_idx = axes
            ax.plot(traj.seq[:, u_idx], traj.seq[:, v_idx], linewidth=1.0, alpha=0.7, color=color, linestyle="--" if traj.group == "noise" else "-", label=label)

    if seen_labels:
        ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
IMPORTANT_FEATURES = [
    "bbox_volume",
    "path_length",
    "radius_std",
    "radius_mean",
    "var_x",
    "var_y",
    "var_z",
]


def plot_box_and_hist(df_T: pd.DataFrame, out_dir: Path, prefix: str, title_prefix: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for feat in IMPORTANT_FEATURES:
        fig, ax = plt.subplots()
        data_clean = df_T[df_T["group"] == "original"][feat]
        data_noise = df_T[df_T["group"] == "noise"][feat]
        ax.set_title(f"{title_prefix}{feat} - Boxplot (original vs noise)")
        ax.boxplot([data_clean.values, data_noise.values], tick_labels=["original", "noise"], showmeans=True)
        ax.set_ylabel(feat)
        fig.tight_layout()
        fig.savefig(out_dir / f"boxplot_{prefix}_{feat}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.set_title(f"{title_prefix}{feat} - Histogram (original vs noise)")
        ax.hist(data_clean, bins=5, alpha=0.6, label="original")
        ax.hist(data_noise, bins=5, alpha=0.6, label="noise")
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{prefix}_{feat}.png", dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_pca() -> None:
    ensure_output_dirs()
    trajs = load_all()
    print(f"[INFO] Loaded {len(trajs)} trajectories (original={sum(t.group=='original' for t in trajs)}, noise={sum(t.group=='noise' for t in trajs)})")

    norm_trajs: list[Trajectory] = []
    for traj in trajs:
        aligned, center, scale, eigvecs = normalize_and_pca(traj.seq)
        norm_trajs.append(Trajectory(seq=aligned, file_id=traj.file_id, group=traj.group, path=traj.path))

    # 개별 플롯 (정규화)
    for traj in norm_trajs:
        base = f"horizontal_norm_{traj.group}_{traj.file_id:02d}"
        plot_3d(traj.seq, title=f"{base} - 3D", save_path=OUT_NORM_INDIVIDUAL / f"{base}_3d.png", margin=0.5)
        plot_projection(traj.seq, (0, 1), ("X'", "Y'"), title=f"{base} - XY'", save_path=OUT_NORM_INDIVIDUAL / f"{base}_xy.png", margin=0.5)
        plot_projection(traj.seq, (1, 2), ("Y'", "Z'"), title=f"{base} - YZ'", save_path=OUT_NORM_INDIVIDUAL / f"{base}_yz.png", margin=0.5)
        plot_projection(traj.seq, (0, 2), ("X'", "Z'"), title=f"{base} - XZ'", save_path=OUT_NORM_INDIVIDUAL / f"{base}_xz.png", margin=0.5)

    # 그룹 오버레이 (정규화)
    plot_group_overlay(norm_trajs, "Normalized Original vs Noise - 3D", OUT_NORM_GROUP / "group_norm_original_vs_noise_3d.png", margin=0.5)
    plot_group_overlay(norm_trajs, "Normalized Original vs Noise - XY'", OUT_NORM_GROUP / "group_norm_original_vs_noise_xy.png", margin=0.5, axes=(0, 1))
    plot_group_overlay(norm_trajs, "Normalized Original vs Noise - YZ'", OUT_NORM_GROUP / "group_norm_original_vs_noise_yz.png", margin=0.5, axes=(1, 2))
    plot_group_overlay(norm_trajs, "Normalized Original vs Noise - XZ'", OUT_NORM_GROUP / "group_norm_original_vs_noise_xz.png", margin=0.5, axes=(0, 2))

    # 특징 표/통계 + 박스플롯/히스토그램 (정규화)
    df_T = save_feature_tables(norm_trajs, OUT_NORM_FEATURES, prefix="horizontal_norm")
    plot_box_and_hist(df_T, OUT_NORM_FEATURES, prefix="horizontal_norm", title_prefix="[Norm] ")
    print(f"[INFO] Normalized artifacts saved to {OUT_NORM_FEATURES.parent}")


if __name__ == "__main__":
    run_pca()
