"""모든 궤적 TXT를 3D/2D 이미지로 저장.

실행하면 draw_line/<noise|original>/<class>/ 파일명.png 로 저장됩니다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def load_xyz(path: Path) -> Tuple[List[float], List[float], List[float]]:
    xs, ys, zs = [], [], []
    with path.open() as f:
        for line in f:
            cols = line.strip().split(",")
            if len(cols) <= 6:
                continue
            xyz_str = cols[6].strip()
            parts = xyz_str.split("/")
            if len(parts) != 3:
                continue
            x, y, z = map(float, parts)
            xs.append(x)
            ys.append(y)
            zs.append(z)
    return xs, ys, zs


def compute_limits(vals: List[float], margin: float = 10.0) -> Tuple[float, float]:
    if not vals:
        return -1, 1
    return min(vals) - margin, max(vals) + margin


def plot_3d(xs: List[float], ys: List[float], zs: List[float], title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, "-o", markersize=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlim(*compute_limits(xs))
    ax.set_ylim(*compute_limits(ys))
    ax.set_zlim(*compute_limits(zs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_xy(xs: List[float], ys: List[float], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xs, ys, "-o", markersize=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*compute_limits(xs))
    ax.set_ylim(*compute_limits(ys))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    def find_dir(candidates):
        for cand in candidates:
            if cand.exists():
                return cand
        return None

    noise_dir = find_dir(
        [
            root / "Data" / "Machine Learning(noise)",
            root / "Data" / "Machine Learning(Noise)",
        ]
    )
    orig_dir = find_dir(
        [
            root / "Data" / "Machine Learing(Original)",
            root / "Data" / "machine_learning(Original)",
            root / "Data" / "Machine Learning(Original)",
        ]
    )
    data_sets = {}
    if noise_dir:
        data_sets["noise"] = noise_dir
    if orig_dir:
        data_sets["original"] = orig_dir
    out_root = root / "draw_line"

    for label, base_dir in data_sets.items():
        if not base_dir.exists():
            continue
        for class_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
            for txt in sorted(class_dir.glob("*.txt")):
                xs, ys, zs = load_xyz(txt)
                if not xs:
                    continue
                rel_name = txt.stem
                class_name = class_dir.name
                title = f"{label} - {class_name} - {rel_name}"
                out_dir = out_root / label / class_name
                plot_3d(xs, ys, zs, title, out_dir / f"{rel_name}_3d.png")
                plot_xy(xs, ys, title, out_dir / f"{rel_name}_xy.png")
                print(f"Saved {label}/{class_name}/{rel_name}")


if __name__ == "__main__":
    main()
