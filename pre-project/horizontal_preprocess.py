"""Extract raw xyz from horizontal TXT files into a CSV (kwanwoo-style)."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from preprocess.loader import load_sequence


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


def ensure_output_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading
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
# Export
# ---------------------------------------------------------------------------
def export_raw_csv(trajs: List[Trajectory], out_path: Path) -> None:
    rows = []
    for traj in trajs:
        for t, (x, y, z) in enumerate(traj.seq):
            rows.append(
                {
                    "group": traj.group,
                    "file_id": traj.file_id,
                    "t": t,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    ensure_output_dirs()
    trajs = load_all()
    out_csv = OUT_ROOT / "horizontal_raw_xyz.csv"
    export_raw_csv(trajs, out_csv)
    print(f"[INFO] Raw xyz saved to {out_csv}")


if __name__ == "__main__":
    main()
