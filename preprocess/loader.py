"""공통 로더/정규화/리샘플 유틸."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

CoordSeq = np.ndarray


def _parse_coord_field(field: str) -> Tuple[float, float, float] | None:
    try:
        x, y, z = [float(v) for v in field.strip().split("/")]
        return x, y, z
    except Exception:
        return None


def load_sequence(path: Path, coord_col: int = 6) -> CoordSeq:
    """TXT 파일 한 개를 읽어 (T,3) numpy 배열로 반환."""
    coords: List[Tuple[float, float, float]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) <= coord_col:
                continue
            coord = _parse_coord_field(row[coord_col])
            if coord:
                coords.append(coord)
    if len(coords) < 2:
        raise ValueError(f"{path}에서 좌표를 충분히 읽지 못했습니다.")
    return np.asarray(coords, dtype=np.float32)


def resample_sequence(seq: CoordSeq, target_len: int) -> CoordSeq:
    """선형 보간으로 길이를 target_len으로 맞춤."""
    if len(seq) < 2:
        raise ValueError("리샘플하려면 길이가 최소 2 이상이어야 합니다.")
    orig_idx = np.linspace(0, 1, num=len(seq))
    new_idx = np.linspace(0, 1, num=target_len)
    resampled = np.stack(
        [np.interp(new_idx, orig_idx, seq[:, i]) for i in range(seq.shape[1])],
        axis=1,
    )
    return resampled.astype(np.float32)


def center_and_scale(
    seq: CoordSeq, mode: str = "instance", global_std: Tuple[float, float, float] | None = None
) -> CoordSeq:
    """평균 0, 표준편차 1 정규화."""
    eps = 1e-8
    centered = seq - seq.mean(axis=0, keepdims=True)
    if mode == "instance":
        std = centered.std(axis=0, keepdims=True) + eps
    elif mode == "global" and global_std is not None:
        std = np.asarray(global_std, dtype=np.float32).reshape(1, -1) + eps
    else:
        raise ValueError("mode는 'instance' 또는 global_std와 함께 'global' 이어야 합니다.")
    return centered / std


def iter_class_files(base_dir: Path) -> Iterable[Tuple[str, Path]]:
    """클래스별 txt 파일 목록을 (label, path)로 순회."""
    for class_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        label = class_dir.name
        for txt in sorted(class_dir.glob("*.txt")):
            yield label, txt


def load_dataset(
    base_dir: Path, target_len: int, norm_mode: str = "instance", global_std: Tuple[float, float, float] | None = None
) -> List[Tuple[CoordSeq, str]]:
    """디렉터리(클래스별 하위 폴더)를 읽어 정규화·리샘플된 (seq, label) 리스트를 반환."""
    data: List[Tuple[CoordSeq, str]] = []
    for label, path in iter_class_files(base_dir):
        seq = load_sequence(path)
        seq = resample_sequence(seq, target_len=target_len)
        seq = center_and_scale(seq, mode=norm_mode, global_std=global_std)
        data.append((seq, label))
    return data


def compute_noise_profile(
    original_dir: Path, noise_dir: Path, target_len: int, norm_mode: str = "instance"
) -> Dict[str, float]:
    """오리지널 vs 노이즈 데이터의 분산 차이를 바탕으로 노이즈 표준편차 추정."""
    orig = load_dataset(original_dir, target_len=target_len, norm_mode=norm_mode)
    noise = load_dataset(noise_dir, target_len=target_len, norm_mode=norm_mode)
    if not orig or not noise:
        raise ValueError("노이즈 프로파일 계산을 위해 두 데이터 세트가 모두 필요합니다.")

    def _stack_all(seqs: Sequence[CoordSeq]) -> np.ndarray:
        return np.concatenate(seqs, axis=0)

    orig_all = _stack_all([s for s, _ in orig])
    noise_all = _stack_all([s for s, _ in noise])

    orig_std = orig_all.std(axis=0)
    noise_std = noise_all.std(axis=0)
    extra_std = np.maximum(noise_std - orig_std, 0.0)

    return {"x": float(extra_std[0]), "y": float(extra_std[1]), "z": float(extra_std[2])}


def class_counts(base_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label, _ in iter_class_files(base_dir):
        counts[label] = counts.get(label, 0) + 1
    return counts

