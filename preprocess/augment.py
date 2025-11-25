"""증강 및 노이즈 주입 유틸."""

from __future__ import annotations

import math
import random
from typing import Dict

import numpy as np

from .loader import resample_sequence, CoordSeq


def random_rotation(seq: CoordSeq, max_degrees: float) -> CoordSeq:
    """세 축에 대해 최대 max_degrees 범위의 작은 회전을 적용."""
    if max_degrees <= 0:
        return seq
    deg = math.radians(random.uniform(-max_degrees, max_degrees))
    yaw = math.radians(random.uniform(-max_degrees, max_degrees))
    pitch = math.radians(random.uniform(-max_degrees, max_degrees))

    def rot_x(a: float) -> np.ndarray:
        return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

    def rot_y(a: float) -> np.ndarray:
        return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

    def rot_z(a: float) -> np.ndarray:
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

    rot = rot_z(deg) @ rot_y(yaw) @ rot_x(pitch)
    return (seq @ rot.T).astype(np.float32)


def time_stretch(seq: CoordSeq, stretch_range: tuple[float, float], target_len: int) -> CoordSeq:
    """시간 축을 늘이거나 줄인 뒤 다시 고정 길이로 리샘플."""
    low, high = stretch_range
    if not (0 < low <= high):
        return seq
    factor = random.uniform(low, high)
    stretched_len = max(2, int(len(seq) * factor))
    stretched = resample_sequence(seq, target_len=stretched_len)
    return resample_sequence(stretched, target_len=target_len)


def gaussian_noise(seq: CoordSeq, sigma: float | tuple[float, float, float]) -> CoordSeq:
    """좌표에 가우시안 노이즈 추가."""
    if isinstance(sigma, tuple):
        noise = np.stack(
            [np.random.normal(0, sigma[i], size=len(seq)) for i in range(3)],
            axis=1,
        ).astype(np.float32)
    else:
        noise = np.random.normal(0, sigma, size=seq.shape).astype(np.float32)
    return seq + noise


def mask_segment(seq: CoordSeq, max_ratio: float = 0.1) -> CoordSeq:
    """연속 구간을 마스킹(평균으로 채움)."""
    if max_ratio <= 0:
        return seq
    length = len(seq)
    seg_len = max(1, int(length * random.uniform(0, max_ratio)))
    start = random.randint(0, length - seg_len)
    masked = seq.copy()
    masked[start : start + seg_len] = seq.mean(axis=0, keepdims=True)
    return masked


def apply_noise_profile(seq: CoordSeq, profile: Dict[str, float], strength: float = 1.0) -> CoordSeq:
    """오리지널 시퀀스에 노이즈 프로파일 기반 노이즈를 주입."""
    sigma = (
        profile.get("x", 0.0) * strength,
        profile.get("y", 0.0) * strength,
        profile.get("z", 0.0) * strength,
    )
    return gaussian_noise(seq, sigma=sigma)


def augment_sequence(
    seq: CoordSeq,
    *,
    rotation_deg: float = 0.0,
    stretch_range: tuple[float, float] | None = None,
    noise_sigma: float | tuple[float, float, float] | None = None,
    mask_ratio: float = 0.0,
    target_len: int | None = None,
    noise_profile: Dict[str, float] | None = None,
    noise_strength: float = 1.0,
) -> CoordSeq:
    """구성 가능한 증강 조합."""
    out = seq
    if rotation_deg:
        out = random_rotation(out, rotation_deg)
    if stretch_range:
        if target_len is None:
            raise ValueError("time_stretch를 사용하려면 target_len이 필요합니다.")
        out = time_stretch(out, stretch_range, target_len=target_len)
    if mask_ratio:
        out = mask_segment(out, max_ratio=mask_ratio)
    if noise_profile:
        out = apply_noise_profile(out, profile=noise_profile, strength=noise_strength)
    if noise_sigma:
        out = gaussian_noise(out, sigma=noise_sigma)
    return out

