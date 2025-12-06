"""
Trajectory augmentation script.

기본 파이프라인
---------------
1) project_root/data/result 에서 전처리된 궤적 데이터(X.npy, y.npy) 로드
2) 각 샘플(궤적)에 대해 다음 증강 기법을 적용하여 새로운 샘플 생성:
- Gaussian noise 추가
- time shift (시간축 이동)
- random crop + resample (부분 구간 잘라서 다시 T 길이로 보간)
- random masking (일부 구간을 0으로 마스킹)
- XY 평면에서 작은 각도로 회전
- same-class mixup (같은 클래스 궤적끼리 선형 결합)
3) 원본 샘플들은 그대로 보존하고, 증강된 샘플들을 뒤에 이어 붙여 X_aug, y_aug 구성
4) 결과를 project_root/data/augmented/X.npy, y.npy 로 저장

"""

import argparse
import os
import random
from typing import Tuple

import numpy as np


# ────────────────────────────────────────────────
#   개별 증강 함수들
# ────────────────────────────────────────────────

def add_noise(x: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """
    궤적 전체에 Gaussian noise를 추가한다.

    Parameters
    ----------
    x : np.ndarray
        shape (T, C)의 궤적 데이터
    sigma : float, optional
        가우시안 노이즈의 표준편차 (기본값: 0.01)

    Returns
    -------
    np.ndarray
        노이즈가 추가된 궤적, shape (T, C)
    """
    noise = np.random.normal(0.0, sigma, size=x.shape)
    return x + noise


def time_shift(x: np.ndarray, max_shift: int = 10) -> np.ndarray:
    """
    궤적을 시간축 방향으로 랜덤하게 이동시킨다.

    - 양수 shift: 앞부분을 잘라 뒤로 밀고, 시작 구간은 첫 프레임 값으로 채운다.
    - 음수 shift: 뒷부분을 잘라 앞으로 당기고, 끝 구간은 마지막 프레임 값으로 채운다.

    Parameters
    ----------
    x : np.ndarray
        shape (T, C)의 궤적 데이터
    max_shift : int, optional
        최대 이동 프레임 수 (기본값: 10)

    Returns
    -------
    np.ndarray
        시간축이 랜덤하게 이동된 궤적, shape (T, C)
    """
    T = x.shape[0]
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return x.copy()

    out = np.empty_like(x)
    if shift > 0:
        out[shift:] = x[:-shift]
        out[:shift] = x[0]
    else:
        s = -shift
        out[:-s] = x[s:]
        out[-s:] = x[-1]
    return out


def _resample(traj: np.ndarray, target_len: int) -> np.ndarray:
    """
    주어진 궤적(traj)을 선형 보간하여 target_len 길이로 리샘플링한다.

    Parameters
    ----------
    traj : np.ndarray
        shape (T, C)의 궤적
    target_len : int
        리샘플링할 길이

    Returns
    -------
    np.ndarray
        리샘플링된 궤적, shape (target_len, C)
    """
    T = traj.shape[0]
    if T == target_len:
        return traj.copy()

    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_len)

    out = np.zeros((target_len, traj.shape[1]), dtype=np.float32)
    for dim in range(traj.shape[1]):
        out[:, dim] = np.interp(new_idx, old_idx, traj[:, dim])
    return out


def random_crop(x: np.ndarray, min_ratio: float = 0.7) -> np.ndarray:
    """
    궤적의 일부 구간을 랜덤하게 잘라(min_ratio ~ 1.0 비율) 다시 원래 길이로 보간한다.

    Parameters
    ----------
    x : np.ndarray
        shape (T, C)의 궤적 데이터
    min_ratio : float, optional
        잘라낼 최소 길이 비율 (기본값: 0.7)

    Returns
    -------
    np.ndarray
        random crop + resample이 적용된 궤적, shape (T, C)
    """
    T = x.shape[0]
    min_len = int(T * min_ratio)
    crop_len = np.random.randint(min_len, T + 1)
    start = np.random.randint(0, T - crop_len + 1)
    window = x[start : start + crop_len]
    return _resample(window, T)


def random_mask(x: np.ndarray, max_ratio: float = 0.1) -> np.ndarray:
    """
    궤적의 연속된 일부 구간을 0으로 마스킹한다.

    Parameters
    ----------
    x : np.ndarray
        shape (T, C)의 궤적 데이터
    max_ratio : float, optional
        마스킹 구간이 전체 길이에서 차지하는 최대 비율 (기본값: 0.1)

    Returns
    -------
    np.ndarray
        일부 구간이 0으로 마스킹된 궤적, shape (T, C)
    """
    T = x.shape[0]
    max_len = max(1, int(T * max_ratio))
    mask_len = np.random.randint(1, max_len + 1)
    start = np.random.randint(0, T - mask_len + 1)

    out = x.copy()
    out[start : start + mask_len] = 0.0
    return out


def random_rotate_xy(x: np.ndarray, max_angle_deg: float = 20.0) -> np.ndarray:
    """
    XY 평면에서 궤적을 작은 각도로 랜덤 회전시킨다.

    - 좌표가 2차원 이상일 때만 적용되며, 첫 두 축(x, y)에 회전을 적용한다.

    Parameters
    ----------
    x : np.ndarray
        shape (T, C)의 궤적 데이터
    max_angle_deg : float, optional
        회전 각도의 최대 절대값(도 단위) (기본값: 20.0)

    Returns
    -------
    np.ndarray
        XY 평면 회전이 적용된 궤적, shape (T, C)
    """
    if x.shape[1] < 2:
        return x.copy()

    angle = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    out = x.copy()
    out_xy = out[:, :2] @ R.T
    out[:, :2] = out_xy
    return out


def mixup_same_class(x1: np.ndarray, x2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    같은 클래스 내 두 궤적 x1, x2를 mixup 방식으로 선형 결합한다.

    lam ~ Beta(alpha, alpha)
    x_mix = lam * x1 + (1 - lam) * x2

    Parameters
    ----------
    x1 : np.ndarray
        shape (T, C)의 궤적
    x2 : np.ndarray
        shape (T, C)의 궤적
    alpha : float, optional
        Beta 분포의 파라미터 (기본값: 0.5).
        - alpha가 클수록 lam이 0.5 근처에 많이 분포한다.

    Returns
    -------
    np.ndarray
        mixup이 적용된 궤적, shape (T, C)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5
    return lam * x1 + (1.0 - lam) * x2


# ────────────────────────────────────────────────
#   증강 데이터셋 빌더
# ────────────────────────────────────────────────

def build_augmented_dataset(
    X: np.ndarray,
    y: np.ndarray,
    noise_sigma: float = 0.01,
    mask_max_ratio: float = 0.1,
    mixup_alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    원본 데이터(X, y)에 여러 증강 기법을 적용하여 증강 데이터셋을 생성한다.

    처리 순서
    ---------
    1) X, y의 원본 샘플을 그대로 X_list, y_list에 먼저 복사
    2) 각 샘플 x_i, label_i에 대해 다음 증강 샘플들을 생성하여 리스트에 추가:
    - add_noise(x_i)
    - time_shift(x_i)
    - random_crop(x_i)
    - random_mask(x_i)
    - random_rotate_xy(x_i)
    - 같은 클래스 내 다른 샘플과 mixup_same_class(x_i, x_j)
    3) 최종적으로 X_list, y_list를 numpy 배열로 변환하여 반환

    Parameters
    ----------
    X : np.ndarray
        shape (N, T, C)의 궤적 데이터
    y : np.ndarray
        shape (N,)의 정수 라벨 배열
    noise_sigma : float, optional
        Gaussian noise 표준편차 (기본값: 0.01)
    mask_max_ratio : float, optional
        random_mask에서 마스킹 구간의 최대 길이 비율 (기본값: 0.1)
    mixup_alpha : float, optional
        same-class mixup의 Beta 분포 파라미터 (기본값: 0.5)

    Returns
    -------
    X_aug : np.ndarray
        증강이 포함된 궤적 데이터, shape (N_aug, T, C)
    y_aug : np.ndarray
        증강이 포함된 라벨 배열, shape (N_aug,)
    """
    # 1) 원본 샘플을 그대로 보존
    X_list = [x.copy() for x in X]
    y_list = [int(label) for label in y]

    # same-class mixup을 위한 인덱스 맵: label -> [indices...]
    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(y):
        class_to_indices.setdefault(int(label), []).append(idx)

    N = X.shape[0]

    # 2) 각 샘플에 대해 증강 생성
    for i in range(N):
        x = X[i]
        label = int(y[i])

        # noise
        X_list.append(add_noise(x, sigma=noise_sigma))
        y_list.append(label)

        # time shift
        X_list.append(time_shift(x))
        y_list.append(label)

        # random crop + resample
        X_list.append(random_crop(x))
        y_list.append(label)

        # random mask
        X_list.append(random_mask(x, max_ratio=mask_max_ratio))
        y_list.append(label)

        # XY rotation
        X_list.append(random_rotate_xy(x))
        y_list.append(label)

        # same-class mixup (같은 클래스가 2개 이상 있을 때만)
        same_class_idx = class_to_indices.get(label, [])
        if len(same_class_idx) > 1 and mixup_alpha > 0:
            j = i
            # i와 다른 샘플 하나를 랜덤 선택
            while j == i:
                j = random.choice(same_class_idx)
            x2 = X[j]
            X_list.append(mixup_same_class(x, x2, alpha=mixup_alpha))
            y_list.append(label)

    X_aug = np.stack(X_list, axis=0).astype(np.float32)
    y_aug = np.array(y_list, dtype=np.int64)
    return X_aug, y_aug


# ────────────────────────────────────────────────
#   main & argument parsing
# ────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    커맨드라인 인자를 파싱한다.

    Returns
    -------
    argparse.Namespace
        파싱된 인자 객체
    """
    parser = argparse.ArgumentParser(
        description="Augment trajectories from project_root/data/result and save to project_root/data/augmented",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.01,
        help="Stddev for Gaussian noise (default: 0.01)",
    )
    parser.add_argument(
        "--mask-max-ratio",
        type=float,
        default=0.1,
        help="Max masked segment ratio (default: 0.1)",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.5,
        help="Beta(alpha, alpha) parameter for same-class mixup (default: 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    """
    커맨드라인에서 실행할 때의 진입점 함수.

    예시
    ----
    python augment.py \\
        --noise-sigma 0.01 \\
        --mask-max-ratio 0.1 \\
        --mixup-alpha 0.5

    - 입력:  project_root/data/result/X.npy, y.npy
    - 출력:  project_root/data/augmented/X.npy, y.npy
    """
    args = parse_args()

    # 재현성을 위한 시드 고정
    np.random.seed(0)
    random.seed(0)

    # 전처리 코드와 동일한 방식으로 project_root 계산
    current_dir = os.path.dirname(os.path.abspath(__file__))        # .../src
    project_root = os.path.dirname(current_dir)                     # 프로젝트 루트

    base_dir = os.path.join(project_root, "data", "result")
    out_dir = os.path.join(project_root, "data", "augmented")

    x_path = os.path.join(base_dir, "X.npy")
    y_path = os.path.join(base_dir, "y.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Preprocessed files not found in '{base_dir}'. "
            "Run the preprocessing script (preprocess.py) first."
        )

    X = np.load(x_path)
    y = np.load(y_path)

    print("Original dataset:", X.shape, y.shape)
    print(f"noise_sigma={args.noise_sigma}, mask_max_ratio={args.mask_max_ratio}, mixup_alpha={args.mixup_alpha}")

    X_aug, y_aug = build_augmented_dataset(
        X,
        y,
        noise_sigma=args.noise_sigma,
        mask_max_ratio=args.mask_max_ratio,
        mixup_alpha=args.mixup_alpha,
    )

    print("Augmented dataset:", X_aug.shape, y_aug.shape)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X_aug)
    np.save(os.path.join(out_dir, "y.npy"), y_aug)

    print(f"Saved augmented dataset to '{out_dir}'")


if __name__ == "__main__":
    main()