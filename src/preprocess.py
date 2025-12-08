"""
Label-aware preprocessing with per-class axis scaling/selection.

기본 파이프라인
---------------
1) Data/{label}/*.txt에서 7번째 컬럼(인덱스 6, "X/Y/Z" 문자열)을 파싱하여 (T, 3) 궤적 로드
2) 시작점을 원점으로 이동 → 스케일 정규화(원점으로부터의 최대 거리 = 1) → 길이 128로 선형 보간
3) 라벨별 축 가중치/제거 적용 (비율은 max=1 기준, 전체 강도는 weight_scale로 조절)
4) 결과를 save_dir/X.npy, save_dir/y.npy 로 저장

"""

import glob
import os
from typing import List, Tuple

import numpy as np


# 사용할 클래스 라벨(고정 순서)
LABELS: List[str] = ["circle", "diagonal_left", "diagonal_right", "horizontal", "vertical"]


def load_trajectory(file_path: str) -> np.ndarray:
    """
    하나의 txt 파일에서 end-effector 궤적을 읽어 (T, 3) 배열로 반환한다.

    - 각 줄은 콤마(,)로 구분된 여러 컬럼으로 구성되어 있으며, 7번째 컬럼(인덱스 6)이 "X/Y/Z" 형태의 문자열이라고 가정한다.
    - s, S, # 로 시작하는 줄은 헤더/코멘트로 간주하고 무시한다.

    Parameters
    ----------
    file_path : str
        입력 파일 경로

    Returns
    -------
    traj : np.ndarray
        shape (T, 3)의 궤적 배열
    """
    xs, ys, zs = [], [], []

    with open(file_path, "r") as f:
        for line in f:
            cols = line.strip().split(",")
            if len(cols) <= 6:
                continue

            col = cols[6].strip()
            # 빈 값 또는 헤더/코멘트는 무시
            if not col or col[0] in ("s", "S", "#"):
                continue

            try:
                x, y, z = map(float, col.split("/"))
            except ValueError:
                # 좌표 파싱 실패 시 해당 줄은 건너뜀
                continue

            xs.append(x)
            ys.append(y)
            zs.append(z)

    if not xs:
        raise ValueError(f"No valid trajectory data found in {file_path}")

    return np.stack([xs, ys, zs], axis=1)  # (T, 3)


def normalize_origin(traj: np.ndarray) -> np.ndarray:
    """
    궤적의 시작점을 원점(0, 0, 0)으로 이동한다.

    Parameters
    ----------
    traj : np.ndarray
        shape (T, 3), 원본 궤적

    Returns
    -------
    np.ndarray
        시작점이 (0, 0, 0)으로 이동된 궤적 (T, 3)
    """
    return traj - traj[0]


def normalize_scale(traj: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    궤적을 "원점으로부터의 최대 거리" 기준으로 스케일 정규화한다.

    - 각 시점의 거리 d_t = ||p_t|| 를 계산하고, max(d_t)를 최대 거리로 사용
    - 전체 궤적을 (max_dist + eps)로 나누어, 가장 먼 점의 거리가 1에 가깝도록 만든다.

    Parameters
    ----------
    traj : np.ndarray
        shape (T, 3), 원점 정렬된 궤적
    eps : float, optional
        0 나누기 방지를 위한 작은 값

    Returns
    -------
    np.ndarray
        스케일 정규화된 궤적 (T, 3)
    """
    dists = np.linalg.norm(traj, axis=1)
    max_dist = np.max(dists)
    return traj / (max_dist + eps)


def resample_trajectory(traj: np.ndarray, target_len: int = 128) -> np.ndarray:
    """
    궤적을 선형 보간을 통해 고정 길이(target_len)로 리샘플링한다.

    - 기존 인덱스 0 ~ T-1 구간을 연속적인 좌표로 보고, 그 위에 target_len개의 균등 분할된 지점을 찍어 각 축별로 선형 보간(np.interp)을 수행한다.

    Parameters
    ----------
    traj : np.ndarray
        shape (T, 3), 스케일 정규화된 궤적
    target_len : int, optional
        리샘플링 후의 길이 (기본값: 128)

    Returns
    -------
    np.ndarray
        리샘플링된 궤적, shape (target_len, 3)
    """
    T = len(traj)
    if T == target_len:
        return traj.copy()

    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_len)

    out = np.zeros((target_len, 3), dtype=np.float32)
    for dim in range(3):
        out[:, dim] = np.interp(new_idx, old_idx, traj[:, dim])

    return out


def apply_label_weights(
    traj: np.ndarray,
    label: str,
    scale: float = 1.0,
) -> np.ndarray:
    """
    라벨별 축 가중치/제거를 적용한다. (비율은 max=1 기준, 전체 강도는 scale로 조절)

    base 비율(라벨별 기본 가중치, scale=1.0일 때)
    ------------------------------------------------
    circle        : X=1.00, Y=1.00, Z=1.00
    diagonal_left : X=0.16, Y=1.00, Z=0.86   (Y 강조)
    diagonal_right: X=0.16, Y=0.86, Z=1.00   (Z 강조)
    horizontal    : X=1.00, Y=0.15, Z=0.40   (X 강조)
    vertical      : X=0.40, Y=0.15, Z=1.00   (Z 강조)

    Parameters
    ----------
    traj : np.ndarray
        shape (T, 3), 리샘플링된 궤적
    label : str
        궤적의 라벨 이름
    scale : float, optional
        전체 축 가중치의 배율(강도). 1.0이면 base 비율 그대로 사용.

    Returns
    -------
    np.ndarray
        라벨별 가중치/제거가 적용된 궤적
    """
    base = {
        "circle": (1.0, 1.0, 1.0),
        "diagonal_left":  (0.16, 1.00, 0.86),  # Y 강조
        "diagonal_right": (0.16, 0.86, 1.00),  # Z 강조
        "horizontal": (1.0, 0.15, 0.4),        # X 강조
        "vertical":   (0.4, 0.15, 1.0),        # Z 강조
    }

    if label not in base:
        return traj.astype(np.float32)

    weights = np.asarray(base[label], dtype=np.float32) * float(scale)

    return (traj * weights).astype(np.float32)


def build_dataset(
    data_root: str,
    save_dir: str,
    target_len: int = 128,
    weight_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    전체 데이터셋을 구축하고 X, y 배열을 저장/반환한다.

    디렉토리 구조 가정
    ------------------
    data_root/
    circle/
        1.txt, 2.txt, ...
    diagonal_left/
        ...
    diagonal_right/
        ...
    horizontal/
        ...
    vertical/
        ...

    처리 순서
    ---------
    1) 각 라벨 폴더에서 *.txt 파일을 순회
    2) load_trajectory → normalize_origin → normalize_scale → resample_trajectory
    3) 가중치 없는 버전과, apply_label_weights(라벨별 축 가중치/제거 적용) 버전을 모두 생성
    4) X, y를 npy 파일로 저장

    Parameters
    ----------
    data_root : str
        원본 txt 파일이 들어 있는 상위 디렉토리 경로
    save_dir : str
        npy 결과를 저장할 디렉토리 경로
    target_len : int, optional
        리샘플링 후 각 궤적의 길이 (기본값: 128)
    weight_scale : float, optional
        축 가중치의 전체 강도 배율

    Returns
    -------
    X : np.ndarray
        shape (N, target_len, 3), 전처리된 궤적 데이터
    y : np.ndarray
        shape (N,), 정수 라벨 인덱스(0 ~ 4)
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    label_to_idx = {label: i for i, label in enumerate(LABELS)}

    for label in LABELS:
        folder = os.path.join(data_root, label)
        paths = sorted(glob.glob(os.path.join(folder, "*.txt")))

        for p in paths:
            traj = load_trajectory(p)
            traj = normalize_origin(traj)
            traj = normalize_scale(traj)
            traj_resampled = resample_trajectory(traj, target_len=target_len)

            # (1) 가중치 없는 버전
            X_list.append(traj_resampled.astype(np.float32))
            y_list.append(label_to_idx[label])

            # (2) 가중치 적용된 버전
            traj_weighted = apply_label_weights(traj_resampled, label, scale=weight_scale)
            X_list.append(traj_weighted)
            y_list.append(label_to_idx[label])


    if not X_list:
        raise ValueError(f"No trajectories built from {data_root}")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)

    return X, y


def main() -> None:
    """
    커맨드라인에서 실행할 때의 진입점 함수.

    예시
    ----
    python preprocess.py \\
        --data-root ../data \\
        --weight-scale 1.0

    - data_root 기본값은 프로젝트 루트 기준 "data"
    - save_dir 기본값은 프로젝트 루트 기준 "data/result"
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Label-aware preprocessing with per-class axis weights",
    )

    # 현재 파일(src/preprocess.py)의 위치 기준으로 프로젝트 루트 계산
    current_dir = os.path.dirname(os.path.abspath(__file__))      # .../src
    project_root = os.path.dirname(current_dir)                   # 프로젝트 루트
    default_data_root = os.path.join(project_root, "data/raw")        # .../data/raw
    default_save_dir = os.path.join(project_root, "data", "result")  # .../data/result

    parser.add_argument(
        "--data-root",
        default=default_data_root,
        help="입력 txt 데이터 루트 디렉토리 (기본값: project_root/data)",
    )
    parser.add_argument(
        "--save-dir",
        default=default_save_dir,
        help="X.npy, y.npy를 저장할 디렉토리 (기본값: project_root/data/result)",
    )
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        help="축 가중치 전체 강도 배율 (기본값: 1.0)",
    )

    args = parser.parse_args()

    X, y = build_dataset(
        data_root=args.data_root,
        save_dir=args.save_dir,
        target_len=128,
        weight_scale=args.weight_scale,
    )

    print(f"Saved preprocessed dataset to '{args.save_dir}'")
    print("X shape:", X.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    main()
