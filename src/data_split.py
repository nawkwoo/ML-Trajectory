"""
Split augmented trajectory data into train/test sets.

기본 파이프라인
---------------
1) project_root/data/augmented 에서 증강된 궤적 데이터(X.npy, y.npy) 로드
2) 라벨 y를 기준으로 stratified 8:2(train:test) 분할 수행
- sklearn.model_selection.train_test_split 사용
- test_size=0.2, random_state=0, stratify=y
3) 분할된 데이터를 project_root/data/split 디렉토리에 저장:
- X_train.npy, y_train.npy
- X_test.npy, y_test.npy

"""

import os

import numpy as np
from sklearn.model_selection import train_test_split


def main() -> None:
    """
    커맨드라인에서 실행할 때의 진입점 함수.

    예시
    ----
    python split_data.py

    - 입력:  project_root/data/augmented/X.npy, y.npy
    - 출력:  project_root/data/split/X_train.npy, y_train.npy, X_test.npy, y_test.npy
    """
    # 현재 파일(src/split_data.py)의 위치 기준으로 프로젝트 루트 계산
    current_dir = os.path.dirname(os.path.abspath(__file__))     # .../src
    project_root = os.path.dirname(current_dir)                  # 프로젝트 루트

    aug_dir = os.path.join(project_root, "data", "augmented")
    split_dir = os.path.join(project_root, "data", "split")

    x_path = os.path.join(aug_dir, "X.npy")
    y_path = os.path.join(aug_dir, "y.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Augmented files not found in '{aug_dir}'. "
            "Run the augmentation script (augment.py) first."
        )

    # 1) 증강 데이터 로드
    X = np.load(x_path)
    y = np.load(y_path)

    print("Augmented dataset:", X.shape, y.shape)

    # 2) stratified 8:2 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )

    print("Train set:", X_train.shape, y_train.shape)
    print("Test  set:", X_test.shape, y_test.shape)

    # 3) 결과 저장
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, "X_train.npy"), X_train)
    np.save(os.path.join(split_dir, "y_train.npy"), y_train)
    np.save(os.path.join(split_dir, "X_test.npy"), X_test)
    np.save(os.path.join(split_dir, "y_test.npy"), y_test)

    print(f"Saved split data to '{split_dir}'")


if __name__ == "__main__":
    main()