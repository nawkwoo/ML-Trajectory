"""클래식 ML 베이스라인 학습/평가 스크립트.

필요 패키지: numpy, scikit-learn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from joblib import dump

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
    from sklearn.svm import SVC
except ImportError as exc:  # pragma: no cover - 의존성 안내용
    sys.stderr.write("scikit-learn이 필요합니다. pip install scikit-learn\n")
    raise

from preprocess.augment import apply_noise_profile
from preprocess.loader import compute_noise_profile, load_dataset

CoordSeq = np.ndarray


def extract_features(seq: CoordSeq, sample_points: int = 32) -> np.ndarray:
    """시퀀스에서 기하·동역학 특징을 추출."""
    eps = 1e-8
    diffs = np.diff(seq, axis=0)
    speed = np.linalg.norm(diffs, axis=1)
    accel = np.diff(diffs, axis=0)
    jerk = np.diff(accel, axis=0)

    bbox = seq.max(axis=0) - seq.min(axis=0)
    path_len = speed.sum()

    direction = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + eps)
    turn = np.diff(direction, axis=0)
    turn_mag = np.linalg.norm(turn, axis=1)

    def stats(arr: np.ndarray) -> List[float]:
        return [float(arr.mean()), float(arr.std()), float(arr.max()), float(arr.min())]

    idx = np.linspace(0, len(seq) - 1, num=sample_points).astype(int)
    sampled = seq[idx].flatten()

    jerk_stats = stats(np.linalg.norm(jerk, axis=1)) if len(jerk) else [0.0, 0.0, 0.0, 0.0]
    turn_stats = stats(turn_mag) if len(turn_mag) else [0.0, 0.0, 0.0, 0.0]

    feats = [
        path_len,
        *bbox.tolist(),
        *stats(speed),
        *stats(np.linalg.norm(accel, axis=1)),
        *jerk_stats,
        *turn_stats,
    ]
    feats.extend(sampled.tolist())
    return np.asarray(feats, dtype=np.float32)


def load_base_sets(
    data_root: Path,
    target_len: int,
    norm_mode: str,
) -> Tuple[List[Tuple[CoordSeq, str]], List[Tuple[CoordSeq, str]], dict]:
    """오리지널/노이즈 세트와 노이즈 프로파일을 반환."""
    orig_dir = data_root / "Data" / "Machine Learing(Original)"
    noise_dir = data_root / "Data" / "Machine Learning(noise)"

    original = load_dataset(orig_dir, target_len=target_len, norm_mode=norm_mode)
    noisy = load_dataset(noise_dir, target_len=target_len, norm_mode=norm_mode)
    profile = compute_noise_profile(original_dir=orig_dir, noise_dir=noise_dir, target_len=target_len, norm_mode=norm_mode)
    return original, noisy, profile


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="클래식 ML 베이스라인 학습/평가")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1], help="프로젝트 루트 경로")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "ml",
        help="모델/리포트를 저장할 경로",
    )
    parser.add_argument("--target-len", type=int, default=256, help="리샘플 길이")
    parser.add_argument("--norm", type=str, default="instance", choices=["instance", "global"], help="정규화 모드")
    parser.add_argument("--synthetic-from-original", type=int, default=1, help="오리지널 기반 노이즈 증강 샘플 수")
    parser.add_argument("--noise-strength", type=float, default=1.0, help="노이즈 프로파일 적용 강도")
    parser.add_argument("--test-size", type=float, default=0.2, help="검증 세트 비율")
    parser.add_argument("--C", type=float, default=10.0, help="SVM C")
    parser.add_argument("--gamma", type=str, default="scale", help="SVM gamma")
    parser.add_argument(
        "--search",
        action="store_true",
        help="GridSearchCV로 SVM 하이퍼파라미터 탐색 활성화",
    )
    parser.add_argument(
        "--C-grid",
        type=str,
        default="1,10,100",
        help="search 시 사용할 C 목록(콤마 구분)",
    )
    parser.add_argument(
        "--gamma-grid",
        type=str,
        default="scale,auto,0.01,0.001",
        help="search 시 사용할 gamma 목록(콤마 구분)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="GridSearchCV 폴드 수",
    )
    args = parser.parse_args()

    original, noisy, profile = load_base_sets(
        data_root=args.data_root,
        target_len=args.target_len,
        norm_mode=args.norm,
    )

    base_sequences = original + noisy
    sequences = [seq for seq, _ in base_sequences]
    labels = [label for _, label in base_sequences]

    X_train, X_test, y_train, y_test = train_test_split(
        np.stack([extract_features(seq) for seq in sequences], axis=0),
        labels,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    # 합성 노이즈 샘플은 학습 전용으로만 추가
    if args.synthetic_from_original > 0:
        synth_features: List[np.ndarray] = []
        synth_labels: List[str] = []
        for seq, label in original:
            for _ in range(args.synthetic_from_original):
                augmented = apply_noise_profile(seq, profile=profile, strength=args.noise_strength)
                synth_features.append(extract_features(augmented))
                synth_labels.append(label)
        if synth_features:
            X_train = np.concatenate([X_train, np.stack(synth_features)], axis=0)
            y_train = list(y_train) + synth_labels

    ensure_dir(args.save_dir)

    search_result = None
    if args.search:
        def parse_grid(raw: str):
            out = []
            for v in raw.split(","):
                v = v.strip()
                if v in ("scale", "auto"):
                    out.append(v)
                else:
                    try:
                        out.append(float(v))
                    except ValueError:
                        pass
            return out

        param_grid = {
            "C": parse_grid(args.C_grid),
            "gamma": parse_grid(args.gamma_grid),
            "kernel": ["rbf"],
            "probability": [True],
        }
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator=SVC(),
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        clf = grid.best_estimator_
        search_result = {
            "best_params": grid.best_params_,
            "best_score": grid.best_score_,
            "cv_results": grid.cv_results_,
        }
    else:
        clf = SVC(C=args.C, gamma=args.gamma, kernel="rbf", probability=True)
        clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    report = classification_report(y_test, preds, digits=4)
    report_dict = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()

    print("Classification report:")
    print(report)
    print("Confusion matrix:")
    print(np.array(cm))

    # 저장: 모델, 라벨 매핑, 리포트
    model_path = args.save_dir / "svm.joblib"
    dump(clf, model_path)

    meta = {
        "labels": sorted(set(labels)),
        "target_len": args.target_len,
        "norm": args.norm,
        "synthetic_from_original": args.synthetic_from_original,
        "noise_strength": args.noise_strength,
        "test_size": args.test_size,
        "C": args.C,
        "gamma": args.gamma,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "search_enabled": args.search,
        "search_result": None,
        "y_true": list(y_test),
        "y_pred": list(preds),
        "y_proba": proba.tolist(),
    }
    if search_result:
        # GridSearchCV 결과에서 numpy 타입을 파이썬 기본 타입으로 변환
        cv_res = {}
        for k, v in search_result["cv_results"].items():
            if hasattr(v, "tolist"):
                cv_res[k] = v.tolist()
            else:
                cv_res[k] = v
        meta["search_result"] = {
            "best_params": search_result["best_params"],
            "best_score": float(search_result["best_score"]),
            "cv_results": cv_res,
        }
    with (args.save_dir / "svm_report.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"모델 저장: {model_path}")
    print(f"리포트 저장: {args.save_dir / 'svm_report.json'}")


if __name__ == "__main__":
    main()
