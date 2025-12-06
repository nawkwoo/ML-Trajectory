# ML-Trajectory (ML-only)

3D 로봇 end-effector 궤적을 5개 클래스(circle, diagonal_left, diagonal_right, horizontal, vertical)로 분류하는 파이프라인입니다. 현재는 ML(SVM/RF)만 사용합니다.

## 디렉토리
- `Data/` : 원본 TXT (`Data/{label}/*.txt`, 7번째 컬럼에 `X/Y/Z`)
- `src/` : 스크립트 (`preprocess.py`, `augment_data.py`, `ml_model.py`, `pca_diagonal_right.py`, `visualize_diagonal_right.py`)
- `results/` : 전처리·증강 결과
  - `results/preprocessed_data/` : `X.npy (N,100,3)`, `y.npy`, `quality.npy`
  - `results/augmented_data/`     : 증강된 `X.npy`, `y.npy`, `quality.npy`
- `models/` : 학습된 ML 모델(`ml_model.pkl`)

## 전처리 (TXT → NPY)
- 스크립트: `src/preprocess.py`
- 내용: 원점 이동 → 스케일 정규화(최대거리=1) → 길이 100 보간
- 출력: `results/preprocessed_data/X.npy, y.npy, quality.npy` (quality: 0=1st, 1=2nd)
- 실행:
```bash
python src/preprocess.py
```

## 증강
- 스크립트: `src/augment_data.py`
- 입력: `results/preprocessed_data/*.npy`
- 기본: 1st(quality=0)만 증강, 원본 1st/2nd는 그대로 포함
- 기법: noise, time shift, crop+resample, masking, XY 회전, same-class mixup
- 옵션: `--quality-filter {0,1,all}`, `--noise-sigma`, `--mask-max-ratio`
- 출력: `results/augmented_data/X.npy, y.npy, quality.npy`
- 실행:
```bash
python src/augment_data.py
```

## 모델 (ML: SVM / RandomForest)
- 스크립트: `src/ml_model.py`
- 데이터: `results/augmented_data` (필수)
- 특징: 궤적 `(100,3)`을 평탄화해 300차원 벡터 사용
- 평가:
  - 1st train → 2nd test (일반화)
  - 동일 분포 5-fold 교차검증
- 저장: `models/ml_model.pkl` (SVM/RF dict)
- 실행:
```bash
python src/ml_model.py
```

## 추가 분석/시각화
- `src/pca_diagonal_right.py` : diagonal_right 샘플 PCA, XY/XZ/YZ 뷰 확인 (라벨 상수 설정 필요)
- `src/visualize_diagonal_right.py` : `Data/diagonal_right/*.txt`를 3D + XY/XZ/YZ로 시각화 (원점 정규화 옵션)
