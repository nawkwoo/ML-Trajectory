# ML-Trajectory

3D end-effector 궤적을 분류하기 위한 전처리·증강·모델 파이프라인 정리.

## 1) Data 전처리
- 원본 구조: `Data/{label}/*.txt`, label ∈ {circle, diagonal_left, diagonal_right, horizontal, vertical}
- 사용 컬럼: 7번째 컬럼(index 6, X/Y/Z 텍스트)만 파싱 → `(T,3)` 궤적
- 전처리 단계 (`src/preprocess.py`):
  1) 원점 이동(normalize_origin)
  2) 스케일 정규화(최대 거리=1)
  3) 길이 100으로 보간(resample_trajectory)
  4) `results/preprocessed_data/X.npy (N,100,3)`, `y.npy (N,)`, `quality.npy (N,)` 생성  
     - `quality`: 0=1st(고품질), 1=2nd(노이즈 포함)
- 실행: `python src/preprocess.py`

## 2) Data 증강
- 스크립트: `src/augment_data.py` (기본: 1st만 증강, 원본은 1st+2nd 모두 포함)
- 주요 증강: noise, time shift, crop+resample, masking, XY 회전, same-class mixup
- 옵션:
  - `--quality-filter {0,1,all}` (기본 0: 1st만 증강)
  - `--noise-sigma` (기본 0.01), `--mask-max-ratio` (기본 0.1)
- 입력: `results/preprocessed_data/*.npy`  
- 출력: `results/augmented_data/X.npy, y.npy, quality.npy`
- 실행: `python src/augment_data.py`

## 3) ML 모델 (SVM / RandomForest)
- 스크립트: `src/ml_model.py`
- 데이터: `results/augmented_data`(필수; 없으면 오류)
- 입력: `(100,3)` → flatten 300차원
- 평가:  
  - 1st train → 2nd test  
  - 동일 분포 5-fold CV
- 최신 결과(augmented_data 기준):
  - 1st→2nd: SVM acc **0.964**, RF acc **0.929**
  - 5-fold: SVM **0.985 ± 0.014**, RF **0.989 ± 0.009**
- 저장: 실행 시 `models/ml_model.pkl` (SVM+RF dict)
- 실행: `python src/ml_model.py`

## 4) DL1 모델 (GRU, 1st→2nd)
- 스크립트: `src/dl_model_1.py`
- 데이터: `results/augmented_data`(필수; 없으면 오류)
- 전처리: 채널별 mean/std 정규화
- 평가:
  - 동일 분포 5-fold (40 epochs)
  - 1st train → 2nd test (80 epochs)
- 최신 결과(augmented_data 기준):
  - 1st→2nd: acc **0.786**
  - 5-fold: **0.962 ± 0.024**
- 저장: 실행 시 `models/dl_model_1.pt`
- 실행: `python src/dl_model_1.py`

## 5) DL2 모델 (GRU, k-fold + 전체 재학습)
- 스크립트: `src/dl_model_2.py`
- 데이터: `results/augmented_data`(필수; 없으면 오류)
- 전처리: 채널별 mean/std 정규화
- 평가:
  - 동일 분포 5-fold (60 epochs)
  - 1st train → 2nd test (80 epochs)
- 최신 결과(augmented_data 기준):
  - 1st→2nd: acc **0.786**
  - 5-fold: **0.970 ± 0.019**
- 저장: 실행 시 전체 데이터로 재학습 후 `models/dl_model_2.pt`
- 실행: `python src/dl_model_2.py`

## 기타
- 모델 파일 저장 위치: `models/`
- 불필요 파일 정리 완료: 구버전 모델/스크립트 삭제됨
