# ML-Trajectory (ML-only)

3D 로봇 end-effector 궤적을 5개 클래스(circle, diagonal_left, diagonal_right, horizontal, vertical)로 분류하는 파이프라인입니다. 현재는 ML(SVM/RF)만 사용합니다.

## 디렉토리
- `Data/` : 원본 TXT (`Data/{label}/*.txt`, 7번째 컬럼에 `X/Y/Z`)
- `src/` : 스크립트 (`preprocess.py`, `augment_data.py`, `ml_model.py`, `pca_diagonal_right.py`, `visualize_diagonal_right.py`)
- `results/` : 전처리·증강 결과
  - `results/preprocessed_data/` : `X.npy (N,100,3)`, `y.npy`, `quality.npy`
  - `results/augmented_data/`     : 증강된 `X.npy`, `y.npy`, `quality.npy`
- `models/` : 학습된 ML 모델(`ml_model.pkl`)

## 1안: 기본 파이프라인 (전역 동일 전처리/증강)
- 전처리: `src/preprocess.py`  
  - 원점 이동 → 스케일 정규화 → 길이 100 보간  
  - 출력: `results/preprocessed_data/X.npy, y.npy, quality.npy`  
  - 실행: `python src/preprocess.py`
- 증강: `src/augment_data.py`  
  - 입력: `results/preprocessed_data/*.npy`  
  - 기본: 1st만 증강, noise/shift/crop/mask/XY회전/mixup  
  - 출력: `results/augmented_data/*.npy`  
  - 실행: `python src/augment_data.py`
- 모델(ML): `src/ml_model.py`  
  - 데이터: `results/augmented_data`  
  - 평가: 1st→2nd, 5-fold CV  
  - 저장: `models/ml_model.pkl`  
  - 실행: `python src/ml_model.py`

## 2안: 라벨별 축 가중/제거 반영 파이프라인
- 전처리2: `src/preprocess2.py`  
  - 라벨별 축 스케일 적용 후 길이 100 보간  
  - diagonal_right: `--dr-mode A`(X*0.2, Y*0.7, Z=0) 또는 `B`(XY만, Z삭제)  
  - 라벨별 축 규칙 요약  
    - circle        : X=1.0, Y=1.0, Z=0.5  
    - diagonal_left : X=0.5, Y=1.0, Z=0.7  
    - diagonal_right: A) X=0.2, Y=0.7, Z=0  /  B) XY만 사용(Z 삭제)  
    - horizontal    : X=1.0, Y=0.3, Z=0.3  
    - vertical      : X=0.4, Y=0.0, Z=1.0  
  - 출력: `results/preprocessed_data2/*.npy`  
  - 실행: `python src/preprocess2.py --dr-mode A`
- 증강2: `src/augment_data2.py`  
  - 입력: `results/preprocessed_data2/*.npy`  
  - 기본: 1st만 증강, 동일 기법  
  - 출력: `results/augmented_data2/*.npy`  
  - 실행: `python src/augment_data2.py`
- 모델2(ML): `src/ml_model2.py`  
  - 데이터: `results/augmented_data2`  
  - 평가: 1st→2nd, 5-fold CV  
  - 저장: `models/ml_model2.pkl`  
  - 실행: `python src/ml_model2.py`

## 추가 분석/시각화
- `src/pca_diagonal_right.py` : diagonal_right PCA 및 PC 계수 확인 (라벨 상수 설정 필요)
- `src/visualize_diagonal_right.py` : `Data/diagonal_right/*.txt` 3D + XY/XZ/YZ 시각화 (원점 정규화 옵션)
