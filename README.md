# ML-Trajectory (ML only)

3D 로봇 end-effector 궤적을 5개 클래스(circle, diagonal_left, diagonal_right, horizontal, vertical)로 분류합니다. 현재 파이프라인은 라벨별 축 가중/제거를 반영한 전처리 → 증강 → 분할 → ML(SVM/RF) 학습으로 구성됩니다.

## 디렉토리
- `Data/` : 원본 TXT (`Data/{label}/*.txt`, 7번째 컬럼의 `X/Y/Z`)
- `src/`  : 스크립트 (`preprocess.py`, `augment_data.py`, `data_split.py`, `ml_model.py`, `pca_diagonal_right.py`, `visualize_diagonal_right.py`)
- `Data/results/` : 전처리·증강·분할 결과  
  - `Data/results/preprocessed_data/`  
  - `Data/results/augmented_data/`  
  - `Data/results/split_data/`  
- `models/` : 학습된 ML 모델 (`ml_modle.pkl`)

## 라벨별 축 규칙 (전처리)
- circle        : X=1.0, Y=1.0, Z=0.5  
- diagonal_left : X=0.5, Y=1.0, Z=0.7  
- diagonal_right: `--dr-mode A` → X=0.2, Y=0.7, Z=0 / `--dr-mode B` → XY만 사용(Z 삭제)  
- horizontal    : X=1.0, Y=0.3, Z=0.3  
- vertical      : X=0.4, Y=0.0, Z=1.0  
공통 전처리: 원점 이동 → 스케일 정규화(최대거리=1) → 길이 100 보간. Z가 삭제된 경우 패딩으로 채널 수를 맞춥니다.

## 실행 순서
1) 전처리:  
```bash
python src/preprocess.py --dr-mode A   # 또는 B
```
출력: `Data/results/preprocessed_data/X.npy, y.npy, quality.npy`

2) 증강(전체 품질 대상):  
```bash
python src/augment_data.py
```
출력: `Data/results/augmented_data/*.npy`

3) 데이터 분할(8:2, stratified):  
```bash
python src/data_split.py
```
출력: `Data/results/split_data/X_train.npy, X_test.npy, ...`

4) ML 학습/평가(SVM/RF):  
```bash
python src/ml_model.py
```
- 학습: train 세트
- 평가: train 5-fold CV + test 세트
  - 저장: `models/ml_modle.pkl` (SVM/RF dict)

## 추가 분석/시각화
- `src/pca_diagonal_right.py` : diagonal_right PCA 및 PC 계수 확인(라벨 상수 설정 필요)
- `src/visualize_diagonal_right.py` : `Data/diagonal_right/*.txt` 3D + XY/XZ/YZ 시각화 (원점 정규화 옵션)
