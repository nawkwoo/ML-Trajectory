# ML-Trajectory (ML only)

3D 로봇 end-effector 궤적을 5개 클래스(circle, diagonal_left, diagonal_right, horizontal, vertical)로 분류합니다. 라벨별 축 가중/제거를 전처리 단계에 적용한 뒤, 증강 → 분할 → SVM/RF 학습을 수행합니다.

## 디렉토리
- `Data/` : 원본 TXT (`Data/{label}/*.txt`, 7번째 컬럼에 `X/Y/Z`)
- `src/`  : 스크립트 (`preprocess.py`, `augment_data.py`, `data_split.py`, `ml_model.py`, `pca_diagonal_right.py`, `visualize_diagonal_right.py`)
- `Data/results/` : 전처리·증강·분할 결과  
  - `Data/results/preprocessed_data/`  
  - `Data/results/augmented_data/`  
  - `Data/results/split_data/`  
- `models/` : 학습된 ML 모델 (`ml_modle.pkl`)

## 라벨별 축 규칙 (전처리)
- circle        : X=0.2,  Y=1.0, Z=0.7  
- diagonal_left : X=0.55, Y=1.0, Z=0.9  
- diagonal_right: `--dr-mode A` → X=0.16, Y=0.86, Z=1.0 / `--dr-mode B` → XY만 사용(Z 제거)  
- horizontal    : X=0.5,  Y=1.0, Z=0.0  
- vertical      : X=0.4,  Y=0.15, Z=1.0  
공통: 원점 이동 → 스케일 정규화 → 길이 100 보간. Z가 없어지면 패딩으로 채널을 맞춥니다. quality는 모두 0으로 저장.

## 실행 순서
1) 전처리  
```bash
python src/preprocess.py --dr-mode A   # 또는 B
```
출력: `Data/results/preprocessed_data/X.npy, y.npy, quality.npy`

2) 증강 (전체 증강)  
```bash
python src/augment_data.py
```
출력: `Data/results/augmented_data/*.npy`

3) 분할 (8:2, stratified)  
```bash
python src/data_split.py
```
출력: `Data/results/split_data/X_train.npy, X_test.npy, ...`

4) ML 학습/평가 (SVM/RF)  
```bash
python src/ml_model.py
```
- 학습: train 세트
- 평가: train 5-fold CV + test 세트
- 저장: `models/ml_modle.pkl` (SVM/RF dict)

## 최신 학습 결과 (증강 후 8:2 분할, dr-mode A)
- Test set (87 samples):  
  - SVM 정확도: **0.9885**  
  - RF 정확도: **1.0000**
- Train 5-fold CV:  
  - SVM: **0.983 ± 0.014**  
  - RF : **0.997 ± 0.006**

## 추가 분석/시각화
- `src/pca_diagonal_right.py` : diagonal_right PCA 및 PC 계수 확인(라벨 상수 설정 필요)
- `src/visualize_diagonal_right.py` : `Data/diagonal_right/*.txt` 3D + XY/XZ/YZ 시각화 (원점 정규화 옵션)
