# ML-Trajectory
Machine learning project for classifying robotic motion trajectories using 3D end-effector data

## 데이터 구조
- `Data/Machine Learing(Original)`: 노이즈가 거의 없는 기본 세트.
- `Data/Machine Learning(noise)`: 노이즈가 포함된 세트.
- 각 클래스(`circle`, `diagonal_left`, `diagonal_right`, `horizontal`, `vertical`)별로 TXT가 있으며, 7번째 컬럼([6])의 `x/y/z` 좌표 시계열만 사용.

## 공통 전처리 파이프라인
- 로딩: 콤마로 split → [6] 컬럼을 `x/y/z`로 파싱해 `(T,3)` 시퀀스 생성.
- 리샘플: 선형 보간으로 고정 길이(`--target-len`, 기본 256)로 정규화.
- 정규화: 시퀀스별 평균 0, 표준편차 1(`--norm instance` 기본). 필요 시 전체 std를 사용할 수도 있음.
- 노이즈 프로파일: `preprocess/loader.py::compute_noise_profile`가 오리지널 vs 노이즈 세트의 축별 std 차이를 계산해, 오리지널에 “노이즈 성향”을 주입하는 데 사용.

## 증강 전략 (왜 이렇게 했나)
- 데이터가 적으므로 **오리지널을 기준으로 노이즈 특성을 주입**하고, 실제 노이즈 세트는 증강을 최소화해 신호를 유지.
- 오리지널 증강: 작은 회전(±5~8°), 가우시안 노이즈(노이즈 프로파일 기반), 시간 스트레치(0.95~1.05), 짧은 구간 마스킹 등으로 “노이즈스러운” 샘플을 생성.
- 노이즈 세트: 추가 노이즈 주입은 최소/없음. 정규화·리샘플만 적용해 신호 희석을 방지.
- 커리큘럼 느낌: 오리지널 + 노이즈-주입 샘플로 기반 학습 → 실제 노이즈 샘플을 섞어 분포 적응.
- 검증 분리: 깨끗한/노이즈 샘플을 모두 포함한 검증으로 과도한 증강으로 인한 성능 하락 여부 확인.

## 폴더/스크립트
- `preprocess/loader.py`: 로딩, 리샘플, 정규화, 노이즈 프로파일 계산.
- `preprocess/augment.py`: 회전, 시간 스트레치, 가우시안 노이즈, 구간 마스킹, 노이즈 프로파일 기반 주입.
- `ml/train_ml.py`: 특징 추출(경로 길이, bbox, 속도/가속도/저크 통계, 방향 변화량, 시간 정규화 샘플 32개) → SVM(RBF) 학습/평가 + 선택적 GridSearchCV(C, gamma 그리드). 합성 노이즈 샘플은 학습 전용, 평가는 오리지널+실제 노이즈만 사용. 결과/모델을 `experiments/ml/`에 저장.
- `dl/train_dl.py`: PyTorch(MPS 우선) 기반 CNN+BiLSTM 베이스라인. 공통 전처리 후 DataLoader에서 가벼운 증강을 on-the-fly로 적용. 최적(Val Loss) 상태를 `experiments/dl/`에 저장.
- `analysis/plot_results.py`: 리포트(JSON)를 읽어 혼동행렬, 정밀도/재현율/F1 바 차트, PR 곡선을 생성. (`matplotlib`, `scikit-learn` 필요)

## 실행 예시
- ML (SVM, 그리드 탐색 포함):  
  - 기본 학습: `python ml/train_ml.py --target-len 256 --synthetic-from-original 1 --noise-strength 1.0`  
  - 그리드 탐색: `python ml/train_ml.py --target-len 256 --synthetic-from-original 1 --noise-strength 1.0 --search --C-grid "1,10,100" --gamma-grid "scale,auto,0.01,0.001" --cv-folds 5`
  - 저장: 모델 `experiments/ml/svm.joblib`, 리포트 `svm_report.json` (메타 정보 + confusion matrix + grid search 결과).
- DL (MPS 확인 후): `python dl/train_dl.py --target-len 256 --epochs 100 --synthetic-from-original 1 --noise-strength 1.0`  
  - 저장: 체크포인트 `experiments/dl/cnn_bilstm.pt`, 리포트 `dl_report.json` (레이블, 하이퍼파라미터, best val loss 등).
- 시각화(혼동행렬/지표/PR): `python analysis/plot_results.py --report experiments/ml/svm_report.json --out-prefix experiments/ml/svm`

## 설계 이유 요약
- **전처리 공통화**로 ML/DL 모두 동일한 입력 분포를 보게 함.
- **노이즈 프로파일 기반 증강**으로 오리지널을 “노이즈화”하지만 실제 노이즈 세트는 최소 변형 → 신호 보존과 분포 적응을 동시에 노림.
- **작은 모델/약한 규제**를 기본값으로 설정해 소량 데이터에서 언더피팅을 줄이고, 증강 강도를 설정값으로 쉽게 조절 가능.
- **하이퍼파라미터 탐색(GridSearchCV)**을 ML 쪽에 추가해 소규모 데이터에서도 C/gamma 조합을 빠르게 검증하고, 결과를 리포트에 남김.

## 전처리/증강 상세와 선택 이유
- 입력은 7번째 컬럼([6]) `x/y/z` 시계열만 사용 → 모터/센서 잡음 영향 최소화.
- 리샘플(선형 보간, 기본 256 스텝)로 길이 가변 문제 해결 → ML/DL 모두 고정 입력.
- 정규화: 시퀀스별 평균 0, 표준편차 1 → 개별 궤적의 위치/스케일 편차 보정.
- 노이즈 프로파일: 오리지널 vs 노이즈의 축별 표준편차 차이를 추정 → 오리지널에 “노이즈 느낌”을 주입해 합성 데이터 생성.
- 증강 전략:
  - 오리지널: 소각도 회전, 시간 스트레치, 가우시안 노이즈(프로파일 기반), 구간 마스킹 → 데이터가 적을 때 일반화 성능 향상.
  - 노이즈 세트: 추가 노이즈 없이 전처리만 → 신호 희석을 방지.
  - 합성 노이즈 샘플은 학습에만 사용, 평가는 실제 데이터로만 진행 → 과도한 합성 편향을 막음.

## 데이터 분할/학습 흐름
- 공통: 오리지널+실제 노이즈를 합쳐 stratified split (기본 8:2) → 테스트는 실제 데이터만 포함.
- ML: (옵션) 오리지널 기반 합성 노이즈 샘플을 학습 세트에만 추가 → 특징 추출 후 SVM RBF 학습. `--search` 시 GridSearchCV(k-fold)로 C/gamma 조합 탐색.
- DL: 같은 전처리/정규화 → CNN+BiLSTM (소형) 학습. DataLoader에서 가벼운 증강 on-the-fly. val loss 기준 early stopping.

## 모델 선택/장점
- ML (SVM RBF):
  - 적은 데이터에서 강한 결정 경계를 학습 가능, 특징 중요도 분석(퍼뮤테이션/SHAP) 용이.
  - 특징 세트가 물리적 의미(경로 길이, 속도/가속도/저크, 곡률)를 반영 → 해석성 확보.
- DL (CNN+BiLSTM):
  - CNN으로 짧은 패턴 감지 → BiLSTM이 전체 시퀀스 문맥 파악 → 소형 구조로 과적합 완화.
  - PyTorch MPS로 M3 Pro에서 빠르게 실험 가능.

## 저장되는 산출물로 할 수 있는 것
- `svm_report.json`: 클래스별 정밀도/재현율/F1, 혼동행렬, 그리드 탐색 결과, 테스트 y_true/y_pred/y_proba → 혼동 클래스 및 PR 곡선 확인.
- `svm.joblib`: 추론에 바로 사용 가능(동일 전처리 후 `predict`/`predict_proba`).
- `cnn_bilstm.pt` + `dl_report.json`: DL 모델 추론/재학습에 활용, best val loss, val set 지표/혼동행렬, y_true/y_pred/y_proba 포함 → PR 곡선, 지표 시각화 가능.
- `analysis/plot_results.py`로 혼동행렬, 정밀도/재현율/F1 바 차트, PR 곡선을 생성해 보고서/발표에 삽입.

## 앞으로 해야 할 일(추천)
- 데이터 분할 신뢰성 강화: k-fold 교차검증으로 ML 성능 평균/분산 기록, 깨끗/노이즈 세트를 별도로 평가하는 코드 추가.
- 증강 재현성: 합성/증강 샘플을 `experiments/augmented/` 등에 저장해 추후 재사용 가능하게 하기.
- DL 하이퍼파라미터 실험: 리샘플 길이(256/384), 학습률(1e-3/5e-4), hidden 크기(32/64), 증강 강도(회전/노이즈) 조합을 몇 가지 스윕해 안정적 설정 찾기.
- 강건성 체크: 깨끗/노이즈 각각을 검증·시각화하여 어떤 분포에서 취약한지 확인.
- 리포트/데모: `plot_results.py`로 이미지 생성 → 보고서/발표 자료에 삽입. 데모 스크립트(단일 txt 예측) 추가 고려.

## 실행 파라미터 가이드
- 공통 전처리
  - `--target-len`: 리샘플 길이. 256(기본) → 속도 빠름, 384/512 → 더 많은 시퀀스 정보 보존.
  - `--norm`: `instance`(기본) 시퀀스별 표준화, `global`은 전체 std 사용(값을 따로 넣어야 활용 의미 있음).
  - `--synthetic-from-original`: 오리지널에서 노이즈 프로파일 주입한 합성 샘플 개수(학습 전용). 값 ↑ → 데이터 늘지만 합성 편향 주의.
  - `--noise-strength`: 노이즈 프로파일의 강도 스케일. 값 ↑ → 더 거칠게 노이즈화.
- ML(SVM)
  - `--search`: GridSearchCV 활성화. `--C-grid`, `--gamma-grid`, `--cv-folds`로 탐색 범위/폴드 조정.
  - `--C`, `--gamma`: `--search` 미사용 시 직접 설정. C↑ → 결정경계 복잡, gamma↑ → 국소적 경계.
  - 출력/저장: `experiments/ml/svm.joblib`, `svm_report.json`, `plot_results.py`로 `_cm.png/_metrics.png/_pr.png`.
- DL(CNN+BiLSTM)
  - `--epochs`: 최대 에폭. early stopping으로 자동 중단.
  - `--batch-size`: 8~32 권장(데이터 적음). 값↑ → 덜 noisy한 배치 통계, 값↓ → regularization 효과.
  - `--lr`: 학습률. 1e-3(기본), 더 안정 필요 시 5e-4.
  - `--val-ratio`: 검증 비율. 데이터 적어도 0.2 이상 확보 권장.
  - `--synthetic-from-original`, `--noise-strength`: 학습 세트에 추가되는 합성 노이즈 샘플 수/강도.
  - 출력/저장: `experiments/dl/cnn_bilstm.pt`, `dl_report.json`, `plot_results.py`로 `_cm.png/_metrics.png/_pr.png`.

## 참고: 현재 분할/증강 규칙 요약
- 오리지널+실제 노이즈를 stratified split(ML: train/test, DL: train/val). 합성 노이즈 샘플은 학습에만 사용, 평가에는 포함하지 않음.
- 오리지널만 적극 증강(노이즈 프로파일 기반 노이즈, 소각도 회전, 시간 스트레치, 구간 마스킹). 노이즈 세트는 추가 노이즈 없이 전처리만 적용해 신호 희석 방지.
