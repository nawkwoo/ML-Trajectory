# ML-Trajectory
3D end-effector 궤적 데이터를 이용해 로봇 동작 클래스를 분류하는 프로젝트.

## 1. 데이터 구조와 전처리

- 원본 데이터 구조  
  - `Data/{label}/*.txt`  
  - 라벨: `circle`, `diagonal_left`, `diagonal_right`, `horizontal`, `vertical`
- 사용 컬럼  
  - 각 파일의 7번째 컬럼(index 6, 문자열 `X/Y/Z`)만 사용해서 궤적을 만든다.

### 전처리 파이프라인 (`preprocess.py`)

1. `load_trajectory`  
   - 텍스트 파일에서 `X/Y/Z` 문자열을 파싱해 `(T, 3)` 형태의 궤적 배열을 생성.
2. `normalize_origin`  
   - 궤적의 첫 점을 원점(0,0,0)으로 이동해 translation 제거.
3. `normalize_scale`  
   - 원점으로부터의 최대 거리를 1로 정규화해 스케일 맞춤.
4. `resample_trajectory`  
   - 모든 궤적을 동일 길이 `target_len=100` 으로 선형 보간.
5. `build_dataset`  
   - 폴더 전체를 읽어 `X`, `y`, `quality` 배열을 생성.

루트 디렉터리에서 다음 명령을 실행하면 전처리된 데이터가 만들어진다.

```bash
python preprocess.py
```

실행 결과, `preprocessed_data/` 폴더에 다음 파일들이 생성된다.

- `X.npy` : `(N, 100, 3)` 전처리된 궤적
- `y.npy` : `(N,)` 클래스 레이블 (0~4)
- `quality.npy` : `(N,)` 데이터 품질 플래그  
  - `0` = 1st (노이즈가 적은 고품질 데이터)  
  - `1` = 2nd (노이즈가 포함된 데이터)

### 1st / 2nd 품질 구분 규칙

파일 번호로 품질을 나눈다.

- `circle`           : 1–8 → 1st, 9–16 → 2nd  
- `diagonal_left`    : 1–7 → 1st, 8–12 → 2nd  
- `diagonal_right`   : 1–7 → 1st, 8–12 → 2nd  
- `horizontal`       : 1–6 → 1st, 7–11 → 2nd  
- `vertical`         : 1–6 → 1st, 7–11 → 2nd  

현재 데이터 기준으로  
1st = 34개, 2nd = 28개, 총 62개 샘플이 생성된다.

---

## 2. Step 1 – 전통적인 ML 기반 분류 (SVM / RandomForest)

실험 코드는 `preprocessed_data/experiment.py` 에 구현되어 있다.

### 특징 구성

- 전처리된 궤적 `(100, 3)`을 평탄화(flatten)해서 300차원 벡터로 사용.

### 데이터 분할

- Train: 1st 고품질 세트 (`quality == 0`, 34개)
- Test: 2nd 노이즈 세트 (`quality == 1`, 28개)

### 모델과 결과 (예시)

- SVM (RBF 커널), RandomForest 사용
- 현재 데이터 기준 대략적인 결과
  - SVM  : 정확도 약 0.93
  - RF   : 정확도 약 0.89
  - 클래스별 precision/recall에 약간의 편차가 존재 → 일부 클래스는 노이즈에 더 취약

### 해석 포인트

- 간단한 전처리와 feature(flatten된 시퀀스)만으로도 전통적인 ML이 꽤 높은 성능을 달성한다.
- 하지만:
  - 시퀀스 구조나 시간 변형(time warp)을 직접 표현하지 못하고
  - 새로운 사람 / 환경에서의 일반화에 대한 불확실성이 남는다.
- 이 한계를 Step 2에서 DL로 보완하는 방향으로 설계했다.

---

## 3. Step 2 – 시퀀스 인지(Sequence-aware) 딥러닝

### 3-1. 1st train → 2nd test (기본 GRU 실험)

- 코드: `preprocessed_data/dl_experiment.py`
- 입력: `(batch, seq_len=100, channels=3)`
- 모델: 단일 GRU 레이어 (hidden size 32) + 최종 시점 hidden state를 사용하는 선형 분류기
- 데이터 분할: Step 1과 동일하게
  - Train: 1st (quality=0)
  - Test : 2nd (quality=1)

실행:

```bash
python preprocessed_data/dl_experiment.py
```

현재 기본 설정에서는 test accuracy가 대략 0.3~0.4 수준으로,  
SVM/RF보다 낮게 나온다.

해석:

- 샘플 수가 매우 적은 상황(Train 34개)에서 파라미터가 많은 시퀀스 모델이 불리하다.
- “딥러닝이라고 항상 전통적인 ML보다 더 잘 되는 것은 아니다”라는 점을 보여준다.
- 모델 구조, 하이퍼파라미터, 증강 전략(time warping, rotation, noise 등)을 어떻게 설계하느냐가 중요하다.

### 3-2. 1st+2nd 전체에 대한 5-fold GRU 평가

- 코드: `preprocessed_data/d1_experiment2.py`
- 사용 데이터:
  - 기본적으로 `augmented_data/`가 존재하면 그 데이터를 사용
  - 없으면 `preprocessed_data/`를 사용
- 방법:
  - 전체 궤적을 채널별 표준화 후
  - GRU 기반 분류기에 대해 5-fold cross-validation 수행

증강 없이 `preprocessed_data`만 사용할 때,  
평균 정확도는 약 0.58 수준(±0.10)으로,  
증강 전 GRU의 기본적인 한계를 보여준다.

---

## 4. 데이터 증강과 Augmented Data

증강 코드는 루트의 `augment_data.py`에 구현되어 있다.  
입력은 `preprocessed_data/X.npy`, `y.npy`, `quality.npy`이며,  
출력은 `augmented_data/X.npy`, `y.npy`, `quality.npy`이다.

실행:

```bash
python augment_data.py
```

### 적용한 증강 기법 요약

각 궤적 `(100, 3)`에 대해 다음 변형들을 생성해 원본에 추가한다.

1. **Noise injection**  
   - 작은 Gaussian 노이즈를 좌표에 더해 센서 노이즈를 모사.
2. **Temporal shift**  
   - 시퀀스 전체를 앞뒤로 몇 스텝 이동시키고 빈 부분은 처음/마지막 값으로 채움.
3. **Crop + resample**  
   - 연속된 구간을 잘라내고 다시 100 스텝으로 보간해 “빠르게/느리게 수행된” 동작을 생성.
4. **Masking**  
   - 일부 구간을 0으로 마스킹해 부분적인 누락/가림 현상 모사.
5. **XY-plane rotation**  
   - XY 평면 상에서 작은 각도(±20도 내외)로 회전시켜 자세/시점 차이를 반영.
6. **Mixup (same class)**  
   - 같은 클래스 샘플 두 개를 선형 결합해 중간 형태의 궤적 생성.  
   - 레이블은 그대로 유지.

이렇게 생성된 증강 샘플을 모두 포함해 `augmented_data/`를 만든다.

### 4-1. 증강 데이터 기반 GRU 5-fold 결과

`augmented_data`를 사용하도록 설정한 뒤,

```bash
python preprocessed_data/d1_experiment2.py
```

를 실행하면,  
평균 정확도 약 0.98(표준편차 약 0.01)의 매우 높은 5-fold 결과가 나온다.

해석:

- 증강으로 데이터 양과 다양성을 크게 늘리면 GRU가 해당 분포 안에서는 거의 완벽하게 동작을 구분할 수 있다는 것을 보여준다.
- 다만, 같은 원본 궤적에서 파생된 증강 샘플들이 서로 다른 fold에 섞여 있을 수 있어,
  “완전히 새로운 궤적”에 대한 일반화 성능은 다소 낙관적으로 평가되었을 가능성이 있다.
- 따라서:
  - **공정한 비교 기준**: Step 1과 동일한 1st train → 2nd test 실험 (SVM/RF vs GRU)
  - **증강 5-fold 결과**: “모델 용량과 증강 효과를 확인하기 위한 추가 실험”으로 제시

---

## 5. 실행 요약

1. 전처리 데이터 생성  
   ```bash
   python preprocess.py
   ```
2. 전통적인 ML 실험 (SVM / RandomForest)  
   ```bash
   python preprocessed_data/experiment.py
   ```
3. GRU – 1st train / 2nd test 실험  
   ```bash
   python preprocessed_data/dl_experiment.py
   ```
4. 증강 데이터 생성  
   ```bash
   python augment_data.py
   ```
5. GRU – 5-fold cross-validation (증강 데이터가 있으면 우선 사용)  
   ```bash
   python preprocessed_data/d1_experiment2.py
   ```

이 구조를 기반으로, 보고서에서는
1) 데이터 분석 및 전처리,  
2) 전통적인 ML의 장점과 한계,  
3) 시퀀스 인지 딥러닝 + 데이터 증강 설계와 결과  
를 순서대로 정리하면 된다.

