# 🧭 ML-Trajectory

3D 로봇 end-effector 궤적을 5개 클래스로 분류하는 머신러닝 프로젝트

---

## 📁 Project Structure

```
ML-Trajectory/
├── src/
│   ├── preprocess.py       # 전처리 + 축 가중치 적용
│   ├── augment.py          # 데이터 증강
│   ├── split_data.py       # Train/Test 분할
│   └── train.py            # RandomForest 학습
├── data/
│   ├── raw/                # Raw TXT 데이터
│   ├── result/             # 전처리 결과
│   ├── augmented/          # 증강 데이터
│   └── split/              # Train/Test 데이터
├── models/                 # 학습된 모델 및 그래프
└── demo/
    ├── data/               # 데모용 raw TXT 궤적
    ├── goal/               # 데모 결과 PNG (예측 요약, 신뢰도 히스토그램, 궤적 플롯)
    └── predict.py          # 예측 스크립트
```

---

## 🧩 Classes

- `0` circle
- `1` diagonal_left
- `2` diagonal_right
- `3` horizontal
- `4` vertical

---

## 🎯 Label-wise Axis Weighting

각 클래스별로 X/Y/Z 축에 다른 가중치를 적용하여 궤적 패턴을 강조합니다.

| Class           | X    | Y    | Z    |
|-----------------|------|------|------|
| circle          | 1.00 | 1.00 | 1.00 |
| diagonal_left   | 0.16 | 1.00 | 0.86 |
| diagonal_right  | 0.16 | 0.86 | 1.00 |
| horizontal      | 0.40 | 1.00 | 0.15 |
| vertical        | 0.40 | 0.15 | 1.00 |

---

## 🚀 Usage

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Preprocessing

```bash
python src/preprocess.py --dr-mode A --weight-scale 1.0
```

### 3. Augmentation

```bash
python src/augment.py
```

### 4. Train/Test Split

```bash
python src/split_data.py
```

### 5. Training

```bash
python src/train.py
```

### 6. Prediction

```bash
python demo/predict.py
```

---

## 📊 Results

- **Test Accuracy**: 0.99+
- **5-Fold CV**: 0.99 ± 0.005
- **Model**: RandomForestClassifier

---

## 📝 Notes

- 각 TXT 파일의 7번째 컬럼은 "X/Y/Z" 형식의 3D 좌표
- 궤적 길이는 128로 리샘플링