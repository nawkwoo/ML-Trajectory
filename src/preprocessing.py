import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

print("="*60)
print("데이터 전처리 시작")
print("="*60)

# ==========================================
# 1. 데이터 로딩
# ==========================================
print("\n[1] 데이터 로딩 중...")

def load_trajectory(file_path):
    """단일 궤적 로딩"""
    try:
        data = pd.read_csv(file_path, header=None)
        if data.shape[1] <= 6:
            return None
        positions = data[6].str.split('/', expand=True)
        if positions.shape[1] != 3:
            return None
        positions.columns = ['X', 'Y', 'Z']
        positions = positions.astype(float)
        if positions.isnull().any().any():
            positions = positions.dropna()
        if len(positions) < 10:
            return None
        return positions.values
    except:
        return None

data_dir = Path("Data")
class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

trajectories = []
labels = []
class_names = [d.name for d in class_dirs]

for class_idx, class_dir in enumerate(class_dirs):
    files = list(class_dir.glob("*.txt")) + list(class_dir.glob("*.csv"))
    success = 0
    for file_path in files:
        traj = load_trajectory(file_path)
        if traj is not None:
            trajectories.append(traj)
            labels.append(class_idx)
            success += 1
    print(f"  {class_dir.name}: {success}개 로딩")

print(f"\n✓ 총 {len(trajectories)}개 궤적 로딩 완료")
print(f"  클래스: {class_names}")

# ==========================================
# 2. 데이터 분할 (Train/Val/Test)
# ==========================================
print("\n[2] Train/Val/Test 분할")

# Train+Val / Test 분리 (70+15 / 15)
X_temp, X_test, y_temp, y_test = train_test_split(
    trajectories, labels,
    test_size=0.15,
    random_state=42,
    stratify=labels
)

# Train / Val 분리
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.176,  # 0.15 / 0.85 ≈ 0.176
    random_state=42,
    stratify=y_temp
)

print(f"  Train: {len(X_train)}개 ({len(X_train)/len(trajectories)*100:.1f}%)")
print(f"  Val:   {len(X_val)}개 ({len(X_val)/len(trajectories)*100:.1f}%)")
print(f"  Test:  {len(X_test)}개 ({len(X_test)/len(trajectories)*100:.1f}%)")

# 클래스 분포 확인
print(f"\n  Train 클래스 분포: {np.bincount(y_train)}")
print(f"  Val 클래스 분포:   {np.bincount(y_val)}")
print(f"  Test 클래스 분포:  {np.bincount(y_test)}")

# ==========================================
# 3. 시퀀스 길이 정규화 (리샘플링)
# ==========================================
print("\n[3] 시퀀스 길이 정규화 (리샘플링)")

def resample_trajectory(traj, target_length):
    """선형 보간으로 리샘플링"""
    current_len = len(traj)
    if current_len == target_length:
        return traj
    
    old_indices = np.linspace(0, current_len - 1, current_len)
    new_indices = np.linspace(0, current_len - 1, target_length)
    
    resampled = np.zeros((target_length, 3))
    for i in range(3):  # X, Y, Z
        resampled[:, i] = np.interp(new_indices, old_indices, traj[:, i])
    
    return resampled

# 목표 길이 설정 (평균값 사용)
train_lengths = [len(traj) for traj in X_train]
target_length = int(np.mean(train_lengths))
print(f"  목표 길이: {target_length}")
print(f"  (Train 평균 길이 기준)")

# 리샘플링 적용
X_train_resampled = [resample_trajectory(traj, target_length) for traj in X_train]
X_val_resampled = [resample_trajectory(traj, target_length) for traj in X_val]
X_test_resampled = [resample_trajectory(traj, target_length) for traj in X_test]

print(f"✓ 모든 시퀀스 길이: {target_length}")

# ==========================================
# 4. 정규화 (Z-score)
# ==========================================
print("\n[4] 정규화 (Z-score Scaling)")

# Train 데이터로 스케일러 학습
all_train_points = np.vstack(X_train_resampled)
print(f"  Train 데이터 통계:")
print(f"    평균: X={all_train_points[:, 0].mean():.2f}, "
      f"Y={all_train_points[:, 1].mean():.2f}, "
      f"Z={all_train_points[:, 2].mean():.2f}")
print(f"    표준편차: X={all_train_points[:, 0].std():.2f}, "
      f"Y={all_train_points[:, 1].std():.2f}, "
      f"Z={all_train_points[:, 2].std():.2f}")

# StandardScaler 학습
scaler = StandardScaler()
scaler.fit(all_train_points)

# 스케일링 적용
X_train_scaled = []
for traj in X_train_resampled:
    scaled = scaler.transform(traj)
    X_train_scaled.append(scaled)

X_val_scaled = []
for traj in X_val_resampled:
    scaled = scaler.transform(traj)
    X_val_scaled.append(scaled)

X_test_scaled = []
for traj in X_test_resampled:
    scaled = scaler.transform(traj)
    X_test_scaled.append(scaled)

# 확인
all_train_scaled = np.vstack(X_train_scaled)
print(f"\n  정규화 후 Train 통계:")
print(f"    평균: X={all_train_scaled[:, 0].mean():.4f}, "
      f"Y={all_train_scaled[:, 1].mean():.4f}, "
      f"Z={all_train_scaled[:, 2].mean():.4f}")
print(f"    표준편차: X={all_train_scaled[:, 0].std():.4f}, "
      f"Y={all_train_scaled[:, 1].std():.4f}, "
      f"Z={all_train_scaled[:, 2].std():.4f}")

print(f"✓ 정규화 완료 (평균≈0, 표준편차≈1)")

# ==========================================
# 5. Numpy Array로 변환
# ==========================================
print("\n[5] 최종 형태로 변환")

X_train = np.array(X_train_scaled)
X_val = np.array(X_val_scaled)
X_test = np.array(X_test_scaled)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

print(f"  X_train: {X_train.shape} (샘플 수, 시퀀스 길이, 특징 수)")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_val:   {y_val.shape}")
print(f"  y_test:  {y_test.shape}")

# ==========================================
# 6. 저장
# ==========================================
print("\n[6] 저장 중...")

save_dir = Path("data/processed")
save_dir.mkdir(parents=True, exist_ok=True)

# 데이터 저장
np.save(save_dir / 'X_train.npy', X_train)
np.save(save_dir / 'X_val.npy', X_val)
np.save(save_dir / 'X_test.npy', X_test)
np.save(save_dir / 'y_train.npy', y_train)
np.save(save_dir / 'y_val.npy', y_val)
np.save(save_dir / 'y_test.npy', y_test)

# 스케일러 저장
with open(save_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 메타 정보 저장
metadata = {
    'target_length': target_length,
    'n_classes': len(class_names),
    'class_names': class_names,
    'n_train': len(X_train),
    'n_val': len(X_val),
    'n_test': len(X_test)
}

import json
with open(save_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ 저장 완료: {save_dir}/")
print(f"  ├── X_train.npy")
print(f"  ├── X_val.npy")
print(f"  ├── X_test.npy")
print(f"  ├── y_train.npy")
print(f"  ├── y_val.npy")
print(f"  ├── y_test.npy")
print(f"  ├── scaler.pkl")
print(f"  └── metadata.json")

print("\n" + "="*60)
print("✓ 전처리 완료!")
print("="*60)