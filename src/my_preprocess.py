"""
데이터 전처리 (윤형님 버전)
- 시퀀스 길이 100으로 통일
- 최대 거리 정규화
- augmented_data/ 형식으로 저장
"""
import numpy as np
import pandas as pd
from pathlib import Path

print("="*60)
print("데이터 전처리")
print("="*60)

# ==========================================
# 1. 데이터 로딩
# ==========================================
print("\n[1] 데이터 로딩")

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
# processed 폴더 제외!
class_dirs = sorted([d for d in data_dir.iterdir() 
                     if d.is_dir() and d.name != 'processed'])

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
    print(f"  {class_dir.name}: {success}개")

print(f"\n✓ 총 {len(trajectories)}개 로딩")

# ==========================================
# 2. 원점 이동 및 스케일 정규화
# ==========================================
print("\n[2] 정규화")

def normalize_trajectory(traj):
    """원점 이동 + 최대 거리 정규화"""
    # 원점을 첫 번째 점으로
    traj = traj - traj[0:1]
    
    # 최대 거리로 정규화
    max_dist = np.sqrt((traj**2).sum(axis=1)).max()
    if max_dist > 0:
        traj = traj / max_dist
    
    return traj

trajectories_norm = [normalize_trajectory(traj) for traj in trajectories]
print("✓ 원점 이동 및 스케일 정규화 완료")

# ==========================================
# 3. 시퀀스 길이 통일 (100)
# ==========================================
print("\n[3] 시퀀스 길이 통일 (100)")

def resample_trajectory(traj, target_len=100):
    """선형 보간"""
    current_len = len(traj)
    if current_len == target_len:
        return traj
    
    old_indices = np.linspace(0, current_len - 1, current_len)
    new_indices = np.linspace(0, current_len - 1, target_len)
    
    resampled = np.zeros((target_len, 3))
    for i in range(3):
        resampled[:, i] = np.interp(new_indices, old_indices, traj[:, i])
    
    return resampled

trajectories_resampled = [resample_trajectory(traj, 100) for traj in trajectories_norm]
X = np.array(trajectories_resampled)
y = np.array(labels)

print(f"✓ 최종 shape: {X.shape}")

# ==========================================
# 4. 저장 (augmented_data 형식)
# ==========================================
print("\n[4] 저장")

save_dir = Path("augmented_data")
save_dir.mkdir(exist_ok=True)

# quality: 모두 1st (고품질)로 설정
quality = np.zeros(len(X), dtype=int)

np.save(save_dir / "X.npy", X)
np.save(save_dir / "y.npy", y)
np.save(save_dir / "quality.npy", quality)

print(f"✓ 저장 완료: {save_dir}/")
print(f"  - X.npy: {X.shape}")
print(f"  - y.npy: {y.shape}")
print(f"  - quality.npy: {quality.shape}")

print("\n" + "="*60)
print("✓ 전처리 완료!")
print("="*60)