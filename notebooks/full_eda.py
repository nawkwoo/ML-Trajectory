"""
전체 데이터 탐색적 분석 v2 (에러 처리 강화)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

sns.set_style("whitegrid")

# ==========================================
# 1. 데이터 로딩 (강화된 에러 처리)
# ==========================================
print("="*60)
print("1. 데이터 로딩 중...")
print("="*60)

def load_trajectory(file_path):
    """단일 궤적 로딩 (에러 처리 강화)"""
    try:
        data = pd.read_csv(file_path, header=None)
        
        if data.shape[1] <= 6:
            return None
            
        positions = data[6].str.split('/', expand=True)
        
        if positions.shape[1] != 3:
            return None
        
        positions.columns = ['X', 'Y', 'Z']
        positions = positions.astype(float)
        
        # NaN 제거
        if positions.isnull().any().any():
            positions = positions.dropna()
        
        if len(positions) < 10:
            return None
            
        return positions.values
        
    except Exception as e:
        return None

# 모든 데이터 로딩
data_dir = Path("Data")
class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

trajectories = []
labels = []
class_names = [d.name for d in class_dirs]
failed_files = []

for class_idx, class_dir in enumerate(class_dirs):
    files = list(class_dir.glob("*.txt")) + list(class_dir.glob("*.csv"))
    print(f"Loading {class_dir.name}: {len(files)} files", end=" ")
    
    success_count = 0
    for file_path in files:
        traj = load_trajectory(file_path)
        if traj is not None:
            trajectories.append(traj)
            labels.append(class_idx)
            success_count += 1
        else:
            failed_files.append(file_path.name)
    
    print(f"→ {success_count}개 성공")

print(f"\n✓ 로딩 완료: {len(trajectories)}개 궤적")
print(f"클래스: {class_names}")

if failed_files:
    print(f"\n⚠️  로딩 실패한 파일 ({len(failed_files)}개):")
    for f in failed_files[:5]:  # 최대 5개만 표시
        print(f"  - {f}")
    if len(failed_files) > 5:
        print(f"  ... 외 {len(failed_files)-5}개")

# ==========================================
# 2. 기본 통계
# ==========================================
print("\n" + "="*60)
print("2. 기본 통계")
print("="*60)

class_counts = [labels.count(i) for i in range(len(class_names))]
for name, count in zip(class_names, class_counts):
    print(f"  {name}: {count}개")

lengths = [len(traj) for traj in trajectories]
print(f"\n시퀀스 길이:")
print(f"  평균: {np.mean(lengths):.2f}")
print(f"  최소: {min(lengths)}")
print(f"  최대: {max(lengths)}")
print(f"  표준편차: {np.std(lengths):.2f}")

all_coords = np.vstack(trajectories)
print(f"\n좌표 범위:")
print(f"  X: [{np.nanmin(all_coords[:, 0]):.1f}, {np.nanmax(all_coords[:, 0]):.1f}] mm")
print(f"  Y: [{np.nanmin(all_coords[:, 1]):.1f}, {np.nanmax(all_coords[:, 1]):.1f}] mm")
print(f"  Z: [{np.nanmin(all_coords[:, 2]):.1f}, {np.nanmax(all_coords[:, 2]):.1f}] mm")

# ==========================================
# 3. 시각화 1: 클래스 분포
# ==========================================
print("\n" + "="*60)
print("3. 클래스 분포 시각화")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(class_names, class_counts, color='skyblue', edgecolor='navy', linewidth=2)
axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(class_counts):
    axes[0].text(i, v + 0.5, str(v), ha='center', fontweight='bold', fontsize=11)

colors = plt.cm.Set3(range(len(class_names)))
axes[1].pie(class_counts, labels=class_names, autopct='%1.1f%%', 
           startangle=90, colors=colors, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1].set_title('Class Distribution (%)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/01_class_distribution.png', dpi=150, bbox_inches='tight')
print("✓ 저장: results/figures/01_class_distribution.png")
plt.close()

# ==========================================
# 4. 시각화 2: 시퀀스 길이
# ==========================================
print("\n" + "="*60)
print("4. 시퀀스 길이 분석")
print("="*60)

lengths_by_class = {i: [] for i in range(len(class_names))}
for traj, label in zip(trajectories, labels):
    lengths_by_class[label].append(len(traj))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(lengths, bins=20, color='skyblue', edgecolor='navy', alpha=0.7)
axes[0].axvline(np.mean(lengths), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(lengths):.1f}')
axes[0].set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Sequence Length Distribution', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

length_data = [lengths_by_class[i] for i in range(len(class_names))]
bp = axes[1].boxplot(length_data, labels=class_names, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[1].set_xlabel('Class', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Sequence Length', fontsize=12, fontweight='bold')
axes[1].set_title('Sequence Length by Class', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/02_sequence_length.png', dpi=150, bbox_inches='tight')
print("✓ 저장: results/figures/02_sequence_length.png")
plt.close()

# ==========================================
# 5. 시각화 3: 모든 클래스 3D
# ==========================================
print("\n" + "="*60)
print("5. 모든 클래스 3D 시각화")
print("="*60)

fig = plt.figure(figsize=(15, 10))

for class_idx in range(len(class_names)):
    sample_idx = labels.index(class_idx)
    trajectory = trajectories[sample_idx]
    
    ax = fig.add_subplot(2, 3, class_idx + 1, projection='3d')
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
           linewidth=2, alpha=0.8, color=colors[class_idx])
    
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
              c='green', s=100, marker='o', edgecolor='black', linewidth=2)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
              c='red', s=100, marker='X', edgecolor='black', linewidth=2)
    ax.scatter(0, 0, 0, c='black', s=100, marker='*')
    
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(f'{class_names[class_idx]}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/03_all_classes_3d.png', dpi=150, bbox_inches='tight')
print("✓ 저장: results/figures/03_all_classes_3d.png")
plt.close()

# ==========================================
# 6. 시각화 4: 좌표 분포
# ==========================================
print("\n" + "="*60)
print("6. 좌표 분포 분석")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].hist(all_coords[:, 0], bins=50, color='red', alpha=0.7, edgecolor='darkred')
axes[0, 0].set_xlabel('X (mm)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 0].set_title('X-axis Distribution', fontsize=12, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

axes[0, 1].hist(all_coords[:, 1], bins=50, color='green', alpha=0.7, edgecolor='darkgreen')
axes[0, 1].set_xlabel('Y (mm)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Y-axis Distribution', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

axes[0, 2].hist(all_coords[:, 2], bins=50, color='blue', alpha=0.7, edgecolor='darkblue')
axes[0, 2].set_xlabel('Z (mm)', fontsize=11, fontweight='bold')
axes[0, 2].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 2].set_title('Z-axis Distribution', fontsize=12, fontweight='bold')
axes[0, 2].grid(axis='y', alpha=0.3)

axes[1, 0].scatter(all_coords[:, 0], all_coords[:, 1], alpha=0.1, s=1, c='navy')
axes[1, 0].scatter(0, 0, c='red', s=100, marker='*', label='Origin')
axes[1, 0].set_xlabel('X (mm)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Y (mm)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('XY Plane', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(all_coords[:, 0], all_coords[:, 2], alpha=0.1, s=1, c='navy')
axes[1, 1].scatter(0, 0, c='red', s=100, marker='*', label='Origin')
axes[1, 1].set_xlabel('X (mm)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Z (mm)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('XZ Plane', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].scatter(all_coords[:, 1], all_coords[:, 2], alpha=0.1, s=1, c='navy')
axes[1, 2].scatter(0, 0, c='red', s=100, marker='*', label='Origin')
axes[1, 2].set_xlabel('Y (mm)', fontsize=11, fontweight='bold')
axes[1, 2].set_ylabel('Z (mm)', fontsize=11, fontweight='bold')
axes[1, 2].set_title('YZ Plane', fontsize=12, fontweight='bold')
axes[1, 2].legend(fontsize=10)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/04_coordinate_distribution.png', dpi=150, bbox_inches='tight')
print("✓ 저장: results/figures/04_coordinate_distribution.png")
plt.close()

# ==========================================
# 7. 클래스별 통계
# ==========================================
print("\n" + "="*60)
print("7. 클래스별 통계 계산")
print("="*60)

class_stats = []
for class_idx, class_name in enumerate(class_names):
    class_trajs = [traj for traj, label in zip(trajectories, labels) if label == class_idx]
    all_class_coords = np.vstack(class_trajs)
    lengths_class = [len(traj) for traj in class_trajs]
    
    stats = {
        'class': class_name,
        'n_samples': len(class_trajs),
        'avg_length': np.mean(lengths_class),
        'std_length': np.std(lengths_class),
        'min_length': min(lengths_class),
        'max_length': max(lengths_class),
        'x_mean': np.nanmean(all_class_coords[:, 0]),
        'y_mean': np.nanmean(all_class_coords[:, 1]),
        'z_mean': np.nanmean(all_class_coords[:, 2]),
        'x_std': np.nanstd(all_class_coords[:, 0]),
        'y_std': np.nanstd(all_class_coords[:, 1]),
        'z_std': np.nanstd(all_class_coords[:, 2]),
    }
    class_stats.append(stats)

df_stats = pd.DataFrame(class_stats)
df_stats.to_csv('results/class_statistics.csv', index=False)
print("\n클래스별 통계:")
print(df_stats.to_string(index=False))
print("\n✓ 저장: results/class_statistics.csv")

print("\n" + "="*60)
print("✓ EDA 완료!")
print("="*60)
print("\n생성된 파일:")
print("  - results/figures/01_class_distribution.png")
print("  - results/figures/02_sequence_length.png")
print("  - results/figures/03_all_classes_3d.png")
print("  - results/figures/04_coordinate_distribution.png")
print("  - results/class_statistics.csv")