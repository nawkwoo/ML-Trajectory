"""
Circle 클래스 심화 분석
Author: 관우 담당 (윤형 작성)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*60)
print("Circle 클래스 심화 분석")
print("="*60)

def load_trajectory(file_path):
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

# 데이터 로딩
print("\n[1] 데이터 로딩")
trajectories = []
qualities = []

# Original
data_dir_original = Path("Data/original/circle")
if data_dir_original.exists():
    files = list(data_dir_original.glob("*.txt")) + list(data_dir_original.glob("*.csv"))
    print(f"Original circle:")
    success = 0
    for file_path in files:
        traj = load_trajectory(file_path)
        if traj is not None:
            trajectories.append(traj)
            qualities.append(0)
            success += 1
    print(f"  {success}개")

# Noise
data_dir_noise = Path("Data/noise/circle")
if data_dir_noise.exists():
    files = list(data_dir_noise.glob("*.txt")) + list(data_dir_noise.glob("*.csv"))
    print(f"Noise circle:")
    success = 0
    for file_path in files:
        traj = load_trajectory(file_path)
        if traj is not None:
            trajectories.append(traj)
            qualities.append(1)
            success += 1
    print(f"  {success}개")

print(f"\n✓ 총 {len(trajectories)}개")

# 축별 변화량 분석
print("\n[2] 축별 변화량 분석")
x_changes, y_changes, z_changes = [], [], []

for traj in trajectories:
    traj_norm = traj - traj[0:1]
    x_changes.append(traj_norm[:, 0].max() - traj_norm[:, 0].min())
    y_changes.append(traj_norm[:, 1].max() - traj_norm[:, 1].min())
    z_changes.append(traj_norm[:, 2].max() - traj_norm[:, 2].min())

x_changes = np.array(x_changes)
y_changes = np.array(y_changes)
z_changes = np.array(z_changes)

print(f"\n  X축: {x_changes.mean():.2f} ± {x_changes.std():.2f} mm")
print(f"  Y축: {y_changes.mean():.2f} ± {y_changes.std():.2f} mm")
print(f"  Z축: {z_changes.mean():.2f} ± {z_changes.std():.2f} mm")

total_change = x_changes + y_changes + z_changes
x_ratio = (x_changes / total_change).mean()
y_ratio = (y_changes / total_change).mean()
z_ratio = (z_changes / total_change).mean()

print(f"\n  ⭐ 중요도:")
print(f"    X: {x_ratio*100:.1f}%")
print(f"    Y: {y_ratio*100:.1f}%")
print(f"    Z: {z_ratio*100:.1f}%")

# 시각화
print("\n[3] 시각화 생성")
fig = plt.figure(figsize=(20, 12))

# 3D
ax1 = fig.add_subplot(3, 3, 1, projection='3d')
for traj, quality in zip(trajectories, qualities):
    traj_norm = traj - traj[0:1]
    color = 'blue' if quality == 0 else 'red'
    alpha = 0.7 if quality == 0 else 0.4
    ax1.plot(traj_norm[:, 0], traj_norm[:, 1], traj_norm[:, 2], 
            alpha=alpha, linewidth=2, color=color)
ax1.scatter(0, 0, 0, c='green', s=150, marker='o', label='Start')
ax1.set_xlabel('X (mm)', fontweight='bold')
ax1.set_ylabel('Y (mm)', fontweight='bold')
ax1.set_zlabel('Z (mm)', fontweight='bold')
ax1.set_title('3D (Blue=Original, Red=Noise)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.view_init(elev=20, azim=45)

# XY
ax2 = fig.add_subplot(3, 3, 2)
for traj, quality in zip(trajectories, qualities):
    traj_norm = traj - traj[0:1]
    color = 'blue' if quality == 0 else 'red'
    alpha = 0.7 if quality == 0 else 0.4
    ax2.plot(traj_norm[:, 0], traj_norm[:, 1], alpha=alpha, linewidth=2, color=color)
ax2.scatter(0, 0, c='green', s=150, marker='o', zorder=5)
ax2.set_xlabel('X (mm)', fontweight='bold')
ax2.set_ylabel('Y (mm)', fontweight='bold')
ax2.set_title('XY Plane', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_aspect('equal')

# XZ
ax3 = fig.add_subplot(3, 3, 3)
for traj, quality in zip(trajectories, qualities):
    traj_norm = traj - traj[0:1]
    color = 'blue' if quality == 0 else 'red'
    alpha = 0.7 if quality == 0 else 0.4
    ax3.plot(traj_norm[:, 0], traj_norm[:, 2], alpha=alpha, linewidth=2, color=color)
ax3.scatter(0, 0, c='green', s=150, marker='o', zorder=5)
ax3.set_xlabel('X (mm)', fontweight='bold')
ax3.set_ylabel('Z (mm)', fontweight='bold')
ax3.set_title('XZ Plane', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='black', linewidth=0.5)
ax3.axvline(0, color='black', linewidth=0.5)
ax3.set_aspect('equal')

# YZ
ax4 = fig.add_subplot(3, 3, 4)
for traj, quality in zip(trajectories, qualities):
    traj_norm = traj - traj[0:1]
    color = 'blue' if quality == 0 else 'red'
    alpha = 0.7 if quality == 0 else 0.4
    ax4.plot(traj_norm[:, 1], traj_norm[:, 2], alpha=alpha, linewidth=2, color=color)
ax4.scatter(0, 0, c='green', s=150, marker='o', zorder=5)
ax4.set_xlabel('Y (mm)', fontweight='bold')
ax4.set_ylabel('Z (mm)', fontweight='bold')
ax4.set_title('YZ Plane', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(0, color='black', linewidth=0.5)
ax4.axvline(0, color='black', linewidth=0.5)
ax4.set_aspect('equal')

# 축별 변화량
ax5 = fig.add_subplot(3, 3, 5)
axes_data = [x_changes, y_changes, z_changes]
axes_names = ['X', 'Y', 'Z']
colors = ['red', 'green', 'blue']
bp = ax5.boxplot(axes_data, labels=axes_names, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax5.set_ylabel('Range (mm)', fontweight='bold')
ax5.set_title('Axis Movement Range', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 중요도
ax6 = fig.add_subplot(3, 3, 6)
ratios = [x_ratio*100, y_ratio*100, z_ratio*100]
bars = ax6.bar(axes_names, ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for bar, ratio in zip(bars, ratios):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax6.set_ylabel('Importance (%)', fontweight='bold')
ax6.set_title('Relative Importance', fontsize=12, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
ax6.set_ylim(0, 100)

# Original Only XY
ax7 = fig.add_subplot(3, 3, 7)
original_idx = [i for i, q in enumerate(qualities) if q == 0]
if len(original_idx) > 0:
    for i in original_idx:
        traj_norm = trajectories[i] - trajectories[i][0:1]
        ax7.plot(traj_norm[:, 0], traj_norm[:, 1], 'b-', alpha=0.7, linewidth=1.5)
    ax7.scatter(0, 0, c='green', s=150, marker='o', zorder=5)
ax7.set_xlabel('X (mm)', fontweight='bold')
ax7.set_ylabel('Y (mm)', fontweight='bold')
ax7.set_title('Original Only (XY)', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.axhline(0, color='black', linewidth=0.5)
ax7.axvline(0, color='black', linewidth=0.5)
ax7.set_aspect('equal')

# Original Only YZ
ax8 = fig.add_subplot(3, 3, 8)
if len(original_idx) > 0:
    for i in original_idx:
        traj_norm = trajectories[i] - trajectories[i][0:1]
        ax8.plot(traj_norm[:, 1], traj_norm[:, 2], 'b-', alpha=0.7, linewidth=1.5)
    ax8.scatter(0, 0, c='green', s=150, marker='o', zorder=5)
ax8.set_xlabel('Y (mm)', fontweight='bold')
ax8.set_ylabel('Z (mm)', fontweight='bold')
ax8.set_title('Original Only (YZ)', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.axhline(0, color='black', linewidth=0.5)
ax8.axvline(0, color='black', linewidth=0.5)
ax8.set_aspect('equal')

# Original Only XZ
ax9 = fig.add_subplot(3, 3, 9)
if len(original_idx) > 0:
    for i in original_idx:
        traj_norm = trajectories[i] - trajectories[i][0:1]
        ax9.plot(traj_norm[:, 0], traj_norm[:, 2], 'b-', alpha=0.7, linewidth=1.5)
    ax9.scatter(0, 0, c='green', s=150, marker='o', zorder=5)
ax9.set_xlabel('X (mm)', fontweight='bold')
ax9.set_ylabel('Z (mm)', fontweight='bold')
ax9.set_title('Original Only (XZ)', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)
ax9.axhline(0, color='black', linewidth=0.5)
ax9.axvline(0, color='black', linewidth=0.5)
ax9.set_aspect('equal')

plt.tight_layout()
plt.savefig('results/figures/circle_analysis.png', dpi=150, bbox_inches='tight')
print("✓ 저장: results/figures/circle_analysis.png")
plt.close()

# 결론
print("\n" + "="*60)
print("✓ Circle 분석 완료!")
print("="*60)
importance = {'X': x_ratio, 'Y': y_ratio, 'Z': z_ratio}
sorted_axes = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print(f"\n💡 중요도 순위:")
print(f"  1순위: {sorted_axes[0][0]}축 ({sorted_axes[0][1]*100:.1f}%)")
print(f"  2순위: {sorted_axes[1][0]}축 ({sorted_axes[1][1]*100:.1f}%)")
print(f"  3순위: {sorted_axes[2][0]}축 ({sorted_axes[2][1]*100:.1f}%)")
if sorted_axes[2][1] < 0.25:
    print(f"\n  ✅ {sorted_axes[2][0]}축 제거 또는 강한 증강 가능!")
print("="*60)