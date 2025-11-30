"""
Diagonal_Left 클래스 심화 분석
Author: 윤형

- X, Y, Z 중 의미있는 축 찾기
- 클래스 특성 분석
- 맞춤형 증강 방법 제안
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*60)
print("Diagonal_Left 클래스 심화 분석")
print("="*60)

# ==========================================
# 1. 데이터 로딩
# ==========================================
print("\n[1] 데이터 로딩")

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

# Original + Noise 둘 다 로딩
trajectories = []
qualities = []  # 0=original, 1=noise

# Original 데이터
data_dir_original = Path("Data/original/diagonal_left")
if data_dir_original.exists():
    files = list(data_dir_original.glob("*.txt")) + list(data_dir_original.glob("*.csv"))
    print(f"Original diagonal_left:")
    success = 0
    for file_path in files:
        traj = load_trajectory(file_path)
        if traj is not None:
            trajectories.append(traj)
            qualities.append(0)
            success += 1
    print(f"  {success}개 로딩")

# Noise 데이터
data_dir_noise = Path("Data/noise/diagonal_left")
if data_dir_noise.exists():
    files = list(data_dir_noise.glob("*.txt")) + list(data_dir_noise.glob("*.csv"))
    print(f"\nNoise diagonal_left:")
    success = 0
    for file_path in files:
        traj = load_trajectory(file_path)
        if traj is not None:
            trajectories.append(traj)
            qualities.append(1)
            success += 1
    print(f"  {success}개 로딩")

print(f"\n✓ 총 {len(trajectories)}개")
print(f"  Original: {sum(q == 0 for q in qualities)}개")
print(f"  Noise:    {sum(q == 1 for q in qualities)}개")

# ==========================================
# 2. 각 축별 변화량 분석
# ==========================================
print("\n[2] 축별 변화량 분석")

x_changes = []
y_changes = []
z_changes = []

for traj in trajectories:
    traj_norm = traj - traj[0:1]
    
    x_range = traj_norm[:, 0].max() - traj_norm[:, 0].min()
    y_range = traj_norm[:, 1].max() - traj_norm[:, 1].min()
    z_range = traj_norm[:, 2].max() - traj_norm[:, 2].min()
    
    x_changes.append(x_range)
    y_changes.append(y_range)
    z_changes.append(z_range)

x_changes = np.array(x_changes)
y_changes = np.array(y_changes)
z_changes = np.array(z_changes)

# 통계
print(f"\n  X축 변화량:")
print(f"    평균: {x_changes.mean():.2f} mm")
print(f"    표준편차: {x_changes.std():.2f} mm")
print(f"    범위: [{x_changes.min():.2f}, {x_changes.max():.2f}]")

print(f"\n  Y축 변화량:")
print(f"    평균: {y_changes.mean():.2f} mm")
print(f"    표준편차: {y_changes.std():.2f} mm")
print(f"    범위: [{y_changes.min():.2f}, {y_changes.max():.2f}]")

print(f"\n  Z축 변화량:")
print(f"    평균: {z_changes.mean():.2f} mm")
print(f"    표준편차: {z_changes.std():.2f} mm")
print(f"    범위: [{z_changes.min():.2f}, {z_changes.max():.2f}]")

# 상대적 중요도
total_change = x_changes + y_changes + z_changes
x_ratio = (x_changes / total_change).mean()
y_ratio = (y_changes / total_change).mean()
z_ratio = (z_changes / total_change).mean()

print(f"\n  ⭐ 상대적 중요도:")
print(f"    X: {x_ratio*100:.1f}%")
print(f"    Y: {y_ratio*100:.1f}%")
print(f"    Z: {z_ratio*100:.1f}%")

# 판단
print(f"\n  💡 분석:")
importance = {'X': x_ratio, 'Y': y_ratio, 'Z': z_ratio}
sorted_axes = sorted(importance.items(), key=lambda x: x[1], reverse=True)

for i, (axis, ratio) in enumerate(sorted_axes):
    if ratio > 0.4:
        print(f"    {axis}축: 매우 중요 ({ratio*100:.1f}%) ⭐⭐⭐")
    elif ratio > 0.25:
        print(f"    {axis}축: 중요 ({ratio*100:.1f}%) ⭐⭐")
    else:
        print(f"    {axis}축: 덜 중요 ({ratio*100:.1f}%) ⭐")

# ==========================================
# 3. 대각선 방향 분석
# ==========================================
print("\n[3] 대각선 방향 분석")

directions = []
for traj in trajectories:
    start = traj[0]
    end = traj[-1]
    direction = end - start
    directions.append(direction)

directions = np.array(directions)

mean_direction = directions.mean(axis=0)
print(f"\n  평균 이동 방향:")
print(f"    X: {mean_direction[0]:+.2f} mm {'→' if mean_direction[0] > 0 else '←'}")
print(f"    Y: {mean_direction[1]:+.2f} mm {'↑' if mean_direction[1] > 0 else '↓'}")
print(f"    Z: {mean_direction[2]:+.2f} mm {'앞' if mean_direction[2] > 0 else '뒤'}")

xy_angles = np.arctan2(directions[:, 1], directions[:, 0]) * 180 / np.pi
print(f"\n  XY 평면 각도:")
print(f"    평균: {xy_angles.mean():.1f}°")
print(f"    표준편차: {xy_angles.std():.1f}°")
print(f"    (0°=우, 90°=위, 180°=좌, -90°=아래)")

# ==========================================
# 4. 시각화
# ==========================================
print("\n[4] 시각화 생성")

fig = plt.figure(figsize=(20, 12))

# 4-1. 3D 궤적
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
ax1.set_title('3D Trajectories\n(Blue=Original, Red=Noise)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.view_init(elev=20, azim=45)

# 4-2. XY 평면
ax2 = fig.add_subplot(3, 3, 2)
for traj, quality in zip(trajectories, qualities):
    traj_norm = traj - traj[0:1]
    color = 'blue' if quality == 0 else 'red'
    alpha = 0.7 if quality == 0 else 0.4
    ax2.plot(traj_norm[:, 0], traj_norm[:, 1], alpha=alpha, linewidth=2, color=color)

ax2.scatter(0, 0, c='green', s=150, marker='o', zorder=5, label='Start')
ax2.set_xlabel('X (mm)', fontweight='bold')
ax2.set_ylabel('Y (mm)', fontweight='bold')
ax2.set_title('XY Plane View', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_aspect('equal')
ax2.legend()

# 4-3. XZ 평면
ax3 = fig.add_subplot(3, 3, 3)
for traj, quality in zip(trajectories, qualities):
    traj_norm = traj - traj[0:1]
    color = 'blue' if quality == 0 else 'red'
    alpha = 0.7 if quality == 0 else 0.4
    ax3.plot(traj_norm[:, 0], traj_norm[:, 2], alpha=alpha, linewidth=2, color=color)

ax3.scatter(0, 0, c='green', s=150, marker='o', zorder=5)
ax3.set_xlabel('X (mm)', fontweight='bold')
ax3.set_ylabel('Z (mm)', fontweight='bold')
ax3.set_title('XZ Plane View', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='black', linewidth=0.5)
ax3.axvline(0, color='black', linewidth=0.5)
ax3.set_aspect('equal')

# 4-4. YZ 평면 (새로 추가!) ⭐
ax4_new = fig.add_subplot(3, 3, 4)
for traj, quality in zip(trajectories, qualities):
    traj_norm = traj - traj[0:1]
    color = 'blue' if quality == 0 else 'red'
    alpha = 0.7 if quality == 0 else 0.4
    ax4_new.plot(traj_norm[:, 1], traj_norm[:, 2], alpha=alpha, linewidth=2, color=color)

ax4_new.scatter(0, 0, c='green', s=150, marker='o', zorder=5, label='Start')
ax4_new.set_xlabel('Y (mm)', fontweight='bold')
ax4_new.set_ylabel('Z (mm)', fontweight='bold')
ax4_new.set_title('YZ Plane View', fontsize=12, fontweight='bold')
ax4_new.grid(True, alpha=0.3)
ax4_new.axhline(0, color='black', linewidth=0.5)
ax4_new.axvline(0, color='black', linewidth=0.5)
ax4_new.set_aspect('equal')
ax4_new.legend()

# 4-5. 축별 변화량 비교
ax4 = fig.add_subplot(3, 3, 5)
axes_data = [x_changes, y_changes, z_changes]
axes_names = ['X', 'Y', 'Z']
colors = ['red', 'green', 'blue']

bp = ax4.boxplot(axes_data, labels=axes_names, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax4.set_ylabel('Range (mm)', fontweight='bold')
ax4.set_title('Axis Movement Range', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 4-6. 축별 중요도
ax5 = fig.add_subplot(3, 3, 6)
ratios = [x_ratio*100, y_ratio*100, z_ratio*100]
bars = ax5.bar(axes_names, ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

for bar, ratio in zip(bars, ratios):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax5.set_ylabel('Importance (%)', fontweight='bold')
ax5.set_title('Relative Importance of Each Axis', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
ax5.set_ylim(0, 100)

# 4-7. 각도 분포
ax6 = fig.add_subplot(3, 3, 7)
ax6.hist(xy_angles, bins=15, color='purple', alpha=0.7, edgecolor='black')
ax6.axvline(xy_angles.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {xy_angles.mean():.1f}°')
ax6.set_xlabel('Angle (degrees)', fontweight='bold')
ax6.set_ylabel('Frequency', fontweight='bold')
ax6.set_title('XY Plane Angle Distribution', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 4-8. Original vs Noise 비교 (XY)
ax7 = fig.add_subplot(3, 3, 8)
original_idx = [i for i, q in enumerate(qualities) if q == 0]
noise_idx = [i for i, q in enumerate(qualities) if q == 1]

if len(original_idx) > 0:
    original_trajs = [trajectories[i] for i in original_idx]
    for traj in original_trajs:
        traj_norm = traj - traj[0:1]
        ax7.plot(traj_norm[:, 0], traj_norm[:, 1], 'b-', alpha=0.7, linewidth=1.5)
    ax7.scatter(0, 0, c='green', s=150, marker='o', zorder=5)

ax7.set_xlabel('X (mm)', fontweight='bold')
ax7.set_ylabel('Y (mm)', fontweight='bold')
ax7.set_title('Original Only (XY)', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.axhline(0, color='black', linewidth=0.5)
ax7.axvline(0, color='black', linewidth=0.5)
ax7.set_aspect('equal')

# 4-9. Original vs Noise 비교 (YZ)
ax8 = fig.add_subplot(3, 3, 9)
if len(original_idx) > 0:
    for traj in original_trajs:
        traj_norm = traj - traj[0:1]
        ax8.plot(traj_norm[:, 1], traj_norm[:, 2], 'b-', alpha=0.7, linewidth=1.5)
    ax8.scatter(0, 0, c='green', s=150, marker='o', zorder=5)

ax8.set_xlabel('Y (mm)', fontweight='bold')
ax8.set_ylabel('Z (mm)', fontweight='bold')
ax8.set_title('Original Only (YZ)', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.axhline(0, color='black', linewidth=0.5)
ax8.axvline(0, color='black', linewidth=0.5)
ax8.set_aspect('equal')

plt.tight_layout()
plt.savefig('results/figures/diagonal_left_analysis.png', dpi=150, bbox_inches='tight')
print("✓ 저장: results/figures/diagonal_left_analysis.png")
plt.close()

# ==========================================
# 5. 증강 방법 제안
# ==========================================
print("\n[5] 맞춤형 증강 방법 제안")

print("\n  📋 추천 증강 기법:")

if z_ratio < 0.2:
    print("\n  1️⃣ Z축 노이즈 추가 (강) ⭐⭐⭐")
    print(f"     → Z축 변화 매우 작음 ({z_ratio*100:.1f}%)")
    print("     → 범위: ±20~30mm")
elif z_ratio < 0.3:
    print("\n  1️⃣ Z축 노이즈 추가 (중) ⭐⭐")
    print(f"     → Z축 중요도: {z_ratio*100:.1f}%")
    print("     → 범위: ±10~15mm")
else:
    print("\n  1️⃣ Z축 노이즈 추가 (약) ⭐")
    print(f"     → Z축도 중요 ({z_ratio*100:.1f}%)")
    print("     → 범위: ±5mm")

if x_ratio > 0.35 and y_ratio > 0.35:
    print("\n  2️⃣ XY 평면 회전 ⭐⭐⭐")
    print(f"     → X, Y가 핵심 (X:{x_ratio*100:.1f}%, Y:{y_ratio*100:.1f}%)")
    print("     → 각도: ±10~15°")
else:
    print("\n  2️⃣ XY 평면 회전 (제한) ⭐")
    print("     → 각도: ±5°")

print("\n  3️⃣ 시간 축 증강 ⭐⭐")
print("     → Time Warping, Time Shift")

print("\n  4️⃣ 스케일 변화 ⭐⭐")
print("     → 0.9~1.1배")

# 최종 요약
print("\n" + "="*60)
print("✓ 분석 완료!")
print("="*60)
print(f"\n💡 핵심 결론:")
print(f"  1순위: {sorted_axes[0][0]}축 ({sorted_axes[0][1]*100:.1f}%) ⭐⭐⭐")
print(f"  2순위: {sorted_axes[1][0]}축 ({sorted_axes[1][1]*100:.1f}%) ⭐⭐")
print(f"  3순위: {sorted_axes[2][0]}축 ({sorted_axes[2][1]*100:.1f}%) ⭐")

if z_ratio < 0.25:
    print(f"\n  ✅ Z축 제거 또는 강한 증강 가능!")
    print(f"     → Z축 기여도: {z_ratio*100:.1f}%")
else:
    print(f"\n  ⚠️ 모든 축 중요 - 보수적 증강 필요")

print(f"\n  📊 시각화: results/figures/diagonal_left_analysis.png")
print("="*60)