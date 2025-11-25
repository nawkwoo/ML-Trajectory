import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def load_trajectory(file_path):
    data = pd.read_csv(file_path, header=None)
    positions = data[6].str.split('/', expand=True).astype(float)
    positions.columns = ['X', 'Y', 'Z']
    return positions.values

# 첫 번째 파일 로딩
data_dir = Path("Data")
class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
first_class = class_dirs[0]
files = list(first_class.glob("*.txt")) + list(first_class.glob("*.csv"))
trajectory = load_trajectory(files[0])

# 3D 플롯
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 궤적 그리기
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
        'b-', linewidth=2, alpha=0.7)

# 시작점, 끝점
ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
          c='green', s=200, marker='o', label='Start', edgecolor='black', linewidth=2)
ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
          c='red', s=200, marker='X', label='End', edgecolor='black', linewidth=2)

# 원점 (어깨)
ax.scatter(0, 0, 0, c='black', s=200, marker='*', label='Origin (Shoulder)')

ax.set_xlabel('X (mm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Y (mm)', fontsize=14, fontweight='bold')
ax.set_zlabel('Z (mm)', fontsize=14, fontweight='bold')
ax.set_title(f'Trajectory: {first_class.name}', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/first_trajectory.png', dpi=150, bbox_inches='tight')
print("✓ 저장 완료: results/figures/first_trajectory.png")
plt.show()