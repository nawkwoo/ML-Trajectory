"""
전처리 결과 확인
"""
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'   # 맥 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

print("="*60)
print("전처리 결과 확인")
print("="*60)

# 로딩
data_dir = Path("data/processed")

X_train = np.load(data_dir / 'X_train.npy')
X_val = np.load(data_dir / 'X_val.npy')
X_test = np.load(data_dir / 'X_test.npy')
y_train = np.load(data_dir / 'y_train.npy')
y_val = np.load(data_dir / 'y_val.npy')
y_test = np.load(data_dir / 'y_test.npy')

with open(data_dir / 'metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"\n메타데이터:")
print(f"  시퀀스 길이: {metadata['target_length']}")
print(f"  클래스 수: {metadata['n_classes']}")
print(f"  클래스 이름: {metadata['class_names']}")

print(f"\n데이터 shape:")
print(f"  X_train: {X_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")

print(f"\n라벨 분포:")
print(f"  Train: {np.bincount(y_train)}")
print(f"  Val:   {np.bincount(y_val)}")
print(f"  Test:  {np.bincount(y_test)}")

print(f"\n정규화 확인:")
print(f"  X_train 평균: {X_train.mean():.6f}")
print(f"  X_train 표준편차: {X_train.std():.6f}")
print(f"  X_train 최소값: {X_train.min():.2f}")
print(f"  X_train 최대값: {X_train.max():.2f}")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 첫 번째 샘플 시각화
axes[0, 0].plot(X_train[0, :, 0], label='X', alpha=0.7)
axes[0, 0].plot(X_train[0, :, 1], label='Y', alpha=0.7)
axes[0, 0].plot(X_train[0, :, 2], label='Z', alpha=0.7)
axes[0, 0].set_title('첫 번째 Train 샘플 (정규화 후)', fontweight='bold')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Normalized Value')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 값 분포 히스토그램
axes[0, 1].hist(X_train.flatten(), bins=50, alpha=0.7, edgecolor='black')
axes[0, 1].set_title('전체 값 분포', fontweight='bold')
axes[0, 1].set_xlabel('Normalized Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(0, color='red', linestyle='--', label='Mean=0')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 클래스별 샘플 수
class_names = metadata['class_names']
train_counts = np.bincount(y_train)
val_counts = np.bincount(y_val)
test_counts = np.bincount(y_test)

x = np.arange(len(class_names))
width = 0.25

axes[1, 0].bar(x - width, train_counts, width, label='Train', alpha=0.8)
axes[1, 0].bar(x, val_counts, width, label='Val', alpha=0.8)
axes[1, 0].bar(x + width, test_counts, width, label='Test', alpha=0.8)
axes[1, 0].set_xlabel('Class')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('클래스별 데이터 분포', fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. 각 축별 분포
axes[1, 1].hist(X_train[:, :, 0].flatten(), bins=30, alpha=0.5, label='X', color='red')
axes[1, 1].hist(X_train[:, :, 1].flatten(), bins=30, alpha=0.5, label='Y', color='green')
axes[1, 1].hist(X_train[:, :, 2].flatten(), bins=30, alpha=0.5, label='Z', color='blue')
axes[1, 1].set_title('축별 값 분포', fontweight='bold')
axes[1, 1].set_xlabel('Normalized Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/05_preprocessing_result.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 저장: results/figures/05_preprocessing_result.png")
plt.show()

print("\n" + "="*60)
print("✓ 확인 완료!")
print("="*60)