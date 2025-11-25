import os
from pathlib import Path

# Data 폴더 경로
data_dir = Path("Data")

print("=" * 60)
print("데이터 구조 확인")
print("=" * 60)

# 클래스 폴더들 찾기
class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
class_dirs = sorted(class_dirs)

print(f"\n발견된 클래스 수: {len(class_dirs)}")
print("\n클래스별 파일 수:")

total_files = 0
for class_dir in class_dirs:
    # txt, csv 파일 찾기
    files = list(class_dir.glob("*.txt")) + list(class_dir.glob("*.csv"))
    file_count = len(files)
    total_files += file_count
    
    print(f"  {class_dir.name}: {file_count}개 파일")
    
    # 첫 번째 파일 미리보기
    if files:
        print(f"    └─ 예시: {files[0].name}")

print(f"\n총 파일 수: {total_files}개")

# 첫 번째 파일 열어서 구조 확인
if class_dirs and files:
    first_file = files[0]
    print(f"\n파일 미리보기: {first_file}")
    print("-" * 60)
    
    with open(first_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:  # 처음 5줄만
                print(f"Line {i+1}: {line.strip()}")
            else:
                break
    
    print("-" * 60)
