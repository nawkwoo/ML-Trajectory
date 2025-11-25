import pandas as pd
import numpy as np
from pathlib import Path

def load_trajectory(file_path):
    """궤적 데이터 로딩 테스트"""
    print(f"\n파일 로딩: {file_path}")
    
    try:
        # CSV 읽기
        data = pd.read_csv(file_path, header=None)
        print(f"✓ CSV 로딩 성공")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {data.shape[1]}개")
        
        # Column[6] 확인
        if data.shape[1] > 6:
            print(f"\n  Column[6] 샘플:")
            print(f"  {data[6].head()}")
            
            # X/Y/Z 분리 테스트
            positions = data[6].str.split('/', expand=True).astype(float)
            positions.columns = ['X', 'Y', 'Z']
            print(f"\n✓ 좌표 분리 성공")
            print(f"  Shape: {positions.shape}")
            print(f"\n  첫 5개 좌표:")
            print(positions.head())
            
            return positions.values
        else:
            print(f"✗ Column[6]이 없습니다! (총 {data.shape[1]}개 컬럼)")
            return None
            
    except Exception as e:
        print(f"✗ 에러 발생: {e}")
        return None

# 테스트 실행
data_dir = Path("Data")
class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

if class_dirs:
    first_class = sorted(class_dirs)[0]
    files = list(first_class.glob("*.txt")) + list(first_class.glob("*.csv"))
    
    if files:
        trajectory = load_trajectory(files[0])
        
        if trajectory is not None:
            print(f"\n" + "="*60)
            print("✓ 데이터 로딩 성공!")
            print("="*60)