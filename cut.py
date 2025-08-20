import numpy as np
import os

# 1. 실제 데이터 불러오기
original_array = np.load(r"C:\Users\jykong\Desktop\pp\250818\npy\new00__SIR_20250819_181010.npy")
print(f"원본 배열의 shape: {original_array.shape}")  # (2042, 104)

# 2. 앞에서부터 자를 크기 지정
cut_size = 500

# 3. 배열 자르기 (2차원 인덱싱 사용)
cut_part = original_array[:, :cut_size]
remaining_part = original_array[cut_size:, :]

print(f"잘라낸 앞부분의 shape: {cut_part.shape}")
print(f"남은 뒷부분의 shape: {remaining_part.shape}")

# 4. 저장할 경로 지정
save_dir = r"C:\Users\jykong\Desktop\pp"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "cut_data.npy")

# 5. 잘라낸 앞부분 저장
np.save(save_path, cut_part)
print(f"\n'{save_path}' 파일로 잘라낸 부분이 저장되었습니다.")

# (선택) 저장된 파일 불러와서 확인하기
loaded_cut_part = np.load(save_path)
print(f"저장된 파일을 다시 불러온 shape: {loaded_cut_part.shape}")
