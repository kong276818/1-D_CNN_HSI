import numpy as np
import matplotlib.pyplot as plt

# 파일 로드
file1 = np.load(r"C:\Users\jykong\Desktop\pp\npy\class4\sample04__SIR_20250818_145200.npy")
file2 = np.load(r"C:\Users\jykong\Desktop\pp\npy\class5\sample05__SIR_20250818_145605.npy")

print("File1 shape:", file1.shape)
print("File2 shape:", file2.shape)

# 중심 좌표 픽셀 선택
x1, y1 = file1.shape[0] // 2, file1.shape[1] // 2
x2, y2 = file2.shape[0] // 2, file2.shape[1] // 2

spectrum1 = file1[x1, y1, :]
spectrum2 = file2[x2, y2, :]

# 파장 번호 (밴드 인덱스)
bands = np.arange(file1.shape[2])

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(bands, spectrum1, label="File1 Spectrum (Center Pixel)")
plt.plot(bands, spectrum2, label="File2 Spectrum (Center Pixel)")
plt.xlabel("Band Index")
plt.ylabel("Intensity")
plt.title("Spectral Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()