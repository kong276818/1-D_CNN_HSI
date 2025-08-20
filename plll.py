import numpy as np
import matplotlib.pyplot as plt

arr = np.load(r"C:\Users\jykong\Desktop\pp\250818\npy_label\sample__SIR_20250820_104101.npy")
print("shape:", arr.shape)   # (2042, 104)

# 예: 0번 픽셀의 스펙트럼
pixel_idx = 20
spectrum = arr[pixel_idx, :]

plt.plot(spectrum)
plt.xlabel("Band index")
plt.ylabel("Value")
plt.title(f"Spectrum of pixel {pixel_idx}")
plt.show()
