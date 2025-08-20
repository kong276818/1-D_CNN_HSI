import os
import numpy as np
from SviParser import SviParser

base_dir = "./250818"
bin_dir = os.path.join(base_dir, "bin")
output_dir = os.path.join(base_dir, "npy")
os.makedirs(output_dir, exist_ok=True)

bin_files = [f for f in os.listdir(bin_dir) if f.lower().endswith(".bin")]

for bin_file in bin_files:
    bin_path = os.path.join(bin_dir, bin_file)
    base_name = os.path.splitext(bin_file)[0]

    try:
        hsi = SviParser(bin_path)
        hsi.parse()
        image = hsi.images  # shape: (H, C, W)

        transposed = image.transpose(2, 0, 1)  # → (W, H, C)
        save_path = os.path.join(output_dir, base_name + ".npy")
        np.save(save_path, transposed)

        print(f"[✓] Saved {save_path} | shape: {transposed.shape}")

    except Exception as e:
        print(f"[X] Error processing {bin_file}: {str(e)}")
