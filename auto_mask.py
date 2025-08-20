import os, glob, json
import numpy as np

# ===== 경로/클래스 설정 =====
BASE_DIR = r"C:\Users\jykong\Desktop\pp\npy"   # class0~class7 폴더가 들어있는 루트
OUT_ROOT = r"C:\Users\jykong\Desktop\pp\masked"
CLASS_NAMES = [f"class{i}" for i in range(8)]      # class0 ~ class7
os.makedirs(OUT_ROOT, exist_ok=True)

# ===== 파라미터 =====
TOPK_BANDS = 10
PCT_LOW, PCT_HIGH = 1, 99
MIN_OBJ_PIXELS = 500
MORPH_KERNEL = 5
CROP_PAD = 8

def percentile_stretch(img, p1=1, p99=99):
    lo, hi = np.percentile(img, [p1, p99])
    if hi <= lo: return np.zeros_like(img, dtype=np.float32)
    x = (img - lo) / (hi - lo)
    return np.clip(x, 0, 1)

def band_snr(cube):
    mu = cube.mean(axis=(0,1))
    sd = cube.std(axis=(0,1)) + 1e-6
    return (mu / sd)

def make_mask_from_cube(cube):
    H, W, B = cube.shape
    cube_f = cube.astype(np.float32)
    mu = cube_f.mean(axis=(0,1), keepdims=True)
    sd = cube_f.std(axis=(0,1), keepdims=True) + 1e-6
    zn = (cube_f - mu) / sd

    snr = band_snr(cube_f)
    k = min(TOPK_BANDS, B)
    top_idx = np.argsort(snr)[-k:]
    inten = np.linalg.norm(zn[:, :, top_idx], axis=2)
    inten = percentile_stretch(inten, PCT_LOW, PCT_HIGH)

    # 간단 Otsu
    hist, _ = np.histogram((inten*255).astype(np.uint8), bins=256, range=(0,255))
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m  = np.cumsum(hist * np.arange(256)); m1 = m[-1] - m
    with np.errstate(divide='ignore', invalid='ignore'):
        mu0 = m / np.maximum(w0, 1)
        mu1 = m1 / np.maximum(w1, 1)
        sigma_b2 = w0 * w1 * (mu0 - mu1)**2
    th = int(np.nanargmax(sigma_b2))
    mask = (inten*255).astype(np.uint8) > th

    # 모폴로지 + 작은 성분 제거
    from scipy.ndimage import binary_opening, binary_closing, label, find_objects
    mask = binary_opening(mask, structure=np.ones((MORPH_KERNEL, MORPH_KERNEL)))
    mask = binary_closing(mask, structure=np.ones((MORPH_KERNEL, MORPH_KERNEL)))

    lab, nlab = label(mask)
    if nlab > 0:
        sizes = np.bincount(lab.ravel())
        rm = np.where(sizes < MIN_OBJ_PIXELS)[0]
        for rid in rm:
            if rid == 0: continue
            mask[lab == rid] = False

    sl = find_objects(label(mask)[0])
    if not sl:
        return mask.astype(np.uint8), (0, H, 0, W)

    # 가장 큰 박스 + 패딩
    best = max(sl, key=lambda s: (s[0].stop - s[0].start) * (s[1].stop - s[1].start))
    y0, y1 = best[0].start, best[0].stop
    x0, x1 = best[1].start, best[1].stop
    y0 = max(0, y0 - CROP_PAD); y1 = min(H, y1 + CROP_PAD)
    x0 = max(0, x0 - CROP_PAD); x1 = min(W, x1 + CROP_PAD)
    return mask.astype(np.uint8), (y0, y1, x0, x1)

def process_one(cube_path, out_dir):
    base = os.path.splitext(os.path.basename(cube_path))[0].replace("_cube","")
    os.makedirs(out_dir, exist_ok=True)

    cube = np.load(cube_path)  # (H,W,B)
    mask, (y0,y1,x0,x1) = make_mask_from_cube(cube)

    masked_cube = cube.copy()
    masked_cube[mask == 0] = 0
    crop_cube = cube[y0:y1, x0:x1, :]

    np.save(os.path.join(out_dir, base + "_mask.npy"), mask)
    np.save(os.path.join(out_dir, base + "_cube_masked.npy"), masked_cube)
    np.save(os.path.join(out_dir, base + "_cube_cropped.npy"), crop_cube)

    meta = {
        "input_cube": cube_path,
        "mask_path": os.path.join(out_dir, base + "_mask.npy"),
        "cube_masked": os.path.join(out_dir, base + "_cube_masked.npy"),
        "cube_cropped": os.path.join(out_dir, base + "_cube_cropped.npy"),
        "bbox": [int(y0), int(y1), int(x0), int(x1)]
    }
    with open(os.path.join(out_dir, base + "_mask_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta

def main():
    total = 0
    for cls in CLASS_NAMES:
        in_glob = os.path.join(BASE_DIR, cls, "**", "*_cube.npy")
        files = glob.glob(in_glob, recursive=True)
        if not files:
            print(f"[INFO] skip {cls}: no *_cube.npy")
            continue

        out_dir = os.path.join(OUT_ROOT, cls)
        os.makedirs(out_dir, exist_ok=True)

        for i, cf in enumerate(sorted(files), 1):
            try:
                meta = process_one(cf, out_dir)
                print(f"[{cls}] {i}/{len(files)} OK -> {meta['cube_cropped']}")
                total += 1
            except Exception as e:
                print(f"[{cls}] {i}/{len(files)} FAIL: {cf} -> {e}")
    print(f"Done. processed files: {total}")

if __name__ == "__main__":
    main()
