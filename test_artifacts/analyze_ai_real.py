import os
import cv2
import numpy as np
from PIL import Image

def analyze_image_properties(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Laplacian Noise Residual
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_std = np.std(lap)
    lap_var = np.var(lap)
    
    # 2. High-pass filter via Gaussian blur subtraction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_res = gray.astype(np.float64) - blurred.astype(np.float64)
    res_std = np.std(noise_res)
    res_kurt = float(np.mean(noise_res**4) / (res_std**4 + 1e-7))
    
    # 3. FFT High-Frequency Ratio
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    mag_spec = np.abs(dft_shift)
    cy, cx = h // 2, w // 2
    r_inner = min(h, w) // 8
    r_outer = min(h, w) // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    hf_mask = (dist_from_center > r_inner) & (dist_from_center < r_outer)
    lf_mask = dist_from_center <= r_inner
    
    hf_energy = np.mean(mag_spec[hf_mask]) if np.any(hf_mask) else 0
    lf_energy = np.mean(mag_spec[lf_mask]) if np.any(lf_mask) else 1
    hf_ratio = hf_energy / (lf_energy + 1e-7)
    
    # 4. Color channel correlation in high frequency
    b_lap = cv2.Laplacian(img[:,:,0], cv2.CV_64F)
    g_lap = cv2.Laplacian(img[:,:,1], cv2.CV_64F)
    r_lap = cv2.Laplacian(img[:,:,2], cv2.CV_64F)
    corr_rg = float(np.corrcoef(r_lap.ravel(), g_lap.ravel())[0, 1])
    corr_gb = float(np.corrcoef(g_lap.ravel(), b_lap.ravel())[0, 1])
    
    return {
        'lap_std': round(lap_std, 2),
        'res_std': round(res_std, 2),
        'res_kurt': round(res_kurt, 2),
        'hf_ratio': round(hf_ratio, 5),
        'corr_rg': round(corr_rg, 3),
        'corr_gb': round(corr_gb, 3),
    }

print("=== REAL PHOTOS ===")
real_samples = [
    'data/new_images/femalecow.jpg',
    'data/new_images/malecow.jpg',
    'data/new_images/femalebuffolo2.jpg',
    'data/new_images/malebuffolo1.jpg',
    'Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg',
    'buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg',
    'test_non_livestock_samples/car.jpg',
    'test_non_livestock_samples/dog.jpg',
]
for p in real_samples:
    props = analyze_image_properties(p)
    print(f"{os.path.basename(p):35s} -> {props}")

print("\n=== AI GENERATED IMAGES ===")
ai_samples = [
    'test_artifacts/ai_generated_samples/ai_cow.jpg',
    'test_artifacts/ai_generated_samples/ai_buffalo.jpg',
    'test_artifacts/ai_generated_samples/ai_male_cow.jpg',
    'test_artifacts/ai_generated_samples/ai_female_cow.jpg',
    'test_artifacts/ai_generated_samples/ai_male_buffalo.jpg',
    'test_artifacts/ai_generated_samples/ai_female_buffalo.jpg',
]
for p in ai_samples:
    props = analyze_image_properties(p)
    print(f"{os.path.basename(p):35s} -> {props}")
