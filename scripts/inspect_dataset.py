import os
import hashlib
from PIL import Image
import numpy as np

CATTLE_DIR = "Cattle Breeds"
BUFFALO_DIR = "buffalo"
DUMMY_DIR = "dummy_unidentified"

def calculate_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def estimate_blur(img_gray):
    # Simple Laplacian variance approximation using numpy
    gy, gx = np.gradient(img_gray.astype(float))
    gnorm = np.sqrt(gx**2 + gy**2)
    return float(np.var(gnorm))

def inspect_dataset_directory(base_dir, species_name):
    print(f"\n==================================================")
    print(f"   INSPECTING {species_name.upper()} DATASET: {base_dir}")
    print(f"==================================================")
    
    breeds = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    overall_total = 0
    overall_usable = 0
    overall_corrupted = 0
    overall_duplicates = 0
    overall_low_res = 0
    overall_blurry = 0
    
    seen_hashes = {}
    report_rows = []
    
    for breed in breeds:
        folder = os.path.join(base_dir, breed)
        files = os.listdir(folder)
        
        b_total = 0
        b_usable = 0
        b_corrupt = 0
        b_dup = 0
        b_lowres = 0
        b_blur = 0
        
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                continue
            
            fpath = os.path.join(folder, fname)
            b_total += 1
            
            # Check corruption
            try:
                with Image.open(fpath) as img:
                    img_rgb = img.convert('RGB')
                    w, h = img_rgb.size
                    img_arr = np.array(img_rgb)
            except Exception as e:
                b_corrupt += 1
                continue
            
            # Check duplicate by hash
            file_hash = calculate_md5(fpath)
            if file_hash in seen_hashes:
                b_dup += 1
                continue
            seen_hashes[file_hash] = fpath
            
            # Check low resolution
            if w < 64 or h < 64:
                b_lowres += 1
                continue
            
            # Check blur / degenerate image
            gray = np.dot(img_arr[...,:3], [0.2989, 0.5870, 0.1140])
            blur_score = estimate_blur(gray)
            if blur_score < 5.0 or np.std(gray) < 5.0: # degenerate or extreme flat color
                b_blur += 1
                continue
            
            b_usable += 1
            
        overall_total += b_total
        overall_usable += b_usable
        overall_corrupted += b_corrupt
        overall_duplicates += b_dup
        overall_low_res += b_lowres
        overall_blurry += b_blur
        
        flagged = b_corrupt + b_dup + b_lowres + b_blur
        report_rows.append((breed, b_total, b_usable, flagged, b_dup, b_corrupt, b_lowres, b_blur))
    
    print(f"{'Breed Name':<28} | {'Total':<6} | {'Usable':<6} | {'Flagged':<7} | {'Dups':<5} | {'Corrupt':<7} | {'LowRes':<6} | {'Blurry':<6}")
    print("-" * 88)
    for r in report_rows:
        print(f"{r[0]:<28} | {r[1]:<6} | {r[2]:<6} | {r[3]:<7} | {r[4]:<5} | {r[5]:<7} | {r[6]:<6} | {r[7]:<6}")
    print("-" * 88)
    print(f"{'TOTAL':<28} | {overall_total:<6} | {overall_usable:<6} | {overall_total - overall_usable:<7} | {overall_duplicates:<5} | {overall_corrupted:<7} | {overall_low_res:<6} | {overall_blurry:<6}\n")
    
    return report_rows, seen_hashes

if __name__ == '__main__':
    c_rows, _ = inspect_dataset_directory(CATTLE_DIR, "Cattle")
    b_rows, _ = inspect_dataset_directory(BUFFALO_DIR, "Buffalo")
