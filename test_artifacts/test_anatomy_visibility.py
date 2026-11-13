import os
import cv2
import numpy as np
from PIL import Image

def analyze_ventral_anatomy(img_path, box=None):
    img = cv2.imread(img_path)
    if img is None:
        return False, "Failed to load image"
    h, w = img.shape[:2]
    
    if box is None:
        # Default ventral region if full image
        y1, y2 = int(h * 0.45), int(h * 0.90)
        x1, x2 = int(w * 0.15), int(w * 0.85)
    else:
        bx1, by1, bx2, by2 = box
        bw, bh = bx2 - bx1, by2 - by1
        y1, y2 = int(by1 + 0.45 * bh), int(by1 + 0.90 * bh)
        x1, x2 = int(bx1 + 0.15 * bw), int(bx1 + 0.85 * bw)
        
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)
    
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "Empty anatomical region"
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 1. Darkness / Under-exposure check (clipped black shadow)
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    dark_ratio = float(np.mean(gray < 25))
    bright_ratio = float(np.mean(gray > 240))
    
    # 2. Structural Edge & Contour Detail in Anatomical ROI
    # Use Canny and Sobel to detect underbelly contours and leg boundaries
    edges = cv2.Canny(gray, 30, 100)
    edge_density = float(np.mean(edges > 0))
    
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y_energy = float(np.mean(np.abs(sobel_y)))
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x_energy = float(np.mean(np.abs(sobel_x)))
    
    # 3. Contour continuity (underbelly profile)
    # The underbelly should produce horizontal edge energy (sobel_y)
    # The legs should produce vertical edge energy (sobel_x)
    
    # Occlusion or insufficient visibility checks:
    # A. Pitch black or blown out:
    if dark_ratio > 0.80 or bright_ratio > 0.80:
        return False, f"Severe lighting occlusion (dark={dark_ratio:.2f}, bright={bright_ratio:.2f})"
        
    # B. Flat / blank / low texture:
    if std_brightness < 12.0 or edge_density < 0.035:
        return False, f"Insufficient anatomical texture (std={std_brightness:.1f}, edge_density={edge_density:.4f})"
        
    # C. Missing vertical leg structures or horizontal underbelly line:
    if sobel_y_energy < 5.0 or sobel_x_energy < 4.0:
        return False, f"Missing anatomical contour structures (sobel_y={sobel_y_energy:.1f}, sobel_x={sobel_x_energy:.1f})"
        
    return True, f"SUFFICIENT (std={std_brightness:.1f}, edges={edge_density:.3f}, sy={sobel_y_energy:.1f}, sx={sobel_x_energy:.1f})"

samples = [
    ('data/new_images/femalecow.jpg', 'Full Cow Female'),
    ('data/new_images/malecow.jpg', 'Full Cow Male'),
    ('data/new_images/femalebuffolo2.jpg', 'Full Buffalo Female'),
    ('data/new_images/malebuffolo1.jpg', 'Full Buffalo Male'),
    ('test_artifacts/half_female_cow.jpg', 'Half Cow'),
    ('gender_dataset/uncertain/cow_head_only.jpg', 'Head Only'),
]

for p, name in samples:
    ok, reason = analyze_ventral_anatomy(p)
    print(f"{name:22s} ({p:35s}) -> Visible: {str(ok):5s} | {reason}")
