import os
import sys
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform for ResNet
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
resnet.fc = nn.Identity()

def extract_single_feature(img_pil, img_cv):
    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 1. Deep representation (512)
    t = tf(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        deep_feat = resnet(t).cpu().numpy().flatten()
        
    # 2. Physical & Frequency features
    # Gaussian blur noise residual
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float64) - blurred.astype(np.float64)
    res_std = float(np.std(noise))
    res_kurt = float(np.mean(noise**4) / (res_std**4 + 1e-7))
    
    # Patch-wise noise variance uniformity
    patch_size = 32
    patch_vars = []
    for r in range(0, h - patch_size, patch_size):
        for c in range(0, w - patch_size, patch_size):
            p_noise = noise[r:r+patch_size, c:c+patch_size]
            patch_vars.append(np.var(p_noise))
    var_of_vars = float(np.var(patch_vars)) if patch_vars else 0.0
    mean_of_vars = float(np.mean(patch_vars)) if patch_vars else 0.0
    rel_var_unif = var_of_vars / (mean_of_vars**2 + 1e-7)
    
    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_std = float(np.std(lap))
    
    # 2D FFT High Frequency
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    mag_spec = np.abs(dft_shift)
    cy, cx = h // 2, w // 2
    r_inner = min(h, w) // 8
    r_outer = min(h, w) // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    hf_mask = (dist > r_inner) & (dist < r_outer)
    lf_mask = dist <= r_inner
    hf_ratio = float(np.mean(mag_spec[hf_mask]) / (np.mean(mag_spec[lf_mask]) + 1e-7))
    
    # Color channel high-pass correlation
    b_res = img_cv[:,:,0].astype(np.float64) - cv2.GaussianBlur(img_cv[:,:,0], (5, 5), 0).astype(np.float64)
    g_res = img_cv[:,:,1].astype(np.float64) - cv2.GaussianBlur(img_cv[:,:,1], (5, 5), 0).astype(np.float64)
    r_res = img_cv[:,:,2].astype(np.float64) - cv2.GaussianBlur(img_cv[:,:,2], (5, 5), 0).astype(np.float64)
    corr_rg = float(np.corrcoef(r_res.ravel(), g_res.ravel())[0, 1])
    corr_gb = float(np.corrcoef(g_res.ravel(), b_res.ravel())[0, 1])
    corr_rb = float(np.corrcoef(r_res.ravel(), b_res.ravel())[0, 1])
    
    # Bayer CFA demosaicing periodicity metric
    cfa_diff = g_res - 0.5 * (r_res + b_res)
    dft_cfa = np.fft.fft2(cfa_diff)
    dft_cfa_shift = np.fft.fftshift(dft_cfa)
    mag_cfa = np.abs(dft_cfa_shift)
    cfa_peak = float(np.max(mag_cfa) / (np.mean(mag_cfa) + 1e-7))
    
    phys_feat = np.array([
        res_std, res_kurt, rel_var_unif, lap_std, hf_ratio,
        corr_rg, corr_gb, corr_rb, cfa_peak
    ], dtype=np.float32)
    
    return np.concatenate([deep_feat, phys_feat])

def extract_image_samples(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return []
    h, w = img_bgr.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
    samples = []
    # Full image
    samples.append((img_pil, img_bgr))
    
    # Center crop (85%)
    cw, ch = int(w * 0.85), int(h * 0.85)
    x1, y1 = (w - cw) // 2, (h - ch) // 2
    c_bgr = img_bgr[y1:y1+ch, x1:x1+cw]
    c_pil = Image.fromarray(cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB))
    samples.append((c_pil, c_bgr))
    
    # Slightly resized / compressed version
    r_bgr = cv2.resize(img_bgr, (int(w * 0.75), int(h * 0.75)))
    r_pil = Image.fromarray(cv2.cvtColor(r_bgr, cv2.COLOR_BGR2RGB))
    samples.append((r_pil, r_bgr))
    
    feats = []
    for p_pil, p_bgr in samples:
        f = extract_single_feature(p_pil, p_bgr)
        feats.append(f)
    return feats

print("Collecting dataset...")
# Class 0: REAL_PHOTO, Class 1: AI_GENERATED
X, y = [], []

# 1. Real photos
real_dirs = [
    'data/new_images',
    'test_non_livestock_samples',
    'Cattle Breeds/Ayrshire cattle',
    'Cattle Breeds/Brown Swiss cattle',
    'Cattle Breeds/Holstein Friesian cattle',
    'Cattle Breeds/Jersey cattle',
    'Cattle Breeds/Red Dane cattle',
    'buffalo/banni',
    'buffalo/murrah',
    'buffalo/Jaffarabadi',
    'buffalo/surti',
    'buffalo/mehsana',
]

real_paths = []
for d in real_dirs:
    if os.path.exists(d):
        for f in os.listdir(d)[:8]:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                real_paths.append(os.path.join(d, f))

print(f"Total real base images: {len(real_paths)}")
for p in real_paths:
    for feat in extract_image_samples(p):
        X.append(feat)
        y.append(0)

# 2. AI photos
ai_dir = 'test_artifacts/ai_generated_samples'
ai_paths = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Total AI base images: {len(ai_paths)}")

# Extract more deterministic variations for AI to balance samples
for p in ai_paths:
    img_bgr = cv2.imread(p)
    if img_bgr is None: continue
    h, w = img_bgr.shape[:2]
    # Variations
    crops = [
        img_bgr,
        img_bgr[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)],
        img_bgr[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)],
        img_bgr[0:int(h*0.8), 0:int(w*0.8)],
        img_bgr[int(h*0.2):h, int(w*0.2):w],
        cv2.resize(img_bgr, (int(w*0.8), int(h*0.8))),
        cv2.resize(img_bgr, (int(w*0.6), int(h*0.6))),
        cv2.flip(img_bgr, 1),
    ]
    for c_bgr in crops:
        c_pil = Image.fromarray(cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB))
        feat = extract_single_feature(c_pil, c_bgr)
        X.append(feat)
        y.append(1)

X = np.array(X)
y = np.array(y)
print(f"Total dataset: X={X.shape}, REAL (0)={sum(y==0)}, AI (1)={sum(y==1)}")

# Train model
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced', random_state=42))
])

clf.fit(X, y)

train_preds = clf.predict(X)
acc = accuracy_score(y, train_preds)
print(f"Training Accuracy: {acc:.4f}")
print(classification_report(y, train_preds, target_names=['REAL_PHOTO', 'AI_GENERATED']))

# Save model
os.makedirs('models', exist_ok=True)
model_path = os.path.join('models', 'real_vs_ai_classifier.pkl')
bundle = {
    'version': '1.0.0',
    'pipeline': clf,
    'classes': ['REAL_PHOTO', 'AI_GENERATED'],
    'threshold': 0.50
}
with open(model_path, 'wb') as f:
    pickle.dump(bundle, f)
print(f"Saved real vs AI model to {model_path}!")
