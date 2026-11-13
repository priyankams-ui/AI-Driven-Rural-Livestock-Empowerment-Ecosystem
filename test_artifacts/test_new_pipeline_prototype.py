import os
import sys
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Models
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
resnet.fc = nn.Identity()

# Load real vs AI model
with open('models/real_vs_ai_classifier.pkl', 'rb') as f:
    bundle_ai = pickle.load(f)
real_vs_ai_clf = bundle_ai['pipeline']

# Load full body model
with open('models/full_body_classifier.pkl', 'rb') as f:
    bundle_fb = pickle.load(f)
full_body_clf = bundle_fb['pipeline']

def extract_real_ai_features(img_pil, img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    t = tf(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        deep_feat = resnet(t).cpu().numpy().flatten()
        
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float64) - blurred.astype(np.float64)
    res_std = float(np.std(noise))
    res_kurt = float(np.mean(noise**4) / (res_std**4 + 1e-7))
    
    patch_size = 32
    patch_vars = []
    for r in range(0, h - patch_size, patch_size):
        for c in range(0, w - patch_size, patch_size):
            p_noise = noise[r:r+patch_size, c:c+patch_size]
            patch_vars.append(np.var(p_noise))
    var_of_vars = float(np.var(patch_vars)) if patch_vars else 0.0
    mean_of_vars = float(np.mean(patch_vars)) if patch_vars else 0.0
    rel_var_unif = var_of_vars / (mean_of_vars**2 + 1e-7)
    
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_std = float(np.std(lap))
    
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
    
    b_res = img_bgr[:,:,0].astype(np.float64) - cv2.GaussianBlur(img_bgr[:,:,0], (5, 5), 0).astype(np.float64)
    g_res = img_bgr[:,:,1].astype(np.float64) - cv2.GaussianBlur(img_bgr[:,:,1], (5, 5), 0).astype(np.float64)
    r_res = img_bgr[:,:,2].astype(np.float64) - cv2.GaussianBlur(img_bgr[:,:,2], (5, 5), 0).astype(np.float64)
    corr_rg = float(np.corrcoef(r_res.ravel(), g_res.ravel())[0, 1])
    corr_gb = float(np.corrcoef(g_res.ravel(), b_res.ravel())[0, 1])
    corr_rb = float(np.corrcoef(r_res.ravel(), b_res.ravel())[0, 1])
    
    cfa_diff = g_res - 0.5 * (r_res + b_res)
    dft_cfa = np.fft.fft2(cfa_diff)
    dft_cfa_shift = np.fft.fftshift(dft_cfa)
    mag_cfa = np.abs(dft_cfa_shift)
    cfa_peak = float(np.max(mag_cfa) / (np.mean(mag_cfa) + 1e-7))
    
    phys_feat = np.array([
        res_std, res_kurt, rel_var_unif, lap_std, hf_ratio,
        corr_rg, corr_gb, corr_rb, cfa_peak
    ], dtype=np.float32)
    
    return np.concatenate([deep_feat, phys_feat]).reshape(1, -1)

def run_real_ai_check(img_pil, img_bgr):
    feat = extract_real_ai_features(img_pil, img_bgr)
    probs = real_vs_ai_clf.predict_proba(feat)[0]
    p_real = float(probs[0])
    p_ai = float(probs[1])
    is_real = (p_real >= 0.50)
    return is_real, p_real, p_ai

def run_full_body_check(img_pil):
    t = tf(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(t).cpu().numpy().flatten().reshape(1, -1)
    probs = full_body_clf.predict_proba(feat)[0]
    p_full = float(probs[0])
    p_partial = float(probs[1])
    is_full = (p_full >= 0.50)
    return is_full, p_full, p_partial

print("Prototype module initialized successfully.")
