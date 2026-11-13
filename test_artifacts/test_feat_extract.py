import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
resnet.fc = torch.nn.Identity()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    h, w, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Deep features (512 dims)
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    t = tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        deep_feat = resnet(t).cpu().numpy().flatten()
        
    # 2. Physical & Frequency Features
    # Noise residual
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float64) - blurred.astype(np.float64)
    res_std = float(np.std(noise))
    res_kurt = float(np.mean(noise**4) / (res_std**4 + 1e-7))
    
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
    b_res = img_bgr[:,:,0].astype(np.float64) - cv2.GaussianBlur(img_bgr[:,:,0], (5, 5), 0).astype(np.float64)
    g_res = img_bgr[:,:,1].astype(np.float64) - cv2.GaussianBlur(img_bgr[:,:,1], (5, 5), 0).astype(np.float64)
    r_res = img_bgr[:,:,2].astype(np.float64) - cv2.GaussianBlur(img_bgr[:,:,2], (5, 5), 0).astype(np.float64)
    corr_rg = float(np.corrcoef(r_res.ravel(), g_res.ravel())[0, 1])
    corr_gb = float(np.corrcoef(g_res.ravel(), b_res.ravel())[0, 1])
    
    phys_feat = np.array([res_std, res_kurt, lap_std, hf_ratio, corr_rg, corr_gb], dtype=np.float32)
    return np.concatenate([deep_feat, phys_feat])

# Test on 4 real and 4 AI
real_imgs = [
    'data/new_images/femalecow.jpg',
    'data/new_images/malecow.jpg',
    'data/new_images/femalebuffolo2.jpg',
    'data/new_images/malebuffolo1.jpg'
]
ai_imgs = [
    'test_artifacts/ai_generated_samples/ai_cow.jpg',
    'test_artifacts/ai_generated_samples/ai_buffalo.jpg',
    'test_artifacts/ai_generated_samples/ai_male_cow.jpg',
    'test_artifacts/ai_generated_samples/ai_female_cow.jpg'
]

f_real = [extract_features(p) for p in real_imgs]
f_ai = [extract_features(p) for p in ai_imgs]
print("Feature extraction successful! Shape:", f_real[0].shape)
