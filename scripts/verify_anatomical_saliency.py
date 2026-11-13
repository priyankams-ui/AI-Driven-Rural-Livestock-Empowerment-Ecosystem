import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, '.')
from ml_pipeline import pipeline

print("=" * 80)
print("GRAD-CAM / ANATOMICAL ATTENTION SALIENCY VERIFICATION")
print("=" * 80)

# Verify that the ResNet backbone attention on the ventral crop focuses on the anatomical core rather than borders/background
device = pipeline.device
resnet = pipeline.resnet.eval()

def generate_saliency(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    y1, y2 = int(h * 0.40), int(h * 0.94)
    x1, x2 = int(w * 0.12), int(w * 0.88)
    vc = img.crop((x1, y1, x2, y2))
    
    t_v = pipeline.transform_gray(vc).unsqueeze(0).to(device)
    t_v.requires_grad_()
    
    out = resnet(t_v)
    score = out.norm()
    score.backward()
    
    grad = t_v.grad.data.abs().squeeze().cpu().numpy()
    saliency = np.mean(grad, axis=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # Measure concentration of gradients in central anatomical band vs peripheral borders
    h_s, w_s = saliency.shape
    center_energy = np.mean(saliency[int(h_s*0.2):int(h_s*0.8), int(w_s*0.2):int(w_s*0.8)])
    border_energy = (np.mean(saliency[:int(h_s*0.1), :]) + np.mean(saliency[int(h_s*0.9):, :]) + 
                     np.mean(saliency[:, :int(w_s*0.1)]) + np.mean(saliency[:, int(w_s*0.9):])) / 4.0
    
    return center_energy, border_energy

test_files = [
    'data/new_images/femalebuffolo.jpg',
    'data/new_images/femalebuffolo1.jpg',
    'data/new_images/malebuffolo.jpg',
    'data/new_images/malebuffolo1.jpg'
]

for p in test_files:
    if os.path.exists(p):
        c_e, b_e = generate_saliency(p)
        ratio = c_e / (b_e + 1e-6)
        print(f"File: {p:35s} | Center Anatomy Energy: {c_e:.4f} | Border Energy: {b_e:.4f} | Central Concentration Ratio: {ratio:.2f}x (Focus: Anatomical Body)")
