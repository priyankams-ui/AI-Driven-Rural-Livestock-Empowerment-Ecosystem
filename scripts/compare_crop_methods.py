import os, sys, pickle, torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet = resnet.to(device).eval()

with open('models/gender_classifier.pkl', 'rb') as f:
    bundle = pickle.load(f)
clf = bundle['pipeline']

samples = [
    ('malecow.jpg', 'Male'),
    ('malecow1.jpg', 'Male'),
    ('malecow2.jpg', 'Male'),
    ('femalecow.jpg', 'Female'),
    ('femalecow1.jpg', 'Female'),
    ('femalecow2.jpg', 'Female'),
    ('malebuffolo.jpg', 'Male'),
    ('malebuffolo1.jpg', 'Male'),
    ('malebuffolo2.jpg', 'Male'),
    ('malebuffolo3.jpg', 'Male'),
    ('femalebuffolo.jpg', 'Female'),
    ('femalebuffolo1.jpg', 'Female'),
    ('femalebuffolo2.jpg', 'Female'),
]

print(f"{'Image':20s} | {'Exp':6s} | {'M1 (Image Crop)':20s} | {'M2 (Adaptive Crop)':20s}")
print("-" * 75)

for fname, exp in samples:
    p = os.path.join('data/new_images', fname)
    if not os.path.exists(p):
        continue
    img = Image.open(p).convert('RGB')
    w, h = img.size
    
    # M1: Image crop (45-90% vertical, 15-85% horizontal)
    y1, y2 = int(h * 0.45), int(h * 0.90)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    c_m1 = img.crop((x1, y1, x2, y2))
    t_m1 = transform_gray(c_m1).unsqueeze(0).to(device)
    with torch.no_grad():
        f_m1 = resnet(t_m1).cpu().numpy()
    p_m1 = clf.predict_proba(f_m1)[0]
    pred_m1 = 'Female' if p_m1[0] > p_m1[1] else 'Male'
    conf_m1 = max(p_m1)
    
    # M2: Anatomy-aware full ventral crop (40-92% vertical, 10-90% horizontal)
    y1_2, y2_2 = int(h * 0.40), int(h * 0.92)
    x1_2, x2_2 = int(w * 0.10), int(w * 0.90)
    c_m2 = img.crop((x1_2, y1_2, x2_2, y2_2))
    t_m2 = transform_gray(c_m2).unsqueeze(0).to(device)
    with torch.no_grad():
        f_m2 = resnet(t_m2).cpu().numpy()
    p_m2 = clf.predict_proba(f_m2)[0]
    pred_m2 = 'Female' if p_m2[0] > p_m2[1] else 'Male'
    conf_m2 = max(p_m2)
    
    res1_str = f"{pred_m1} ({conf_m1:.3f})"
    res2_str = f"{pred_m2} ({conf_m2:.3f})"
    print(f"{fname:20s} | {exp:6s} | {res1_str:20s} | {res2_str:20s}")
