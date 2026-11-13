import sys
sys.path.insert(0, '.')
import os
import torch
import numpy as np
from PIL import Image
from ml_pipeline import pipeline

samples = [
    # Female Cows
    ('Female Cow 0', 'data/new_images/femalecow.jpg', 'Cow', 'Female'),
    ('Female Cow 1', 'data/new_images/femalecow1.jpg', 'Cow', 'Female'),
    ('Female Cow 2', 'data/new_images/femalecow2.jpg', 'Cow', 'Female'),
    ('Female Holstein Cow', 'Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 'Cow', 'Female'),
    ('Female Jersey Cow', 'Cattle Breeds/Jersey cattle/Jerseycattle0.jpg', 'Cow', 'Female'),
    ('Female Brown Swiss', 'Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg', 'Cow', 'Female'),
    ('Female Red Dane Cow', 'Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 'Cow', 'Female'),
    
    # Male Cows
    ('Male Cow 0', 'data/new_images/malecow.jpg', 'Cow', 'Male'),
    ('Male Cow 1', 'data/new_images/malecow1.jpg', 'Cow', 'Male'),
    ('Male Cow 2', 'data/new_images/malecow2.jpg', 'Cow', 'Male'),
    ('Male Hereford WebP', 'data/new_images/malecow2.webp', 'Cow', 'Male'),
    ('Male Holstein Steer', 'static/uploads/ec95ac1e5e694f6eb9a39efa0151e85c.jpg', 'Cow', 'Male'),
    ('Male Ayrshire Bull', 'Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 'Cow', 'Male'),

    # Female Buffaloes
    ('Female Buffalo 0', 'data/new_images/femalebuffolo.jpg', 'Buffalo', 'Female'),
    ('Female Buffalo 1', 'data/new_images/femalebuffolo1.jpg', 'Buffalo', 'Female'),
    ('Female Buffalo 2', 'data/new_images/femalebuffolo2.jpg', 'Buffalo', 'Female'),
    ('Female Murrah Buffalo', 'buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 'Buffalo', 'Female'),
    ('Female Banni Buffalo', 'buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 'Buffalo', 'Female'),

    # Male Buffaloes
    ('Male Buffalo 0', 'data/new_images/malebuffolo.jpg', 'Buffalo', 'Male'),
    ('Male Buffalo 1', 'data/new_images/malebuffolo1.jpg', 'Buffalo', 'Male'),
    ('Male Buffalo 2', 'data/new_images/malebuffolo2.jpg', 'Buffalo', 'Male'),
    ('Male Buffalo 3', 'data/new_images/malebuffolo3.jpg', 'Buffalo', 'Male'),
    ('Male Banni Bull', 'buffalo/banni/baani-buffalo-bulls_0.jpg', 'Buffalo', 'Male'),
    ('Male Bhadawari Bull', 'buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg', 'Buffalo', 'Male'),
    ('Male Jafarabadi Bull', 'buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 'Buffalo', 'Male'),
]

print("=" * 110)
print(f"{'Sample Name':24s} | {'Exp Species':11s} | {'Pred Species':12s} | {'Class 0 (Female)':16s} | {'Class 1 (Male)':14s} | {'Pred Gender':11s} | {'Match'}")
print("=" * 110)

all_passed = True
for name, path, exp_sp, exp_g in samples:
    if not os.path.exists(path):
        print(f"Skipping {name}: file not found {path}")
        continue
    res = pipeline.predict_image(path)
    
    # Detailed anatomical crop inspect
    im = Image.open(path).convert('RGB')
    w, h = im.size
    y1, y2 = int(h * 0.45), int(h * 0.90)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    c0 = im.crop((x1, y1, x2, y2))
    c1 = c0.transpose(Image.FLIP_LEFT_RIGHT)
    t0 = pipeline.transform_gray(c0).unsqueeze(0).to(pipeline.device)
    t1 = pipeline.transform_gray(c1).unsqueeze(0).to(pipeline.device)
    with torch.no_grad():
        f0 = pipeline.resnet(t0).cpu().numpy()
        f1 = pipeline.resnet(t1).cpu().numpy()
    p0 = pipeline.gender_clf.predict_proba(f0)[0]
    p1 = pipeline.gender_clf.predict_proba(f1)[0]
    avg = (p0 + p1) / 2.0
    p_fem, p_male = float(avg[0]), float(avg[1])
    
    pred_sp = res.get('species')
    pred_g = res.get('gender')
    match = (pred_sp == exp_sp and pred_g == exp_g)
    if not match:
        all_passed = False
    print(f"{name:24s} | {exp_sp:11s} | {str(pred_sp):12s} | {p_fem:16.4f} | {p_male:14.4f} | {str(pred_g):11s} | {match}")

print("=" * 110)
print(f"All Samples Passed: {all_passed}")
