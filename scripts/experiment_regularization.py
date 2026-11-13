import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

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

def extract_features(p):
    img = Image.open(p).convert('RGB')
    w, h = img.size
    y1, y2 = int(h * 0.45), int(h * 0.92)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    c0 = img.crop((x1, y1, x2, y2))
    c1 = c0.transpose(Image.FLIP_LEFT_RIGHT)
    t0 = transform_gray(c0).unsqueeze(0).to(device)
    t1 = transform_gray(c1).unsqueeze(0).to(device)
    with torch.no_grad():
        f0 = resnet(t0).cpu().numpy().flatten()
        f1 = resnet(t1).cpu().numpy().flatten()
    return (f0 + f1) / 2.0

# 26 Verified Animals
samples = [
    # Females (0)
    ('data/new_images/femalecow.jpg', 0, 'FCow0'),
    ('data/new_images/femalecow1.jpg', 0, 'FCow1'),
    ('data/new_images/femalecow2.jpg', 0, 'FCow2'),
    ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 0, 'FHolstein'),
    ('Cattle Breeds/Jersey cattle/Jerseycattle0.jpg', 0, 'FJersey'),
    ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg', 0, 'FBrownSwiss'),
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 0, 'FAyrshire'),
    ('Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 0, 'FRedDane'),
    ('data/new_images/femalebuffolo.jpg', 0, 'FBuff0'),
    ('data/new_images/femalebuffolo1.jpg', 0, 'FBuff1'),
    ('data/new_images/femalebuffolo2.jpg', 0, 'FBuff2'),
    ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 0, 'FMurrah'),
    ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 0, 'FBanni'),

    # Males (1)
    ('data/new_images/malecow.jpg', 1, 'MCow0'),
    ('data/new_images/malecow1.jpg', 1, 'MCow1'),
    ('data/new_images/malecow2.jpg', 1, 'MCow2'),
    ('static/uploads/ec95ac1e5e694f6eb9a39efa0151e85c.jpg', 1, 'MSteer'),
    ('static/uploads/7fa0f07b81354ed68bac87e2704437c2.jpg', 1, 'MBull'),
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 1, 'MAyrshire'),
    ('data/new_images/malebuffolo.jpg', 1, 'MBuff0'),
    ('data/new_images/malebuffolo1.jpg', 1, 'MBuff1'),
    ('data/new_images/malebuffolo2.jpg', 1, 'MBuff2'),
    ('data/new_images/malebuffolo3.jpg', 1, 'MBuff3'),
    ('buffalo/banni/baani-buffalo-bulls_0.jpg', 1, 'MBanniBull'),
    ('buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg', 1, 'MBhadawari'),
    ('buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 1, 'MJafarabadi'),
]

X, y, names = [], [], []
for p, label, name in samples:
    X.append(extract_features(p))
    y.append(label)
    names.append(name)
X = np.array(X)
y = np.array(y)
print(f"Dataset: X={X.shape}, y={y.shape}")

# Test Leave-One-Out Cross Validation across various pipelines
for C_val in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=C_val, max_iter=1000, random_state=42))
    ])
    loo = LeaveOneOut()
    scores = cross_val_score(pipe, X, y, cv=loo)
    print(f"C={C_val:<6f} -> Leave-One-Out CV Accuracy: {scores.mean()*100:.1f}% ({int(scores.sum())}/{len(y)})")

# Test with PCA
for n_comp in [8, 12, 16, 20]:
    pipe_pca = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_comp, random_state=42)),
        ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    loo = LeaveOneOut()
    scores = cross_val_score(pipe_pca, X, y, cv=loo)
    print(f"PCA({n_comp}) + LR(0.1) -> LOO CV Accuracy: {scores.mean()*100:.1f}% ({int(scores.sum())}/{len(y)})")
