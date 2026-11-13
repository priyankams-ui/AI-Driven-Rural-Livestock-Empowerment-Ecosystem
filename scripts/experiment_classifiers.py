import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_rgb = transforms.Compose([
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

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
resnet18.fc = nn.Identity()

resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()
resnet50.fc = nn.Identity()

def extract_feat(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    y1, y2 = int(h * 0.40), int(h * 0.94)
    x1, x2 = int(w * 0.12), int(w * 0.88)
    vc = img.crop((x1, y1, x2, y2))
    
    t_full_0 = transform_rgb(img).unsqueeze(0).to(device)
    t_full_1 = transform_rgb(img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
    t_vrgb_0 = transform_rgb(vc).unsqueeze(0).to(device)
    t_vrgb_1 = transform_rgb(vc.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
    t_vgray_0 = transform_gray(vc).unsqueeze(0).to(device)
    t_vgray_1 = transform_gray(vc.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        fb_0 = torch.flatten(resnet50(t_full_0), 1).cpu().numpy().flatten()
        fb_1 = torch.flatten(resnet50(t_full_1), 1).cpu().numpy().flatten()
        vr_0 = resnet18(t_vrgb_0).cpu().numpy().flatten()
        vr_1 = resnet18(t_vrgb_1).cpu().numpy().flatten()
        vg_0 = resnet18(t_vgray_0).cpu().numpy().flatten()
        vg_1 = resnet18(t_vgray_1).cpu().numpy().flatten()
        
    fb = (fb_0 + fb_1) / 2.0
    vr = (vr_0 + vr_1) / 2.0
    vg = (vg_0 + vg_1) / 2.0
    return np.concatenate([fb, vr, vg])

# Load training data from previous run or re-extract
with open('models/authoritative_buffalo_gender_catalog.json', 'r') as f:
    catalog = json.load(f)

ref_paths = [
    ('data/new_images/femalebuffolo.jpg', 0, 'Female 0'),
    ('data/new_images/femalebuffolo1.jpg', 0, 'Female 1'),
    ('data/new_images/femalebuffolo2.jpg', 0, 'Female 2'),
    ('data/new_images/malebuffolo.jpg', 1, 'Male 0'),
    ('data/new_images/malebuffolo1.jpg', 1, 'Male 1'),
    ('data/new_images/malebuffolo2.jpg', 1, 'Male 2'),
    ('data/new_images/malebuffolo3.jpg', 1, 'Male 3')
]

# Extract ref feats
ref_feats = {}
for p, lbl, desc in ref_paths:
    ref_feats[p] = extract_feat(p)

# Let's inspect what happens with different classifiers
print("Reference features extracted successfully!")
