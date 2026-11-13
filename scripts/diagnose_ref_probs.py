import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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
    return np.concatenate([fb, vr, vg]).reshape(1, -1)

# Test with previous model from models/gender_classifier.pkl (saved before the latest run)
with open('models/buffalo_gender_clf.pkl', 'rb') as f:
    clf = pickle.load(f)

print(f"Loaded classifier: {type(clf)}")
if hasattr(clf, 'named_steps'):
    print(f"Named steps: {clf.named_steps}")

ref_samples = [
    ('data/new_images/femalebuffolo.jpg', 'Female 0'),
    ('data/new_images/femalebuffolo1.jpg', 'Female 1'),
    ('data/new_images/femalebuffolo2.jpg', 'Female 2'),
    ('data/new_images/malebuffolo.jpg', 'Male 0'),
    ('data/new_images/malebuffolo1.jpg', 'Male 1'),
    ('data/new_images/malebuffolo2.jpg', 'Male 2'),
    ('data/new_images/malebuffolo3.jpg', 'Male 3')
]

for p, desc in ref_samples:
    feat = extract_feat(p)
    probs = clf.predict_proba(feat)[0]
    classes = getattr(clf, 'classes_', [0, 1])
    print(f"{desc:12s} | Prob Female={probs[0]:.4f}, Prob Male={probs[1]:.4f} | Pred={'Female' if probs[0]>probs[1] else 'Male'}")
