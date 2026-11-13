import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

print("=" * 80)
print("TESTING CONSISTENT ANATOMICAL FEATURE EXTRACTION FOR BUFFALO")
print("=" * 80)

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

def get_crops(img, box=None):
    w, h = img.size
    if box is not None:
        bx1, by1, bx2, by2 = box
        bw, bh = bx2 - bx1, by2 - by1
        # Ventral & hindquarter focus:
        # Preputial sheath in males extends from mid-ventrum (0.45*bh) forward;
        # Scrotum & udder are between/in front of hind legs (0.60*bh to 0.95*bh).
        y1 = max(0, int(by1 + 0.38 * bh))
        y2 = min(h, int(by1 + 0.96 * bh))
        x1 = max(0, int(bx1 + 0.10 * bw))
        x2 = min(w, int(bx1 + 0.90 * bw))
        body_crop = img.crop((max(0, int(bx1)), max(0, int(by1)), min(w, int(bx2)), min(h, int(by2))))
    else:
        y1, y2 = int(h * 0.40), int(h * 0.94)
        x1, x2 = int(w * 0.12), int(w * 0.88)
        body_crop = img
    ventral_crop = img.crop((x1, y1, x2, y2))
    return body_crop, ventral_crop

def extract_features(img, box=None):
    body_crop, ventral_crop = get_crops(img, box)
    
    # Body features (ResNet-50) with bilateral flip
    t_body_0 = transform_rgb(body_crop).unsqueeze(0).to(device)
    t_body_1 = transform_rgb(body_crop.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
    
    # Ventral RGB features (ResNet-18)
    t_vrgb_0 = transform_rgb(ventral_crop).unsqueeze(0).to(device)
    t_vrgb_1 = transform_rgb(ventral_crop.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
    
    # Ventral Grayscale features (ResNet-18) - invariant to skin texture/mud
    t_vgray_0 = transform_gray(ventral_crop).unsqueeze(0).to(device)
    t_vgray_1 = transform_gray(ventral_crop.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        fb_0 = torch.flatten(resnet50(t_body_0), 1).cpu().numpy().flatten()
        fb_1 = torch.flatten(resnet50(t_body_1), 1).cpu().numpy().flatten()
        vr_0 = resnet18(t_vrgb_0).cpu().numpy().flatten()
        vr_1 = resnet18(t_vrgb_1).cpu().numpy().flatten()
        vg_0 = resnet18(t_vgray_0).cpu().numpy().flatten()
        vg_1 = resnet18(t_vgray_1).cpu().numpy().flatten()
        
    fb = (fb_0 + fb_1) / 2.0
    vr = (vr_0 + vr_1) / 2.0
    vg = (vg_0 + vg_1) / 2.0
    
    return np.concatenate([fb, vr, vg])

# Test on reference images
ref_samples = [
    ('data/new_images/femalebuffolo.jpg', 0, 'Female Buffalo 0'),
    ('data/new_images/femalebuffolo1.jpg', 0, 'Female Buffalo 1'),
    ('data/new_images/femalebuffolo2.jpg', 0, 'Female Buffalo 2'),
    ('data/new_images/malebuffolo.jpg', 1, 'Male Buffalo 0'),
    ('data/new_images/malebuffolo1.jpg', 1, 'Male Buffalo 1'),
    ('data/new_images/malebuffolo2.jpg', 1, 'Male Buffalo 2'),
    ('data/new_images/malebuffolo3.jpg', 1, 'Male Buffalo 3'),
]

print("Reference samples loaded:")
for p, lbl, desc in ref_samples:
    im = Image.open(p).convert('RGB')
    feat = extract_features(im)
    print(f"  {desc:20s} | Feature vector shape: {feat.shape} | Norm: {np.linalg.norm(feat):.2f}")
