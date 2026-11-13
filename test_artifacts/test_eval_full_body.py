import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
resnet.fc = nn.Identity()

with open('models/full_body_classifier.pkl', 'rb') as f:
    bundle = pickle.load(f)
clf = bundle['pipeline']

def check_full_body(path):
    im = Image.open(path).convert('RGB')
    t = tf(im).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(t).cpu().numpy().flatten().reshape(1, -1)
    probs = clf.predict_proba(feat)[0]
    p_full = probs[0]
    p_partial = probs[1]
    is_full = (p_full > 0.5)
    return is_full, p_full, p_partial

print("=== FULL BODY IMAGES (Should be TRUE / FULL) ===")
full_imgs = [
    'data/new_images/femalecow.jpg',
    'data/new_images/malecow.jpg',
    'data/new_images/femalebuffolo2.jpg',
    'data/new_images/malebuffolo1.jpg',
]
for p in full_imgs:
    is_full, pf, pp = check_full_body(p)
    print(f"{os.path.basename(p):32s} -> is_full={is_full} (p_full={pf:.3f}, p_partial={pp:.3f})")

print("\n=== PARTIAL / HEAD / HALF CROPS (Should be FALSE / PARTIAL) ===")
partial_imgs = [
    'gender_dataset/uncertain/cow_head_only.jpg',
    'gender_dataset/uncertain/buffalo_head_only.jpg',
    'gender_dataset/uncertain/calf_headshot_fence.jpg',
    'test_artifacts/half_female_cow.jpg',
    'test_artifacts/half_male_cow.jpg',
    'test_artifacts/half_female_buffalo.jpg',
    'test_artifacts/half_male_buffalo.jpg',
    'test_artifacts/test_head_horns_landscape.jpg',
    'test_artifacts/test_m_head_horns.jpg',
    'test_artifacts/test_m_front_half.jpg',
    'test_artifacts/test_m_rear_half.jpg',
    'test_artifacts/test_torso_only_no_head.jpg',
    'test_artifacts/test_front_half.jpg',
    'test_artifacts/test_rear_half.jpg',
    'test_artifacts/test_legs_only.jpg',
]
for p in partial_imgs:
    if os.path.exists(p):
        is_full, pf, pp = check_full_body(p)
        print(f"{os.path.basename(p):32s} -> is_full={is_full} (p_full={pf:.3f}, p_partial={pp:.3f})")
