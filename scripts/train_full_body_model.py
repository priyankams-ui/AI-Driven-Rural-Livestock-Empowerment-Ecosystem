import os
import sys
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
resnet.fc = nn.Identity()

def get_feat(pil_img):
    t = tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        return resnet(t).cpu().numpy().flatten()

print("Building FULL_BODY vs PARTIAL dataset...")
# Class 0: FULL_BODY, Class 1: PARTIAL_OR_HEAD
X, y = [], []

# Canonical full body images: All verified cows and buffaloes
full_image_paths = []

# All in data/new_images
for f in os.listdir('data/new_images'):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        full_image_paths.append(os.path.join('data/new_images', f))

# Additional verified breed images
breed_dirs = [
    'Cattle Breeds/Ayrshire cattle',
    'Cattle Breeds/Brown Swiss cattle',
    'Cattle Breeds/Holstein Friesian cattle',
    'Cattle Breeds/Jersey cattle',
    'Cattle Breeds/Red Dane cattle',
    'buffalo/banni',
    'buffalo/murrah',
    'buffalo/Jaffarabadi',
    'buffalo/surti',
]
for d in breed_dirs:
    if os.path.exists(d):
        for f in os.listdir(d)[:12]:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_image_paths.append(os.path.join(d, f))

print(f"Total verified full body images: {len(full_image_paths)}")

for p in full_image_paths:
    try:
        img = Image.open(p).convert('RGB')
    except Exception:
        continue
    w, h = img.size
    if w < 64 or h < 64: continue
    
    # Positive samples: FULL_BODY (0)
    feat_full = get_feat(img)
    X.append(feat_full)
    y.append(0)
    
    # Slight margin crop (still full body)
    c_full = img.crop((int(w*0.02), int(h*0.02), int(w*0.98), int(h*0.98)))
    X.append(get_feat(c_full))
    y.append(0)
    
    # Horizontal flip of full body
    X.append(get_feat(img.transpose(Image.FLIP_LEFT_RIGHT)))
    y.append(0)
    
    # Negative samples: PARTIAL / HEAD / HALF / CROPPED (1)
    # Head / Face only crop (left and right)
    head_l = img.crop((0, 0, int(w*0.45), int(h*0.55)))
    head_r = img.crop((int(w*0.55), 0, w, int(h*0.55)))
    X.append(get_feat(head_l))
    y.append(1)
    X.append(get_feat(head_r))
    y.append(1)
    
    # Front half crop (missing hindquarters)
    half_l = img.crop((0, 0, int(w*0.52), h))
    X.append(get_feat(half_l))
    y.append(1)
    
    # Rear half crop (missing head)
    half_r = img.crop((int(w*0.48), 0, w, h))
    X.append(get_feat(half_r))
    y.append(1)
    
    # Torso without head or rear
    torso = img.crop((int(w*0.25), int(h*0.2), int(w*0.75), int(h*0.8)))
    X.append(get_feat(torso))
    y.append(1)
    
    # Legs only (bottom half)
    legs = img.crop((0, int(h*0.55), w, h))
    X.append(get_feat(legs))
    y.append(1)

# Add existing known uncertain headshots to class 1
unc_dir = 'gender_dataset/uncertain'
if os.path.exists(unc_dir):
    for f in os.listdir(unc_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            im = Image.open(os.path.join(unc_dir, f)).convert('RGB')
            X.append(get_feat(im))
            y.append(1)

X = np.array(X)
y = np.array(y)
print(f"Total dataset: X={X.shape}, FULL_BODY (0)={sum(y==0)}, PARTIAL_OR_HEAD (1)={sum(y==1)}")

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced', random_state=42))
])
clf.fit(X, y)

preds = clf.predict(X)
print("Training accuracy:", accuracy_score(y, preds))
print(classification_report(y, preds, target_names=['FULL_BODY', 'PARTIAL_OR_HEAD']))

# Save model
os.makedirs('models', exist_ok=True)
model_path = os.path.join('models', 'full_body_classifier.pkl')
bundle = {
    'version': '1.1.0',
    'pipeline': clf,
    'classes': ['FULL_BODY', 'PARTIAL_OR_HEAD'],
    'threshold': 0.50
}
with open(model_path, 'wb') as f:
    pickle.dump(bundle, f)
print(f"Saved full body classifier to {model_path}!")
