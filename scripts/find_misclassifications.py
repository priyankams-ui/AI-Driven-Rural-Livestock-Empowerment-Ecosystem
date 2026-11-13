import os, sys, glob, pickle, torch
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
classes = bundle['classes']

# Let's test all images in gender_dataset, Cattle Breeds, buffalo, and test_artifacts
print("Searching for images...")

# 1. Gender dataset ground truth
print("\n--- Testing gender_dataset ---")
correct, total = 0, 0
for sp in ['cow', 'buffalo']:
    for gender_label, g_idx in [('female', 0), ('male', 1)]:
        dir_path = os.path.join('gender_dataset', sp, gender_label)
        if not os.path.exists(dir_path): continue
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if not fpath.endswith(('.jpg', '.jpeg', '.png', '.webp')): continue
            im = Image.open(fpath).convert('RGB')
            w, h = im.size
            # Test default ventral crop
            y1, y2 = int(h * 0.45), int(h * 0.90)
            x1, x2 = int(w * 0.15), int(w * 0.85)
            c0 = im.crop((x1, y1, x2, y2))
            c1 = c0.transpose(Image.FLIP_LEFT_RIGHT)
            t0 = transform_gray(c0).unsqueeze(0).to(device)
            t1 = transform_gray(c1).unsqueeze(0).to(device)
            with torch.no_grad():
                f0 = resnet(t0).cpu().numpy()
                f1 = resnet(t1).cpu().numpy()
            p0 = clf.predict_proba(f0)[0]
            p1 = clf.predict_proba(f1)[0]
            avg_p = (p0 + p1) / 2.0
            pred_idx = int(np.argmax(avg_p))
            pred_gender = classes[pred_idx]
            expected_gender = 'Female' if g_idx == 0 else 'Male'
            conf = max(avg_p[0], avg_p[1])
            is_correct = (pred_idx == g_idx)
            if is_correct:
                correct += 1
            else:
                print(f"MISCLASSIFICATION: {fpath} Expected: {expected_gender} Got: {pred_gender} (conf: {conf:.4f})")
            total += 1

print(f"gender_dataset accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
