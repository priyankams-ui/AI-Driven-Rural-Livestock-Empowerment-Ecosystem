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

def test_dir(folder, label):
    files = glob.glob(os.path.join(folder, '**', '*.jpg'), recursive=True) + \
            glob.glob(os.path.join(folder, '**', '*.webp'), recursive=True)
    print(f"\n--- Testing {label} ({len(files)} files) ---")
    female_cnt, male_cnt = 0, 0
    for p in files[:30]:
        try:
            im = Image.open(p).convert('RGB')
            w, h = im.size
            if w < 50 or h < 50:
                continue
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
            pred = 'Male' if avg_p[1] > avg_p[0] else 'Female'
            conf = max(avg_p[0], avg_p[1])
            if pred == 'Female':
                female_cnt += 1
            else:
                male_cnt += 1
            print(f"  {os.path.basename(p):35s} -> {pred} ({conf:.4f}) [F:{avg_p[0]:.2f}, M:{avg_p[1]:.2f}]")
        except Exception as e:
            pass
    print(f"Summary for {label}: Female={female_cnt}, Male={male_cnt}")

if __name__ == '__main__':
    test_dir('gender_dataset/cow/female', 'gender_dataset/cow/female')
    test_dir('gender_dataset/cow/male', 'gender_dataset/cow/male')
    test_dir('gender_dataset/buffalo/female', 'gender_dataset/buffalo/female')
    test_dir('gender_dataset/buffalo/male', 'gender_dataset/buffalo/male')
