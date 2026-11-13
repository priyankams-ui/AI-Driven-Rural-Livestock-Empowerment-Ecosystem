import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

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

ref_images = [
    ('data/new_images/femalebuffolo.jpg', 'Female 0'),
    ('data/new_images/femalebuffolo1.jpg', 'Female 1'),
    ('data/new_images/femalebuffolo2.jpg', 'Female 2'),
    ('data/new_images/malebuffolo.jpg', 'Male 0'),
    ('data/new_images/malebuffolo1.jpg', 'Male 1'),
    ('data/new_images/malebuffolo2.jpg', 'Male 2'),
    ('data/new_images/malebuffolo3.jpg', 'Male 3')
]

for p, desc in ref_images:
    im = Image.open(p).convert('RGB')
    w, h = im.size
    # Ventral crop
    y1, y2 = int(h * 0.40), int(h * 0.94)
    x1, x2 = int(w * 0.12), int(w * 0.88)
    vc = im.crop((x1, y1, x2, y2))
    
    # Also save visual crops for verification
    vc.save(f"test_artifacts/ventral_{desc.replace(' ', '_')}.jpg")
    print(f"{desc:10s} | Size: {w}x{h} | Ventral Crop: ({x1},{y1},{x2},{y2}) -> {vc.size}")

print("Saved ventral crops to test_artifacts/")
