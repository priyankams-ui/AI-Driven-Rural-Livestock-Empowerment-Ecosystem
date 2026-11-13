import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import urllib.request

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

url = 'https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json'
with urllib.request.urlopen(url) as resp:
    data = json.loads(resp.read().decode())
labels = {int(k): v[1] for k, v in data.items()}

# Bovid / Bovine ImageNet classes:
# 345: ox (domestic cattle), 346: water_buffalo, 347: bison, 690: oxcart
BOVID_CLASSES = {345, 346, 347, 690}

test_images = {
    'Male Cow': r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle4.jpg',
    'Female Cow': r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle0.jpg',
    'Male Buffalo': r'buffalo\Jaffarabadi\Best_Jafarabadi_bull_.jpg',
    'Female Buffalo': r'buffalo\banni\0caae52ef510154bb215e99a7db35607.jpg',
}

for name in ['car', 'bus', 'bike', 'truck', 'dog', 'cat', 'horse', 'goat', 'sheep', 'building', 'landscape', 'food', 'random_object', 'person']:
    test_images[name] = os.path.join('test_non_livestock_samples', f'{name}.jpg')

print(f"{'Image Name':16} | {'Top-1 ImageNet Class':24} | {'Top-1 Prob':10} | {'Bovid Prob':10} | Is Bovine?")
print('-'*80)

for name, path in test_images.items():
    img = Image.open(path).convert('RGB')
    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = resnet(tensor)
        probs = torch.softmax(out, dim=1)[0]
        top1_prob, top1_idx = torch.topk(probs, 1)
        bovid_prob = sum(probs[c].item() for c in BOVID_CLASSES)
    
    top1_name = labels[top1_idx.item()]
    is_bovine = bovid_prob > 0.05 or any(top1_idx.item() == c for c in BOVID_CLASSES)
    print(f"{name:16} | {top1_name:24} | {top1_prob.item():10.4f} | {bovid_prob:10.4f} | {is_bovine}")
