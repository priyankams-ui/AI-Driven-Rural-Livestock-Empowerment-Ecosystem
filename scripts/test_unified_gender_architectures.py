import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Transforms
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

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet18.fc = nn.Identity()
resnet18 = resnet18.to(device).eval()

resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet50.fc = nn.Identity()
resnet50 = resnet50.to(device).eval()

# Verified Buffalo Dataset (All verified genuine, single animal, real photos)
buffalo_females = [
    'data/new_images/femalebuffolo.jpg',
    'data/new_images/femalebuffolo1.jpg',
    'buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg',
    'buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg',
    'buffalo/surti/Milking_Murrah_Buffalo_Female_750kg_.jpg',
    'buffalo/murrah/High_Milking_Pure_Breed_Murrah_Buffalo_.jpg',
    'buffalo/murrah/Pregnant_Murrah_Buffalo_Prime_Dairy_.jpg',
    'buffalo/murrah/Dairy_Murrah_Buffalo_for_Dairy_Farming_.jpg',
    'buffalo/murrah/Dairy_Murrah_Buffalo_at_70000_per_.jpg',
    'buffalo/murrah/Dairy_Murrah_Buffalo_for_Sale_at_an_.jpg',
    'buffalo/murrah/Murrah_Dairy_Knowledge_Portal.jpg',
    'buffalo/murrah/131_Murrah_Buffalo_for_Dairy_Farming_.jpg',
    'buffalo/Jaffarabadi/Milking_Jafrabadi_Buffalo_Farm_.jpg',
    'buffalo/Jaffarabadi/Chandani_dairy_and_farming_-_My_.jpg',
    'buffalo/Jaffarabadi/jafarabadi_buffalo_dairyfarm_cattle_.jpg',
    'buffalo/bargur/Black_Murrah_BuffaloVijay_Dairy_Farm.jpg',
    'buffalo/bargur/Murrah_BuffaloSuresh_Dairy_Farm.jpg',
    'buffalo/bhadwari/Mehsana_Dairy_Knowledge_Portal.jpg',
    'buffalo/bhadwari/Surti_Dairy_Knowledge_Portal.jpg',
]

buffalo_males = [
    'data/new_images/malebuffolo.jpg',
    'data/new_images/malebuffolo1.jpg',
    'data/new_images/malebuffolo3.jpg',
    'buffalo/Jaffarabadi/Black_Jafarabadi_Bull_Semen_at_Best_.jpg',
    'buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg',
    'buffalo/Jaffarabadi/Jafarabadi_buffalo_bull_for_sales_4_.jpg',
    'buffalo/Jaffarabadi/Jafarabadi_bull_from_Gujarat_-_YouTube.jpg',
    'buffalo/Jaffarabadi/Tulsidas_-_BAIF_Sire_Directory.jpg',
    'buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg',
    'buffalo/Jaffarabadi/JAFFRABADI_BUFFALO_BULL.jpg',
    'buffalo/bhadwari/Bhadavari_Archives_-_BAIF_Sire_Directory.jpg',
    'buffalo/murrah/Murrah_BullMurrah_Buffalo_BullsBull_.jpg',
    'buffalo/murrah/Murrah_buffalo_bulls_to_be_used_for_.jpg',
    'buffalo/nagpuri/Thakya_-_BAIF_Sire_Directory.jpg',
    'buffalo/nili-ravi/Neeli_Ravi_buffalo_bull_.jpg',
    'buffalo/nili-ravi/Neeli_Ravi_buffalo_bull_ (2).jpg',
    'buffalo/nili-ravi/Nili_ravi_bull_Quality_full_niliravi_.jpg',
]

# Verified Cow Dataset
cow_females = [
    'data/new_images/femalecow.jpg',
    'data/new_images/femalecow1.jpg',
    'gender_dataset/cow/female/ayrshire_female_1.jpg',
    'gender_dataset/cow/female/brownswiss_female_0.jpg',
    'gender_dataset/cow/female/holstein_female_1.jpg',
    'gender_dataset/cow/female/jersey_female_0.jpg',
    'gender_dataset/cow/female/reddane_female_0.jpg',
    'Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg',
    'Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg',
    'Cattle Breeds/Brown Swiss cattle/BrownSwisscattle1.jpg',
    'Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg',
    'Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle11.jpg',
    'Cattle Breeds/Jersey cattle/Jerseycattle0.jpg',
    'Cattle Breeds/Jersey cattle/Jerseycattle1.jpg',
    'Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg',
]

cow_males = [
    'data/new_images/malecow.jpg',
    'data/new_images/malecow1.jpg',
    'data/new_images/malecow2.jpg',
    'data/new_images/malecow2.webp',
    'gender_dataset/cow/male/ayrshire_bull_4.jpg',
    'gender_dataset/cow/male/holstein_bull_male.jpg',
    'gender_dataset/cow/male/holstein_steer_male.jpg',
    'Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg',
    'Cattle Breeds/Brown Swiss cattle/BrownSwisscattle13.jpg',
    'Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle13.jpg',
    'Cattle Breeds/Jersey cattle/Jerseycattle13.jpg',
    'static/uploads/7fa0f07b81354ed68bac87e2704437c2.jpg',
    'static/uploads/ec95ac1e5e694f6eb9a39efa0151e85c.jpg',
]

print(f"Verified dataset: Buffalo {len(buffalo_females)}F / {len(buffalo_males)}M, Cow {len(cow_females)}F / {len(cow_males)}M")

# Precompute representations for each image:
# 1. Ventral Grayscale (ResNet18: 512)
# 2. Ventral RGB (ResNet18: 512)
# 3. Full Body RGB (ResNet50: 2048)
# 4. Multi-scale = Full Body (2048) + Ventral RGB (512) = 2560
def extract_all(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    y1, y2 = int(h * 0.40), int(h * 0.95)
    x1, x2 = int(w * 0.10), int(w * 0.90)
    v_crop = img.crop((x1, y1, x2, y2))
    
    t_v_gray = transform_gray(v_crop).unsqueeze(0).to(device)
    t_v_rgb = transform_rgb(v_crop).unsqueeze(0).to(device)
    t_full_rgb = transform_rgb(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        f_v_gray = resnet18(t_v_gray).cpu().numpy().flatten()
        f_v_rgb = resnet18(t_v_rgb).cpu().numpy().flatten()
        f_full = resnet50(t_full_rgb).cpu().numpy().flatten()
        
    return {
        'v_gray': f_v_gray,
        'v_rgb': f_v_rgb,
        'full': f_full,
        'multiscale': np.concatenate([f_full, f_v_rgb])
    }

b_feats = None
c_feats = None

def get_b_feats():
    global b_feats
    if b_feats is None:
        all_buffalo = buffalo_females + buffalo_males
        b_feats = [extract_all(p) for p in all_buffalo]
    return b_feats

def get_c_feats():
    global c_feats
    if c_feats is None:
        all_cow = cow_females + cow_males
        c_feats = [extract_all(p) for p in all_cow]
    return c_feats

b_labels = np.array([0] * len(buffalo_females) + [1] * len(buffalo_males))
c_labels = np.array([0] * len(cow_females) + [1] * len(cow_males))

if __name__ == '__main__':
    print("\nExtracting features...")
    b_feats = get_b_feats()
    c_feats = get_c_feats()
    print("Extraction complete. Running 5-fold CV comparisons...")

def test_feature_set(name, feat_key, feats_list, labels):
    X = np.array([f[feat_key] for f in feats_list])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for c_val in [0.01, 0.05, 0.1, 0.5]:
        y_true, y_pred = [], []
        for tr, te in skf.split(X, labels):
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(C=c_val, max_iter=1000, random_state=42))
            ])
            pipe.fit(X[tr], labels[tr])
            preds = pipe.predict(X[te])
            y_true.extend(labels[te])
            y_pred.extend(preds)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print(f"  {name:25s} | C={c_val:<4} | Acc: {acc*100:.2f}% | CM: [[{cm[0,0]},{cm[0,1]}],[{cm[1,0]},{cm[1,1]}]]")

print("\n--- BUFFALO GENDER EVALUATION ---")
test_feature_set("ResNet18 Ventral Gray", 'v_gray', b_feats, b_labels)
test_feature_set("ResNet18 Ventral RGB", 'v_rgb', b_feats, b_labels)
test_feature_set("ResNet50 Full Body", 'full', b_feats, b_labels)
test_feature_set("Multi-scale Full+Ventral", 'multiscale', b_feats, b_labels)

print("\n--- COW GENDER EVALUATION ---")
test_feature_set("ResNet18 Ventral Gray", 'v_gray', c_feats, c_labels)
test_feature_set("ResNet18 Ventral RGB", 'v_rgb', c_feats, c_labels)
test_feature_set("ResNet50 Full Body", 'full', c_feats, c_labels)
test_feature_set("Multi-scale Full+Ventral", 'multiscale', c_feats, c_labels)
