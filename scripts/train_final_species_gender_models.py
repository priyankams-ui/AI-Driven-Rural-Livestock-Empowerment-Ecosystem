import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
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

# -------------------------------------------------------------
# 1. VERIFIED AUDITED DATASETS
# -------------------------------------------------------------

# Cow Dataset (Bos taurus / Bos indicus)
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

# Buffalo Dataset (Bubalus bubalis)
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

# Feature Extractors
def extract_cattle_features(p):
    """512-d ResNet-18 Grayscale Ventral Anatomical Crop"""
    img = Image.open(p).convert('RGB')
    w, h = img.size
    y1, y2 = int(h * 0.42), int(h * 0.94)
    x1, x2 = int(w * 0.12), int(w * 0.88)
    v_crop = img.crop((x1, y1, x2, y2))
    t0 = transform_gray(v_crop).unsqueeze(0).to(device)
    t1 = transform_gray(v_crop.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
    with torch.no_grad():
        f0 = resnet18(t0).cpu().numpy().flatten()
        f1 = resnet18(t1).cpu().numpy().flatten()
    return (f0 + f1) / 2.0

def extract_buffalo_features(p):
    """3072-d Multi-Scale Feature: ResNet-50 Full Body (2048) + ResNet-18 Ventral RGB (512) + ResNet-18 Ventral Gray (512)"""
    img = Image.open(p).convert('RGB')
    w, h = img.size
    y1, y2 = int(h * 0.38), int(h * 0.95)
    x1, x2 = int(w * 0.10), int(w * 0.90)
    v_crop = img.crop((x1, y1, x2, y2))
    
    t_full = transform_rgb(img).unsqueeze(0).to(device)
    t_vrgb = transform_rgb(v_crop).unsqueeze(0).to(device)
    t_vgray = transform_gray(v_crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        f_full = resnet50(t_full).cpu().numpy().flatten()
        f_vrgb = resnet18(t_vrgb).cpu().numpy().flatten()
        f_vgray = resnet18(t_vgray).cpu().numpy().flatten()
        
    return np.concatenate([f_full, f_vrgb, f_vgray])

# Cattle Dataset (Bos taurus / Bos indicus)
cattle_data = [
    # Female Cows (Class 0)
    ('gender_dataset/cow/female/ayrshire_female_1.jpg', 0, 'Female Ayrshire Cow 1'),
    ('gender_dataset/cow/female/brownswiss_female_0.jpg', 0, 'Female Brown Swiss Cow 0'),
    ('gender_dataset/cow/female/femalecow.jpg', 0, 'Female Cow Sample'),
    ('gender_dataset/cow/female/femalecow1.jpg', 0, 'Female Cow 1 Sample'),
    ('gender_dataset/cow/female/femalecow2.jpg', 0, 'Female Cow 2 Sample'),
    ('gender_dataset/cow/female/holstein_female_1.jpg', 0, 'Female Holstein Cow 1'),
    ('gender_dataset/cow/female/jersey_female_0.jpg', 0, 'Female Jersey Cow 0'),
    ('gender_dataset/cow/female/reddane_female_0.jpg', 0, 'Female Red Dane Cow 0'),
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 0, 'Female Ayrshire Cow'),
    ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg', 0, 'Female Brown Swiss Cow'),
    ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 0, 'Female Holstein Cow'),
    ('Cattle Breeds/Jersey cattle/Jerseycattle0.jpg', 0, 'Female Jersey Cow'),
    
    # Male Cows (Class 1)
    ('gender_dataset/cow/male/ayrshire_bull_4.jpg', 1, 'Male Ayrshire Bull 4'),
    ('gender_dataset/cow/male/holstein_bull_male.jpg', 1, 'Male Holstein Bull'),
    ('gender_dataset/cow/male/holstein_steer_male.jpg', 1, 'Male Holstein Steer'),
    ('gender_dataset/cow/male/malecow.jpg', 1, 'Male Cow Sample'),
    ('gender_dataset/cow/male/malecow1.jpg', 1, 'Male Cow 1 Sample'),
    ('gender_dataset/cow/male/malecow2.jpg', 1, 'Male Cow 2 Sample'),
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 1, 'Male Ayrshire Bull'),
    ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle13.jpg', 1, 'Male Brown Swiss Bull'),
    ('Cattle Breeds/Jersey cattle/Jerseycattle13.jpg', 1, 'Male Jersey Bull'),
    ('static/uploads/7fa0f07b81354ed68bac87e2704437c2.jpg', 1, 'Male Holstein Bull Upload'),
    ('static/uploads/ec95ac1e5e694f6eb9a39efa0151e85c.jpg', 1, 'Male Holstein Steer Upload'),
]

def extract_cattle_features_aug(img_path, augment=True, n_aug=12):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    y1, y2 = int(h * 0.42), int(h * 0.94)
    x1, x2 = int(w * 0.12), int(w * 0.88)
    base = img.crop((x1, y1, x2, y2))
    crops = [base, base.transpose(Image.FLIP_LEFT_RIGHT)]
    if augment:
        cw, ch = x2 - x1, y2 - y1
        for _ in range(n_aug):
            jx1 = max(0, int(x1 + np.random.uniform(-0.03, 0.03) * cw))
            jx2 = min(w, int(x2 + np.random.uniform(-0.03, 0.03) * cw))
            jy1 = max(0, int(y1 + np.random.uniform(-0.03, 0.03) * ch))
            jy2 = min(h, int(y2 + np.random.uniform(-0.02, 0.03) * ch))
            c = img.crop((jx1, jy1, jx2, jy2))
            crops.append(c)
            crops.append(c.transpose(Image.FLIP_LEFT_RIGHT))
    feats = []
    for c in crops:
        t = transform_gray(c).unsqueeze(0).to(device)
        with torch.no_grad():
            f = resnet18(t).cpu().numpy().flatten()
        feats.append(f)
    return feats

# -------------------------------------------------------------
# 2. TRAINING & EVALUATING CATTLE GENDER CLASSIFIER
# -------------------------------------------------------------
print("=" * 80)
print("TRAINING CATTLE GENDER CLASSIFIER")
print(f"Cow Dataset: {len(cattle_data)} animals")
print("=" * 80)

# 5-Fold Stratified Cross-Validation for Cattle
cattle_paths = [p for p, _, _ in cattle_data]
cattle_labels = [l for _, l, _ in cattle_data]
skf_cow = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cow_preds, cow_true = [], []

for tr, te in skf_cow.split(cattle_paths, cattle_labels):
    X_tr_c, y_tr_c = [], []
    for idx in tr:
        for f in extract_cattle_features_aug(cattle_paths[idx], augment=True, n_aug=10):
            X_tr_c.append(f)
            y_tr_c.append(cattle_labels[idx])
            
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    clf.fit(np.array(X_tr_c), np.array(y_tr_c))
    
    for idx in te:
        base_f = extract_cattle_features_aug(cattle_paths[idx], augment=False)[:2]
        avg_f = (base_f[0] + base_f[1]) / 2.0
        p = clf.predict([avg_f])[0]
        cow_preds.append(p)
        cow_true.append(cattle_labels[idx])

cow_acc = accuracy_score(cow_true, cow_preds)
cow_prec, cow_rec, cow_f1, _ = precision_recall_fscore_support(cow_true, cow_preds, average='weighted')
cow_cm = confusion_matrix(cow_true, cow_preds)
print(f"Cattle 5-Fold CV Accuracy: {cow_acc*100:.2f}% | Precision: {cow_prec*100:.2f}% | Recall: {cow_rec*100:.2f}% | F1: {cow_f1*100:.2f}%")
print(f"Confusion Matrix (Row 0=Female, Row 1=Male):\n{cow_cm}")

# Final Cattle Model trained on all augmented cattle data
X_all_c, y_all_c = [], []
for p, l, _ in cattle_data:
    for f in extract_cattle_features_aug(p, augment=True, n_aug=12):
        X_all_c.append(f)
        y_all_c.append(l)

cattle_gender_model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
])
cattle_gender_model.fit(np.array(X_all_c), np.array(y_all_c))

# -------------------------------------------------------------
# 3. TRAINING & EVALUATING BUFFALO GENDER CLASSIFIER
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("TRAINING BUFFALO GENDER CLASSIFIER")
print(f"Buffalo Dataset: {len(buffalo_females)} Females, {len(buffalo_males)} Males (36 total)")
print("=" * 80)

X_buf = np.array([extract_buffalo_features(p) for p in buffalo_females + buffalo_males])
y_buf = np.array([0] * len(buffalo_females) + [1] * len(buffalo_males))

# Buffalo Ensemble: Random Forest + Extra Trees + Logistic Regression
def make_buffalo_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('voting', VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=120, max_depth=4, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=120, max_depth=4, random_state=42)),
            ('lr', LogisticRegression(C=0.02, max_iter=1000, random_state=42))
        ], voting='soft'))
    ])

# 5-Fold Cross-Validation for Buffalo
skf_buf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
buf_preds, buf_true = [], []
for tr, te in skf_buf.split(X_buf, y_buf):
    clf = make_buffalo_pipeline()
    clf.fit(X_buf[tr], y_buf[tr])
    buf_preds.extend(clf.predict(X_buf[te]))
    buf_true.extend(y_buf[te])

buf_acc = accuracy_score(buf_true, buf_preds)
buf_prec, buf_rec, buf_f1, _ = precision_recall_fscore_support(buf_true, buf_preds, average='weighted')
buf_cm = confusion_matrix(buf_true, buf_preds)
print(f"Buffalo 5-Fold CV Accuracy: {buf_acc*100:.2f}% | Precision: {buf_prec*100:.2f}% | Recall: {buf_rec*100:.2f}% | F1: {buf_f1*100:.2f}%")
print(f"Confusion Matrix (Row 0=Female, Row 1=Male):\n{buf_cm}")

# Final Buffalo Model
buffalo_gender_model = make_buffalo_pipeline()
buffalo_gender_model.fit(X_buf, y_buf)

# -------------------------------------------------------------
# 4. SAVE CANONICAL MODELS
# -------------------------------------------------------------
os.makedirs('models', exist_ok=True)

# Save individual models
with open('models/cattle_gender_clf.pkl', 'wb') as f:
    pickle.dump(cattle_gender_model, f)
print("\n[Saved] models/cattle_gender_clf.pkl")

with open('models/buffalo_gender_clf.pkl', 'wb') as f:
    pickle.dump(buffalo_gender_model, f)
print("[Saved] models/buffalo_gender_clf.pkl")

# Save unified bundle with both pipelines
unified_bundle = {
    'version': '3.0.0',
    'classes': ['Female', 'Male'],
    'confidence_threshold': 0.65,
    'min_margin': 0.12,
    'cattle_pipeline': cattle_gender_model,
    'buffalo_pipeline': buffalo_gender_model,
    'pipeline': cattle_gender_model, # fallback
    'feature_extractor': 'Species-Aware Multi-Scale Anatomical & Ventral Representation'
}
with open('models/gender_classifier.pkl', 'wb') as f:
    pickle.dump(unified_bundle, f)
print("[Saved] models/gender_classifier.pkl")

print("\nModel training and artifact creation complete!")
