import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

print("=" * 80)
print("EXPERIMENTING WITH BUFFALO GENDER ARCHITECTURES & HYPERPARAMETERS")
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

def extract_consistent_features(img_path):
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
    return np.concatenate([fb, vr, vg])

def extract_augmented_features(img_path, n_aug=6):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    feats = [extract_consistent_features(img_path)]
    
    for _ in range(n_aug):
        bright_factor = np.random.uniform(0.85, 1.15)
        contrast_factor = np.random.uniform(0.85, 1.15)
        aug_img = ImageEnhance.Brightness(img).enhance(bright_factor)
        aug_img = ImageEnhance.Contrast(aug_img).enhance(contrast_factor)
        
        cw, ch = int(w * 0.76), int(h * 0.54)
        jx1 = max(0, int(w * 0.12 + np.random.uniform(-0.02, 0.02) * cw))
        jx2 = min(w, int(w * 0.88 + np.random.uniform(-0.02, 0.02) * cw))
        jy1 = max(0, int(h * 0.40 + np.random.uniform(-0.02, 0.02) * ch))
        jy2 = min(h, int(h * 0.94 + np.random.uniform(-0.02, 0.02) * ch))
        vc = aug_img.crop((jx1, jy1, jx2, jy2))
        
        t_full_0 = transform_rgb(aug_img).unsqueeze(0).to(device)
        t_full_1 = transform_rgb(aug_img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
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
        feats.append(np.concatenate([fb, vr, vg]))
        
    return feats

# Load Authoritative Catalog
with open('models/authoritative_buffalo_gender_catalog.json', 'r') as f:
    catalog = json.load(f)

female_animals = [a for a in catalog if a['gender'] == 'Female']
male_animals = [a for a in catalog if a['gender'] == 'Male']

np.random.seed(42)
fem_entries = list(female_animals)
male_entries = list(male_animals)
np.random.shuffle(fem_entries)
np.random.shuffle(male_entries)

# 80 / 10 / 10 split
n_f_test = max(1, int(len(fem_entries) * 0.10))
n_f_val = max(1, int(len(fem_entries) * 0.10))
test_fem = fem_entries[:n_f_test]
val_fem = fem_entries[n_f_test:n_f_test + n_f_val]
train_fem = fem_entries[n_f_test + n_f_val:]

n_m_test = max(1, int(len(male_entries) * 0.10))
n_m_val = max(1, int(len(male_entries) * 0.10))
test_male = male_entries[:n_m_test]
val_male = male_entries[n_m_test:n_m_test + n_m_val]
train_male = male_entries[n_m_test + n_m_val:]

train_animals = train_fem + train_male
val_animals = val_fem + val_male
test_animals = test_fem + test_male

print(f"Dataset split: Train={len(train_animals)}, Val={len(val_animals)}, Test={len(test_animals)}")

def load_split(split_list, is_train=True):
    X, y = [], []
    for a in split_list:
        label = 0 if a['gender'] == 'Female' else 1
        p = a['primary']
        if not os.path.exists(p):
            continue
        if is_train:
            n_aug = 12 if label == 1 else 4
            sample_feats = extract_augmented_features(p, n_aug=n_aug)
        else:
            sample_feats = [extract_consistent_features(p)]
        for f in sample_feats:
            X.append(f)
            y.append(label)
    return np.array(X), np.array(y)

print("Extracting features...")
X_train, y_train = load_split(train_animals, is_train=True)
X_val, y_val = load_split(val_animals, is_train=False)
X_test, y_test = load_split(test_animals, is_train=False)

print(f"X_train shape: {X_train.shape}, Females={np.sum(y_train==0)}, Males={np.sum(y_train==1)}")
print(f"X_val shape: {X_val.shape}, Females={np.sum(y_val==0)}, Males={np.sum(y_val==1)}")
print(f"X_test shape: {X_test.shape}, Females={np.sum(y_test==0)}, Males={np.sum(y_test==1)}")

# Reference samples
ref_samples = [
    ('data/new_images/femalebuffolo.jpg', 0, 'Female 0'),
    ('data/new_images/femalebuffolo1.jpg', 0, 'Female 1'),
    ('data/new_images/femalebuffolo2.jpg', 0, 'Female 2'),
    ('data/new_images/malebuffolo.jpg', 1, 'Male 0'),
    ('data/new_images/malebuffolo1.jpg', 1, 'Male 1'),
    ('data/new_images/malebuffolo2.jpg', 1, 'Male 2'),
    ('data/new_images/malebuffolo3.jpg', 1, 'Male 3')
]

ref_feats = [(desc, true_lbl, extract_consistent_features(p).reshape(1, -1)) for p, true_lbl, desc in ref_samples]

# Candidates to evaluate
candidates = {
    'LR_C0.02': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.02, class_weight='balanced', max_iter=1000, random_state=42))]),
    'LR_C0.05': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.05, class_weight='balanced', max_iter=1000, random_state=42))]),
    'LR_C0.1': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42))]),
    'ET_d6': Pipeline([('scaler', StandardScaler()), ('clf', ExtraTreesClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42))]),
    'RF_d6': Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42))]),
    'SVC_RBF': Pipeline([('scaler', StandardScaler()), ('clf', SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42))]),
    'Ensemble_LR_ET_RF': Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(C=0.05, class_weight='balanced', max_iter=1000, random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42))
            ],
            voting='soft'
        ))
    ]),
    'Ensemble_LR_SVC_ET': Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(C=0.05, class_weight='balanced', max_iter=1000, random_state=42)),
                ('svc', SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42))
            ],
            voting='soft'
        ))
    ])
}

print("\n" + "=" * 80)
print("RESULTS ACROSS CANDIDATE ARCHITECTURES")
print("=" * 80)

for name, model in candidates.items():
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(y_val, val_preds, average='weighted', zero_division=0)
    
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average='weighted', zero_division=0)
    
    # Check reference pattern
    ref_matches = []
    for desc, true_lbl, f_vec in ref_feats:
        probs = model.predict_proba(f_vec)[0]
        p_male = probs[1]
        p_fem = probs[0]
        pred = 1 if p_male > p_fem else 0
        ref_matches.append(pred == true_lbl)
    ref_pass = sum(ref_matches)
    
    print(f"  {name:20s} | Val F1: {val_f1*100:5.1f}% | Test F1: {test_f1*100:5.1f}% (Acc: {test_acc*100:5.1f}%) | Ref Pass: {ref_pass}/7")

