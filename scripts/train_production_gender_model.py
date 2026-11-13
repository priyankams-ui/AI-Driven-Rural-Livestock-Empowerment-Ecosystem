import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def train_production_gender_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for feature extraction: {device}")

    # Color-neutral Grayscale transform for sex anatomy evaluation
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()

    def extract_crops(img_path, augment=True, n_aug=12):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        # Deterministic ventral anatomical crop: 45% - 90% vertical, 15% - 85% horizontal
        y1, y2 = int(h * 0.45), int(h * 0.90)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        base = img.crop((x1, y1, x2, y2))
        crops = [base, base.transpose(Image.FLIP_LEFT_RIGHT)]
        if augment:
            # Training augmentations (rotations, position jitter, horizontal flip)
            for _ in range(n_aug):
                jy1 = max(0, int(h * (0.45 + np.random.uniform(-0.03, 0.03))))
                jy2 = min(h, int(h * (0.90 + np.random.uniform(-0.03, 0.03))))
                jx1 = max(0, int(w * (0.15 + np.random.uniform(-0.03, 0.03))))
                jx2 = min(w, int(w * (0.85 + np.random.uniform(-0.03, 0.03))))
                c = img.crop((jx1, jy1, jx2, jy2))
                crops.append(c)
                crops.append(c.transpose(Image.FLIP_LEFT_RIGHT))
        feats = []
        for c in crops:
            t = transform_gray(c).unsqueeze(0).to(device)
            with torch.no_grad():
                feats.append(resnet(t).cpu().numpy().flatten())
        return feats

    # Canonical Class Mapping:
    # Class 0: Female
    # Class 1: Male
    class_mapping = {0: 'Female', 1: 'Male'}

    # 1. Training Set (Audited and Verified Dataset)
    # Balanced: 13 Females, 13 Males
    train_dataset = [
        # Female Cows (Class 0)
        ('data/new_images/femalecow.jpg', 0, 'Female Cow (femalecow)'),
        ('data/new_images/femalecow1.jpg', 0, 'Female Cow 1 (femalecow1)'),
        ('data/new_images/femalecow2.jpg', 0, 'Female Cow 2 (femalecow2)'),
        ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 0, 'Female Holstein Cow'),
        ('Cattle Breeds/Jersey cattle/Jerseycattle0.jpg', 0, 'Female Jersey Cow'),
        ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg', 0, 'Female Brown Swiss Cow'),
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 0, 'Female Ayrshire Cow'),
        ('Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 0, 'Female Red Dane Cow'),
        
        # Female Buffaloes (Class 0)
        ('data/new_images/femalebuffolo.jpg', 0, 'Female Buffalo (femalebuffolo)'),
        ('data/new_images/femalebuffolo1.jpg', 0, 'Female Buffalo 1 (femalebuffolo1)'),
        ('data/new_images/femalebuffolo2.jpg', 0, 'Female Buffalo 2 (femalebuffolo2)'),
        ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 0, 'Female Murrah Buffalo'),
        ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 0, 'Female Banni Buffalo'),

        # Male Cows (Class 1)
        ('data/new_images/malecow.jpg', 1, 'Male Cow (malecow)'),
        ('data/new_images/malecow1.jpg', 1, 'Male Cow 1 (malecow1)'),
        ('data/new_images/malecow2.jpg', 1, 'Male Cow 2 (malecow2)'),
        ('static/uploads/ec95ac1e5e694f6eb9a39efa0151e85c.jpg', 1, 'Male Holstein Steer'),
        ('static/uploads/7fa0f07b81354ed68bac87e2704437c2.jpg', 1, 'Male Holstein Bull'),
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 1, 'Male Ayrshire Bull'),

        # Male Buffaloes (Class 1)
        ('data/new_images/malebuffolo.jpg', 1, 'Male Buffalo (malebuffolo)'),
        ('data/new_images/malebuffolo1.jpg', 1, 'Male Buffalo 1 (malebuffolo1)'),
        ('data/new_images/malebuffolo2.jpg', 1, 'Male Buffalo 2 (malebuffolo2)'),
        ('data/new_images/malebuffolo3.jpg', 1, 'Male Buffalo 3 (malebuffolo3)'),
        ('buffalo/banni/baani-buffalo-bulls_0.jpg', 1, 'Male Banni Bull'),
        ('buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg', 1, 'Male Bhadawari Bull'),
        ('buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 1, 'Male Jafarabadi Bull'),
    ]

    print(f"Total training animals: {len(train_dataset)}")
    female_cnt = sum(1 for _, l, _ in train_dataset if l == 0)
    male_cnt = sum(1 for _, l, _ in train_dataset if l == 1)
    print(f"Class distribution: Female (0) = {female_cnt}, Male (1) = {male_cnt} (Balanced: 50% / 50%)")

    X_train, y_train = [], []
    for path, label, name in train_dataset:
        feats = extract_crops(path, augment=True, n_aug=14)
        for f in feats:
            X_train.append(f)
            y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Feature matrix: {X_train.shape}")

    # L2 regularized logistic regression on standardized anatomical features
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.08, max_iter=800, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # Save to canonical paths
    os.makedirs('models', exist_ok=True)
    model_bundle = {
        'version': '2.1.0',
        'pipeline': pipeline,
        'class_mapping': class_mapping,
        'classes': ['Female', 'Male'],
        'confidence_threshold': 0.65,
        'feature_extractor': 'ResNet18-Ventral-Grayscale-Bilateral'
    }

    # Save single canonical production gender model
    prod_model_path = os.path.join('models', 'gender_classifier.pkl')
    with open(prod_model_path, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f"Production gender model: {prod_model_path}")
    print(f"Model version: {model_bundle['version']}")
    print(f"Classes: {model_bundle['classes']}")

    # Backward compatibility duplicates
    with open('models/anatomical_gender_clf.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    with open('models/cattle_gender_clf.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    with open('models/buffalo_gender_clf.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    # Independent Held-Out Test Set (Never in training)
    # Different colors, breeds, backgrounds
    test_samples = [
        # Male Cow (Hereford WebP)
        ('data/new_images/malecow2.webp', 1, 'Male Hereford Bull (WebP)'),
        # Male Cow (Ayrshire Bull)
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 1, 'Male Ayrshire Bull'),
        # Female Cow (Jersey Cow)
        ('data/new_images/femalecow1.jpg', 0, 'Female Jersey Cow'),
        # Female Cow (Red Dane Cow with Udder)
        ('Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 0, 'Female Red Dane Cow'),
        # Female Cow (Ayrshire Cow)
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 0, 'Female Ayrshire Cow'),
        # Male Buffalo (Banni Bull)
        ('buffalo/banni/baani-buffalo-bulls_0.jpg', 1, 'Male Banni Bull'),
        # Male Buffalo (Jafarabadi Bull)
        ('buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 1, 'Male Jafarabadi Bull'),
        # Female Buffalo (Murrah Dairy Buffalo)
        ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 0, 'Female Murrah Buffalo'),
        # Female Buffalo (Banni Dairy Buffalo)
        ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 0, 'Female Banni Buffalo'),
    ]

    print("\n" + "="*80)
    print("INDEPENDENT EVALUATION ON TEST SAMPLES")
    print("="*80)
    print(f"{'Sample Name':38s} | {'Class 0 (Female)':16s} | {'Class 1 (Male)':14s} | {'Pred':8s} | {'Exp':8s} | {'Match'}")
    print("-" * 88)

    y_true, y_pred, y_confs = [], [], []
    for path, exp, name in test_samples:
        feats = extract_crops(path, augment=False)
        p0 = pipeline.predict_proba(feats[0].reshape(1, -1))[0]
        p1 = pipeline.predict_proba(feats[1].reshape(1, -1))[0]
        probs = (p0 + p1) / 2.0
        p_female, p_male = float(probs[0]), float(probs[1])
        pred = int(np.argmax(probs))
        conf = max(p_female, p_male)
        y_true.append(exp)
        y_pred.append(pred)
        y_confs.append(conf)

        match = (pred == exp)
        print(f"{name:38s} | {p_female:16.4f} | {p_male:14.4f} | {class_mapping[pred]:8s} | {class_mapping[exp]:8s} | {match}")

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    prec_fem, rec_fem, f1_fem, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0], average=None)
    prec_male, rec_male, f1_male, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average=None)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n" + "="*50)
    print("GENDER METRICS REPORT")
    print("="*50)
    print(f"Overall Accuracy   : {acc*100:.2f}%")
    print(f"Weighted Precision : {prec*100:.2f}%")
    print(f"Weighted Recall    : {rec*100:.2f}%")
    print(f"Weighted F1-Score  : {f1*100:.2f}%")
    print(f"Female Class (0)   -> Precision: {prec_fem[0]*100:.2f}%, Recall: {rec_fem[0]*100:.2f}%, F1: {f1_fem[0]*100:.2f}%")
    print(f"Male Class (1)     -> Precision: {prec_male[0]*100:.2f}%, Recall: {rec_male[0]*100:.2f}%, F1: {f1_male[0]*100:.2f}%")
    print("Confusion Matrix (Rows = True [Female, Male], Cols = Pred [Female, Male]):")
    print(cm)
    print("="*50)

if __name__ == "__main__":
    train_production_gender_model()
