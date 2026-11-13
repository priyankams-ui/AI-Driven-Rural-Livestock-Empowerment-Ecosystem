import os
import sys
import pickle
import numpy as np
import torch
import cv2
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline import pipeline

def extract_ventral_features(img_path, augment=True, n_aug=12):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    # 1. Animal-centric detection bounding box
    count, cow_dets, other, single, r = pipeline.detect_and_count_animals(img, w, h)
    cow_box = cow_dets[0][2] if cow_dets else None
    if cow_box is not None:
        bx1, by1, bx2, by2 = cow_box
        bw, bh = bx2 - bx1, by2 - by1
        y1 = int(by1 + 0.40 * bh)
        y2 = min(h, max(int(by1 + 0.98 * bh), int(by2 + 0.05 * bh), int(h * 0.94)))
        x1 = int(bx1 + 0.12 * bw)
        x2 = int(bx1 + 0.88 * bw)
    else:
        y1, y2 = int(h * 0.42), int(h * 0.94)
        x1, x2 = int(w * 0.12), int(w * 0.88)
    
    y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
    x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
    
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
        t = pipeline.transform_gray(c).unsqueeze(0).to(pipeline.device)
        with torch.no_grad():
            f = pipeline.resnet(t).cpu().numpy().flatten()
        feats.append(f)
    return feats

def run():
    print("=" * 80)
    print("TRAINING & EVALUATING SPECIES-AWARE GENDER CLASSIFIERS")
    print("=" * 80)

    # 1. Verified Cattle Dataset (Bos taurus / Bos indicus)
    cattle_data = [
        # Female Cows (Class 0: 8 animals)
        ('gender_dataset/cow/female/ayrshire_female_1.jpg', 0, 'Female Ayrshire Cow 1'),
        ('gender_dataset/cow/female/brownswiss_female_0.jpg', 0, 'Female Brown Swiss Cow 0'),
        ('gender_dataset/cow/female/femalecow.jpg', 0, 'Female Cow Sample'),
        ('gender_dataset/cow/female/femalecow1.jpg', 0, 'Female Cow 1 Sample'),
        ('gender_dataset/cow/female/femalecow2.jpg', 0, 'Female Cow 2 Sample'),
        ('gender_dataset/cow/female/holstein_female_1.jpg', 0, 'Female Holstein Cow 1'),
        ('gender_dataset/cow/female/jersey_female_0.jpg', 0, 'Female Jersey Cow 0'),
        ('gender_dataset/cow/female/reddane_female_0.jpg', 0, 'Female Red Dane Cow 0'),
        
        # Male Cows (Class 1: 6 animals)
        ('gender_dataset/cow/male/ayrshire_bull_4.jpg', 1, 'Male Ayrshire Bull 4'),
        ('gender_dataset/cow/male/holstein_bull_male.jpg', 1, 'Male Holstein Bull'),
        ('gender_dataset/cow/male/holstein_steer_male.jpg', 1, 'Male Holstein Steer'),
        ('gender_dataset/cow/male/malecow.jpg', 1, 'Male Cow Sample'),
        ('gender_dataset/cow/male/malecow1.jpg', 1, 'Male Cow 1 Sample'),
        ('gender_dataset/cow/male/malecow2.jpg', 1, 'Male Cow 2 Sample'),
    ]

    # 2. Verified Buffalo Dataset (Bubalus bubalis)
    buffalo_data = [
        # Female Buffaloes (Class 0: 5 animals)
        ('gender_dataset/buffalo/female/banni_female_0caae.jpg', 0, 'Female Banni Buffalo 0caae'),
        ('gender_dataset/buffalo/female/femalebuffolo.jpg', 0, 'Female Buffalo Sample'),
        ('gender_dataset/buffalo/female/femalebuffolo1.jpg', 0, 'Female Buffalo 1 Sample'),
        ('gender_dataset/buffalo/female/femalebuffolo2.jpg', 0, 'Female Buffalo 2 Sample'),
        ('gender_dataset/buffalo/female/murrah_female_original.jpg', 0, 'Female Murrah Buffalo Original'),
        
        # Male Buffaloes (Class 1: 7 animals)
        ('gender_dataset/buffalo/male/banni_bull_0.jpg', 1, 'Male Banni Bull 0'),
        ('gender_dataset/buffalo/male/bhadawari_bull.jpg', 1, 'Male Bhadawari Bull'),
        ('gender_dataset/buffalo/male/jafarabadi_bull_punjab.jpg', 1, 'Male Jafarabadi Bull Punjab'),
        ('gender_dataset/buffalo/male/malebuffolo.jpg', 1, 'Male Buffalo Sample'),
        ('gender_dataset/buffalo/male/malebuffolo1.jpg', 1, 'Male Buffalo 1 Sample'),
        ('gender_dataset/buffalo/male/malebuffolo2.jpg', 1, 'Male Buffalo 2 Sample'),
        ('gender_dataset/buffalo/male/malebuffolo3.jpg', 1, 'Male Buffalo 3 Sample'),
    ]

    # Train Cattle Gender Classifier
    print("\n--- TRAINING CATTLE GENDER CLASSIFIER ---")
    Xc, yc = [], []
    for p, label, name in cattle_data:
        feats = extract_ventral_features(p, augment=True, n_aug=12)
        for f in feats:
            Xc.append(f)
            yc.append(label)
    Xc, yc = np.array(Xc), np.array(yc)
    cattle_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    cattle_clf.fit(Xc, yc)
    print(f"Cattle training complete: {Xc.shape[0]} samples.")

    # Train Buffalo Gender Classifier
    print("\n--- TRAINING BUFFALO GENDER CLASSIFIER ---")
    Xb, yb = [], []
    for p, label, name in buffalo_data:
        feats = extract_ventral_features(p, augment=True, n_aug=14)
        for f in feats:
            Xb.append(f)
            yb.append(label)
    Xb, yb = np.array(Xb), np.array(yb)
    buffalo_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    buffalo_clf.fit(Xb, yb)
    print(f"Buffalo training complete: {Xb.shape[0]} samples.")

    # Train Universal / Fallback Classifier
    print("\n--- TRAINING UNIVERSAL GENDER CLASSIFIER ---")
    Xu = np.vstack([Xc, Xb])
    yu = np.concatenate([yc, yb])
    universal_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.08, max_iter=1000, random_state=42))
    ])
    universal_clf.fit(Xu, yu)
    print(f"Universal training complete: {Xu.shape[0]} samples.")

    # Save models
    os.makedirs('models', exist_ok=True)
    with open('models/cattle_gender_clf.pkl', 'wb') as f:
        pickle.dump(cattle_clf, f)
    with open('models/buffalo_gender_clf.pkl', 'wb') as f:
        pickle.dump(buffalo_clf, f)
        
    bundle = {
        'version': '2.3.0',
        'pipeline': universal_clf,
        'cattle_pipeline': cattle_clf,
        'buffalo_pipeline': buffalo_clf,
        'classes': ['Female', 'Male'],
        'confidence_threshold': 0.70,
        'min_margin': 0.15,
        'feature_extractor': 'ResNet18-Ventral-Grayscale-Bilateral-BoundingBox-Aligned'
    }
    with open('models/gender_classifier.pkl', 'wb') as f:
        pickle.dump(bundle, f)
    with open('models/anatomical_gender_clf.pkl', 'wb') as f:
        pickle.dump(universal_clf, f)
        
    print("\n[SUCCESS] Saved cattle_gender_clf.pkl, buffalo_gender_clf.pkl, and gender_classifier.pkl!")

    # Evaluate Cattle Classifier on Cattle
    print("\n" + "=" * 65)
    print("EVALUATION: CATTLE GENDER CLASSIFIER ON COW SAMPLES")
    print("=" * 65)
    y_true_c, y_pred_c = [], []
    for p, exp, name in cattle_data:
        feats = extract_ventral_features(p, augment=False)
        p0 = cattle_clf.predict_proba(feats[0].reshape(1, -1))[0]
        p1 = cattle_clf.predict_proba(feats[1].reshape(1, -1))[0]
        avg = (p0 + p1) / 2.0
        pred = int(np.argmax(avg))
        conf = max(avg)
        y_true_c.append(exp)
        y_pred_c.append(pred)
        match = (pred == exp)
        print(f"  {name:32s} | Exp: {'Female' if exp==0 else 'Male':6s} | Got: {'Female' if pred==0 else 'Male':6s} ({conf:.3f}) | {match}")
    acc_c = accuracy_score(y_true_c, y_pred_c)
    print(f"Cattle Accuracy: {acc_c * 100:.2f}%")

    # Evaluate Buffalo Classifier on Buffalo
    print("\n" + "=" * 65)
    print("EVALUATION: BUFFALO GENDER CLASSIFIER ON BUFFALO SAMPLES")
    print("=" * 65)
    y_true_b, y_pred_b = [], []
    for p, exp, name in buffalo_data:
        feats = extract_ventral_features(p, augment=False)
        p0 = buffalo_clf.predict_proba(feats[0].reshape(1, -1))[0]
        p1 = buffalo_clf.predict_proba(feats[1].reshape(1, -1))[0]
        avg = (p0 + p1) / 2.0
        pred = int(np.argmax(avg))
        conf = max(avg)
        y_true_b.append(exp)
        y_pred_b.append(pred)
        match = (pred == exp)
        print(f"  {name:32s} | Exp: {'Female' if exp==0 else 'Male':6s} | Got: {'Female' if pred==0 else 'Male':6s} ({conf:.3f}) | {match}")
    acc_b = accuracy_score(y_true_b, y_pred_b)
    print(f"Buffalo Accuracy: {acc_b * 100:.2f}%")

if __name__ == '__main__':
    run()
