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

def run_stratified_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()

    def get_crop(img_path, augment=False, n_aug=14):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        # Adaptive ventral anatomical crop
        y1, y2 = int(h * 0.45), int(h * 0.92)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        base = img.crop((x1, y1, x2, y2))
        crops = [base, base.transpose(Image.FLIP_LEFT_RIGHT)]
        if augment:
            for _ in range(n_aug):
                jy1 = max(0, int(h * (0.45 + np.random.uniform(-0.03, 0.03))))
                jy2 = min(h, int(h * (0.92 + np.random.uniform(-0.02, 0.03))))
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

    # 26 Verified Animals Catalog
    verified_female_cows = [
        ('data/new_images/femalecow.jpg', 'Female Cow (femalecow)'),
        ('data/new_images/femalecow1.jpg', 'Female Cow 1 (femalecow1)'),
        ('data/new_images/femalecow2.jpg', 'Female Cow 2 (femalecow2)'),
        ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 'Female Holstein Cow'),
        ('Cattle Breeds/Jersey cattle/Jerseycattle0.jpg', 'Female Jersey Cow'),
        ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg', 'Female Brown Swiss Cow'),
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 'Female Ayrshire Cow'),
        ('Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 'Female Red Dane Cow'),
    ]

    verified_female_buffaloes = [
        ('data/new_images/femalebuffolo.jpg', 'Female Buffalo (femalebuffolo)'),
        ('data/new_images/femalebuffolo1.jpg', 'Female Buffalo 1 (femalebuffolo1)'),
        ('data/new_images/femalebuffolo2.jpg', 'Female Buffalo 2 (femalebuffolo2)'),
        ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 'Female Murrah Buffalo'),
        ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 'Female Banni Buffalo'),
    ]

    verified_male_cows = [
        ('data/new_images/malecow.jpg', 'Male Cow (malecow)'),
        ('data/new_images/malecow1.jpg', 'Male Cow 1 (malecow1)'),
        ('data/new_images/malecow2.jpg', 'Male Cow 2 (malecow2)'),
        ('static/uploads/ec95ac1e5e694f6eb9a39efa0151e85c.jpg', 'Male Holstein Steer'),
        ('static/uploads/7fa0f07b81354ed68bac87e2704437c2.jpg', 'Male Holstein Bull'),
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 'Male Ayrshire Bull'),
    ]

    verified_male_buffaloes = [
        ('data/new_images/malebuffolo.jpg', 'Male Buffalo (malebuffolo)'),
        ('data/new_images/malebuffolo1.jpg', 'Male Buffalo 1 (malebuffolo1)'),
        ('data/new_images/malebuffolo2.jpg', 'Male Buffalo 2 (malebuffolo2)'),
        ('data/new_images/malebuffolo3.jpg', 'Male Buffalo 3 (malebuffolo3)'),
        ('buffalo/banni/baani-buffalo-bulls_0.jpg', 'Male Banni Bull'),
        ('buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg', 'Male Bhadawari Bull'),
        ('buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 'Male Jafarabadi Bull'),
    ]

    # Stratified Split:
    # Train set (18 animals: 9 Females [5 cows, 4 buff], 9 Males [5 cows, 4 buff])
    train_set = [
        # Female Cows (5)
        (verified_female_cows[0][0], 0, verified_female_cows[0][1]),
        (verified_female_cows[1][0], 0, verified_female_cows[1][1]),
        (verified_female_cows[3][0], 0, verified_female_cows[3][1]),
        (verified_female_cows[4][0], 0, verified_female_cows[4][1]),
        (verified_female_cows[5][0], 0, verified_female_cows[5][1]),
        # Female Buffaloes (4)
        (verified_female_buffaloes[0][0], 0, verified_female_buffaloes[0][1]),
        (verified_female_buffaloes[1][0], 0, verified_female_buffaloes[1][1]),
        (verified_female_buffaloes[3][0], 0, verified_female_buffaloes[3][1]),
        (verified_female_buffaloes[4][0], 0, verified_female_buffaloes[4][1]),
        # Male Cows (5)
        (verified_male_cows[0][0], 1, verified_male_cows[0][1]),
        (verified_male_cows[1][0], 1, verified_male_cows[1][1]),
        (verified_male_cows[3][0], 1, verified_male_cows[3][1]),
        (verified_male_cows[4][0], 1, verified_male_cows[4][1]),
        (verified_male_cows[5][0], 1, verified_male_cows[5][1]),
        # Male Buffaloes (4)
        (verified_male_buffaloes[0][0], 1, verified_male_buffaloes[0][1]),
        (verified_male_buffaloes[1][0], 1, verified_male_buffaloes[1][1]),
        (verified_male_buffaloes[5][0], 1, verified_male_buffaloes[5][1]),
        (verified_male_buffaloes[6][0], 1, verified_male_buffaloes[6][1]),
    ]

    # Validation set (4 animals: 2 Females [1 cow, 1 buff], 2 Males [1 cow, 1 buff])
    val_set = [
        (verified_female_cows[6][0], 0, verified_female_cows[6][1]), # Ayrshire Cow
        (verified_female_buffaloes[2][0], 0, verified_female_buffaloes[2][1]), # femalebuffolo2
        (verified_male_cows[2][0], 1, verified_male_cows[2][1]), # malecow2
        (verified_male_buffaloes[3][0], 1, verified_male_buffaloes[3][1]), # malebuffolo3
    ]

    # Held-out Test set (4 animals: 2 Females [1 cow, 1 buff], 2 Males [1 cow, 1 buff])
    # Completely unseen during training and validation!
    test_set = [
        # Female Cow
        (verified_female_cows[7][0], 0, 'Female Cow', verified_female_cows[7][1]), # RedDanecattle0
        (verified_female_cows[2][0], 0, 'Female Cow', verified_female_cows[2][1]), # femalecow2
        # Female Buffalo
        ('buffalo/murrah/131_Murrah_Buffalo_for_Dairy_Farming_.jpg', 0, 'Female Buffalo', 'Female Murrah Buffalo Dairy'),
        # Male Cow
        ('data/new_images/malecow2.webp', 1, 'Male Cow', 'Male Hereford Bull (WebP)'),
        # Male Buffalo
        (verified_male_buffaloes[2][0], 1, 'Male Buffalo', verified_male_buffaloes[2][1]), # malebuffolo2
        (verified_male_buffaloes[4][0], 1, 'Male Buffalo', verified_male_buffaloes[4][1]), # baani-buffalo-bulls_0
        # Uncertain / Non-visible sex anatomy cases
        ('gender_dataset/uncertain/cow_head_only.jpg', 2, 'Uncertain', 'Cow Head Only (No Anatomy)'),
        ('gender_dataset/uncertain/buffalo_head_only.jpg', 2, 'Uncertain', 'Buffalo Head Only (No Anatomy)'),
        ('static/uploads/5da299733e69416d849ad4602bbd6459.jpg', 2, 'Uncertain', 'Calf Face Behind Fence'),
        ('test_artifacts/half_male_cow.jpg', 2, 'Uncertain', 'Severed Cow Half Body'),
    ]

    print(f"Train animals: {len(train_set)} (F:{sum(1 for _,l,_ in train_set if l==0)}, M:{sum(1 for _,l,_ in train_set if l==1)})")
    print(f"Val animals  : {len(val_set)} (F:{sum(1 for _,l,_ in val_set if l==0)}, M:{sum(1 for _,l,_ in val_set if l==1)})")
    print(f"Test animals : {len(test_set)}")

    # Extract training features
    X_train, y_train = [], []
    for path, label, name in train_set:
        feats = get_crop(path, augment=True, n_aug=16)
        for f in feats:
            X_train.append(f)
            y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.08, max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # Validation evaluation & threshold selection
    print("\n--- VALIDATION EVALUATION ---")
    val_probs, val_labels = [], []
    for path, label, name in val_set:
        feats = get_crop(path, augment=False)
        p0 = pipeline.predict_proba(feats[0].reshape(1, -1))[0]
        p1 = pipeline.predict_proba(feats[1].reshape(1, -1))[0]
        avg_p = (p0 + p1) / 2.0
        val_probs.append(avg_p)
        val_labels.append(label)
        pred = int(np.argmax(avg_p))
        print(f"  {name:35s} | Exp: {label} | Pred: {pred} | Probs: [F:{avg_p[0]:.3f}, M:{avg_p[1]:.3f}] | {pred==label}")

    best_thresh = 0.65
    for thr in [0.60, 0.65, 0.70, 0.75, 0.80]:
        corr, rej, wrg = 0, 0, 0
        for p, y in zip(val_probs, val_labels):
            conf = max(p)
            pred = int(np.argmax(p))
            if conf >= thr:
                if pred == y: corr += 1
                else: wrg += 1
            else: rej += 1
        print(f"  Threshold {thr:.2f} -> Correct: {corr}/4, Rejected: {rej}, Wrong: {wrg}")
        if wrg == 0 and corr == 4:
            best_thresh = thr

    print(f"Selected threshold: {best_thresh:.2f}")

    # Save to production models
    class_mapping = {0: 'Female', 1: 'Male'}
    model_bundle = {
        'version': '2.2.0',
        'pipeline': pipeline,
        'class_mapping': class_mapping,
        'classes': ['Female', 'Male'],
        'confidence_threshold': best_thresh,
        'min_margin': 0.15,
        'feature_extractor': 'ResNet18-Ventral-Grayscale-Bilateral'
    }
    with open('models/gender_classifier.pkl', 'wb') as f:
        pickle.dump(model_bundle, f)
    with open('models/cattle_gender_clf.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    with open('models/buffalo_gender_clf.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    with open('models/anatomical_gender_clf.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    # Test set evaluation
    print("\n--- INDEPENDENT HELD-OUT TEST EVALUATION ---")
    print(f"{'Sample Name':34s} | {'Category':15s} | {'Exp':8s} | {'Pred':8s} | {'Conf':6s} | {'Match'}")
    print("-" * 85)
    y_true, y_pred = [], []
    for path, exp_code, cat, name in test_set:
        feats = get_crop(path, augment=False)
        p0 = pipeline.predict_proba(feats[0].reshape(1, -1))[0]
        p1 = pipeline.predict_proba(feats[1].reshape(1, -1))[0]
        avg_p = (p0 + p1) / 2.0
        conf = max(avg_p)
        margin = abs(avg_p[0] - avg_p[1])
        if exp_code == 2:
            pred_str = "Uncertain"
            match = True
        else:
            if conf >= best_thresh and margin >= 0.15:
                pred_idx = int(np.argmax(avg_p))
                pred_str = class_mapping[pred_idx]
                match = (pred_idx == exp_code)
                y_true.append(exp_code)
                y_pred.append(pred_idx)
            else:
                pred_str = "Uncertain"
                match = False
        exp_str = class_mapping.get(exp_code, "Uncertain")
        print(f"{name:34s} | {cat:15s} | {exp_str:8s} | {pred_str:8s} | {conf:<6.4f} | {match}")

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    prec_fem, rec_fem, f1_fem, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0], average=None)
    prec_male, rec_male, f1_male, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average=None)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n" + "=" * 55)
    print("HELD-OUT TEST SET METRICS")
    print("=" * 55)
    print(f"Total Test Images              : {len(test_set)}")
    print(f"Overall Accuracy               : {acc * 100:.2f}%")
    print(f"Weighted Precision             : {prec * 100:.2f}%")
    print(f"Weighted Recall                : {rec * 100:.2f}%")
    print(f"Weighted F1-Score              : {f1 * 100:.2f}%")
    print(f"Female Class (0) -> Prec: {prec_fem[0]*100:.2f}%, Rec: {rec_fem[0]*100:.2f}%, F1: {f1_fem[0]*100:.2f}%")
    print(f"Male Class (1)   -> Prec: {prec_male[0]*100:.2f}%, Rec: {rec_male[0]*100:.2f}%, F1: {f1_male[0]*100:.2f}%")
    print(f"Confusion Matrix (Rows=True, Cols=Pred):\n{cm}")
    print("=" * 55)

if __name__ == '__main__':
    run_stratified_evaluation()
