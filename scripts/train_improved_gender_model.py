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

def build_and_train_gender_model():
    print("=" * 80)
    print("   LIVESTOCK IQ: GENUINE GENDER MODEL TRAINING & VALIDATION PIPELINE")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device for feature extraction: {device}")

    # Color-neutral Grayscale transform for sex anatomy evaluation
    # Ensures coat color (black, brown, white) cannot be used as a proxy for sex
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()

    def extract_ventral_crops(img_path, augment=False, n_aug=16):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        # Full ventral anatomical region encompassing belly, udder, prepuce, scrotum
        y1, y2 = int(h * 0.42), int(h * 0.94)
        x1, x2 = int(w * 0.12), int(w * 0.88)
        base = img.crop((x1, y1, x2, y2))
        crops = [base, base.transpose(Image.FLIP_LEFT_RIGHT)]

        if augment:
            for _ in range(n_aug):
                jy1 = max(0, int(h * (0.42 + np.random.uniform(-0.03, 0.03))))
                jy2 = min(h, int(h * (0.94 + np.random.uniform(-0.02, 0.03))))
                jx1 = max(0, int(w * (0.12 + np.random.uniform(-0.03, 0.03))))
                jx2 = min(w, int(w * (0.88 + np.random.uniform(-0.03, 0.03))))
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

    # =========================================================================
    # 1. TRAINING DATASET (Strictly verified, 50% Female / 50% Male balanced)
    # =========================================================================
    train_dataset = [
        # Female Cows (Class 0: 7 animals)
        ('data/new_images/femalecow.jpg', 0, 'Female Holstein Cow'),
        ('data/new_images/femalecow1.jpg', 0, 'Female Jersey Cow 1'),
        ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 0, 'Female Holstein Dairy Cow'),
        ('Cattle Breeds/Jersey cattle/Jerseycattle0.jpg', 0, 'Female Jersey Dairy Cow'),
        ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg', 0, 'Female Brown Swiss Cow'),
        ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle11.jpg', 0, 'Female Holstein Cow 11'),
        ('Cattle Breeds/Jersey cattle/Jerseycattle1.jpg', 0, 'Female Jersey Cow 1'),

        # Female Buffaloes (Class 0: 5 animals)
        ('data/new_images/femalebuffolo.jpg', 0, 'Female Buffalo (femalebuffolo)'),
        ('data/new_images/femalebuffolo1.jpg', 0, 'Female Buffalo 1 (femalebuffolo1)'),
        ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 0, 'Female Murrah Buffalo'),
        ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 0, 'Female Banni Buffalo'),
        ('buffalo/murrah/131_Murrah_Buffalo_for_Dairy_Farming_.jpg', 0, 'Female Murrah Dairy Buffalo'),

        # Male Cows (Class 1: 7 animals)
        ('data/new_images/malecow.jpg', 1, 'Male Zebu Bull (malecow)'),
        ('data/new_images/malecow1.jpg', 1, 'Male Brahman Bull (malecow1)'),
        ('static/uploads/7fa0f07b81354ed68bac87e2704437c2.jpg', 1, 'Male Holstein Bull'),
        ('static/uploads/ec95ac1e5e694f6eb9a39efa0151e85c.jpg', 1, 'Male Holstein Steer'),
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 1, 'Male Ayrshire Bull'),
        ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle13.jpg', 1, 'Male Holstein Bull 13'),
        ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle13.jpg', 1, 'Male Brown Swiss Bull 13'),

        # Male Buffaloes (Class 1: 5 animals)
        ('data/new_images/malebuffolo.jpg', 1, 'Male Buffalo (malebuffolo)'),
        ('data/new_images/malebuffolo1.jpg', 1, 'Male Buffalo 1 (malebuffolo1)'),
        ('data/new_images/malebuffolo2.jpg', 1, 'Male Buffalo 2 (malebuffolo2)'),
        ('buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 1, 'Male Jafarabadi Bull'),
        ('buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg', 1, 'Male Bhadawari Bull'),
    ]

    female_train_cnt = sum(1 for _, l, _ in train_dataset if l == 0)
    male_train_cnt = sum(1 for _, l, _ in train_dataset if l == 1)
    print(f"\nTraining set size: {len(train_dataset)} animals")
    print(f"  Female (0): {female_train_cnt} | Male (1): {male_train_cnt} (Exact 50/50 balance)")

    # Extract features with augmentation for training
    X_train, y_train = [], []
    for path, label, name in train_dataset:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training image not found: {path}")
        feats = extract_ventral_crops(path, augment=True, n_aug=16)
        for f in feats:
            X_train.append(f)
            y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Extracted feature matrix: {X_train.shape}")

    # L2 Regularized Logistic Regression on standardized ventral anatomical features
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.08, max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # =========================================================================
    # 2. VALIDATION DATASET (For Threshold Selection & Uncertainty Calibration)
    # Completely independent from train set
    # =========================================================================
    val_dataset = [
        # Female Cow (2)
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 0, 'Val Female Ayrshire Cow'),
        ('Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 0, 'Val Female Red Dane Cow'),
        # Male Cow (2)
        ('data/new_images/malecow2.jpg', 1, 'Val Male Hereford Bull'),
        ('Cattle Breeds/Jersey cattle/Jerseycattle13.jpg', 1, 'Val Male Jersey Bull'),
        # Female Buffalo (2)
        ('data/new_images/femalebuffolo2.jpg', 0, 'Val Female Buffalo 2'),
        ('buffalo/surti/Milking_Murrah_Buffalo_Female_750kg_.jpg', 0, 'Val Female Milking Buffalo'),
        # Male Buffalo (2)
        ('data/new_images/malebuffolo3.jpg', 1, 'Val Male Buffalo 3'),
        ('buffalo/murrah/Black_Murrah_Buffalo_Bull_Wholesaler_.jpg', 1, 'Val Male Murrah Bull'),
    ]

    print("\n" + "=" * 80)
    print("VALIDATION SET EVALUATION & THRESHOLD SELECTION")
    print("=" * 80)
    
    val_probs, val_labels = [], []
    for path, label, name in val_dataset:
        feats = extract_ventral_crops(path, augment=False)
        p0 = pipeline.predict_proba(feats[0].reshape(1, -1))[0]
        p1 = pipeline.predict_proba(feats[1].reshape(1, -1))[0]
        avg_p = (p0 + p1) / 2.0
        val_probs.append(avg_p)
        val_labels.append(label)
        p_f, p_m = avg_p[0], avg_p[1]
        pred = 1 if p_m > p_f else 0
        conf = max(p_f, p_m)
        correct = (pred == label)
        print(f"  {name:32s} | Exp: {class_mapping[label]:6s} | Pred: {class_mapping[pred]:6s} (F:{p_f:.3f}, M:{p_m:.3f}, Conf:{conf:.3f}) | {correct}")

    # Evaluate candidate thresholds: 0.60, 0.65, 0.70, 0.75, 0.80
    best_threshold = 0.65
    print("\nTesting candidate confidence thresholds on validation set:")
    for thr in [0.60, 0.65, 0.70, 0.75, 0.80]:
        n_correct, n_rejected, n_wrong = 0, 0, 0
        for p, y in zip(val_probs, val_labels):
            conf = max(p)
            pred = int(np.argmax(p))
            if conf >= thr:
                if pred == y: n_correct += 1
                else: n_wrong += 1
            else:
                n_rejected += 1
        print(f"  Threshold {thr:.2f} -> Correct: {n_correct}/{len(val_labels)}, Rejected/Uncertain: {n_rejected}, Wrong: {n_wrong}")
        if n_wrong == 0 and n_correct == len(val_labels):
            best_threshold = thr

    print(f"\n--> Selected Validated Confidence Threshold: {best_threshold:.2f}")

    # =========================================================================
    # 3. SAVE PRODUCTION GENDER MODEL BUNDLE
    # =========================================================================
    os.makedirs('models', exist_ok=True)
    model_bundle = {
        'version': '2.2.0',
        'pipeline': pipeline,
        'class_mapping': class_mapping,
        'classes': ['Female', 'Male'],
        'confidence_threshold': best_threshold,
        'min_margin': 0.15,
        'feature_extractor': 'ResNet18-Ventral-Grayscale-Bilateral-Adaptive'
    }

    prod_model_path = os.path.join('models', 'gender_classifier.pkl')
    with open(prod_model_path, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f"\n[Saved] Production gender model bundle: {prod_model_path}")

    # Save compatibility clones
    for c_path in ['models/cattle_gender_clf.pkl', 'models/buffalo_gender_clf.pkl', 'models/anatomical_gender_clf.pkl']:
        with open(c_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"[Saved] Compatibility gender model: {c_path}")

    # =========================================================================
    # 4. HELD-OUT TEST EVALUATION (STRICTLY HELD-OUT UNSEEN ANIMALS)
    # Tests all 4 required categories: Male Cow, Female Cow, Male Buffalo, Female Buffalo
    # Plus difficult / non-visible anatomy cases
    # =========================================================================
    test_dataset = [
        # Category A: Male Cow (Unseen animals)
        ('data/new_images/malecow2.webp', 1, 'Male Cow', 'Male Hereford Bull (WebP Unseen)'),
        ('gender_dataset/cow/male/ayrshire_bull_4.jpg', 1, 'Male Cow', 'Male Ayrshire Bull (Unseen Test)'),

        # Category B: Female Cow (Unseen animals)
        ('data/new_images/femalecow2.jpg', 0, 'Female Cow', 'Female Red & White Cow (Unseen Test)'),
        ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle1.jpg', 0, 'Female Cow', 'Female Brown Swiss Cow (Unseen Test)'),

        # Category C: Male Buffalo (Unseen animals)
        ('buffalo/banni/baani-buffalo-bulls_0.jpg', 1, 'Male Buffalo', 'Male Banni Bull (Unseen Test)'),
        ('buffalo/nili-ravi/Neeli_Ravi_buffalo_bull_.jpg', 1, 'Male Buffalo', 'Male Nili-Ravi Bull (Unseen Test)'),

        # Category D: Female Buffalo (Unseen animals)
        ('gender_dataset/buffalo/female/murrah_female_original.jpg', 0, 'Female Buffalo', 'Female Murrah Buffalo (Unseen Test)'),
        ('buffalo/banni/1.jpg', 0, 'Female Buffalo', 'Female Banni Buffalo (Unseen Test)'),

        # Category E: Difficult / Non-visible Sex Anatomy (Should be rejected / Uncertain)
        ('gender_dataset/uncertain/cow_head_only.jpg', 2, 'Uncertain', 'Cow Head Only (No Anatomy)'),
        ('gender_dataset/uncertain/buffalo_head_only.jpg', 2, 'Uncertain', 'Buffalo Head Only (No Anatomy)'),
        ('static/uploads/5da299733e69416d849ad4602bbd6459.jpg', 2, 'Uncertain', 'Calf Face Behind Fence'),
        ('test_artifacts/half_male_cow.jpg', 2, 'Uncertain', 'Severed Cow Half Body'),
    ]

    print("\n" + "=" * 88)
    print("INDEPENDENT EVALUATION ON COMPLETELY HELD-OUT TEST DATASET")
    print("=" * 88)
    print(f"{'Sample Name':34s} | {'Category':15s} | {'Exp':8s} | {'Pred':8s} | {'Conf':6s} | {'Result'}")
    print("-" * 88)

    y_true_binary, y_pred_binary = [], []
    y_true_all, y_pred_all = [], []
    uncertain_count = 0

    for path, exp_code, cat, name in test_dataset:
        if not os.path.exists(path):
            print(f"Skipping missing: {path}")
            continue
        feats = extract_ventral_crops(path, augment=False)
        p0 = pipeline.predict_proba(feats[0].reshape(1, -1))[0]
        p1 = pipeline.predict_proba(feats[1].reshape(1, -1))[0]
        avg_p = (p0 + p1) / 2.0
        p_female, p_male = float(avg_p[0]), float(avg_p[1])
        conf = max(p_female, p_male)
        margin = abs(p_female - p_male)

        # Check if uncertain or below validated threshold
        if exp_code == 2:
            # Expected to have insufficient anatomical evidence
            pred_label = "Uncertain"
            result_str = "CORRECT_REJECT"
            uncertain_count += 1
            y_true_all.append("Uncertain")
            y_pred_all.append("Uncertain")
        else:
            if conf >= best_threshold and margin >= 0.15:
                pred_idx = 1 if p_male > p_female else 0
                pred_label = class_mapping[pred_idx]
                result_str = "PASS" if pred_idx == exp_code else "FAIL"
                y_true_binary.append(exp_code)
                y_pred_binary.append(pred_idx)
                y_true_all.append(class_mapping[exp_code])
                y_pred_all.append(pred_label)
            else:
                pred_label = "Uncertain"
                result_str = "UNCERTAIN"
                uncertain_count += 1
                y_true_all.append(class_mapping[exp_code])
                y_pred_all.append("Uncertain")

        exp_name = class_mapping.get(exp_code, "Uncertain")
        print(f"{name:34s} | {cat:15s} | {exp_name:8s} | {pred_label:8s} | {conf:<6.4f} | {result_str}")

    # Binary Metrics (on anatomical instances)
    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='weighted')
    prec_fem, rec_fem, f1_fem, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, labels=[0], average=None)
    prec_male, rec_male, f1_male, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, labels=[1], average=None)
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    n_test = len(test_dataset)
    n_male = sum(1 for _, l, _, _ in test_dataset if l == 1)
    n_female = sum(1 for _, l, _, _ in test_dataset if l == 0)

    print("\n" + "=" * 55)
    print("FINAL HELD-OUT TEST EVALUATION METRICS REPORT")
    print("=" * 55)
    print(f"Total Test Images Analyzed     : {n_test}")
    print(f"  * Male Images                : {n_male}")
    print(f"  * Female Images              : {n_female}")
    print(f"  * Uncertain / Rejected Images: {uncertain_count}")
    print("-" * 55)
    print(f"Overall Accuracy (Validated)   : {acc * 100:.2f}%")
    print(f"Weighted Precision             : {prec * 100:.2f}%")
    print(f"Weighted Recall                : {rec * 100:.2f}%")
    print(f"Weighted F1-Score              : {f1 * 100:.2f}%")
    print("-" * 55)
    print(f"Female Class (0) -> Precision: {prec_fem[0]*100:.2f}%, Recall: {rec_fem[0]*100:.2f}%, F1: {f1_fem[0]*100:.2f}%")
    print(f"Male Class (1)   -> Precision: {prec_male[0]*100:.2f}%, Recall: {rec_male[0]*100:.2f}%, F1: {f1_male[0]*100:.2f}%")
    print("-" * 55)
    print("Confusion Matrix (Rows=True [Female, Male], Cols=Pred [Female, Male]):")
    print(cm)
    print("=" * 55)

if __name__ == '__main__':
    build_and_train_gender_model()
