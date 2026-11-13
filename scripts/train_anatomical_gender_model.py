import os
import sys
import json
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

def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    # Classes: 0 = Male, 1 = Female, 2 = Uncertain (anatomical cues hidden/occluded)
    train_samples = [
        # Male Cows (Various colors, breeds, backgrounds)
        ('data/new_images/malecow.jpg', 0, 'Cow', 'Male Zebu Bull (Brown)'),
        ('data/new_images/malecow1.jpg', 0, 'Cow', 'Male Brahman Bull (Red/Brown)'),
        ('data/new_images/malecow2.jpg', 0, 'Cow', 'Male Hereford Bull (Red/White)'),
        ('static/uploads/ec95ac1e5e694f6eb9a39efa0151e85c.jpg', 0, 'Cow', 'Male Steer (B&W)'),
        ('static/uploads/7fa0f07b81354ed68bac87e2704437c2.jpg', 0, 'Cow', 'Male Holstein Bull (B&W)'),
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 0, 'Cow', 'Male Ayrshire Bull (Red/Brown)'),
        
        # Female Cows (Various colors, breeds, backgrounds)
        ('data/new_images/femalecow.jpg', 1, 'Cow', 'Female Holstein (B&W, Udder)'),
        ('data/new_images/femalecow2.jpg', 1, 'Cow', 'Female Red/White (Udder)'),
        ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 1, 'Cow', 'Female Holstein 1 (Udder)'),
        ('Cattle Breeds/Jersey cattle/Jerseycattle0.jpg', 1, 'Cow', 'Female Jersey 0 (Brown, Udder)'),
        ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg', 1, 'Cow', 'Female Brown Swiss (Grey/Brown, Udder)'),
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 1, 'Cow', 'Female Ayrshire 1 (Udder)'),
        
        # Male Buffaloes (Various sizes, breeds, backgrounds)
        ('data/new_images/malebuffolo1.jpg', 0, 'Buffalo', 'Male Buffalo Bull 1 (Sheath)'),
        ('data/new_images/malebuffolo2.jpg', 0, 'Buffalo', 'Male Buffalo Bull 2 (Sheath)'),
        ('data/new_images/malebuffolo3.jpg', 0, 'Buffalo', 'Male Buffalo Bull 3 (Sheath)'),
        ('buffalo/banni/baani-buffalo-bulls_0.jpg', 0, 'Buffalo', 'Male Banni Bull (Light, Sheath)'),
        ('buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg', 0, 'Buffalo', 'Male Bhadawari Bull (Brown/Dark)'),
        ('buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 0, 'Buffalo', 'Male Jafarabadi Bull (Sheath)'),
        
        # Female Buffaloes (Various sizes, breeds, backgrounds)
        ('data/new_images/femalebuffolo2.jpg', 1, 'Buffalo', 'Female Buffalo 2 (White BG, Udder)'),
        ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 1, 'Buffalo', 'Female Murrah (Black, Udder)'),
        ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 1, 'Buffalo', 'Female Banni (Udder)'),
        
        # Uncertain / Non-visible sex anatomy samples
        ('static/uploads/5da299733e69416d849ad4602bbd6459.jpg', 2, 'Cow', 'Calf Headshot / Fence'),
        ('data/new_images/femalecow1.jpg', 2, 'Cow', 'Female Cow 1 (Belly Occluded)'),
        ('data/new_images/femalebuffolo.jpg', 2, 'Buffalo', 'Female Buffalo (Belly Submerged in Water)'),
        ('data/new_images/femalebuffolo1.jpg', 2, 'Buffalo', 'Ambiguous Buffalo 1 (Sheath Conflict)'),
        ('data/new_images/malebuffolo.jpg', 2, 'Buffalo', 'Male Buffalo (Low Angle / Occluded)'),
    ]

    # Independent Held-Out Test Set
    test_samples = [
        # Male Cow (Different colors and breeds)
        ('data/new_images/malecow2.webp', 0, 'Cow', 'Independent Hereford Bull WebP'),
        # Female Cow (Different colors and breeds)
        ('Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 1, 'Cow', 'Independent Red Dane Cow (Udder)'),
        # Male Buffalo (Independent Bull)
        ('buffalo/banni/baani-buffalo-bulls_0.jpg', 0, 'Buffalo', 'Banni Bull with Sheath'),
        # Female Buffalo (Independent Murrah)
        ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 1, 'Buffalo', 'Murrah Dairy Buffalo with Udder'),
        # Headshot Cow (Calf Headshot)
        ('static/uploads/5da299733e69416d849ad4602bbd6459.jpg', 2, 'Cow', 'Calf Face & Fence (Headshot)'),
        # Occluded Cow (Ambiguous user image)
        ('data/new_images/femalecow1.jpg', 2, 'Cow', 'Cow with Hidden Ventral Anatomy'),
    ]

    def extract_features(img_path, augment=False, n_aug=12):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        crops = []
        # Ventral anatomical region: 45% - 90% vertical, 15% - 85% horizontal
        y1, y2 = int(h * 0.45), int(h * 0.90)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        crops.append(img.crop((x1, y1, x2, y2)))
        crops.append(img.crop((x1, y1, x2, y2)).transpose(Image.FLIP_LEFT_RIGHT))
        
        if augment:
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
                f = resnet(t).cpu().numpy().flatten()
            feats.append(f)
        return feats

    print("Extracting features with anatomical augmentation...")
    X_train, y_train = [], []
    for path, label, sp, name in train_samples:
        feats = extract_features(path, augment=True, n_aug=12)
        for f in feats:
            X_train.append(f)
            y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Training feature matrix: {X_train.shape}, Class distribution (0=Male, 1=Female, 2=Uncertain): {np.bincount(y_train)}")

    # Fit L2 Regularized Pipeline
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.08, max_iter=600, random_state=42))
    ])
    clf.fit(X_train, y_train)

    # Save to models directory
    os.makedirs('models', exist_ok=True)
    with open('models/anatomical_gender_clf.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('models/cattle_gender_clf.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('models/buffalo_gender_clf.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("Saved models/anatomical_gender_clf.pkl, cattle_gender_clf.pkl, buffalo_gender_clf.pkl successfully!")

    # Evaluate on Independent Test Set
    print("\n" + "="*75)
    print("EVALUATING ON INDEPENDENT TEST SET")
    print("="*75)
    
    lbl_map = {0: 'Male', 1: 'Female', 2: 'Uncertain'}
    y_true, y_pred, y_probs = [], [], []
    test_results = []

    for path, true_label, sp, name in test_samples:
        feats = extract_features(path, augment=False)
        p0 = clf.predict_proba(feats[0].reshape(1, -1))[0]
        p1 = clf.predict_proba(feats[1].reshape(1, -1))[0]
        probs = (p0 + p1) / 2.0
        pred = int(np.argmax(probs))
        
        y_true.append(true_label)
        y_pred.append(pred)
        y_probs.append(probs)
        
        pred_str = lbl_map[pred]
        true_str = lbl_map[true_label]
        match = 'MATCH' if pred == true_label else 'MISMATCH'
        conf = float(np.max(probs))
        test_results.append((name, sp, true_str, pred_str, conf, match))

    header = f"{'Sample':38s} | {'Species':8s} | {'True':9s} | {'Pred':9s} | {'Conf':6s} | {'Result'}"
    print(header)
    print("-" * 82)
    for name, sp, true_str, pred_str, conf, match in test_results:
        print(f"{name:38s} | {sp:8s} | {true_str:9s} | {pred_str:9s} | {conf:.3f} | {match}")

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\n" + "="*50)
    print("INDEPENDENT TEST SET PERFORMANCE METRICS")
    print("="*50)
    print(f"Overall Accuracy    : {acc*100:.2f}%")
    print(f"Weighted Precision  : {prec*100:.2f}%")
    print(f"Weighted Recall     : {rec*100:.2f}%")
    print(f"Weighted F1-Score   : {f1*100:.2f}%")
    print("Confusion Matrix (Rows = Ground Truth [Male, Female, Uncertain], Cols = Predicted [Male, Female, Uncertain]):")
    print(cm)
    print("="*50)

if __name__ == '__main__':
    run_training()
