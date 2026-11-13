import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet18.fc = nn.Identity()
resnet18 = resnet18.to(device).eval()

resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet50.fc = nn.Identity()
resnet50 = resnet50.to(device).eval()

mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mobilenet_feat = nn.Sequential(mobilenet.features, nn.AdaptiveAvgPool2d((1, 1))).to(device).eval()

# 19 verified dairy females and 17 verified breeding bulls
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

def extract_features(p):
    img = Image.open(p).convert('RGB')
    w, h = img.size
    
    # 1. Full Body RGB
    t_full_rgb = transform_rgb(img).unsqueeze(0).to(device)
    
    # 2. Ventral RGB & Gray
    y1, y2 = int(h * 0.38), int(h * 0.95)
    x1, x2 = int(w * 0.10), int(w * 0.90)
    v_crop = img.crop((x1, y1, x2, y2))
    t_v_rgb = transform_rgb(v_crop).unsqueeze(0).to(device)
    t_v_gray = transform_gray(v_crop).unsqueeze(0).to(device)
    
    # 3. Posterior / Rear Underbelly (focusing on udder/scrotum area)
    # The hindquarters/udder area is in the rear half of the ventral region
    # We take left-rear and right-rear depending on orientation
    y_rear1, y_rear2 = int(h * 0.45), int(h * 0.95)
    x_rear_l1, x_rear_l2 = int(w * 0.08), int(w * 0.55)
    x_rear_r1, x_rear_r2 = int(w * 0.45), int(w * 0.92)
    crop_rear_l = img.crop((x_rear_l1, y_rear1, x_rear_l2, y_rear2))
    crop_rear_r = img.crop((x_rear_r1, y_rear1, x_rear_r2, y_rear2))
    t_rear_l = transform_rgb(crop_rear_l).unsqueeze(0).to(device)
    t_rear_r = transform_rgb(crop_rear_r).unsqueeze(0).to(device)
    
    with torch.no_grad():
        f50_full = resnet50(t_full_rgb).cpu().numpy().flatten()
        f18_full = resnet18(t_full_rgb).cpu().numpy().flatten()
        f18_v_rgb = resnet18(t_v_rgb).cpu().numpy().flatten()
        f18_v_gray = resnet18(t_v_gray).cpu().numpy().flatten()
        f_mob = torch.flatten(mobilenet_feat(t_full_rgb), 1).cpu().numpy().flatten()
        
    return {
        'f50_full': f50_full,                                              # 2048
        'f18_full': f18_full,                                              # 512
        'f18_v_rgb': f18_v_rgb,                                            # 512
        'f18_v_gray': f18_v_gray,                                          # 512
        'f_mob': f_mob,                                                    # 1280
        'f18_dual': np.concatenate([f18_full, f18_v_rgb]),                 # 1024
        'f50_f18_dual': np.concatenate([f50_full, f18_v_rgb]),             # 2560
        'all_combined': np.concatenate([f50_full, f18_v_rgb, f18_v_gray]), # 3072
        'mob_f18_dual': np.concatenate([f_mob, f18_v_rgb]),                # 1792
    }

print("Extracting features for 36 buffaloes...")
all_paths = buffalo_females + buffalo_males
labels = np.array([0] * len(buffalo_females) + [1] * len(buffalo_males))
data = [extract_features(p) for p in all_paths]
print("Features extracted.")

if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    feature_keys = [
        ('ResNet-18 Dual (Full+Ventral RGB: 1024-d)', 'f18_dual'),
        ('ResNet-50 Full Body (2048-d)', 'f50_full'),
        ('ResNet-50 + ResNet-18 Ventral (2560-d)', 'f50_f18_dual'),
        ('MobileNet-V2 + ResNet-18 Ventral (1792-d)', 'mob_f18_dual'),
        ('ResNet-50 + ResNet-18 RGB+Gray (3072-d)', 'all_combined'),
    ]

    classifiers = [
    ('LogReg (C=0.01)', lambda: LogisticRegression(C=0.01, max_iter=1000, random_state=42)),
    ('LogReg (C=0.05)', lambda: LogisticRegression(C=0.05, max_iter=1000, random_state=42)),
    ('Linear SVM (C=0.02)', lambda: SVC(kernel='linear', C=0.02, probability=True, random_state=42)),
    ('Linear SVM (C=0.05)', lambda: SVC(kernel='linear', C=0.05, probability=True, random_state=42)),
    ('Random Forest (100)', lambda: RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)),
    ('Extra Trees (100)', lambda: ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=42)),
    ('Ensemble (LR+SVM+ET)', lambda: VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=0.05, max_iter=1000, random_state=42)),
            ('svm', SVC(kernel='linear', C=0.05, probability=True, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=42))
        ], voting='soft'
    )),
    ]

    print("=" * 80)
    print(f"{'Feature Set':35s} | {'Classifier':22s} | {'CV Acc':7s} | {'F1':7s} | {'Confusion Matrix'}")
    print("=" * 80)

    best_acc = 0.0
    best_combo = None

    for feat_name, feat_key in feature_keys:
        X = np.array([d[feat_key] for d in data])
        for clf_name, clf_fn in classifiers:
            y_true, y_pred = [], []
            for tr, te in skf.split(X, labels):
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', clf_fn())
                ])
                pipe.fit(X[tr], labels[tr])
                preds = pipe.predict(X[te])
                y_true.extend(labels[te])
                y_pred.extend(preds)
                
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            cm_str = f"[[{cm[0,0]},{cm[0,1]}],[{cm[1,0]},{cm[1,1]}]]"
            print(f"{feat_name:35s} | {clf_name:22s} | {acc*100:.2f}% | {f1*100:.2f}% | {cm_str}")
            if acc > best_acc:
                best_acc = acc
                best_combo = (feat_name, clf_name, acc, cm)

    print("=" * 80)
    print(f"BEST CONFIGURATION: {best_combo[0]} with {best_combo[1]} -> {best_combo[2]*100:.2f}%")
    print(f"Confusion Matrix:\n{best_combo[3]}")
