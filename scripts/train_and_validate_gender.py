import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# Load models
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet18.fc = nn.Identity()
resnet18 = resnet18.to(device).eval()

resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet50.fc = nn.Identity()
resnet50 = resnet50.to(device).eval()

mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mobilenet_feat = nn.Sequential(mobilenet.features, nn.AdaptiveAvgPool2d((1, 1))).to(device).eval()

# Verified Buffalo Samples (Bubalus bubalis)
buffalo_females = [
    'buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg',
    'buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg',
    'buffalo/murrah/131_Murrah_Buffalo_for_Dairy_Farming_.jpg',
    'buffalo/surti/Milking_Murrah_Buffalo_Female_750kg_.jpg',
    'buffalo/murrah/Murrah_Buffalo_-_Female_Breed_High_.jpg',
    'buffalo/murrah/High_Milking_Pure_Breed_Murrah_Buffalo_.jpg',
    'buffalo/murrah/Pregnant_Murrah_Buffalo_Prime_Dairy_.jpg',
    'buffalo/murrah/Dairy_Murrah_Buffalo_for_Dairy_Farming_.jpg',
    'buffalo/murrah/Dairy_Murrah_Buffalo_at_70000_per_.jpg',
    'buffalo/murrah/Dairy_Murrah_Buffalo_for_Sale_at_an_.jpg',
    'buffalo/murrah/Murrah_Dairy_Knowledge_Portal.jpg',
    'buffalo/Jaffarabadi/Milking_Jafrabadi_Buffalo_Farm_.jpg',
    'buffalo/Jaffarabadi/Chandani_dairy_and_farming_-_My_.jpg',
    'buffalo/Jaffarabadi/Jafarabadi_buffalo_Chandani_dairy_.jpg',
    'buffalo/bhadwari/Mehsana_Dairy_Knowledge_Portal.jpg',
    'buffalo/bhadwari/Surti_Dairy_Knowledge_Portal.jpg',
    'buffalo/bargur/Black_Murrah_BuffaloVijay_Dairy_Farm.jpg',
    'buffalo/bargur/Murrah_BuffaloSuresh_Dairy_Farm.jpg',
    'buffalo/surti/Murrah_BuffaloShri_Ganesh_Dairy_Farm.jpg',
    'buffalo/surti/Murrah_BuffaloSuresh_Dairy_Farm.jpg',
    'buffalo/surti/Murrah_Dairy_Knowledge_Portal.jpg',
    'data/new_images/femalebuffolo.jpg',
    'data/new_images/femalebuffolo1.jpg',
    'data/new_images/femalebuffolo2.jpg',
]

buffalo_males = [
    'buffalo/banni/baani-buffalo-bulls_0.jpg',
    'buffalo/banni/1280px-Water_buffalo_bull-_near_Mehsana-_Gujarat-_India-_3.jpg',
    'buffalo/banni/livestock-breeding-services-frozen-semen-dose-of-banni-buf.jpg',
    'buffalo/banni/Banni-Male-Buffalo-Breed-360x240.png',
    'buffalo/bhadwari/Bhadavari_Archives_-_BAIF_Sire_Directory.jpg',
    'buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg',
    'buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg',
    'buffalo/Jaffarabadi/Black_Jafarabadi_Bull_Semen_at_Best_.jpg',
    'buffalo/Jaffarabadi/Dairy_Farm_with_Jafarabadi_Bull_.jpg',
    'buffalo/Jaffarabadi/Jafarabadi_buffalo_bull_for_sales_4_.jpg',
    'buffalo/Jaffarabadi/Jafarabadi_bull_from_Gujarat_-_YouTube.jpg',
    'buffalo/Jaffarabadi/Tulsidas_-_BAIF_Sire_Directory.jpg',
    'buffalo/murrah/Black_Murrah_Buffalo_Bull_Wholesaler_.jpg',
    'buffalo/murrah/Murrah_buffalo_bulls_to_be_used_for_.jpg',
    'buffalo/murrah/Murrah_BullMurrah_Buffalo_BullsBull_.jpg',
    'buffalo/murrah/Murrah_Bull_In_India_Murrah_Buffalo_Is_ (3).jpg',
    'buffalo/murrah/Murrah_Bull_in_India_Murrah_Buffalo_is_ (4).jpg',
    'buffalo/nagpuri/Thakya_-_BAIF_Sire_Directory.jpg',
    'buffalo/nili-ravi/Neeli_Ravi_buffalo_bull_.jpg',
    'buffalo/nili-ravi/Neeli_Ravi_buffalo_bull_ (2).jpg',
    'buffalo/nili-ravi/Nili_ravi_bull_Quality_full_niliravi_.jpg',
    'data/new_images/malebuffolo.jpg',
    'data/new_images/malebuffolo1.jpg',
    'data/new_images/malebuffolo2.jpg',
]

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    # 1. Full Body Crop (RGB)
    t_full_rgb = transform_rgb(img).unsqueeze(0).to(device)
    # 2. Ventral Anatomical Crop (Lower 55% of height, middle 80% width)
    crop_v = img.crop((int(w * 0.10), int(h * 0.40), int(w * 0.90), int(h * 0.95)))
    t_v_rgb = transform_rgb(crop_v).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # ResNet50 gives rich 2048-dim features
        f50_full = resnet50(t_full_rgb).cpu().numpy().flatten()
        f50_v = resnet50(t_v_rgb).cpu().numpy().flatten()
        # ResNet18
        f18_full = resnet18(t_full_rgb).cpu().numpy().flatten()
        f18_v = resnet18(t_v_rgb).cpu().numpy().flatten()
        
    return {
        'r18_v': f18_v,
        'r18_full': f18_full,
        'r18_dual': np.concatenate([f18_v, f18_full]),
        'r50_v': f50_v,
        'r50_full': f50_full,
        'r50_dual': np.concatenate([f50_v, f50_full]),
    }

def run_tests():
    print("=" * 75)
    print("EXTRACTING DEEP RESNET-18 AND RESNET-50 MULTI-SCALE FEATURES")
    print("=" * 75)
    
    all_paths = buffalo_females + buffalo_males
    labels = np.array([0] * len(buffalo_females) + [1] * len(buffalo_males))
    
    data_dict = {'r18_v': [], 'r18_full': [], 'r18_dual': [], 'r50_v': [], 'r50_full': [], 'r50_dual': []}
    for p in all_paths:
        feats = extract_features(p)
        for k in data_dict:
            data_dict[k].append(feats[k])
            
    for k in data_dict:
        data_dict[k] = np.array(data_dict[k])
        
    print(f"Extracted features for {len(all_paths)} buffalo samples.")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models_to_test = [
        ("Logistic Regression (C=0.1)", lambda: LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ("Logistic Regression (C=0.01)", lambda: LogisticRegression(C=0.01, max_iter=1000, random_state=42)),
        ("Linear SVM (C=0.1)", lambda: SVC(kernel='linear', C=0.1, probability=True, random_state=42)),
        ("RBF SVM (C=1.0, gamma=scale)", lambda: SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
        ("MLP (64 hidden, ReLU, alpha=0.1)", lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, alpha=0.1, random_state=42)),
        ("Random Forest (100 trees)", lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ]
    
    print("\n" + "=" * 75)
    print("5-FOLD STRATIFIED CROSS VALIDATION RESULTS ACROSS FEATURE TYPES & CLASSIFIERS")
    print("=" * 75)
    
    for feat_name, X in data_dict.items():
        print(f"\n>>> FEATURE: {feat_name} (dim: {X.shape[1]}) <<<")
        for model_name, model_fn in models_to_test:
            y_true_all, y_pred_all = [], []
            for train_idx, test_idx in skf.split(X, labels):
                clf = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model_fn())
                ])
                clf.fit(X[train_idx], labels[train_idx])
                preds = clf.predict(X[test_idx])
                y_true_all.extend(labels[test_idx])
                y_pred_all.extend(preds)
            acc = accuracy_score(y_true_all, y_pred_all)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='weighted')
            print(f"  {model_name:38s} | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}%")

if __name__ == '__main__':
    run_tests()
