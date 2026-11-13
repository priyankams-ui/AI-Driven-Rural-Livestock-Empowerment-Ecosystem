import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline import pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet = resnet.to(device).eval()

# Verified Buffalo Samples
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

def extract_features_method_A(img_path):
    """Old Method: Grayscale ventral crop only"""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    crop = img.crop((int(w * 0.12), int(h * 0.42), int(w * 0.88), int(h * 0.94)))
    t = transform_gray(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        return resnet(t).cpu().numpy().flatten()

def extract_features_method_B(img_path):
    """Method B: RGB ventral crop"""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    crop = img.crop((int(w * 0.12), int(h * 0.42), int(w * 0.88), int(h * 0.94)))
    t = transform_rgb(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        return resnet(t).cpu().numpy().flatten()

def extract_features_method_C(img_path):
    """Method C: Multi-scale (Anatomical Ventral Crop + Full Animal Body)"""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    # 1. Full Body Crop
    t_full = transform_rgb(img).unsqueeze(0).to(device)
    
    # 2. Ventral Anatomical Crop
    crop_ventral = img.crop((int(w * 0.10), int(h * 0.40), int(w * 0.90), int(h * 0.94)))
    t_ventral = transform_rgb(crop_ventral).unsqueeze(0).to(device)
    
    with torch.no_grad():
        f_full = resnet(t_full).cpu().numpy().flatten()
        f_ventral = resnet(t_ventral).cpu().numpy().flatten()
    return np.concatenate([f_ventral, f_full])

mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mobilenet_feat = nn.Sequential(mobilenet.features, nn.AdaptiveAvgPool2d((1, 1))).to(device).eval()

def extract_features_method_D(img_path):
    """Method D: Full Body + Ventral using ResNet18 + MobileNetV2 (1792-d each)"""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    t_full = transform_rgb(img).unsqueeze(0).to(device)
    crop_v = img.crop((int(w * 0.10), int(h * 0.40), int(w * 0.90), int(h * 0.94)))
    t_v = transform_rgb(crop_v).unsqueeze(0).to(device)
    
    with torch.no_grad():
        r_f = resnet(t_full).cpu().numpy().flatten()
        m_f = torch.flatten(mobilenet_feat(t_full), 1).cpu().numpy().flatten()
        full_feat = np.concatenate([r_f, m_f])
        
        r_v = resnet(t_v).cpu().numpy().flatten()
        m_v = torch.flatten(mobilenet_feat(t_v), 1).cpu().numpy().flatten()
        v_feat = np.concatenate([r_v, m_v])
        
    return np.concatenate([full_feat, v_feat])

def extract_features_method_E(img_path):
    """Method E: Full Body ResNet18 + MobileNetV2 (1792-d)"""
    img = Image.open(img_path).convert('RGB')
    t_full = transform_rgb(img).unsqueeze(0).to(device)
    with torch.no_grad():
        r_f = resnet(t_full).cpu().numpy().flatten()
        m_f = torch.flatten(mobilenet_feat(t_full), 1).cpu().numpy().flatten()
    return np.concatenate([r_f, m_f])

def run_experiment():
    print("=" * 75)
    print("EMPIRICAL COMPARISON OF FEATURE REPRESENTATIONS ON VERIFIED BUFFALOES")
    print(f"Total dataset: {len(buffalo_females)} Females, {len(buffalo_males)} Males (48 total)")
    print("=" * 75)
    
    labels = np.array([0] * len(buffalo_females) + [1] * len(buffalo_males))
    all_paths = buffalo_females + buffalo_males
    
    methods = [
        ("Method D: Dual-Scale Full + Ventral (ResNet18 + MobileNetV2 3584-d)", extract_features_method_D),
        ("Method E: Full Body (ResNet18 + MobileNetV2 1792-d)", extract_features_method_E),
    ]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, extractor in methods:
        print(f"\n--- Evaluating {name} ---")
        X = np.array([extractor(p) for p in all_paths])
        
        for c_val in [0.01, 0.05, 0.1, 0.5]:
            fold_accs = []
            all_y_true = []
            all_y_pred = []
            
            for train_idx, test_idx in skf.split(X, labels):
                X_tr, y_tr = X[train_idx], labels[train_idx]
                X_te, y_te = X[test_idx], labels[test_idx]
                
                clf = Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(C=c_val, max_iter=1000, random_state=42))
                ])
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_te)
                fold_accs.append(accuracy_score(y_te, preds))
                all_y_true.extend(y_te)
                all_y_pred.extend(preds)
                
            acc = accuracy_score(all_y_true, all_y_pred)
            cm = confusion_matrix(all_y_true, all_y_pred)
            print(f"  C={c_val:<4} -> CV Acc: {acc*100:.2f}%, CM:\n{cm}")

if __name__ == '__main__':
    run_experiment()

