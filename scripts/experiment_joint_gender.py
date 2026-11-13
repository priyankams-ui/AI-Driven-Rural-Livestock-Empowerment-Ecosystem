import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
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

resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet50.fc = nn.Identity()
resnet50 = resnet50.to(device).eval()

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet18.fc = nn.Identity()
resnet18 = resnet18.to(device).eval()

# 1. Verified Cow Data (Bos taurus / Bos indicus)
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

# 2. Verified Buffalo Data (Bubalus bubalis)
buffalo_males = [
    'buffalo/Jaffarabadi/Black_Jafarabadi_Bull_Semen_at_Best_.jpg',
    'buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg',
    'buffalo/Jaffarabadi/Jafarabadi_buffalo_bull_for_sales_4_.jpg',
    'buffalo/Jaffarabadi/Jafarabadi_bull_from_Gujarat_-_YouTube.jpg',
    'buffalo/Jaffarabadi/Tulsidas_-_BAIF_Sire_Directory.jpg',
    'buffalo/banni/1280px-Water_buffalo_bull-_near_Mehsana-_Gujarat-_India-_3.jpg',
    'buffalo/bhadwari/Bhadavari_Archives_-_BAIF_Sire_Directory.jpg',
    'buffalo/murrah/Murrah_BullMurrah_Buffalo_BullsBull_.jpg',
    'buffalo/murrah/Murrah_Bull_In_India_Murrah_Buffalo_Is_ (3).jpg',
    'buffalo/murrah/Murrah_Bull_in_India_Murrah_Buffalo_is_ (4).jpg',
    'buffalo/murrah/Murrah_buffalo_bulls_to_be_used_for_.jpg',
    'buffalo/nagpuri/Thakya_-_BAIF_Sire_Directory.jpg',
    'buffalo/nili-ravi/HSB_Nili_Ravi_Bull-Parass_Mother_ (1).jpg',
    'buffalo/nili-ravi/HSB_Nili_Ravi_Bull-Parass_Mother_ (3).jpg',
    'buffalo/nili-ravi/Neeli_Ravi_buffalo_bull_ (2).jpg',
    'buffalo/nili-ravi/Neeli_Ravi_buffalo_bull_.jpg',
    'buffalo/nili-ravi/Nili_ravi_bull_Quality_full_niliravi_.jpg',
    'data/new_images/malebuffolo.jpg',
    'data/new_images/malebuffolo1.jpg',
    'data/new_images/malebuffolo2.jpg',
    'data/new_images/malebuffolo3.jpg',
    'buffalo/banni/Banni-Male-Buffalo-Breed-360x240.png',
    'buffalo/banni/livestock-breeding-services-frozen-semen-dose-of-banni-buf.jpg',
    'buffalo/Jaffarabadi/Dairy_Farm_with_Jafarabadi_Bull_.jpg',
]

buffalo_females = [
    'buffalo/Jaffarabadi/Animal_Nutrition_-_FoodFeedInfo_Dairy_.jpg',
    'buffalo/Jaffarabadi/Chandani_dairy_and_farming_-_My_.jpg',
    'buffalo/Jaffarabadi/Dhan_dairy_Fram_on_Reels_Facebook.jpg',
    'buffalo/Jaffarabadi/Ghanshyam_dairy_farm_in_Navi_Mumbai_.jpg',
    'buffalo/Jaffarabadi/Gujarat_Kanhaiya_Dairy_Farm_Jamnagar_.jpg',
    'buffalo/Jaffarabadi/Jafarabadi_buffalo_Chandani_dairy_.jpg',
    'buffalo/luit/Jaffarabadi_Dairy_Knowledge_Portal.jpg',
    'buffalo/Jaffarabadi/Milking_Jafrabadi_Buffalo_Farm_.jpg',
    'buffalo/Jaffarabadi/jafarabadi_buffalo_dairyfarm_cattle_.jpg',
    'buffalo/bargur/Black_Murrah_BuffaloVijay_Dairy_Farm.jpg',
    'buffalo/bargur/Murrah_BuffaloSuresh_Dairy_Farm.jpg',
    'buffalo/bhadwari/Mehsana_Dairy_Knowledge_Portal.jpg',
    'buffalo/bhadwari/Surti_Dairy_Knowledge_Portal.jpg',
    'buffalo/chilika/Dairy_Farming_Buffalo_Pics.jpg',
    'buffalo/kalahandi/Luit_Dairy_Knowledge_Portal.jpg',
    'buffalo/kalahandi/kalahandi_Buffalo_dairy_farming_.jpg',
    'buffalo/luit/Dairy_Farming_Buffalo_Pics (1).jpg',
    'buffalo/luit/Jaffarabadi_Dairy_Knowledge_Portal.jpg',
    'buffalo/marathwada/Dairy_Business_Ideas_marathwadi_buffalo_.jpg',
    'buffalo/murrah/353_Murrah_Buffalo_for_Dairy_Farming_.jpg',
    'buffalo/murrah/Dairy_Farm_Murrah_Buffalo_PriceDairy_.jpg',
    'buffalo/murrah/Dairy_Farming_Murrah_Buffalo_Pictures.jpg',
    'buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg',
    'data/new_images/femalebuffolo.jpg',
]

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    t_full = transform_rgb(img).unsqueeze(0).to(device)
    crop_v = img.crop((int(w * 0.10), int(h * 0.40), int(w * 0.90), int(h * 0.95)))
    t_v = transform_rgb(crop_v).unsqueeze(0).to(device)
    
    with torch.no_grad():
        f50_full = resnet50(t_full).cpu().numpy().flatten()
        f50_v = resnet50(t_v).cpu().numpy().flatten()
        f18_full = resnet18(t_full).cpu().numpy().flatten()
        f18_v = resnet18(t_v).cpu().numpy().flatten()
        
    return {
        'r50_full': f50_full,
        'r50_dual': np.concatenate([f50_full, f50_v]),
        'hybrid_all': np.concatenate([f50_full, f50_v, f18_full, f18_v])
    }

def run_experiment():
    print("=" * 75)
    print("JOINT & SPECIES-SPECIFIC EVALUATION ON VERIFIED COW & BUFFALO DATASET")
    print(f"  Cow Female:     {len(cow_females)}")
    print(f"  Cow Male:       {len(cow_males)}")
    print(f"  Buffalo Female: {len(buffalo_females)}")
    print(f"  Buffalo Male:   {len(buffalo_males)}")
    total = len(cow_females) + len(cow_males) + len(buffalo_females) + len(buffalo_males)
    print(f"  Total animals:  {total}")
    print("=" * 75)
    
    # Build dataset records: (path, gender [0=Female, 1=Male], species [0=Cow, 1=Buffalo])
    records = []
    for p in cow_females: records.append((p, 0, 0, 'Cow Female'))
    for p in cow_males: records.append((p, 1, 0, 'Cow Male'))
    for p in buffalo_females: records.append((p, 0, 1, 'Buffalo Female'))
    for p in buffalo_males: records.append((p, 1, 1, 'Buffalo Male'))
    
    feats_r50_full = []
    feats_r50_dual = []
    feats_hybrid = []
    y_gender = []
    y_species = []
    
    for p, g, s, name in records:
        f = extract_features(p)
        feats_r50_full.append(f['r50_full'])
        feats_r50_dual.append(f['r50_dual'])
        feats_hybrid.append(f['hybrid_all'])
        y_gender.append(g)
        y_species.append(s)
        
    X_r50_full = np.array(feats_r50_full)
    X_r50_dual = np.array(feats_r50_dual)
    X_hybrid = np.array(feats_hybrid)
    y_gender = np.array(y_gender)
    y_species = np.array(y_species)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for feat_name, X in [('ResNet-50 Full Body (2048)', X_r50_full), ('ResNet-50 Dual-Scale (4096)', X_r50_dual), ('Hybrid R50+R18 All (5120)', X_hybrid)]:
        print(f"\n" + "=" * 60)
        print(f"FEATURE REPRESENTATION: {feat_name}")
        print("=" * 60)
        
        # Test 1: Unified Gender Classifier (Evaluated across all 75 animals)
        rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        lr = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
        et = ExtraTreesClassifier(n_estimators=150, max_depth=6, random_state=42)
        
        for clf_name, clf_model in [('Random Forest', rf), ('Extra Trees', et), ('Logistic Regression (C=0.05)', lr)]:
            y_true_all, y_pred_all = [], []
            y_sp_all = []
            
            for tr, te in skf.split(X, y_gender):
                pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf_model)])
                pipe.fit(X[tr], y_gender[tr])
                preds = pipe.predict(X[te])
                y_true_all.extend(y_gender[te])
                y_pred_all.extend(preds)
                y_sp_all.extend(y_species[te])
                
            y_true_all = np.array(y_true_all)
            y_pred_all = np.array(y_pred_all)
            y_sp_all = np.array(y_sp_all)
            
            acc = accuracy_score(y_true_all, y_pred_all)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='weighted')
            cm = confusion_matrix(y_true_all, y_pred_all)
            
            # Sub-breakdowns: Cow vs Buffalo
            cow_mask = (y_sp_all == 0)
            buff_mask = (y_sp_all == 1)
            
            cow_acc = accuracy_score(y_true_all[cow_mask], y_pred_all[cow_mask])
            buff_acc = accuracy_score(y_true_all[buff_mask], y_pred_all[buff_mask])
            
            print(f"\n{clf_name}:")
            print(f"  Overall Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")
            print(f"  Cow Gender Accuracy:     {cow_acc*100:.2f}% ({np.sum(y_true_all[cow_mask] == y_pred_all[cow_mask])}/{np.sum(cow_mask)})")
            print(f"  Buffalo Gender Accuracy: {buff_acc*100:.2f}% ({np.sum(y_true_all[buff_mask] == y_pred_all[buff_mask])}/{np.sum(buff_mask)})")
            print(f"  Confusion Matrix (Row 0=Female, Row 1=Male):\n{cm}")

if __name__ == '__main__':
    run_experiment()
