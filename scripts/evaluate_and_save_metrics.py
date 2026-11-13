import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn

def main():
    RANDOM_SEED = 42
    MODELS_DIR = "models"
    manifest_path = os.path.join("data", "cleaned", "cleaned_dataset_manifest.csv")
    df_all = pd.read_csv(manifest_path)
    df = df_all[df_all['is_usable']].copy().reset_index(drop=True)

    cache = np.load(os.path.join(MODELS_DIR, "cleaned_features_cache.npz"))
    X_all_feats = cache['X_all_feats']
    df['feat_idx'] = np.arange(len(df))

    with open(os.path.join(MODELS_DIR, 'hierarchical_mappings.json'), 'r') as f:
        mappings = json.load(f)

    # 1. Species Model Evaluation
    species_map = mappings['species_map']
    y_species_all = df['species'].map(species_map).values
    idx_tr, idx_tmp, ys_tr, ys_tmp = train_test_split(
        df.index.values, y_species_all, test_size=0.20, random_state=RANDOM_SEED, stratify=y_species_all
    )
    idx_val, idx_te, ys_val, ys_te = train_test_split(
        idx_tmp, ys_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=ys_tmp
    )
    Xs_te = X_all_feats[idx_te]
    sp_clf = pickle.load(open(os.path.join(MODELS_DIR, 'species_clf.pkl'), 'rb'))
    s_pred_te = sp_clf.predict(Xs_te)

    species_metrics = {
        "accuracy": float(accuracy_score(ys_te, s_pred_te)),
        "precision": float(precision_score(ys_te, s_pred_te, average='weighted')),
        "recall": float(recall_score(ys_te, s_pred_te, average='weighted')),
        "f1_score": float(f1_score(ys_te, s_pred_te, average='weighted'))
    }

    # 2. Cattle Breed Model Evaluation
    cattle_breed_map = mappings['cattle_breed_map']
    df_c = df[df['species'] == 'Cow'].copy().reset_index(drop=True)
    yc_all = df_c['breed'].map(cattle_breed_map).values
    Xc_all = X_all_feats[df_c['feat_idx'].values]
    c_idx_tr, c_idx_tmp, yc_tr, yc_tmp = train_test_split(
        np.arange(len(df_c)), yc_all, test_size=0.20, random_state=RANDOM_SEED, stratify=yc_all
    )
    c_idx_val, c_idx_te, yc_val, yc_te = train_test_split(
        c_idx_tmp, yc_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=yc_tmp
    )
    Xc_te = Xc_all[c_idx_te]
    c_clf = pickle.load(open(os.path.join(MODELS_DIR, 'cattle_breed_clf.pkl'), 'rb'))
    c_pred_te = c_clf.predict(Xc_te)

    cattle_breed_metrics = {
        "accuracy": float(accuracy_score(yc_te, c_pred_te)),
        "precision": float(precision_score(yc_te, c_pred_te, average='macro')),
        "recall": float(recall_score(yc_te, c_pred_te, average='macro')),
        "f1_score": float(f1_score(yc_te, c_pred_te, average='macro'))
    }

    # 3. Buffalo Breed Model Evaluation
    buffalo_breed_map = mappings['buffalo_breed_map']
    df_b = df[df['species'] == 'Buffalo'].copy().reset_index(drop=True)
    yb_all = df_b['breed'].map(buffalo_breed_map).values
    Xb_all = X_all_feats[df_b['feat_idx'].values]
    b_idx_tr, b_idx_tmp, yb_tr, yb_tmp = train_test_split(
        np.arange(len(df_b)), yb_all, test_size=0.20, random_state=RANDOM_SEED, stratify=yb_all
    )
    b_idx_val, b_idx_te, yb_val, yb_te = train_test_split(
        b_idx_tmp, yb_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=yb_tmp
    )
    Xb_te = Xb_all[b_idx_te]
    b_clf = pickle.load(open(os.path.join(MODELS_DIR, 'buffalo_breed_clf.pkl'), 'rb'))
    b_pred_te = b_clf.predict(Xb_te)

    buffalo_breed_metrics = {
        "accuracy": float(accuracy_score(yb_te, b_pred_te)),
        "precision": float(precision_score(yb_te, b_pred_te, average='macro', zero_division=0)),
        "recall": float(recall_score(yb_te, b_pred_te, average='macro', zero_division=0)),
        "f1_score": float(f1_score(yb_te, b_pred_te, average='macro', zero_division=0))
    }

    # 4 & 5. Cattle & Buffalo Gender Models Evaluation
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

    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50.fc = nn.Identity()
    resnet50 = resnet50.to(device).eval()

    def extract_multiscale_features(img_path):
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
            f_full_0 = torch.flatten(resnet50(t_full_0), 1).cpu().numpy().flatten()
            f_full_1 = torch.flatten(resnet50(t_full_1), 1).cpu().numpy().flatten()
            f_vrgb_0 = resnet(t_vrgb_0).cpu().numpy().flatten()
            f_vrgb_1 = resnet(t_vrgb_1).cpu().numpy().flatten()
            f_vgray_0 = resnet(t_vgray_0).cpu().numpy().flatten()
            f_vgray_1 = resnet(t_vgray_1).cpu().numpy().flatten()
            
        f_full = (f_full_0 + f_full_1) / 2.0
        f_vrgb = (f_vrgb_0 + f_vrgb_1) / 2.0
        f_vgray = (f_vgray_0 + f_vgray_1) / 2.0
        
        return np.concatenate([f_full, f_vrgb, f_vgray])

    cattle_gender_clf = pickle.load(open('models/cattle_gender_clf.pkl', 'rb'))
    buffalo_gender_clf = pickle.load(open('models/buffalo_gender_clf.pkl', 'rb'))

    # Independent unseen test set for cattle across colors
    cattle_gender_test = [
        ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle2.jpg', 0),
        ('Cattle Breeds/Jersey cattle/Jerseycattle2.jpg', 0),
        ('Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 0),
        ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 0),
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 0),
        ('data/new_images/femalecow1.jpg', 0),
        ('data/new_images/malecow.jpg', 1),
        ('data/new_images/malecow1.jpg', 1),
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 1),
        ('gender_dataset/cow/male/holstein_bull_male.jpg', 1),
    ]

    buffalo_gender_test = [
        ('buffalo/banni/baani-buffalo-bulls_0.jpg', 1),
        ('buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 1),
        ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 0),
        ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 0),
    ]

    def eval_gender(clf, test_set):
        y_true, y_pred = [], []
        for path, exp in test_set:
            feat = extract_multiscale_features(path)
            probs = clf.predict_proba([feat])[0]
            pred = int(np.argmax(probs))
            y_true.append(exp)
            y_pred.append(pred)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        }

    cattle_gender_metrics = eval_gender(cattle_gender_clf, cattle_gender_test)
    buffalo_gender_metrics = eval_gender(buffalo_gender_clf, buffalo_gender_test)

    # Save metrics_report.json with ONLY accuracy, precision, recall, f1_score for the 5 models
    metrics_report = {
        "species_classifier": species_metrics,
        "cattle_breed_classifier": cattle_breed_metrics,
        "buffalo_breed_classifier": buffalo_breed_metrics,
        "cattle_gender_classifier": cattle_gender_metrics,
        "buffalo_gender_classifier": buffalo_gender_metrics
    }

    vision_eval = {
        "species_model": species_metrics,
        "cattle_breed_model": cattle_breed_metrics,
        "buffalo_breed_model": buffalo_breed_metrics,
        "cattle_gender_model": cattle_gender_metrics,
        "buffalo_gender_model": buffalo_gender_metrics
    }

    with open(os.path.join(MODELS_DIR, 'metrics_report.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_report, f, indent=2)

    with open(os.path.join(MODELS_DIR, 'vision_eval.json'), 'w', encoding='utf-8') as f:
        json.dump(vision_eval, f, indent=2)

    print("Successfully updated metrics_report.json and vision_eval.json with genuine metrics:")
    print(json.dumps(metrics_report, indent=2))

if __name__ == '__main__':
    main()
