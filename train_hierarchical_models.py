import os
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error
import xgboost as xgb

# Set reproducible seeds
np.random.seed(42)
torch.manual_seed(42)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
CACHE_FILE = os.path.join(MODELS_DIR, "features_cache.npz")

# -------------------------------------------------------------
# 1. Dataset Discovery (Cows, Buffaloes, Dummy/Unknown)
# -------------------------------------------------------------
CATTLE_DIR = os.path.join(".", "Cattle Breeds")
BUFFALO_DIR = os.path.join(".", "buffalo")
DUMMY_DIR = os.path.join(".", "dummy_unidentified")

print("=== Scanning Livestock and Negative/Unknown Image Datasets ===")

cattle_paths, cattle_labels = [], []
buffalo_paths, buffalo_labels = [], []
species_paths, species_labels = [], []

# Scan Cow Breeds
cattle_breeds = sorted([d for d in os.listdir(CATTLE_DIR) if os.path.isdir(os.path.join(CATTLE_DIR, d))])
for breed in cattle_breeds:
    folder = os.path.join(CATTLE_DIR, breed)
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            fpath = os.path.join(folder, fname)
            cattle_paths.append(fpath)
            cattle_labels.append(breed)
            species_paths.append(fpath)
            species_labels.append("Cow")

# Scan Buffalo Breeds
buffalo_breeds = sorted([d for d in os.listdir(BUFFALO_DIR) if os.path.isdir(os.path.join(BUFFALO_DIR, d))])
for breed in buffalo_breeds:
    folder = os.path.join(BUFFALO_DIR, breed)
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            fpath = os.path.join(folder, fname)
            buffalo_paths.append(fpath)
            buffalo_labels.append(breed)
            species_paths.append(fpath)
            species_labels.append("Buffalo")

# Scan Dummy/Unknown Images (~4,000 negative samples)
dummy_paths = []
if os.path.exists(DUMMY_DIR):
    for fname in os.listdir(DUMMY_DIR):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            fpath = os.path.join(DUMMY_DIR, fname)
            dummy_paths.append(fpath)
            species_paths.append(fpath)
            species_labels.append("Unknown")

print(f"Found {len(cattle_paths)} Cow images across {len(cattle_breeds)} breeds.")
print(f"Found {len(buffalo_paths)} Buffalo images across {len(buffalo_breeds)} breeds.")
print(f"Found {len(dummy_paths)} Dummy / Negative images.")
print(f"Total Species Training Pool: {len(species_paths)} images.")

# Label Mappings
cattle_breed_map = {b: i for i, b in enumerate(cattle_breeds)}
rev_cattle_breed_map = {i: b for i, b in enumerate(cattle_breeds)}

buffalo_breed_map = {b: i for i, b in enumerate(buffalo_breeds)}
rev_buffalo_breed_map = {i: b for i, b in enumerate(buffalo_breeds)}

species_map = {"Cow": 0, "Buffalo": 1, "Unknown": 2}
rev_species_map = {0: "Cow", 1: "Buffalo", 2: "Unknown"}

mappings_payload = {
    'species_map': species_map,
    'rev_species_map': rev_species_map,
    'cattle_breeds': cattle_breeds,
    'cattle_breed_map': cattle_breed_map,
    'rev_cattle_breed_map': rev_cattle_breed_map,
    'buffalo_breeds': buffalo_breeds,
    'buffalo_breed_map': buffalo_breed_map,
    'rev_buffalo_breed_map': rev_buffalo_breed_map,
    'species_confidence_threshold': 0.70
}

with open(os.path.join(MODELS_DIR, 'hierarchical_mappings.json'), 'w', encoding='utf-8') as f:
    json.dump(mappings_payload, f, indent=2)

# -------------------------------------------------------------
# 2. PyTorch Deep Feature Extraction or Cache Load
# -------------------------------------------------------------
if os.path.exists(CACHE_FILE):
    print(f"\n[Cache Found] Loading pre-extracted deep features from {CACHE_FILE}...")
    cached = np.load(CACHE_FILE)
    X_species = cached['X_species']
    y_species = cached['y_species']
    X_cattle = cached['X_cattle']
    y_cattle = cached['y_cattle']
    X_buffalo = cached['X_buffalo']
    y_buffalo = cached['y_buffalo']
    print(f"Loaded Species Features: {X_species.shape}, Cow Features: {X_cattle.shape}, Buffalo Features: {X_buffalo.shape}")
else:
    class FastImageDataset(Dataset):
        def __init__(self, paths, labels, label_map, transform=None):
            self.paths = paths
            self.labels = [label_map[l] for l in labels]
            self.transform = transform or transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            try:
                image = Image.open(path).convert('RGB')
            except Exception:
                image = Image.new('RGB', (224, 224), (128, 128, 128))
            return self.transform(image), self.labels[idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for feature extraction: {device}")

    # Dual Pretrained Feature Extractors (ResNet18 + MobileNetV2)
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()

    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    mobilenet_feat = nn.Sequential(mobilenet.features, nn.AdaptiveAvgPool2d((1, 1))).to(device).eval()

    def extract_features(paths, labels, label_map, batch_size=64):
        ds = FastImageDataset(paths, labels, label_map)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        feats_list = []
        targets_list = []
        
        with torch.no_grad():
            for i, (imgs, tgs) in enumerate(loader):
                imgs = imgs.to(device)
                f_res = resnet(imgs).cpu().numpy()
                f_mob = torch.flatten(mobilenet_feat(imgs), 1).cpu().numpy()
                combined = np.hstack([f_res, f_mob])
                feats_list.append(combined)
                targets_list.extend(tgs.numpy())
                if (i + 1) % 20 == 0 or (i + 1) == len(loader):
                    print(f"  Processed {min((i+1)*batch_size, len(paths))}/{len(paths)} images...", flush=True)
                    
        return np.vstack(feats_list), np.array(targets_list)

    print("\n--- Extracting Stage 1 Species Features (Cow, Buffalo, Unknown) ---", flush=True)
    X_species, y_species = extract_features(species_paths, species_labels, species_map)

    print("\n--- Extracting Stage 2A Cow Breed Features ---", flush=True)
    X_cattle, y_cattle = extract_features(cattle_paths, cattle_labels, cattle_breed_map)

    print("\n--- Extracting Stage 2B Buffalo Breed Features ---", flush=True)
    X_buffalo, y_buffalo = extract_features(buffalo_paths, buffalo_labels, buffalo_breed_map)

    # Save to Cache
    np.savez_compressed(
        CACHE_FILE,
        X_species=X_species, y_species=y_species,
        X_cattle=X_cattle, y_cattle=y_cattle,
        X_buffalo=X_buffalo, y_buffalo=y_buffalo
    )
    print(f"Feature cache saved to {CACHE_FILE}")

# -------------------------------------------------------------
# 3. Model Training & Honest Evaluation (Train/Test Splits)
# -------------------------------------------------------------

# Stage 1: Species Classifier (Cow vs Buffalo vs Unknown)
print("\n--- Training Stage 1: Species Classifier ---", flush=True)
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_species, y_species, test_size=0.20, random_state=42, stratify=y_species)

species_clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, max_iter=1000, random_state=42)
)
species_clf.fit(Xs_tr, ys_tr)

ys_pred = species_clf.predict(Xs_te)
s_acc = accuracy_score(ys_te, ys_pred)
s_prec = precision_score(ys_te, ys_pred, average='weighted', zero_division=0)
s_rec = recall_score(ys_te, ys_pred, average='weighted', zero_division=0)
s_f1 = f1_score(ys_te, ys_pred, average='weighted', zero_division=0)

# Stage 2A: Cow Breed Classifier (5 Breeds)
print("\n--- Training Stage 2A: Cow Breed Classifier ---", flush=True)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_cattle, y_cattle, test_size=0.20, random_state=42, stratify=y_cattle)

cattle_clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.8, max_iter=1000, random_state=42)
)
cattle_clf.fit(Xc_tr, yc_tr)

yc_pred = cattle_clf.predict(Xc_te)
c_acc = accuracy_score(yc_te, yc_pred)
c_prec = precision_score(yc_te, yc_pred, average='macro', zero_division=0)
c_rec = recall_score(yc_te, yc_pred, average='macro', zero_division=0)
c_f1 = f1_score(yc_te, yc_pred, average='macro', zero_division=0)

# Stage 2B: Buffalo Breed Classifier (17 Breeds)
print("\n--- Training Stage 2B: Buffalo Breed Classifier ---", flush=True)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_buffalo, y_buffalo, test_size=0.20, random_state=42, stratify=y_buffalo)

clf_lr = LogisticRegression(C=0.5, max_iter=1000, class_weight='balanced', random_state=42)
clf_mlp = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=600, alpha=0.01, random_state=42, early_stopping=True)

buffalo_ensemble = VotingClassifier(
    estimators=[('lr', clf_lr), ('mlp', clf_mlp)],
    voting='soft'
)
buffalo_clf = make_pipeline(StandardScaler(), buffalo_ensemble)
buffalo_clf.fit(Xb_tr, yb_tr)

yb_pred = buffalo_clf.predict(Xb_te)
b_acc = accuracy_score(yb_te, yb_pred)
b_prec = precision_score(yb_te, yb_pred, average='macro', zero_division=0)
b_rec = recall_score(yb_te, yb_pred, average='macro', zero_division=0)
b_f1 = f1_score(yb_te, yb_pred, average='macro', zero_division=0)

# Save Classifiers
with open(os.path.join(MODELS_DIR, 'species_clf.pkl'), 'wb') as f:
    pickle.dump(species_clf, f)

with open(os.path.join(MODELS_DIR, 'cattle_breed_clf.pkl'), 'wb') as f:
    pickle.dump(cattle_clf, f)

with open(os.path.join(MODELS_DIR, 'buffalo_breed_clf.pkl'), 'wb') as f:
    pickle.dump(buffalo_clf, f)

# -------------------------------------------------------------
# 4. Multi-Variable Price Prediction Model (Karnataka Regressor)
# -------------------------------------------------------------
print("\n--- Training Multi-Variable Price Prediction Model ---", flush=True)

def generate_karnataka_price_dataset(num_samples=10000):
    data = []
    base_breed_value = {
        'Ayrshire cattle': 65000, 'Brown Swiss cattle': 70000, 'Holstein Friesian cattle': 78000,
        'Jersey cattle': 68000, 'Red Dane cattle': 72000, 'murrah': 95000, 'Jaffarabadi': 90000,
        'nili-ravi': 88000, 'mehsana': 85000, 'banni': 82000, 'pandharpuri': 78000, 'surti': 75000,
        'bhadwari': 72000, 'nagpuri': 70000, 'marathwada': 68000, 'Chhattisgarhi': 65000,
        'bargur': 62000, 'gojri': 60000, 'chilika': 58000, 'kalahandi': 55000, 'luit': 54000, 'toda': 52000
    }

    districts = [
        "Bagalkote", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban", "Bidar", 
        "Chamarajanagara", "Chikkaballapura", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada", 
        "Davanagere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi", "Kodagu", "Kolar", 
        "Koppala", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumakuru", 
        "Udupi", "Uttara Kannada", "Vijayanagara", "Vijayapura", "Yadgir"
    ]

    district_multipliers = {
        "Bengaluru Urban": 1.25, "Bengaluru Rural": 1.20, "Mysuru": 1.15, "Mandya": 1.12, 
        "Belagavi": 1.10, "Dakshina Kannada": 1.12, "Udupi": 1.10, "Hassan": 1.06, "Kolar": 1.08,
        "Shivamogga": 1.05, "Tumakuru": 1.05, "Dharwad": 1.05
    }

    all_breeds = list(base_breed_value.keys())

    for _ in range(num_samples):
        breed = np.random.choice(all_breeds)
        species = "Cow" if breed in cattle_breeds else "Buffalo"
        gender = np.random.choice(["Female", "Male"], p=[0.85, 0.15])
        age = round(np.random.uniform(1.5, 11.0), 1)

        if gender == "Female":
            base_milk = 18.0 if species == "Cow" else 14.0
            if "Holstein" in breed: base_milk = 23.0
            if "Jersey" in breed: base_milk = 16.0
            if "murrah" in breed or "Jaffarabadi" in breed: base_milk = 17.5
            
            # Age effect on lactation
            age_factor = 1.0 if (3.0 <= age <= 6.5) else (0.85 if age < 3.0 else max(0.45, 1.0 - (age - 6.5) * 0.11))
            milk_production = round(max(2.0, np.random.normal(base_milk * age_factor, 2.0)), 1)
        else:
            milk_production = 0.0

        district = np.random.choice(districts)
        dist_mult = district_multipliers.get(district, 1.0)
        base = base_breed_value[breed]

        if gender == "Female":
            price = base + (milk_production * 3600)
            if age < 3.0: price *= 0.92
            elif age > 6.5: price *= max(0.50, 1.0 - (age - 6.5) * 0.11)
        else:
            price = base * 0.60
            if age > 6.0: price *= 0.70

        price *= dist_mult
        price += np.random.normal(0, price * 0.02)
        price = round(max(15000, price), -2)

        data.append({
            'species': species,
            'breed': breed,
            'gender': gender,
            'age_years': age,
            'milk_production': milk_production,
            'district': district,
            'price': price
        })

    return pd.DataFrame(data)

df_price = generate_karnataka_price_dataset(10000)
features_df = pd.get_dummies(df_price[['species', 'breed', 'gender', 'age_years', 'milk_production', 'district']], drop_first=False)
y_price = df_price['price']

Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(features_df, y_price, test_size=0.20, random_state=42)

price_model = xgb.XGBRegressor(
    n_estimators=450,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42
)
price_model.fit(Xp_tr, yp_tr)

yp_pred = price_model.predict(Xp_te)
r2 = r2_score(yp_te, yp_pred)
mae = mean_absolute_error(yp_te, yp_pred)
within_10pct = np.mean(np.abs((yp_te - yp_pred) / yp_te) < 0.10)

# Save Price Model & Feature Schema
with open(os.path.join(MODELS_DIR, 'price_model.pkl'), 'wb') as f:
    pickle.dump(price_model, f)

with open(os.path.join(MODELS_DIR, 'price_features.json'), 'w', encoding='utf-8') as f:
    json.dump({'feature_names': list(features_df.columns)}, f, indent=2)

# -------------------------------------------------------------
# 5. Metrics Report Output
# -------------------------------------------------------------
metrics_report = {
    'species_classifier': {
        'accuracy': float(s_acc),
        'precision': float(s_prec),
        'recall': float(s_rec),
        'f1_score': float(s_f1)
    },
    'cattle_breed_classifier': {
        'accuracy': float(c_acc),
        'precision': float(c_prec),
        'recall': float(c_rec),
        'f1_score': float(c_f1)
    },
    'buffalo_breed_classifier': {
        'accuracy': float(b_acc),
        'precision': float(b_prec),
        'recall': float(b_rec),
        'f1_score': float(b_f1)
    },
    'price_prediction_model': {
        'r2_score': float(r2),
        'mae': float(mae),
        'accuracy_within_10pct': float(within_10pct)
    }
}

with open(os.path.join(MODELS_DIR, 'metrics_report.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics_report, f, indent=2)

# Terminal Output (Exact format from Requirement 22)
try:
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

print("\n====================================================")
print("AI MODEL EVALUATION")
print("====================================================")
print("Species Classifier")
print(f"Accuracy  : {s_acc*100:.2f}%")
print(f"Precision : {s_prec*100:.2f}%")
print(f"Recall    : {s_rec*100:.2f}%")
print(f"F1 Score  : {s_f1*100:.2f}%")
print("")
print("Cow Breed Classifier")
print(f"Accuracy  : {c_acc*100:.2f}%")
print(f"Precision : {c_prec*100:.2f}%")
print(f"Recall    : {c_rec*100:.2f}%")
print(f"F1 Score  : {c_f1*100:.2f}%")
print("")
print("Buffalo Breed Classifier")
print(f"Accuracy  : {b_acc*100:.2f}%")
print(f"Precision : {b_prec*100:.2f}%")
print(f"Recall    : {b_rec*100:.2f}%")
print(f"F1 Score  : {b_f1*100:.2f}%")
print("")
print("Price Prediction Model")
print(f"R2 Score            : {r2:.4f}")
print(f"MAE                 : ₹{int(mae)}")
print(f"Within 10% Accuracy : {within_10pct*100:.2f}%")
print("====================================================\n")
