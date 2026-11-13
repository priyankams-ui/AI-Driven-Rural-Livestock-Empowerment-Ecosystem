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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error
import xgboost as xgb

# ----------------------------------------------------------------------
# 0. Global Setup & Reproducibility
# ----------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
CACHE_FILE = os.path.join(MODELS_DIR, "features_cache.npz")

print("================================================================================")
print("   LIVESTOCK EMPOWERMENT SYSTEM - 80/10/10 ML TRAINING & EVALUATION PIPELINE    ")
print("================================================================================")
print(f"Protocol: 80% Training | 10% Validation | 10% Independent Testing (Seed: {RANDOM_SEED})\n")

# ----------------------------------------------------------------------
# 1. Dataset Discovery & Label Mappings
# ----------------------------------------------------------------------
CATTLE_DIR = os.path.join(".", "Cattle Breeds")
BUFFALO_DIR = os.path.join(".", "buffalo")
DUMMY_DIR = os.path.join(".", "dummy_unidentified")

cattle_paths, cattle_labels = [], []
buffalo_paths, buffalo_labels = [], []
species_paths, species_labels = [], []

# Scan Cow Breeds (5 Breeds)
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

# Scan Buffalo Breeds (17 Breeds)
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

print(f"[Dataset Summary]")
print(f"  * Cattle Images:  {len(cattle_paths):>5} across {len(cattle_breeds)} breeds")
print(f"  * Buffalo Images: {len(buffalo_paths):>5} across {len(buffalo_breeds)} breeds")
print(f"  * Dummy Images:   {len(dummy_paths):>5} negative / unidentified samples")
print(f"  * Total Images:   {len(species_paths):>5} total species evaluation pool\n")

# Mappings
cattle_breed_map = {b: i for i, b in enumerate(cattle_breeds)}
rev_cattle_breed_map = {i: b for i, b in enumerate(cattle_breeds)}

buffalo_breed_map = {b: i for i, b in enumerate(buffalo_breeds)}
rev_buffalo_breed_map = {i: b for i, b in enumerate(buffalo_breeds)}

species_map = {"Cow": 0, "Buffalo": 1, "Unknown": 2}
rev_species_map = {0: "Cow", 1: "Buffalo", 2: "Unknown"}

# Save hierarchical mappings
hierarchical_mappings = {
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
    json.dump(hierarchical_mappings, f, indent=2)

# Save legacy mappings for backward compatibility
all_unique_breeds = cattle_breeds + buffalo_breeds
with open(os.path.join(MODELS_DIR, 'mappings.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'species_map': {"Cow": 0, "Buffalo": 1},
        'rev_species_map': {0: "Cow", 1: "Buffalo"},
        'breed_map': {b: i for i, b in enumerate(all_unique_breeds)},
        'rev_breed_map': {i: b for i, b in enumerate(all_unique_breeds)},
        'unique_breeds': all_unique_breeds
    }, f, indent=2)

# ----------------------------------------------------------------------
# 2. Deep Feature Extraction & Cache Management
# ----------------------------------------------------------------------
if os.path.exists(CACHE_FILE):
    print(f"[Cache Found] Loading pre-computed deep features from {CACHE_FILE}...")
    cached = np.load(CACHE_FILE)
    X_species = cached['X_species']
    y_species = cached['y_species']
    X_cattle = cached['X_cattle']
    y_cattle = cached['y_cattle']
    X_buffalo = cached['X_buffalo']
    y_buffalo = cached['y_buffalo']
else:
    print("[Extracting Features] Initializing PyTorch Pretrained Feature Extractors...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class FastImageDataset(Dataset):
        def __init__(self, paths, labels, label_map):
            self.paths = paths
            self.labels = [label_map[l] for l in labels]
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224), (128, 128, 128))
            return self.transform(img), self.labels[idx]

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
            for imgs, tgs in loader:
                imgs = imgs.to(device)
                f_res = resnet(imgs).cpu().numpy()
                f_mob = torch.flatten(mobilenet_feat(imgs), 1).cpu().numpy()
                feats_list.append(np.hstack([f_res, f_mob]))
                targets_list.extend(tgs.numpy())
        return np.vstack(feats_list), np.array(targets_list)

    print("  • Extracting Species Features (Cow, Buffalo, Unknown)...")
    X_species, y_species = extract_features(species_paths, species_labels, species_map)
    print("  • Extracting Cattle Breed Features...")
    X_cattle, y_cattle = extract_features(cattle_paths, cattle_labels, cattle_breed_map)
    print("  • Extracting Buffalo Breed Features...")
    X_buffalo, y_buffalo = extract_features(buffalo_paths, buffalo_labels, buffalo_breed_map)

    np.savez_compressed(
        CACHE_FILE,
        X_species=X_species, y_species=y_species,
        X_cattle=X_cattle, y_cattle=y_cattle,
        X_buffalo=X_buffalo, y_buffalo=y_buffalo
    )
    print(f"Features successfully cached to {CACHE_FILE}\n")

# ----------------------------------------------------------------------
# 3. Model 1: Species Classifier (Cow vs Buffalo vs Unknown)
# ----------------------------------------------------------------------
print("================================================================================")
print("   MODEL 1: SPECIES CLASSIFIER (Cow vs Buffalo vs Unknown)                      ")
print("================================================================================")

Xs_tr, Xs_tmp, ys_tr, ys_tmp = train_test_split(
    X_species, y_species, test_size=0.20, random_state=RANDOM_SEED, stratify=y_species
)
Xs_val, Xs_te, ys_val, ys_te = train_test_split(
    Xs_tmp, ys_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=ys_tmp
)

print(f"Data Split: Train={len(ys_tr)} (80%) | Val={len(ys_val)} (10%) | Test={len(ys_te)} (10%)")

species_clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
)
species_clf.fit(Xs_tr, ys_tr)

# Predictions
ys_tr_pred = species_clf.predict(Xs_tr)
ys_val_pred = species_clf.predict(Xs_val)
ys_te_pred = species_clf.predict(Xs_te)

# Metrics
s_tr_metrics = {
    'accuracy': float(accuracy_score(ys_tr, ys_tr_pred)),
    'precision': float(precision_score(ys_tr, ys_tr_pred, average='weighted', zero_division=0)),
    'recall': float(recall_score(ys_tr, ys_tr_pred, average='weighted', zero_division=0)),
    'f1_score': float(f1_score(ys_tr, ys_tr_pred, average='weighted', zero_division=0))
}
s_val_metrics = {
    'accuracy': float(accuracy_score(ys_val, ys_val_pred)),
    'precision': float(precision_score(ys_val, ys_val_pred, average='weighted', zero_division=0)),
    'recall': float(recall_score(ys_val, ys_val_pred, average='weighted', zero_division=0)),
    'f1_score': float(f1_score(ys_val, ys_val_pred, average='weighted', zero_division=0))
}
s_te_metrics = {
    'accuracy': float(accuracy_score(ys_te, ys_te_pred)),
    'precision': float(precision_score(ys_te, ys_te_pred, average='weighted', zero_division=0)),
    'recall': float(recall_score(ys_te, ys_te_pred, average='weighted', zero_division=0)),
    'f1_score': float(f1_score(ys_te, ys_te_pred, average='weighted', zero_division=0))
}

print("\n[Species Classifier Performance]")
print(f"  * Training Set   (80%): Accuracy: {s_tr_metrics['accuracy']*100:6.2f}% | Precision: {s_tr_metrics['precision']*100:6.2f}% | Recall: {s_tr_metrics['recall']*100:6.2f}% | F1: {s_tr_metrics['f1_score']*100:6.2f}%")
print(f"  * Validation Set (10%): Accuracy: {s_val_metrics['accuracy']*100:6.2f}% | Precision: {s_val_metrics['precision']*100:6.2f}% | Recall: {s_val_metrics['recall']*100:6.2f}% | F1: {s_val_metrics['f1_score']*100:6.2f}%")
print(f"  * Testing Set    (10%): Accuracy: {s_te_metrics['accuracy']*100:6.2f}% | Precision: {s_te_metrics['precision']*100:6.2f}% | Recall: {s_te_metrics['recall']*100:6.2f}% | F1: {s_te_metrics['f1_score']*100:6.2f}%\n")

with open(os.path.join(MODELS_DIR, 'species_clf.pkl'), 'wb') as f:
    pickle.dump(species_clf, f)

# ----------------------------------------------------------------------
# 4. Model 2A: Cattle Breed Classifier (5 Breeds)
# ----------------------------------------------------------------------
print("================================================================================")
print("   MODEL 2A: CATTLE BREED CLASSIFIER (5 Breeds)                                 ")
print("================================================================================")

Xc_tr, Xc_tmp, yc_tr, yc_tmp = train_test_split(
    X_cattle, y_cattle, test_size=0.20, random_state=RANDOM_SEED, stratify=y_cattle
)
Xc_val, Xc_te, yc_val, yc_te = train_test_split(
    Xc_tmp, yc_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=yc_tmp
)

print(f"Data Split: Train={len(yc_tr)} (80%) | Val={len(yc_val)} (10%) | Test={len(yc_te)} (10%)")

cattle_clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.8, max_iter=1000, random_state=RANDOM_SEED)
)
cattle_clf.fit(Xc_tr, yc_tr)

# Predictions
yc_tr_pred = cattle_clf.predict(Xc_tr)
yc_val_pred = cattle_clf.predict(Xc_val)
yc_te_pred = cattle_clf.predict(Xc_te)

# Metrics (Macro averaged across 5 balanced breeds)
c_tr_metrics = {
    'accuracy': float(accuracy_score(yc_tr, yc_tr_pred)),
    'precision': float(precision_score(yc_tr, yc_tr_pred, average='macro', zero_division=0)),
    'recall': float(recall_score(yc_tr, yc_tr_pred, average='macro', zero_division=0)),
    'f1_score': float(f1_score(yc_tr, yc_tr_pred, average='macro', zero_division=0))
}
c_val_metrics = {
    'accuracy': float(accuracy_score(yc_val, yc_val_pred)),
    'precision': float(precision_score(yc_val, yc_val_pred, average='macro', zero_division=0)),
    'recall': float(recall_score(yc_val, yc_val_pred, average='macro', zero_division=0)),
    'f1_score': float(f1_score(yc_val, yc_val_pred, average='macro', zero_division=0))
}
c_te_metrics = {
    'accuracy': float(accuracy_score(yc_te, yc_te_pred)),
    'precision': float(precision_score(yc_te, yc_te_pred, average='macro', zero_division=0)),
    'recall': float(recall_score(yc_te, yc_te_pred, average='macro', zero_division=0)),
    'f1_score': float(f1_score(yc_te, yc_te_pred, average='macro', zero_division=0))
}

print("\n[Cattle Breed Classifier Performance]")
print(f"  * Training Set   (80%): Accuracy: {c_tr_metrics['accuracy']*100:6.2f}% | Precision: {c_tr_metrics['precision']*100:6.2f}% | Recall: {c_tr_metrics['recall']*100:6.2f}% | F1: {c_tr_metrics['f1_score']*100:6.2f}%")
print(f"  * Validation Set (10%): Accuracy: {c_val_metrics['accuracy']*100:6.2f}% | Precision: {c_val_metrics['precision']*100:6.2f}% | Recall: {c_val_metrics['recall']*100:6.2f}% | F1: {c_val_metrics['f1_score']*100:6.2f}%")
print(f"  * Testing Set    (10%): Accuracy: {c_te_metrics['accuracy']*100:6.2f}% | Precision: {c_te_metrics['precision']*100:6.2f}% | Recall: {c_te_metrics['recall']*100:6.2f}% | F1: {c_te_metrics['f1_score']*100:6.2f}%\n")

with open(os.path.join(MODELS_DIR, 'cattle_breed_clf.pkl'), 'wb') as f:
    pickle.dump(cattle_clf, f)

# ----------------------------------------------------------------------
# 5. Model 2B: Buffalo Breed Classifier (17 Breeds)
# ----------------------------------------------------------------------
print("================================================================================")
print("   MODEL 2B: BUFFALO BREED CLASSIFIER (17 Breeds)                               ")
print("================================================================================")

Xb_tr, Xb_tmp, yb_tr, yb_tmp = train_test_split(
    X_buffalo, y_buffalo, test_size=0.20, random_state=RANDOM_SEED, stratify=y_buffalo
)
Xb_val, Xb_te, yb_val, yb_te = train_test_split(
    Xb_tmp, yb_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=yb_tmp
)

print(f"Data Split: Train={len(yb_tr)} (80%) | Val={len(yb_val)} (10%) | Test={len(yb_te)} (10%)")

buffalo_clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.001, max_iter=1000, random_state=RANDOM_SEED)
)
buffalo_clf.fit(Xb_tr, yb_tr)

# Predictions
yb_tr_pred = buffalo_clf.predict(Xb_tr)
yb_val_pred = buffalo_clf.predict(Xb_val)
yb_te_pred = buffalo_clf.predict(Xb_te)

# Metrics (Macro averaged across 17 breeds)
b_tr_metrics = {
    'accuracy': float(accuracy_score(yb_tr, yb_tr_pred)),
    'precision': float(precision_score(yb_tr, yb_tr_pred, average='macro', zero_division=0)),
    'recall': float(recall_score(yb_tr, yb_tr_pred, average='macro', zero_division=0)),
    'f1_score': float(f1_score(yb_tr, yb_tr_pred, average='macro', zero_division=0))
}
b_val_metrics = {
    'accuracy': float(accuracy_score(yb_val, yb_val_pred)),
    'precision': float(precision_score(yb_val, yb_val_pred, average='macro', zero_division=0)),
    'recall': float(recall_score(yb_val, yb_val_pred, average='macro', zero_division=0)),
    'f1_score': float(f1_score(yb_val, yb_val_pred, average='macro', zero_division=0))
}
b_te_metrics = {
    'accuracy': float(accuracy_score(yb_te, yb_te_pred)),
    'precision': float(precision_score(yb_te, yb_te_pred, average='macro', zero_division=0)),
    'recall': float(recall_score(yb_te, yb_te_pred, average='macro', zero_division=0)),
    'f1_score': float(f1_score(yb_te, yb_te_pred, average='macro', zero_division=0))
}

print("\n[Buffalo Breed Classifier Performance]")
print(f"  * Training Set   (80%): Accuracy: {b_tr_metrics['accuracy']*100:6.2f}% | Precision: {b_tr_metrics['precision']*100:6.2f}% | Recall: {b_tr_metrics['recall']*100:6.2f}% | F1: {b_tr_metrics['f1_score']*100:6.2f}%")
print(f"  * Validation Set (10%): Accuracy: {b_val_metrics['accuracy']*100:6.2f}% | Precision: {b_val_metrics['precision']*100:6.2f}% | Recall: {b_val_metrics['recall']*100:6.2f}% | F1: {b_val_metrics['f1_score']*100:6.2f}%")
print(f"  * Testing Set    (10%): Accuracy: {b_te_metrics['accuracy']*100:6.2f}% | Precision: {b_te_metrics['precision']*100:6.2f}% | Recall: {b_te_metrics['recall']*100:6.2f}% | F1: {b_te_metrics['f1_score']*100:6.2f}%\n")

with open(os.path.join(MODELS_DIR, 'buffalo_breed_clf.pkl'), 'wb') as f:
    pickle.dump(buffalo_clf, f)

# ----------------------------------------------------------------------
# 6. Model 3: End-to-End Hierarchical Vision Breed Pipeline
# ----------------------------------------------------------------------
print("================================================================================")
print("   MODEL 3: HIERARCHICAL VISION BREED PIPELINE (End-to-End Evaluation)          ")
print("================================================================================")

# Evaluate the combined vision pipeline on Cow and Buffalo livestock
def evaluate_hierarchical_pipeline(Xc, yc, Xb, yb):
    # Cattle evaluation
    pred_c_species = species_clf.predict(Xc)
    pred_c_breed = cattle_clf.predict(Xc)
    # A cattle sample is correct only if species == Cow (0) AND breed == true breed
    correct_c = (pred_c_species == 0) & (pred_c_breed == yc)
    
    # Buffalo evaluation
    pred_b_species = species_clf.predict(Xb)
    pred_b_breed = buffalo_clf.predict(Xb)
    # A buffalo sample is correct only if species == Buffalo (1) AND breed == true breed
    correct_b = (pred_b_species == 1) & (pred_b_breed == yb)

    total_samples = len(yc) + len(yb)
    total_correct = np.sum(correct_c) + np.sum(correct_b)
    acc = total_correct / total_samples

    # Compute macro metrics across all 22 breeds (5 cow + 17 buffalo)
    y_true_all = list(yc) + [b + len(cattle_breeds) for b in yb]
    y_pred_all = []
    
    # Predict for cattle set
    for s_pred, b_pred in zip(pred_c_species, pred_c_breed):
        if s_pred == 0:
            y_pred_all.append(b_pred)
        else:
            y_pred_all.append(-1) # rejected or wrong species branch

    # Predict for buffalo set
    for s_pred, b_pred in zip(pred_b_species, pred_b_breed):
        if s_pred == 1:
            y_pred_all.append(b_pred + len(cattle_breeds))
        else:
            y_pred_all.append(-1) # rejected or wrong species branch

    prec = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

    return acc, prec, rec, f1

v_tr_acc, v_tr_prec, v_tr_rec, v_tr_f1 = evaluate_hierarchical_pipeline(Xc_tr, yc_tr, Xb_tr, yb_tr)
v_val_acc, v_val_prec, v_val_rec, v_val_f1 = evaluate_hierarchical_pipeline(Xc_val, yc_val, Xb_val, yb_val)
v_te_acc, v_te_prec, v_te_rec, v_te_f1 = evaluate_hierarchical_pipeline(Xc_te, yc_te, Xb_te, yb_te)

print(f"Data Split: Train={len(yc_tr)+len(yb_tr)} (80%) | Val={len(yc_val)+len(yb_val)} (10%) | Test={len(yc_te)+len(yb_te)} (10%)")
print("\n[Hierarchical Vision Pipeline Performance]")
print(f"  * Training Set   (80%): Accuracy: {v_tr_acc*100:6.2f}% | Precision: {v_tr_prec*100:6.2f}% | Recall: {v_tr_rec*100:6.2f}% | F1: {v_tr_f1*100:6.2f}%")
print(f"  * Validation Set (10%): Accuracy: {v_val_acc*100:6.2f}% | Precision: {v_val_prec*100:6.2f}% | Recall: {v_val_rec*100:6.2f}% | F1: {v_val_f1*100:6.2f}%")
print(f"  * Testing Set    (10%): Accuracy: {v_te_acc*100:6.2f}% | Precision: {v_te_prec*100:6.2f}% | Recall: {v_te_rec*100:6.2f}% | F1: {v_te_f1*100:6.2f}%\n")

# Save vision_eval.json
vision_eval_payload = {
    "species_accuracy": float(s_te_metrics['accuracy']),
    "species_precision": float(s_te_metrics['precision']),
    "species_recall": float(s_te_metrics['recall']),
    "species_f1": float(s_te_metrics['f1_score']),
    "breed_accuracy": float(v_te_acc),
    "breed_precision": float(v_te_prec),
    "breed_recall": float(v_te_rec),
    "breed_f1": float(v_te_f1)
}
with open(os.path.join(MODELS_DIR, 'vision_eval.json'), 'w', encoding='utf-8') as f:
    json.dump(vision_eval_payload, f, indent=2)

# ----------------------------------------------------------------------
# 7. Model 4: Multi-Variable Price Prediction Model (Karnataka Regressor)
# ----------------------------------------------------------------------
print("================================================================================")
print("   MODEL 4: MULTI-VARIABLE PRICE PREDICTION MODEL (Karnataka Regressor)         ")
print("================================================================================")

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
            
            # Age factor on lactation curve
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

# 80 / 10 / 10 Split for Price Model
Xp_tr, Xp_tmp, yp_tr, yp_tmp = train_test_split(features_df, y_price, test_size=0.20, random_state=RANDOM_SEED)
Xp_val, Xp_te, yp_val, yp_te = train_test_split(Xp_tmp, yp_tmp, test_size=0.50, random_state=RANDOM_SEED)

print(f"Data Split: Train={len(yp_tr)} (80%) | Val={len(yp_val)} (10%) | Test={len(yp_te)} (10%)")

price_model = xgb.XGBRegressor(
    n_estimators=450,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=RANDOM_SEED
)
price_model.fit(Xp_tr, yp_tr)

# Predictions
yp_tr_pred = price_model.predict(Xp_tr)
yp_val_pred = price_model.predict(Xp_val)
yp_te_pred = price_model.predict(Xp_te)

# Regression Metrics
p_tr_metrics = {
    'r2_score': float(r2_score(yp_tr, yp_tr_pred)),
    'mae': float(mean_absolute_error(yp_tr, yp_tr_pred)),
    'accuracy_within_10pct': float(np.mean(np.abs((yp_tr - yp_tr_pred) / yp_tr) < 0.10))
}
p_val_metrics = {
    'r2_score': float(r2_score(yp_val, yp_val_pred)),
    'mae': float(mean_absolute_error(yp_val, yp_val_pred)),
    'accuracy_within_10pct': float(np.mean(np.abs((yp_val - yp_val_pred) / yp_val) < 0.10))
}
p_te_metrics = {
    'r2_score': float(r2_score(yp_te, yp_te_pred)),
    'mae': float(mean_absolute_error(yp_te, yp_te_pred)),
    'accuracy_within_10pct': float(np.mean(np.abs((yp_te - yp_te_pred) / yp_te) < 0.10))
}

print("\n[Price Prediction Model Performance]")
print(f"  * Training Set   (80%): R2 Score: {p_tr_metrics['r2_score']*100:6.2f}% | MAE: INR {p_tr_metrics['mae']:7.2f} | Within 10% Acc: {p_tr_metrics['accuracy_within_10pct']*100:6.2f}%")
print(f"  * Validation Set (10%): R2 Score: {p_val_metrics['r2_score']*100:6.2f}% | MAE: INR {p_val_metrics['mae']:7.2f} | Within 10% Acc: {p_val_metrics['accuracy_within_10pct']*100:6.2f}%")
print(f"  * Testing Set    (10%): R2 Score: {p_te_metrics['r2_score']*100:6.2f}% | MAE: INR {p_te_metrics['mae']:7.2f} | Within 10% Acc: {p_te_metrics['accuracy_within_10pct']*100:6.2f}%\n")

with open(os.path.join(MODELS_DIR, 'price_model.pkl'), 'wb') as f:
    pickle.dump(price_model, f)

with open(os.path.join(MODELS_DIR, 'price_features.json'), 'w', encoding='utf-8') as f:
    json.dump({'feature_names': list(features_df.columns)}, f, indent=2)

# ----------------------------------------------------------------------
# 8. Save Metrics Report (Independent 10% Test Set Verification)
# ----------------------------------------------------------------------
metrics_report_payload = {
    "species_classifier": {
        "accuracy": s_te_metrics['accuracy'],
        "precision": s_te_metrics['precision'],
        "recall": s_te_metrics['recall'],
        "f1_score": s_te_metrics['f1_score']
    },
    "cattle_breed_classifier": {
        "accuracy": c_te_metrics['accuracy'],
        "precision": c_te_metrics['precision'],
        "recall": c_te_metrics['recall'],
        "f1_score": c_te_metrics['f1_score']
    },
    "buffalo_breed_classifier": {
        "accuracy": b_te_metrics['accuracy'],
        "precision": b_te_metrics['precision'],
        "recall": b_te_metrics['recall'],
        "f1_score": b_te_metrics['f1_score']
    },
    "price_prediction_model": {
        "r2_score": p_te_metrics['r2_score'],
        "mae": p_te_metrics['mae'],
        "accuracy_within_10pct": p_te_metrics['accuracy_within_10pct']
    }
}

with open(os.path.join(MODELS_DIR, 'metrics_report.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics_report_payload, f, indent=2)

print("================================================================================")
print("   FINAL CONSOLIDATED EVALUATION REPORT (10% Independent Test Set)             ")
print("================================================================================")
print(f" 1. Species Classifier:        Acc: {s_te_metrics['accuracy']*100:6.2f}% | Prec: {s_te_metrics['precision']*100:6.2f}% | Rec: {s_te_metrics['recall']*100:6.2f}% | F1: {s_te_metrics['f1_score']*100:6.2f}%")
print(f" 2. Cattle Breed Classifier:   Acc: {c_te_metrics['accuracy']*100:6.2f}% | Prec: {c_te_metrics['precision']*100:6.2f}% | Rec: {c_te_metrics['recall']*100:6.2f}% | F1: {c_te_metrics['f1_score']*100:6.2f}%")
print(f" 3. Buffalo Breed Classifier:  Acc: {b_te_metrics['accuracy']*100:6.2f}% | Prec: {b_te_metrics['precision']*100:6.2f}% | Rec: {b_te_metrics['recall']*100:6.2f}% | F1: {b_te_metrics['f1_score']*100:6.2f}%")
print(f" 4. Vision Breed Pipeline:     Acc: {v_te_acc*100:6.2f}% | Prec: {v_te_prec*100:6.2f}% | Rec: {v_te_rec*100:6.2f}% | F1: {v_te_f1*100:6.2f}%")
print(f" 5. Price Prediction Model:    R2:  {p_te_metrics['r2_score']*100:6.2f}% | MAE: INR {p_te_metrics['mae']:7.2f} | Acc (+/-10%): {p_te_metrics['accuracy_within_10pct']*100:6.2f}%")
print("================================================================================")
print("All models and honest performance reports generated successfully!")
