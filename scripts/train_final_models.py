import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
CACHE_FILE = os.path.join(MODELS_DIR, "cleaned_features_cache.npz")

print("================================================================================")
print("   TRAINING & EVALUATION PIPELINE: CLEANED 80/10/10 RIGOROUS PIPELINE           ")
print("================================================================================")

# Load cleaned dataset manifest
manifest_path = os.path.join("data", "cleaned", "cleaned_dataset_manifest.csv")
df_all = pd.read_csv(manifest_path)
df = df_all[df_all['is_usable']].copy().reset_index(drop=True)

print(f"Loaded {len(df)} usable images from cleaned dataset manifest.")
print(f"Species breakdown: {dict(df['species'].value_counts())}\n")

# Mappings
cattle_breeds = sorted(df[df['species'] == 'Cow']['breed'].unique())
buffalo_breeds = sorted(df[df['species'] == 'Buffalo']['breed'].unique())

species_map = {"Cow": 0, "Buffalo": 1, "Unknown": 2}
rev_species_map = {0: "Cow", 1: "Buffalo", 2: "Unknown"}

cattle_breed_map = {b: i for i, b in enumerate(cattle_breeds)}
rev_cattle_breed_map = {i: b for i, b in enumerate(cattle_breeds)}

buffalo_breed_map = {b: i for i, b in enumerate(buffalo_breeds)}
rev_buffalo_breed_map = {i: b for i, b in enumerate(buffalo_breeds)}

hierarchical_mappings = {
    'species_map': species_map,
    'rev_species_map': rev_species_map,
    'cattle_breeds': list(cattle_breeds),
    'cattle_breed_map': cattle_breed_map,
    'rev_cattle_breed_map': rev_cattle_breed_map,
    'buffalo_breeds': list(buffalo_breeds),
    'buffalo_breed_map': buffalo_breed_map,
    'rev_buffalo_breed_map': rev_buffalo_breed_map,
    'species_confidence_threshold': 0.70
}
with open(os.path.join(MODELS_DIR, 'hierarchical_mappings.json'), 'w', encoding='utf-8') as f:
    json.dump(hierarchical_mappings, f, indent=2)

cache = np.load(CACHE_FILE)
X_all_feats = cache['X_all_feats']
df['feat_idx'] = np.arange(len(df))

# ----------------------------------------------------------------------
# 1. Species Classifier (Cow vs Buffalo vs Unknown)
# ----------------------------------------------------------------------
print("================================================================================")
print("   1. SPECIES IDENTIFICATION MODEL (Cow, Buffalo, Unknown)                      ")
print("================================================================================")
y_species_all = df['species'].map(species_map).values
idx_tr, idx_tmp, ys_tr, ys_tmp = train_test_split(
    df.index.values, y_species_all, test_size=0.20, random_state=RANDOM_SEED, stratify=y_species_all
)
idx_val, idx_te, ys_val, ys_te = train_test_split(
    idx_tmp, ys_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=ys_tmp
)

Xs_tr = X_all_feats[idx_tr]
Xs_val = X_all_feats[idx_val]
Xs_te = X_all_feats[idx_te]

species_clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED))
species_clf.fit(Xs_tr, ys_tr)

s_pred_tr = species_clf.predict(Xs_tr)
s_pred_val = species_clf.predict(Xs_val)
s_pred_te = species_clf.predict(Xs_te)

print(f"Data Split: Train={len(ys_tr)} (80%) | Val={len(ys_val)} (10%) | Test={len(ys_te)} (10%)")
print(f"  * Training Set   (80%): Accuracy: {accuracy_score(ys_tr, s_pred_tr)*100:6.2f}% | Precision: {precision_score(ys_tr, s_pred_tr, average='weighted')*100:6.2f}% | Recall: {recall_score(ys_tr, s_pred_tr, average='weighted')*100:6.2f}% | F1: {f1_score(ys_tr, s_pred_tr, average='weighted')*100:6.2f}%")
print(f"  * Validation Set (10%): Accuracy: {accuracy_score(ys_val, s_pred_val)*100:6.2f}% | Precision: {precision_score(ys_val, s_pred_val, average='weighted')*100:6.2f}% | Recall: {recall_score(ys_val, s_pred_val, average='weighted')*100:6.2f}% | F1: {f1_score(ys_val, s_pred_val, average='weighted')*100:6.2f}%")
print(f"  * Testing Set    (10%): Accuracy: {accuracy_score(ys_te, s_pred_te)*100:6.2f}% | Precision: {precision_score(ys_te, s_pred_te, average='weighted')*100:6.2f}% | Recall: {recall_score(ys_te, s_pred_te, average='weighted')*100:6.2f}% | F1: {f1_score(ys_te, s_pred_te, average='weighted')*100:6.2f}%")

with open(os.path.join(MODELS_DIR, 'species_clf.pkl'), 'wb') as f:
    pickle.dump(species_clf, f)

# ----------------------------------------------------------------------
# 2. Cattle Breed Classifier (5 Breeds)
# ----------------------------------------------------------------------
print("\n================================================================================")
print("   2. CATTLE BREED CLASSIFIER (5 Classes)                                       ")
print("================================================================================")
df_c = df[df['species'] == 'Cow'].copy().reset_index(drop=True)
yc_all = df_c['breed'].map(cattle_breed_map).values
Xc_all = X_all_feats[df_c['feat_idx'].values]

c_idx_tr, c_idx_tmp, yc_tr, yc_tmp = train_test_split(
    np.arange(len(df_c)), yc_all, test_size=0.20, random_state=RANDOM_SEED, stratify=yc_all
)
c_idx_val, c_idx_te, yc_val, yc_te = train_test_split(
    c_idx_tmp, yc_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=yc_tmp
)

Xc_tr, Xc_val, Xc_te = Xc_all[c_idx_tr], Xc_all[c_idx_val], Xc_all[c_idx_te]

cattle_clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.8, max_iter=1000, random_state=RANDOM_SEED))
cattle_clf.fit(Xc_tr, yc_tr)

c_pred_tr = cattle_clf.predict(Xc_tr)
c_pred_val = cattle_clf.predict(Xc_val)
c_pred_te = cattle_clf.predict(Xc_te)

print(f"Data Split: Train={len(yc_tr)} (80%) | Val={len(yc_val)} (10%) | Test={len(yc_te)} (10%)")
print(f"  * Training Set   (80%): Accuracy: {accuracy_score(yc_tr, c_pred_tr)*100:6.2f}% | Precision: {precision_score(yc_tr, c_pred_tr, average='macro')*100:6.2f}% | Recall: {recall_score(yc_tr, c_pred_tr, average='macro')*100:6.2f}% | F1: {f1_score(yc_tr, c_pred_tr, average='macro')*100:6.2f}%")
print(f"  * Validation Set (10%): Accuracy: {accuracy_score(yc_val, c_pred_val)*100:6.2f}% | Precision: {precision_score(yc_val, c_pred_val, average='macro')*100:6.2f}% | Recall: {recall_score(yc_val, c_pred_val, average='macro')*100:6.2f}% | F1: {f1_score(yc_val, c_pred_val, average='macro')*100:6.2f}%")
print(f"  * Testing Set    (10%): Accuracy: {accuracy_score(yc_te, c_pred_te)*100:6.2f}% | Precision: {precision_score(yc_te, c_pred_te, average='macro')*100:6.2f}% | Recall: {recall_score(yc_te, c_pred_te, average='macro')*100:6.2f}% | F1: {f1_score(yc_te, c_pred_te, average='macro')*100:6.2f}%")

with open(os.path.join(MODELS_DIR, 'cattle_breed_clf.pkl'), 'wb') as f:
    pickle.dump(cattle_clf, f)

# ----------------------------------------------------------------------
# 3. Buffalo Breed Classifier (17 Breeds)
# ----------------------------------------------------------------------
print("\n================================================================================")
print("   3. BUFFALO BREED CLASSIFIER (17 Classes)                                     ")
print("================================================================================")
df_b = df[df['species'] == 'Buffalo'].copy().reset_index(drop=True)
yb_all = df_b['breed'].map(buffalo_breed_map).values
Xb_all = X_all_feats[df_b['feat_idx'].values]

b_idx_tr, b_idx_tmp, yb_tr, yb_tmp = train_test_split(
    np.arange(len(df_b)), yb_all, test_size=0.20, random_state=RANDOM_SEED, stratify=yb_all
)
b_idx_val, b_idx_te, yb_val, yb_te = train_test_split(
    b_idx_tmp, yb_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=yb_tmp
)

Xb_tr, Xb_val, Xb_te = Xb_all[b_idx_tr], Xb_all[b_idx_val], Xb_all[b_idx_te]

# Regularized Logistic Regression with balanced weighting
buffalo_clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED))
buffalo_clf.fit(Xb_tr, yb_tr)

b_pred_tr = buffalo_clf.predict(Xb_tr)
b_pred_val = buffalo_clf.predict(Xb_val)
b_pred_te = buffalo_clf.predict(Xb_te)

print(f"Data Split: Train={len(yb_tr)} (80%) | Val={len(yb_val)} (10%) | Test={len(yb_te)} (10%)")
print(f"  * Training Set   (80%): Accuracy: {accuracy_score(yb_tr, b_pred_tr)*100:6.2f}% | Precision: {precision_score(yb_tr, b_pred_tr, average='macro', zero_division=0)*100:6.2f}% | Recall: {recall_score(yb_tr, b_pred_tr, average='macro', zero_division=0)*100:6.2f}% | F1: {f1_score(yb_tr, b_pred_tr, average='macro', zero_division=0)*100:6.2f}%")
print(f"  * Validation Set (10%): Accuracy: {accuracy_score(yb_val, b_pred_val)*100:6.2f}% | Precision: {precision_score(yb_val, b_pred_val, average='macro', zero_division=0)*100:6.2f}% | Recall: {recall_score(yb_val, b_pred_val, average='macro', zero_division=0)*100:6.2f}% | F1: {f1_score(yb_val, b_pred_val, average='macro', zero_division=0)*100:6.2f}%")
print(f"  * Testing Set    (10%): Accuracy: {accuracy_score(yb_te, b_pred_te)*100:6.2f}% | Precision: {precision_score(yb_te, b_pred_te, average='macro', zero_division=0)*100:6.2f}% | Recall: {recall_score(yb_te, b_pred_te, average='macro', zero_division=0)*100:6.2f}% | F1: {f1_score(yb_te, b_pred_te, average='macro', zero_division=0)*100:6.2f}%")

with open(os.path.join(MODELS_DIR, 'buffalo_breed_clf.pkl'), 'wb') as f:
    pickle.dump(buffalo_clf, f)

# ----------------------------------------------------------------------
# 4. Multi-Variable Price Model
# ----------------------------------------------------------------------
print("\n================================================================================")
print("   4. MULTI-VARIABLE PRICE PREDICTION REGRESSOR                                ")
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
            
            age_factor = 1.0 if (3.0 <= age <= 6.5) else (0.85 if age < 3.0 else max(0.45, 1.0 - (age - 6.5) * 0.11))
            milk_production = round(max(2.0, np.random.normal(base_milk * age_factor, 2.0)), 1)

            # Pregnancy logic for females
            is_pregnant = int(np.random.choice([0, 1], p=[0.55, 0.45]))
            if is_pregnant == 1:
                pregnancy_month = int(np.random.randint(1, 10))
            else:
                pregnancy_month = 0
        else:
            milk_production = 0.0
            is_pregnant = 0
            pregnancy_month = 0

        district = np.random.choice(districts)
        dist_mult = district_multipliers.get(district, 1.0)
        base = base_breed_value[breed]

        if gender == "Female":
            price = base + (milk_production * 3600)
            if age < 3.0: price *= 0.92
            elif age > 6.5: price *= max(0.50, 1.0 - (age - 6.5) * 0.11)

            if is_pregnant == 1:
                # Pregnancy premium: +₹2,000 base + ₹1,200 per month
                pregnancy_premium = 2000 + (pregnancy_month * 1200)
                price += pregnancy_premium
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
            'is_pregnant': is_pregnant,
            'pregnancy_month': pregnancy_month,
            'district': district,
            'price': price
        })

    return pd.DataFrame(data)

df_price = generate_karnataka_price_dataset(10000)
features_df = pd.get_dummies(df_price[['species', 'breed', 'gender', 'age_years', 'milk_production', 'is_pregnant', 'pregnancy_month', 'district']], drop_first=False)
y_price = df_price['price']

Xp_tr, Xp_tmp, yp_tr, yp_tmp = train_test_split(features_df, y_price, test_size=0.20, random_state=RANDOM_SEED)
Xp_val, Xp_te, yp_val, yp_te = train_test_split(Xp_tmp, yp_tmp, test_size=0.50, random_state=RANDOM_SEED)

price_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=RANDOM_SEED
)
price_model.fit(Xp_tr, yp_tr)

p_pred_te = price_model.predict(Xp_te)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2_te = float(r2_score(yp_te, p_pred_te))
mae_te = float(mean_absolute_error(yp_te, p_pred_te))
rmse_te = float(np.sqrt(mean_squared_error(yp_te, p_pred_te)))
acc_10_te = float(np.mean(np.abs((yp_te - p_pred_te) / yp_te) < 0.10))

print(f"Price Model Split: Train={len(yp_tr)} (80%) | Val={len(yp_val)} (10%) | Test={len(yp_te)} (10%)")
print(f"Price Model Test Set (10%): R2: {r2_te*100:6.2f}% | MAE: INR {mae_te:7.2f} | RMSE: INR {rmse_te:7.2f} | Acc (+/-10%): {acc_10_te*100:6.2f}%")

with open(os.path.join(MODELS_DIR, 'price_model.pkl'), 'wb') as f:
    pickle.dump(price_model, f)
with open(os.path.join(MODELS_DIR, 'price_features.json'), 'w', encoding='utf-8') as f:
    json.dump({'feature_names': list(features_df.columns)}, f, indent=2)

# ----------------------------------------------------------------------
# 5. Write Comprehensive metrics_report.json & vision_eval.json
# ----------------------------------------------------------------------
metrics_report_payload = {
    "species_classifier": {
        "accuracy": float(accuracy_score(ys_te, s_pred_te)),
        "precision": float(precision_score(ys_te, s_pred_te, average='weighted')),
        "recall": float(recall_score(ys_te, s_pred_te, average='weighted')),
        "f1_score": float(f1_score(ys_te, s_pred_te, average='weighted'))
    },
    "cattle_breed_classifier": {
        "accuracy": float(accuracy_score(yc_te, c_pred_te)),
        "precision": float(precision_score(yc_te, c_pred_te, average='macro')),
        "recall": float(recall_score(yc_te, c_pred_te, average='macro')),
        "f1_score": float(f1_score(yc_te, c_pred_te, average='macro'))
    },
    "buffalo_breed_classifier": {
        "accuracy": float(accuracy_score(yb_te, b_pred_te)),
        "precision": float(precision_score(yb_te, b_pred_te, average='macro', zero_division=0)),
        "recall": float(recall_score(yb_te, b_pred_te, average='macro', zero_division=0)),
        "f1_score": float(f1_score(yb_te, b_pred_te, average='macro', zero_division=0))
    },
    "price_prediction_model": {
        "r2_score": r2_te,
        "mae": mae_te,
        "rmse": rmse_te,
        "accuracy_within_10pct": acc_10_te,
        "train_samples": len(yp_tr),
        "val_samples": len(yp_val),
        "test_samples": len(yp_te)
    }
}

with open(os.path.join(MODELS_DIR, 'metrics_report.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics_report_payload, f, indent=2)

vision_eval_payload = {
    "evaluation_methodology": "80% Training / 10% Validation / 10% Testing Split",
    "dataset_clean_manifest": manifest_path,
    "species_model": {
        "test_accuracy": float(accuracy_score(ys_te, s_pred_te)),
        "test_precision_weighted": float(precision_score(ys_te, s_pred_te, average='weighted')),
        "test_recall_weighted": float(recall_score(ys_te, s_pred_te, average='weighted')),
        "test_f1_weighted": float(f1_score(ys_te, s_pred_te, average='weighted'))
    },
    "cattle_breed_model": {
        "classes": cattle_breeds,
        "test_accuracy": float(accuracy_score(yc_te, c_pred_te)),
        "test_precision_macro": float(precision_score(yc_te, c_pred_te, average='macro')),
        "test_recall_macro": float(recall_score(yc_te, c_pred_te, average='macro')),
        "test_f1_macro": float(f1_score(yc_te, c_pred_te, average='macro'))
    },
    "buffalo_breed_model": {
        "classes": buffalo_breeds,
        "test_accuracy": float(accuracy_score(yb_te, b_pred_te)),
        "test_precision_macro": float(precision_score(yb_te, b_pred_te, average='macro', zero_division=0)),
        "test_recall_macro": float(recall_score(yb_te, b_pred_te, average='macro', zero_division=0)),
        "test_f1_macro": float(f1_score(yb_te, b_pred_te, average='macro', zero_division=0))
    }
}

with open(os.path.join(MODELS_DIR, 'vision_eval.json'), 'w', encoding='utf-8') as f:
    json.dump(vision_eval_payload, f, indent=2)

print("\n================================================================================")
print("   TRAINING & EVALUATION COMPLETED SUCCESSFULLY!                                ")
print("================================================================================")
