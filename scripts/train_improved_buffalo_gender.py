import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

print("=" * 80)
print("TRAINING ENHANCED BUFFALO GENDER CLASSIFIER (ANATOMICAL MULTI-SCALE)")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Compute Device: {device}")

# Preprocessing transforms
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

# Feature Extractors (ResNet-50 and ResNet-18)
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
resnet18.fc = nn.Identity()

resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()
resnet50.fc = nn.Identity()

# ------------------------------------------------------------------------------
# 1. LOAD AUTHORITATIVE VERIFIED CLEAN BUFFALO CATALOG
# ------------------------------------------------------------------------------
catalog_path = 'models/authoritative_buffalo_gender_catalog.json'
with open(catalog_path, 'r') as f:
    catalog = json.load(f)

female_animals = [a for a in catalog if a['gender'] == 'Female']
male_animals = [a for a in catalog if a['gender'] == 'Male']

print(f"Loaded Authoritative Catalog: {len(catalog)} Total Animals ({len(female_animals)} Female, {len(male_animals)} Male)")

# ------------------------------------------------------------------------------
# 2. FEATURE EXTRACTION PIPELINE (CONSISTENT WITH INFERENCE)
# ResNet-50 Body Crop (2048) + ResNet-18 Ventral RGB (512) + ResNet-18 Ventral Gray (512)
# Bilateral horizontal flip averaging -> 3072-d
# ------------------------------------------------------------------------------
def get_crops(img, box=None):
    w, h = img.size
    if box is not None:
        bx1, by1, bx2, by2 = box
        bw, bh = bx2 - bx1, by2 - by1
        y1 = max(0, int(by1 + 0.38 * bh))
        y2 = min(h, int(by1 + 0.96 * bh))
        x1 = max(0, int(bx1 + 0.10 * bw))
        x2 = min(w, int(bx1 + 0.90 * bw))
        body_crop = img.crop((max(0, int(bx1)), max(0, int(by1)), min(w, int(bx2)), min(h, int(by2))))
    else:
        y1, y2 = int(h * 0.40), int(h * 0.94)
        x1, x2 = int(w * 0.12), int(w * 0.88)
        body_crop = img
    ventral_crop = img.crop((x1, y1, x2, y2))
    return body_crop, ventral_crop

def extract_buffalo_multiscale(img_path, box=None):
    img = Image.open(img_path).convert('RGB')
    body_crop, ventral_crop = get_crops(img, box)
    
    t_body_0 = transform_rgb(body_crop).unsqueeze(0).to(device)
    t_body_1 = transform_rgb(body_crop.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
    t_vrgb_0 = transform_rgb(ventral_crop).unsqueeze(0).to(device)
    t_vrgb_1 = transform_rgb(ventral_crop.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
    t_vgray_0 = transform_gray(ventral_crop).unsqueeze(0).to(device)
    t_vgray_1 = transform_gray(ventral_crop.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        fb_0 = torch.flatten(resnet50(t_body_0), 1).cpu().numpy().flatten()
        fb_1 = torch.flatten(resnet50(t_body_1), 1).cpu().numpy().flatten()
        vr_0 = resnet18(t_vrgb_0).cpu().numpy().flatten()
        vr_1 = resnet18(t_vrgb_1).cpu().numpy().flatten()
        vg_0 = resnet18(t_vgray_0).cpu().numpy().flatten()
        vg_1 = resnet18(t_vgray_1).cpu().numpy().flatten()
        
    fb = (fb_0 + fb_1) / 2.0
    vr = (vr_0 + vr_1) / 2.0
    vg = (vg_0 + vg_1) / 2.0
    
    return np.concatenate([fb, vr, vg])

# Color, Contrast & Jitter Invariant Augmentation for Training
def extract_augmented_multiscale(img_path, box=None, n_aug=6):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    feats = []
    
    # Base feature
    feats.append(extract_buffalo_multiscale(img_path, box))
    
    for _ in range(n_aug):
        bright_factor = np.random.uniform(0.85, 1.15)
        contrast_factor = np.random.uniform(0.85, 1.15)
        aug_img = ImageEnhance.Brightness(img).enhance(bright_factor)
        aug_img = ImageEnhance.Contrast(aug_img).enhance(contrast_factor)
        
        # Slight jitter to box / crops
        if box is not None:
            bx1, by1, bx2, by2 = box
            bw, bh = bx2 - bx1, by2 - by1
            jbx1 = max(0, int(bx1 + np.random.uniform(-0.02, 0.02) * bw))
            jby1 = max(0, int(by1 + np.random.uniform(-0.02, 0.02) * bh))
            jbx2 = min(w, int(bx2 + np.random.uniform(-0.02, 0.02) * bw))
            jby2 = min(h, int(by2 + np.random.uniform(-0.02, 0.02) * bh))
            jbox = (jbx1, jby1, jbx2, jby2)
        else:
            jbox = None
            
        body_crop, ventral_crop = get_crops(aug_img, jbox)
        
        t_body_0 = transform_rgb(body_crop).unsqueeze(0).to(device)
        t_body_1 = transform_rgb(body_crop.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
        t_vrgb_0 = transform_rgb(ventral_crop).unsqueeze(0).to(device)
        t_vrgb_1 = transform_rgb(ventral_crop.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
        t_vgray_0 = transform_gray(ventral_crop).unsqueeze(0).to(device)
        t_vgray_1 = transform_gray(ventral_crop.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            fb_0 = torch.flatten(resnet50(t_body_0), 1).cpu().numpy().flatten()
            fb_1 = torch.flatten(resnet50(t_body_1), 1).cpu().numpy().flatten()
            vr_0 = resnet18(t_vrgb_0).cpu().numpy().flatten()
            vr_1 = resnet18(t_vrgb_1).cpu().numpy().flatten()
            vg_0 = resnet18(t_vgray_0).cpu().numpy().flatten()
            vg_1 = resnet18(t_vgray_1).cpu().numpy().flatten()
            
        fb = (fb_0 + fb_1) / 2.0
        vr = (vr_0 + vr_1) / 2.0
        vg = (vg_0 + vg_1) / 2.0
        feats.append(np.concatenate([fb, vr, vg]))
        
    return feats

# ------------------------------------------------------------------------------
# 3. ANIMAL-LEVEL 80/10/10 SPLIT (ZERO-LEAKAGE)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SPLITTING DATASET: 80% Train, 10% Validation, 10% Independent Test")
print("=" * 80)

# Set seed for reproducible split
np.random.seed(42)

fem_entries = list(female_animals)
male_entries = list(male_animals)
np.random.shuffle(fem_entries)
np.random.shuffle(male_entries)

# 80 / 10 / 10 split
# 75 females -> 8 test, 8 val, 59 train
# 32 males -> 4 test, 4 val, 24 train
n_f_test = max(1, int(len(fem_entries) * 0.10))
n_f_val = max(1, int(len(fem_entries) * 0.10))
test_fem = fem_entries[:n_f_test]
val_fem = fem_entries[n_f_test:n_f_test + n_f_val]
train_fem = fem_entries[n_f_test + n_f_val:]

n_m_test = max(1, int(len(male_entries) * 0.10))
n_m_val = max(1, int(len(male_entries) * 0.10))
test_male = male_entries[:n_m_test]
val_male = male_entries[n_m_test:n_m_test + n_m_val]
train_male = male_entries[n_m_test + n_m_val:]

train_animals = train_fem + train_male
val_animals = val_fem + val_male
test_animals = test_fem + test_male

print(f"Train Base Animals: {len(train_animals)} ({len(train_fem)} Female, {len(train_male)} Male)")
print(f"Val Base Animals:   {len(val_animals)} ({len(val_fem)} Female, {len(val_male)} Male)")
print(f"Test Base Animals:  {len(test_animals)} ({len(test_fem)} Female, {len(test_male)} Male)")

# Verify zero overlap
train_ids = set(a['base_id'] for a in train_animals)
val_ids = set(a['base_id'] for a in val_animals)
test_ids = set(a['base_id'] for a in test_animals)
assert len(train_ids.intersection(val_ids)) == 0, "Train-Val Leakage detected!"
assert len(train_ids.intersection(test_ids)) == 0, "Train-Test Leakage detected!"
assert len(val_ids.intersection(test_ids)) == 0, "Val-Test Leakage detected!"
print("[VERIFIED] Animal-level leakage check passed with 0 overlaps across all splits!")

# ------------------------------------------------------------------------------
# 4. EXTRACT TRAINING, VALIDATION, AND TEST FEATURES
# ------------------------------------------------------------------------------
print("\nExtracting multi-scale features for Train, Val, Test...")

def load_split_features(split_animals, is_train=True):
    X, y, groups = [], [], []
    for a in split_animals:
        label = 0 if a['gender'] == 'Female' else 1
        p = a['primary']
        box = a.get('box', None)
        if not os.path.exists(p):
            continue
        if is_train:
            # Balanced sampling: equal feature count between male and female in feature space
            # 24 males * 13 aug = 312 male feats; 59 females * 5 aug = 295 female feats
            n_aug = 12 if label == 1 else 4
            sample_feats = extract_augmented_multiscale(p, box, n_aug=n_aug)
        else:
            sample_feats = [extract_buffalo_multiscale(p, box)]
            
        for feat in sample_feats:
            X.append(feat)
            y.append(label)
            groups.append(a['base_id'])
    return np.array(X), np.array(y), groups

X_train, y_train, train_grps = load_split_features(train_animals, is_train=True)
X_val, y_val, val_grps = load_split_features(val_animals, is_train=False)
X_test, y_test, test_grps = load_split_features(test_animals, is_train=False)

print(f"Feature matrix dimensions:")
print(f"  X_train: {X_train.shape} | Labels: Female={np.sum(y_train==0)}, Male={np.sum(y_train==1)}")
print(f"  X_val:   {X_val.shape}   | Labels: Female={np.sum(y_val==0)}, Male={np.sum(y_val==1)}")
print(f"  X_test:  {X_test.shape}  | Labels: Female={np.sum(y_test==0)}, Male={np.sum(y_test==1)}")

# ------------------------------------------------------------------------------
# 5. MODEL TRAINING & COMPARATIVE EVALUATION
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TRAINING & COMPARING MULTIPLE ARCHITECTURES")
print("=" * 80)

# Model 1: Regularized Logistic Regression
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=0.05, penalty='l2', class_weight='balanced', max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train, y_train)

# Model 2: Extra Trees Classifier
et_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', ExtraTreesClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42))
])
et_pipeline.fit(X_train, y_train)

# Model 3: Random Forest Classifier
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42))
])
rf_pipeline.fit(X_train, y_train)

# Model 4: Soft-Voting Ensemble (LR + ET + RF)
ensemble_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ensemble', VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=0.05, penalty='l2', class_weight='balanced', max_iter=1000, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42))
        ],
        voting='soft'
    ))
])
ensemble_pipeline.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 6. VALIDATION EVALUATION & MODEL SELECTION
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("VALIDATION SET EVALUATION (MODEL SELECTION)")
print("=" * 80)

models_dict = {
    'LogisticRegression': lr_pipeline,
    'ExtraTrees': et_pipeline,
    'RandomForest': rf_pipeline,
    'Soft-Voting Ensemble': ensemble_pipeline
}

best_val_f1 = -1.0
selected_model_name = None
selected_model = None

for name, m in models_dict.items():
    val_probs = m.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    acc = accuracy_score(y_val, val_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, val_preds, average='weighted', zero_division=0)
    print(f"  {name:25s} | Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%")
    if f1 > best_val_f1:
        best_val_f1 = f1
        selected_model_name = name
        selected_model = m

print(f"\n[SELECTED] Best Model on Validation: {selected_model_name} (Val F1: {best_val_f1*100:.2f}%)")

# ------------------------------------------------------------------------------
# 7. INDEPENDENT TEST SET EVALUATION (MANDATORY METRICS)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("INDEPENDENT UNSEEN TEST SET EVALUATION (STRICT GENUINE METRICS)")
print("=" * 80)

test_probs = selected_model.predict_proba(X_test)
test_preds = np.argmax(test_probs, axis=1)

test_acc = accuracy_score(y_test, test_preds)
test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average='weighted', zero_division=0)
test_cm = confusion_matrix(y_test, test_preds)

# Per-class metrics
prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(y_test, test_preds, average=None, zero_division=0)

female_prec, female_rec, female_f1 = prec_per_class[0], rec_per_class[0], f1_per_class[0]
male_prec = prec_per_class[1] if len(prec_per_class) > 1 else 0.0
male_rec = rec_per_class[1] if len(rec_per_class) > 1 else 0.0
male_f1 = f1_per_class[1] if len(f1_per_class) > 1 else 0.0

male_pred_as_female = int(test_cm[1, 0]) if test_cm.shape[0] > 1 else 0
female_pred_as_male = int(test_cm[0, 1]) if test_cm.shape[1] > 1 else 0

print(f"Independent Test Accuracy:  {test_acc * 100:.2f}%")
print(f"Independent Test Precision: {test_prec * 100:.2f}%")
print(f"Independent Test Recall:    {test_rec * 100:.2f}%")
print(f"Independent Test F1 Score:  {test_f1 * 100:.2f}%")
print(f"\nFemale Buffalo Metrics -> Prec: {female_prec*100:.2f}% | Rec: {female_rec*100:.2f}% | F1: {female_f1*100:.2f}%")
print(f"Male Buffalo Metrics   -> Prec: {male_prec*100:.2f}% | Rec: {male_rec*100:.2f}% | F1: {male_f1*100:.2f}%")
print(f"\nMisclassification Counts:")
print(f"  Male predicted as Female:   {male_pred_as_female}")
print(f"  Female predicted as Male:   {female_pred_as_male}")

print("\nConfusion Matrix (Row 0 = Actual Female, Row 1 = Actual Male):")
print(f"             Predicted Female   Predicted Male")
print(f"Actual Female      {test_cm[0, 0]:<15}    {test_cm[0, 1]}")
if test_cm.shape[0] > 1:
    print(f"Actual Male        {test_cm[1, 0]:<15}    {test_cm[1, 1]}")

# ------------------------------------------------------------------------------
# 8. VERIFY AGAINST ALL USER REFERENCE SAMPLES
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("VERIFYING SPECIFICALLY AGAINST REFERENCE SAMPLES")
print("=" * 80)

reference_samples = [
    ('data/new_images/femalebuffolo.jpg', 0, 'Female Buffalo 0 (Udder & Teats visible)'),
    ('data/new_images/femalebuffolo1.jpg', 0, 'Female Buffalo 1 (Posterior Udder visible)'),
    ('data/new_images/femalebuffolo2.jpg', 0, 'Female Buffalo 2 (White background, Teats visible)'),
    ('data/new_images/malebuffolo.jpg', 1, 'Male Buffalo 0 (Preputial Sheath visible)'),
    ('data/new_images/malebuffolo1.jpg', 1, 'Male Buffalo 1 (Dirt path, Ventral Prepuce visible)'),
    ('data/new_images/malebuffolo2.jpg', 1, 'Male Buffalo 2 (Open ground, Prepuce visible)'),
    ('data/new_images/malebuffolo3.jpg', 1, 'Male Buffalo 3 (Heavy Bull, Penile Sheath visible)'),
]

ref_results = []
ref_correct = 0
for path, true_label, desc in reference_samples:
    if not os.path.exists(path):
        continue
    f = extract_buffalo_multiscale(path).reshape(1, -1)
    probs = selected_model.predict_proba(f)[0]
    p_fem, p_male = probs[0], probs[1]
    pred = 0 if p_fem > p_male else 1
    conf = max(p_fem, p_male)
    pred_str = "Female" if pred == 0 else "Male"
    true_str = "Female" if true_label == 0 else "Male"
    is_correct = (pred == true_label)
    if is_correct:
        ref_correct += 1
    ref_results.append({
        'path': path,
        'description': desc,
        'true_gender': true_str,
        'predicted_gender': pred_str,
        'confidence': float(conf),
        'passed': bool(is_correct)
    })
    print(f"  {desc:50s} | True:{true_str:6s} | Pred:{pred_str:6s} ({conf*100:.1f}%) | {'PASS' if is_correct else 'FAIL'}")

print(f"\nReference Pattern Result: {ref_correct}/{len(reference_samples)} ({ref_correct/len(reference_samples)*100:.1f}%) correctly classified based on genuine anatomy!")

# ------------------------------------------------------------------------------
# 9. SAVE THE IMPROVED MODEL & UPDATE BUNDLES / METRICS
# ------------------------------------------------------------------------------
os.makedirs('models', exist_ok=True)
model_path = 'models/buffalo_gender_clf.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(selected_model, f)
print(f"\n[SAVED] Successfully saved improved buffalo gender model to: {model_path}")

# Update models/gender_classifier.pkl buffalo_pipeline while preserving cattle_pipeline 100% UNTOUCHED
bundle_path = 'models/gender_classifier.pkl'
if os.path.exists(bundle_path):
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict):
        bundle['buffalo_pipeline'] = selected_model
        bundle['version'] = '3.4.0'
        bundle['confidence_threshold'] = 0.60
        bundle['min_margin'] = 0.10
        with open(bundle_path, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"[SAVED] Successfully updated buffalo_pipeline in {bundle_path} (cattle_pipeline preserved 100% intact).")

# Update models/metrics_report.json with genuine test metrics
metrics_report_path = 'models/metrics_report.json'
if os.path.exists(metrics_report_path):
    with open(metrics_report_path, 'r') as f:
        mr = json.load(f)
    mr['buffalo_gender_classifier'] = {
        'accuracy': float(test_acc),
        'precision': float(test_prec),
        'recall': float(test_rec),
        'f1_score': float(test_f1)
    }
    with open(metrics_report_path, 'w') as f:
        json.dump(mr, f, indent=2)
    print(f"[UPDATED] {metrics_report_path} with genuine buffalo gender test metrics.")

# Update models/vision_eval.json with genuine test metrics
vision_eval_path = 'models/vision_eval.json'
if os.path.exists(vision_eval_path):
    with open(vision_eval_path, 'r') as f:
        ve = json.load(f)
    ve['buffalo_gender_model'] = {
        'accuracy': float(test_acc),
        'precision': float(test_prec),
        'recall': float(test_rec),
        'f1_score': float(test_f1)
    }
    with open(vision_eval_path, 'w') as f:
        json.dump(ve, f, indent=2)
    print(f"[UPDATED] {vision_eval_path} with genuine buffalo gender test metrics.")
