import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, 'scripts')
from test_unified_gender_architectures import (
    buffalo_females, buffalo_males, cow_females, cow_males,
    b_feats, b_labels, c_feats, c_labels
)

# Multi-scale features: 2560 dimensions
X_b = np.array([f['multiscale'] for f in b_feats])
y_b = b_labels

X_c = np.array([f['multiscale'] for f in c_feats])
y_c = c_labels

# Also ResNet50 full body only: 2048 dimensions
X_b_full = np.array([f['full'] for f in b_feats])
X_c_full = np.array([f['full'] for f in c_feats])

# Also ResNet18 ventral gray: 512 dimensions
X_b_v = np.array([f['v_gray'] for f in b_feats])
X_c_v = np.array([f['v_gray'] for f in c_feats])

print("=" * 75)
print("EXPERIMENT 1: JOINT (COW + BUFFALO) TRAINING EVALUATION ON BUFFALO CV")
print(f"Data: Buffalo ({len(y_b)}), Cow ({len(y_c)}), Total = {len(y_b) + len(y_c)}")
print("=" * 75)

# In each fold of Buffalo cross-validation, add ALL Cow samples to the training fold!
# This gives the model strong prior knowledge of bovine sexual anatomy (udders vs sheaths)
# while evaluating purely on held-out Buffalo test samples!

def eval_joint(feature_matrix_b, feature_matrix_c, name):
    print(f"\n--- {name} (Trained on Train_Buffalo + All_Cow, Tested on Held-Out Buffalo) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    clfs = [
        ("LogReg (C=0.01)", Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.01, max_iter=1000, random_state=42))])),
        ("LogReg (C=0.05)", Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.05, max_iter=1000, random_state=42))])),
        ("LogReg (C=0.1)", Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))])),
        ("PCA(32)+LogReg", Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=32, random_state=42)), ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))])),
        ("PCA(48)+LogReg", Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=48, random_state=42)), ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))])),
        ("Linear SVM (C=0.05)", Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear', C=0.05, probability=True, random_state=42))])),
        ("RBF SVM (C=1.0)", Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))])),
        ("Random Forest", Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))])),
        ("Extra Trees", Pipeline([('scaler', StandardScaler()), ('clf', ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42))])),
    ]
    
    for clf_name, pipeline_model in clfs:
        y_true, y_pred = [], []
        for tr_idx, te_idx in skf.split(feature_matrix_b, y_b):
            # Combine Buffalo train fold + ALL Cow samples
            X_tr = np.vstack([feature_matrix_b[tr_idx], feature_matrix_c])
            y_tr = np.concatenate([y_b[tr_idx], y_c])
            
            X_te = feature_matrix_b[te_idx]
            y_te = y_b[te_idx]
            
            pipeline_model.fit(X_tr, y_tr)
            preds = pipeline_model.predict(X_te)
            y_true.extend(y_te)
            y_pred.extend(preds)
            
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print(f"  {clf_name:25s} -> Buffalo CV Acc: {acc*100:.2f}% | CM: [[{cm[0,0]},{cm[0,1]}],[{cm[1,0]},{cm[1,1]}]]")

eval_joint(X_b, X_c, "Multi-scale Full+Ventral (2560-d)")
eval_joint(X_b_full, X_c_full, "ResNet50 Full Body (2048-d)")
eval_joint(X_b_v, X_c_v, "ResNet18 Ventral Gray (512-d)")

print("\n" + "=" * 75)
print("EXPERIMENT 2: JOINT TRAINING EVALUATION ON COW CV (Buffalo + Cow train, Cow test)")
print("=" * 75)
def eval_joint_cow(feature_matrix_b, feature_matrix_c, name):
    print(f"\n--- {name} (Trained on Train_Cow + All_Buffalo, Tested on Held-Out Cow) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    clfs = [
        ("LogReg (C=0.05)", Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.05, max_iter=1000, random_state=42))])),
        ("PCA(32)+LogReg", Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=32, random_state=42)), ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))])),
        ("Linear SVM (C=0.05)", Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear', C=0.05, probability=True, random_state=42))])),
        ("Random Forest", Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))])),
    ]
    
    for clf_name, pipeline_model in clfs:
        y_true, y_pred = [], []
        for tr_idx, te_idx in skf.split(feature_matrix_c, y_c):
            X_tr = np.vstack([feature_matrix_c[tr_idx], feature_matrix_b])
            y_tr = np.concatenate([y_c[tr_idx], y_b])
            
            X_te = feature_matrix_c[te_idx]
            y_te = y_c[te_idx]
            
            pipeline_model.fit(X_tr, y_tr)
            preds = pipeline_model.predict(X_te)
            y_true.extend(y_te)
            y_pred.extend(preds)
            
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print(f"  {clf_name:25s} -> Cow CV Acc: {acc*100:.2f}% | CM: [[{cm[0,0]},{cm[0,1]}],[{cm[1,0]},{cm[1,1]}]]")

eval_joint_cow(X_b, X_c, "Multi-scale Full+Ventral (2560-d)")
eval_joint_cow(X_b_full, X_c_full, "ResNet50 Full Body (2048-d)")
eval_joint_cow(X_b_v, X_c_v, "ResNet18 Ventral Gray (512-d)")
