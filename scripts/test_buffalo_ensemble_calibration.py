import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, 'scripts')
from optimize_buffalo_model import data, labels, all_paths

X = np.array([d['all_combined'] for d in data]) # 3072-d
y = labels

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Test ensemble model
for name, clf_inst in [
    ('ExtraTrees(150)', Pipeline([('scaler', StandardScaler()), ('clf', ExtraTreesClassifier(n_estimators=150, max_depth=5, random_state=42))])),
    ('RandomForest(150)', Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42))])),
    ('LinearSVM(C=0.02)', Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear', C=0.02, probability=True, random_state=42))])),
    ('LogReg(C=0.02)', Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.02, max_iter=1000, random_state=42))])),
    ('VotingEnsemble(ET+RF+LR+SVM)', Pipeline([
        ('scaler', StandardScaler()),
        ('clf', VotingClassifier(estimators=[
            ('et', ExtraTreesClassifier(n_estimators=150, max_depth=5, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)),
            ('lr', LogisticRegression(C=0.02, max_iter=1000, random_state=42)),
            ('svm', SVC(kernel='linear', C=0.02, probability=True, random_state=42))
        ], voting='soft'))
    ]))
]:
    y_true_all, y_pred_all, probs_all = [], [], []
    for tr, te in skf.split(X, y):
        clf_inst.fit(X[tr], y[tr])
        probs = clf_inst.predict_proba(X[te])
        preds = np.argmax(probs, axis=1)
        y_true_all.extend(y[te])
        y_pred_all.extend(preds)
        probs_all.extend(probs)
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    probs_all = np.array(probs_all)
    
    acc = accuracy_score(y_true_all, y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all)
    print(f"\n{name:30s} -> Raw Accuracy: {acc*100:.2f}%, CM:\n{cm}")
    
    # Now evaluate with confidence gating
    for thr in [0.55, 0.60, 0.65, 0.70]:
        confs = np.max(probs_all, axis=1)
        margins = np.abs(probs_all[:, 1] - probs_all[:, 0])
        mask = (confs >= thr) & (margins >= 0.12)
        
        n_pass = np.sum(mask)
        n_unc = len(mask) - n_pass
        if n_pass > 0:
            filt_acc = accuracy_score(y_true_all[mask], y_pred_all[mask])
            filt_cm = confusion_matrix(y_true_all[mask], y_pred_all[mask])
            print(f"  Threshold >= {thr:.2f} (margin >= 0.12): Passed={n_pass}/{len(mask)} ({n_pass/len(mask)*100:.1f}%), Uncertain={n_unc}, Filtered Acc={filt_acc*100:.2f}%, CM:\n{filt_cm}")
