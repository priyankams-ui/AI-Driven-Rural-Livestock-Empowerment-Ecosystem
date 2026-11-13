import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
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

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet18.fc = nn.Identity()
resnet18 = resnet18.to(device).eval()

resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet50.fc = nn.Identity()
resnet50 = resnet50.to(device).eval()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_and_validate_gender import buffalo_females, buffalo_males

# Let's test augmented extraction
def get_features(img_path, augment=False, n_aug=6):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    crops = []
    # 1. Full image
    crops.append((img, img.crop((int(w*0.10), int(h*0.40), int(w*0.90), int(h*0.95)))))
    
    if augment:
        for _ in range(n_aug):
            # Jitter full image slightly
            j_full = img.crop((
                max(0, int(w * np.random.uniform(0.0, 0.04))),
                max(0, int(h * np.random.uniform(0.0, 0.04))),
                min(w, int(w * np.random.uniform(0.96, 1.0))),
                min(h, int(h * np.random.uniform(0.96, 1.0)))
            ))
            j_v = img.crop((
                max(0, int(w * (0.10 + np.random.uniform(-0.03, 0.03)))),
                max(0, int(h * (0.40 + np.random.uniform(-0.03, 0.03)))),
                min(w, int(w * (0.90 + np.random.uniform(-0.03, 0.03)))),
                min(h, int(h * (0.95 + np.random.uniform(-0.02, 0.03))))
            ))
            crops.append((j_full, j_v))
            # Also flipped
            crops.append((j_full.transpose(Image.FLIP_LEFT_RIGHT), j_v.transpose(Image.FLIP_LEFT_RIGHT)))
            
    feats = []
    for full_c, v_c in crops:
        t_f_rgb = transform_rgb(full_c).unsqueeze(0).to(device)
        t_v_rgb = transform_rgb(v_c).unsqueeze(0).to(device)
        t_v_gray = transform_gray(v_c).unsqueeze(0).to(device)
        with torch.no_grad():
            f50 = resnet50(t_f_rgb).cpu().numpy().flatten()
            f18_v_rgb = resnet18(t_v_rgb).cpu().numpy().flatten()
            f18_v_gray = resnet18(t_v_gray).cpu().numpy().flatten()
        # Rich concatenated feature: ResNet50 Full (2048) + ResNet18 Ventral RGB (512) + ResNet18 Ventral Gray (512) = 3072 dims
        feat = np.concatenate([f50, f18_v_rgb, f18_v_gray])
        feats.append(feat)
    return feats

def evaluate():
    print("=" * 75)
    print("EVALUATING HYBRID MULTI-SCALE ANATOMICAL REPRESENTATION")
    print("=" * 75)
    
    all_paths = buffalo_females + buffalo_males
    labels = np.array([0] * len(buffalo_females) + [1] * len(buffalo_males))
    
    # Extract base features for cross-validation evaluation (unaugmented for clean testing)
    X_base = []
    for p in all_paths:
        X_base.append(get_features(p, augment=False)[0])
    X_base = np.array(X_base)
    print(f"Feature matrix shape: {X_base.shape}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test Ensemble
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    et = ExtraTreesClassifier(n_estimators=150, max_depth=6, random_state=42)
    lr = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, alpha=0.5, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('et', et), ('lr', lr), ('mlp', mlp)],
        voting='soft'
    )
    
    models = [
        ("Random Forest", lambda: RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)),
        ("Extra Trees", lambda: ExtraTreesClassifier(n_estimators=150, max_depth=6, random_state=42)),
        ("Logistic Regression (C=0.05)", lambda: LogisticRegression(C=0.05, max_iter=1000, random_state=42)),
        ("MLP (64, 32)", lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, alpha=0.5, random_state=42)),
        ("Soft-Voting Ensemble (RF+ET+LR+MLP)", lambda: VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=150, max_depth=6, random_state=42)),
                ('lr', LogisticRegression(C=0.05, max_iter=1000, random_state=42)),
                ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, alpha=0.5, random_state=42))
            ],
            voting='soft'
        ))
    ]
    
    for name, m_fn in models:
        y_true, y_pred = [], []
        fold_accs = []
        for train_idx, test_idx in skf.split(X_base, labels):
            clf = Pipeline([
                ('scaler', StandardScaler()),
                ('model', m_fn())
            ])
            clf.fit(X_base[train_idx], labels[train_idx])
            preds = clf.predict(X_base[test_idx])
            fold_accs.append(accuracy_score(labels[test_idx], preds))
            y_true.extend(labels[test_idx])
            y_pred.extend(preds)
            
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n{name}:")
        print(f"  5-Fold CV Accuracy: {acc*100:.2f}% (std: {np.std(fold_accs)*100:.2f}%) | F1: {f1*100:.2f}%")
        print(f"  Confusion Matrix (Row 0=Female, Row 1=Male):\n{cm}")

if __name__ == '__main__':
    evaluate()
