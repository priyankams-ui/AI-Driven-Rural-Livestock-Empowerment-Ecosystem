import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import random

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_and_validate_gender import buffalo_females, buffalo_males

# Data Augmentation Transforms for Training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class BuffaloGenderDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform, use_crop=True):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.use_crop = use_crop
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        w, h = img.size
        if self.use_crop:
            # Ventral crop with slight variation
            y1 = int(h * 0.38)
            y2 = int(h * 0.95)
            x1 = int(w * 0.10)
            x2 = int(w * 0.90)
            img = img.crop((x1, y1, x2, y2))
        tensor = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label

def train_and_eval_fold(train_paths, train_labels, val_paths, val_labels, epochs=12):
    train_ds = BuffaloGenderDataset(train_paths, train_labels, train_transforms, use_crop=True)
    val_ds = BuffaloGenderDataset(val_paths, val_labels, val_transforms, use_crop=True)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False)
    
    # Pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze layers 1, 2, 3 and train layer 4 + fc
    for name, p in model.named_parameters():
        if not ('layer4' in name or 'fc' in name):
            p.requires_grad = False
            
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 2)
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-2)
    
    best_val_acc = 0.0
    best_preds = []
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
    # Final eval on val
    model.eval()
    val_preds = []
    val_probs = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            val_preds.extend(preds)
            val_probs.extend(probs)
            
    return val_preds, val_probs

def run_cv():
    all_paths = buffalo_females + buffalo_males
    all_labels = [0] * len(buffalo_females) + [1] * len(buffalo_males)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_true_all, y_pred_all = [], []
    print("=" * 70)
    print("RUNNING 5-FOLD STRATIFIED CV WITH FINE-TUNED RESNET-18 (LAYER 4 + FC)")
    print(f"Total buffaloes: {len(all_paths)}")
    print("=" * 70)
    
    fold = 1
    for tr_idx, te_idx in skf.split(all_paths, all_labels):
        tr_p = [all_paths[i] for i in tr_idx]
        tr_y = [all_labels[i] for i in tr_idx]
        te_p = [all_paths[i] for i in te_idx]
        te_y = [all_labels[i] for i in te_idx]
        
        preds, probs = train_and_eval_fold(tr_p, tr_y, te_p, te_y, epochs=14)
        acc = accuracy_score(te_y, preds)
        print(f"  Fold {fold}: Acc = {acc*100:.2f}% ({sum(a==b for a,b in zip(te_y, preds))}/{len(te_y)})")
        y_true_all.extend(te_y)
        y_pred_all.extend(preds)
        fold += 1
        
    overall_acc = accuracy_score(y_true_all, y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all)
    print("\n" + "=" * 70)
    print(f"OVERALL 5-FOLD CV ACCURACY: {overall_acc*100:.2f}%")
    print(f"CONFUSION MATRIX (Row 0=Female, Row 1=Male):\n{cm}")
    print("=" * 70)

if __name__ == '__main__':
    run_cv()
