import os
import sys
import json
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

sys.path.insert(0, '.')
from ml_pipeline import pipeline

print("=" * 80)
print("COMPREHENSIVE BUFFALO GENDER DIAGNOSTIC SCRIPT")
print("=" * 80)

# 1. Check saved model bundle
print("\n--- 1. INSPECTING SAVED MODEL BUNDLE ---")
with open('models/gender_classifier.pkl', 'rb') as f:
    bundle = pickle.load(f)
print(f"Bundle keys: {list(bundle.keys()) if isinstance(bundle, dict) else 'Not a dict'}")
if isinstance(bundle, dict):
    print(f"Version: {bundle.get('version')}")
    print(f"Classes: {bundle.get('classes')}")
    print(f"Confidence threshold: {bundle.get('confidence_threshold')}")
    print(f"Min margin: {bundle.get('min_margin')}")
    print(f"Buffalo pipeline type: {type(bundle.get('buffalo_pipeline'))}")
    if hasattr(bundle.get('buffalo_pipeline'), 'named_steps'):
        print(f"Buffalo pipeline steps: {bundle.get('buffalo_pipeline').named_steps}")

with open('models/buffalo_gender_clf.pkl', 'rb') as f:
    buff_clf = pickle.load(f)
print(f"standalone buffalo_gender_clf.pkl: {type(buff_clf)}")

# 2. Check mappings.json and hierarchical_mappings.json
print("\n--- 2. INSPECTING MAPPINGS ---")
with open('models/hierarchical_mappings.json', 'r') as f:
    hmap = json.load(f)
print(f"gender_map in hierarchical_mappings: {hmap.get('gender_map')}")
print(f"rev_gender_map in hierarchical_mappings: {hmap.get('rev_gender_map')}")
print(f"gender_confidence_threshold in hierarchical_mappings: {hmap.get('gender_confidence_threshold')}")

# 3. Check reference images predictions
print("\n--- 3. TESTING REFERENCE IMAGES THROUGH PIPELINE ---")
ref_images = [
    'data/new_images/femalebuffolo.jpg',
    'data/new_images/femalebuffolo1.jpg',
    'data/new_images/femalebuffolo2.jpg',
    'data/new_images/malebuffolo.jpg',
    'data/new_images/malebuffolo1.jpg',
    'data/new_images/malebuffolo2.jpg',
    'data/new_images/malebuffolo3.jpg',
]

for img_path in ref_images:
    if not os.path.exists(img_path):
        print(f"MISSING: {img_path}")
        continue
    img_pil = Image.open(img_path).convert('RGB')
    w, h = img_pil.size
    print(f"\nImage: {img_path} ({w}x{h})")
    
    # Run pipeline predict_image
    res = pipeline.predict_image(img_path)
    print(f"  Pipeline result -> Status: {res.get('status')}, Species: {res.get('species')} ({res.get('species_confidence')}), Gender: {res.get('gender')} ({res.get('gender_confidence')}), Valid: {res.get('is_valid')}, Msg: {res.get('gender_message') or res.get('error')}")

    # Inspect the raw model output probabilities directly
    # Check what features are extracted
    # Ventral crop
    y1, y2 = int(h * 0.40), int(h * 0.94)
    x1, x2 = int(w * 0.12), int(w * 0.88)
    vc = img_pil.crop((x1, y1, x2, y2))
    
    t_full_0 = pipeline.transform(img_pil).unsqueeze(0).to(pipeline.device)
    t_full_1 = pipeline.transform(img_pil.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(pipeline.device)
    t_vrgb_0 = pipeline.transform(vc).unsqueeze(0).to(pipeline.device)
    t_vrgb_1 = pipeline.transform(vc.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(pipeline.device)
    t_vgray_0 = pipeline.transform_gray(vc).unsqueeze(0).to(pipeline.device)
    t_vgray_1 = pipeline.transform_gray(vc.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(pipeline.device)
    
    with torch.no_grad():
        f_full_0 = torch.flatten(pipeline.resnet50_feat(t_full_0), 1).cpu().numpy().flatten()
        f_full_1 = torch.flatten(pipeline.resnet50_feat(t_full_1), 1).cpu().numpy().flatten()
        f_vrgb_0 = pipeline.resnet(t_vrgb_0).cpu().numpy().flatten()
        f_vrgb_1 = pipeline.resnet(t_vrgb_1).cpu().numpy().flatten()
        f_vgray_0 = pipeline.resnet(t_vgray_0).cpu().numpy().flatten()
        f_vgray_1 = pipeline.resnet(t_vgray_1).cpu().numpy().flatten()
        
    f_full = (f_full_0 + f_full_1) / 2.0
    f_vrgb = (f_vrgb_0 + f_vrgb_1) / 2.0
    f_vgray = (f_vgray_0 + f_vgray_1) / 2.0
    f_combo = np.concatenate([f_full, f_vrgb, f_vgray]).reshape(1, -1)
    
    raw_probs = buff_clf.predict_proba(f_combo)[0]
    classes = getattr(buff_clf, 'classes_', [0, 1])
    print(f"  Raw classifier classes: {classes}")
    print(f"  Raw classifier probs: idx 0 (Female?) = {raw_probs[0]:.4f}, idx 1 (Male?) = {raw_probs[1]:.4f}")

    # Check estimator breakdown if VotingClassifier
    if hasattr(buff_clf, 'named_steps') and 'ensemble' in buff_clf.named_steps:
        ensemble = buff_clf.named_steps['ensemble']
        scaler = buff_clf.named_steps['scaler']
        f_scaled = scaler.transform(f_combo)
        for est_name, est in ensemble.named_estimators_.items():
            est_probs = est.predict_proba(f_scaled)[0]
            print(f"    - Sub-estimator {est_name}: idx 0={est_probs[0]:.4f}, idx 1={est_probs[1]:.4f}")
