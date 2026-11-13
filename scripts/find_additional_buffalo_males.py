import os
import sys
import json
import hashlib
from PIL import Image
import cv2
import numpy as np
import torch

sys.path.insert(0, '.')
from ml_pipeline import pipeline

# Load existing exclusions and verified clean
with open('models/buffalo_audit_exclusions.json') as f:
    exclusions = json.load(f)

with open('models/buffalo_catalog_verified_clean.json') as f:
    existing_catalog = json.load(f)

existing_paths = set(entry['primary'].replace('\\', '/') for entry in existing_catalog)
excluded_paths = set(e['path'].replace('\\', '/') for e in exclusions)

print(f"Existing catalog paths: {len(existing_paths)}, excluded paths: {len(excluded_paths)}")

def dhash(image, hash_size=8):
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(image.get_flattened_data(), dtype=float).reshape((hash_size, hash_size + 1))
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

# Find all images in buffalo/
all_candidates = []
male_keywords = ['bull', 'sire', 'male', 'he_buffalo', 'breeding', 'yuvraj', 'sultan', 'veera', 'bhima', 'thakya', 'tulsidas']

for root, dirs, files in os.walk('buffalo'):
    for f in files:
        if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue
        fl = f.lower()
        if any(k in fl for k in male_keywords):
            p = os.path.join(root, f).replace('\\', '/')
            if p not in existing_paths and p not in excluded_paths:
                all_candidates.append(p)

print(f"Found {len(all_candidates)} new candidate male buffalo images matching keywords!")

new_verified_males = []

for p in all_candidates:
    fn = os.path.basename(p).lower()
    # Skip obvious cows/calves/collages
    if any(k in fn for k in ['calf', 'heifer', 'cow', 'dairy', 'milk', 'mother', 'female', 'dam']):
        continue
    try:
        im_pil = Image.open(p).convert('RGB')
        w, h = im_pil.size
        im_cv = cv2.imread(p)
        if im_cv is None or w < 150 or h < 150:
            continue
    except Exception:
        continue

    # Quality check
    ok_q, _ = pipeline.check_image_quality(im_pil, im_cv)
    if not ok_q:
        continue

    # Detect animal
    cnt, cow_dets, other, is_single, r = pipeline.detect_and_count_animals(im_pil, w, h)
    if cnt != 1 or len(cow_dets) == 0:
        continue

    # Species check: must be buffalo
    img_t = pipeline.transform(im_pil).unsqueeze(0).to(pipeline.device)
    with torch.no_grad():
        f_res = pipeline.resnet(img_t).cpu().numpy()
        f_mob = torch.flatten(pipeline.mobilenet_feat(img_t), 1).cpu().numpy()
        f_sp = np.hstack([f_res, f_mob])
    sp_pred = pipeline.species_clf.predict(f_sp)[0]
    sp_name = pipeline.rev_species_map.get(int(sp_pred), 'Unknown')
    if sp_name != 'Buffalo':
        continue

    # Full body check
    top_box = cow_dets[0][2]
    is_full, full_reason = pipeline.check_full_body_visibility(im_pil, im_cv, cow_dets, w, h)
    if not is_full:
        continue

    # Anatomical visibility
    is_anat, anat_reason = pipeline.check_anatomical_visibility(im_pil, im_cv, cow_dets)
    if not is_anat:
        continue

    new_verified_males.append({
        'path': p,
        'size': (w, h),
        'box': [float(x) for x in top_box]
    })
    print(f"  [NEW MALE] {p} ({w}x{h}) - Passed all verification!")

print(f"\nTotal new verified male buffaloes found: {len(new_verified_males)}")
