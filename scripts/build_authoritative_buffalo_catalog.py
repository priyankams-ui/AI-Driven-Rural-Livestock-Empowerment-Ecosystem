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

# 1. Load verified clean list from audit
with open('models/buffalo_catalog_verified_clean.json') as f:
    orig_clean = json.load(f)

with open('models/buffalo_audit_exclusions.json') as f:
    exclusions = json.load(f)

excluded_paths = set(e['path'].replace('\\', '/') for e in exclusions)

# Collect all candidates
candidates = []

# Add non-excluded from orig_clean
for c in orig_clean:
    p = c['primary'].replace('\\', '/')
    if p not in excluded_paths and os.path.exists(p):
        candidates.append({
            'base_id': c['base_id'],
            'gender': c['gender'],
            'primary': p,
            'source': 'clean_catalog'
        })

# Check gender_dataset/buffalo/male
for f in os.listdir('gender_dataset/buffalo/male'):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        p = f'gender_dataset/buffalo/male/{f}'
        if p not in excluded_paths and os.path.exists(p):
            candidates.append({
                'base_id': f'GD_Male_{f}',
                'gender': 'Male',
                'primary': p,
                'source': 'gender_dataset'
            })

# Check gender_dataset/buffalo/female
for f in os.listdir('gender_dataset/buffalo/female'):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        p = f'gender_dataset/buffalo/female/{f}'
        if p not in excluded_paths and os.path.exists(p):
            candidates.append({
                'base_id': f'GD_Fem_{f}',
                'gender': 'Female',
                'primary': p,
                'source': 'gender_dataset'
            })

# Add the 4 new verified males
new_males = [
    'buffalo/banni/baani-buffalo-bulls_0.jpg',
    'buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg',
    'buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg',
    'buffalo/nili-ravi/Neeli_Ravi_buffalo_bull_.jpg'
]
for p in new_males:
    if os.path.exists(p):
        candidates.append({
            'base_id': f'New_Male_{os.path.basename(p)}',
            'gender': 'Male',
            'primary': p,
            'source': 'breed_folder'
        })

# Check Semen station bulls
semen_bulls = [
    'buffalo/Jaffarabadi/JF-40493_Alamadhi_Semen_Station.jpg',
    'buffalo/Jaffarabadi/JF-50262_Rahuri_Semen_Station.jpg',
    'buffalo/Jaffarabadi/JF-50400_Rahuri_Semen_Station.jpg',
    'buffalo/Jaffarabadi/Tulsidas_-_BAIF_Sire_Directory.jpg',
    'buffalo/nagpuri/Thakya_-_BAIF_Sire_Directory.jpg',
    'buffalo/bhadwari/Bhadavari_Archives_-_BAIF_Sire_Directory.jpg'
]
for p in semen_bulls:
    if os.path.exists(p):
        candidates.append({
            'base_id': f'Semen_Bull_{os.path.basename(p)}',
            'gender': 'Male',
            'primary': p,
            'source': 'semen_station'
        })

print(f"Total raw candidates collected: {len(candidates)}")

def dhash(image, hash_size=8):
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(image.get_flattened_data(), dtype=float).reshape((hash_size, hash_size + 1))
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

seen_hashes = {}
seen_paths = set()
authoritative_catalog = []

for c in candidates:
    p = c['primary']
    if p in seen_paths:
        continue
    try:
        im = Image.open(p).convert('RGB')
        w, h = im.size
        im_cv = cv2.imread(p)
        if im_cv is None or w < 120 or h < 120:
            continue
    except Exception:
        continue

    # Quality check
    ok_q, _ = pipeline.check_image_quality(im, im_cv)
    if not ok_q:
        continue

    # Animal detection
    cnt, cow_dets, other, is_single, r = pipeline.detect_and_count_animals(im, w, h)
    if cnt != 1 or len(cow_dets) == 0:
        continue

    # Species check: must be Buffalo
    img_t = pipeline.transform(im).unsqueeze(0).to(pipeline.device)
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
    is_full, full_reason = pipeline.check_full_body_visibility(im, im_cv, cow_dets, w, h)
    if not is_full:
        continue

    # Anatomical visibility
    is_anat, anat_reason = pipeline.check_anatomical_visibility(im, im_cv, cow_dets)
    if not is_anat:
        continue

    # Deduplicate
    dh = dhash(im)
    if dh in seen_hashes:
        continue
    seen_hashes[dh] = p
    seen_paths.add(p)

    authoritative_catalog.append({
        'base_id': f"Buff_Auth_{len(authoritative_catalog):03d}",
        'gender': c['gender'],
        'primary': p,
        'box': [float(x) for x in top_box]
    })

fems = [x for x in authoritative_catalog if x['gender'] == 'Female']
males = [x for x in authoritative_catalog if x['gender'] == 'Male']
print(f"\nAuthoritative Verified Clean Buffalo Catalog built:")
print(f"  Total Animals: {len(authoritative_catalog)}")
print(f"  Female Buffaloes: {len(fems)}")
print(f"  Male Buffaloes:   {len(males)}")

with open('models/authoritative_buffalo_gender_catalog.json', 'w') as f:
    json.dump(authoritative_catalog, f, indent=2)

print("Saved to models/authoritative_buffalo_gender_catalog.json")
