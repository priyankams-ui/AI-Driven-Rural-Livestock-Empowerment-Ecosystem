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

print("=" * 80)
print("COMPREHENSIVE BUFFALO GENDER DATASET AUDIT & CLEANING PIPELINE")
print("=" * 80)

# Perceptual hash for deduplication
def dhash(image, hash_size=8):
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(image.get_flattened_data(), dtype=float).reshape((hash_size, hash_size + 1))
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

# MD5 for exact duplicates
def md5_file(fname):
    h = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

# Load existing catalogs
with open('models/buffalo_catalog_verified_clean.json', 'r') as f:
    existing_catalog = json.load(f)

print(f"Loaded existing catalog entries: {len(existing_catalog)}")

# Exclusions log
exclusions = []
verified_clean = []
seen_dhashes = {}
seen_md5s = {}

# Audit each entry in existing catalog first
print("\n--- AUDITING EXISTING CATALOG ENTRIES ---")
for entry in existing_catalog:
    p = entry['primary'].replace('\\', '/')
    bid = entry['base_id']
    claimed_gender = entry['gender']
    
    # Check existence
    if not os.path.exists(p):
        exclusions.append({
            'path': p,
            'base_id': bid,
            'claimed_gender': claimed_gender,
            'reason': 'File does not exist'
        })
        continue
        
    fn = os.path.basename(p).lower()
    
    # 1. Non-buffalo or Cow checks
    if any(k in fn for k in ['ai_cow_dance', 'gir_fighter_cow', 'punganur_cow', 'cross_between_cows', 'gaay_bhens_dekho', 'cow_videos', 'cows_cattles_ox', 'cattles_ox']):
        exclusions.append({
            'path': p,
            'base_id': bid,
            'claimed_gender': claimed_gender,
            'reason': 'Non-buffalo image / Cow / Multi-species collage / AI animation'
        })
        continue

    # 2. Calf / Immature animals where gender cannot be established visually
    if any(k in fn for k in ['calf', 'heifer']):
        exclusions.append({
            'path': p,
            'base_id': bid,
            'claimed_gender': claimed_gender,
            'reason': 'Immature calf/heifer without visible developed reproductive anatomy'
        })
        continue

    # 3. Can PIL and OpenCV open it?
    try:
        im_pil = Image.open(p).convert('RGB')
        w, h = im_pil.size
        im_cv = cv2.imread(p)
        if im_cv is None or w < 120 or h < 120:
            exclusions.append({
                'path': p,
                'base_id': bid,
                'claimed_gender': claimed_gender,
                'reason': f'Low resolution or corrupt image ({w}x{h})'
            })
            continue
    except Exception as e:
        exclusions.append({
            'path': p,
            'base_id': bid,
            'claimed_gender': claimed_gender,
            'reason': f'Cannot open image: {str(e)}'
        })
        continue

    # 4. Exact & Perceptual Duplicates
    m = md5_file(p)
    dh = dhash(im_pil)
    if m in seen_md5s:
        exclusions.append({
            'path': p,
            'base_id': bid,
            'claimed_gender': claimed_gender,
            'reason': f'Exact duplicate of {seen_md5s[m]}'
        })
        continue
    if dh in seen_dhashes:
        # Check hamming distance
        orig_dh, orig_p = seen_dhashes[dh]
        exclusions.append({
            'path': p,
            'base_id': bid,
            'claimed_gender': claimed_gender,
            'reason': f'Near duplicate / perceptual duplicate of {orig_p}'
        })
        continue
        
    # 5. Image quality & Real vs AI check
    is_good_q, q_err = pipeline.check_image_quality(im_pil, im_cv)
    if not is_good_q:
        exclusions.append({
            'path': p,
            'base_id': bid,
            'claimed_gender': claimed_gender,
            'reason': f'Image quality rejected: {q_err}'
        })
        continue

    # 6. Animal detection & Single animal check
    cnt, cow_dets, other, is_single, r = pipeline.detect_and_count_animals(im_pil, w, h)
    if cnt == 0 or len(cow_dets) == 0:
        exclusions.append({
            'path': p,
            'base_id': bid,
            'claimed_gender': claimed_gender,
            'reason': 'No bovine detected'
        })
        continue
    if cnt > 1:
        exclusions.append({
            'path': p,
            'base_id': bid,
            'claimed_gender': claimed_gender,
            'reason': f'Multiple animals detected ({cnt} animals)'
        })
        continue

    # 7. Species check (Must be Buffalo, not Cow)
    img_t = pipeline.transform(im_pil).unsqueeze(0).to(pipeline.device)
    with torch.no_grad():
        f_res = pipeline.resnet(img_t).cpu().numpy()
        f_mob = torch.flatten(pipeline.mobilenet_feat(img_t), 1).cpu().numpy()
        f_sp = np.hstack([f_res, f_mob])
    sp_pred = pipeline.species_clf.predict(f_sp)[0]
    sp_name = pipeline.rev_species_map.get(int(sp_pred), 'Unknown')
    if sp_name != 'Buffalo':
        exclusions.append({
            'path': p,
            'base_id': bid,
            'claimed_gender': claimed_gender,
            'reason': f'Classified as {sp_name}, not Buffalo'
        })
        continue

    # 8. Full body visibility check
    top_box = cow_dets[0][2]
    is_full, full_reason = pipeline.check_full_body_visibility(im_pil, im_cv, cow_dets, w, h)
    if not is_full:
        exclusions.append({
            'path': p,
            'base_id': bid,
            'claimed_gender': claimed_gender,
            'reason': f'Full body visibility failed: {full_reason}'
        })
        continue

    # 9. Anatomical visibility
    is_anat, anat_reason = pipeline.check_anatomical_visibility(im_pil, im_cv, cow_dets)
    if not is_anat:
        exclusions.append({
            'path': p,
            'base_id': bid,
            'claimed_gender': claimed_gender,
            'reason': f'Anatomical visibility failed: {anat_reason}'
        })
        continue

    # Verified passed
    seen_md5s[m] = p
    seen_dhashes[dh] = (dh, p)
    verified_clean.append({
        'base_id': bid,
        'gender': claimed_gender,
        'primary': p,
        'breed': entry.get('breed', 'buffalo'),
        'box': [float(x) for x in top_box]
    })

print(f"\nAudit complete for existing catalog:")
print(f"  - Usable & Verified: {len(verified_clean)}")
print(f"  - Excluded: {len(exclusions)}")
fems = [x for x in verified_clean if x['gender'] == 'Female']
males = [x for x in verified_clean if x['gender'] == 'Male']
print(f"  - Verified Females: {len(fems)}")
print(f"  - Verified Males:   {len(males)}")

with open('models/buffalo_audit_exclusions.json', 'w') as f:
    json.dump(exclusions, f, indent=2)
print("Saved exclusions to models/buffalo_audit_exclusions.json")
