import os
import sys
import json
import hashlib
import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, '.')
from ml_pipeline import pipeline

print("=" * 80)
print("COMPREHENSIVE AUDIT OF ALL BUFFALO IMAGES IN REPOSITORY")
print("=" * 80)

# Check quality check on white background images
def check_image_quality_robust(image_pil, image_cv):
    w, h = image_pil.size
    if w < 80 or h < 80:
        return False, "Image resolution too low"
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    if mean_val < 25.0:
        return False, "Image too dark"
    # Check overexposure: only reject if the animal itself is washed out, not just a white studio background
    # If standard deviation is good (> 25), the image has sharp foreground contrast even with white background
    if mean_val > 245.0 and std_val < 15.0:
        return False, "Image is completely blank or blown out white"
    if std_val < 10.0:
        return False, "Image lacks contrast"
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 30.0:
        return False, "Image too blurry"
    return True, "Quality OK"

# 1. Gather all buffalo images
all_images = []
for root, dirs, files in os.walk('buffalo'):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            all_images.append(os.path.join(root, f))

for root, dirs, files in os.walk('gender_dataset/buffalo'):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            all_images.append(os.path.join(root, f))

for root, dirs, files in os.walk('data/new_images'):
    for f in files:
        if 'buff' in f.lower() and f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            all_images.append(os.path.join(root, f))

all_images = sorted(list(set(all_images)))
print(f"Total candidate buffalo image files found: {len(all_images)}")

# Perceptual hash for near-duplicate detection
def dhash(image, hash_size=8):
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(image.getdata(), dtype=float).reshape((hash_size, hash_size + 1))
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

# Audit results container
audit_results = {
    'total_candidates': len(all_images),
    'quality_failed': 0,
    'non_single_or_unclear': 0,
    'species_not_buffalo': 0,
    'full_body_failed': 0,
    'anatomy_occluded': 0,
    'duplicates_removed': 0,
    'mislabeled_corrected_or_removed': 0,
    'verified_females': [],
    'verified_males': [],
}

print("Running pipeline verification...")
seen_hashes = {}

for idx, p in enumerate(all_images):
    try:
        im_pil = Image.open(p).convert('RGB')
        w, h = im_pil.size
        im_cv = cv2.imread(p)
        if im_cv is None:
            im_cv = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
            
        ok_q, q_msg = check_image_quality_robust(im_pil, im_cv)
        if not ok_q:
            audit_results['quality_failed'] += 1
            continue
            
        cnt, cow_dets, other, is_single, cnt_msg = pipeline.detect_and_count_animals(im_pil, w, h)
        if cnt != 1:
            audit_results['non_single_or_unclear'] += 1
            continue
            
        is_fb, fb_msg = pipeline.check_full_body_visibility(im_pil, im_cv, cow_dets, w, h)
        if not is_fb:
            audit_results['full_body_failed'] += 1
            continue
            
        is_anat, anat_msg = pipeline.check_anatomical_visibility(im_pil, im_cv, cow_dets)
        if not is_anat:
            audit_results['anatomy_occluded'] += 1
            continue
            
        # Species classification check
        img_tensor = pipeline.transform(im_pil).unsqueeze(0).to(pipeline.device)
        with torch.no_grad():
            f_res = pipeline.resnet(img_tensor).cpu().numpy()
            f_mob = torch.flatten(pipeline.mobilenet_feat(img_tensor), 1).cpu().numpy()
            feat = np.hstack([f_res, f_mob])
        sp_pred = pipeline.species_clf.predict(feat)[0]
        sp_name = pipeline.rev_species_map.get(int(sp_pred), 'Unknown')
        if sp_name != 'Buffalo':
            audit_results['species_not_buffalo'] += 1
            continue
            
        # Check duplicate
        dh = dhash(im_pil)
        if dh in seen_hashes:
            audit_results['duplicates_removed'] += 1
            continue
        seen_hashes[dh] = p
        
        # Determine verified gender from keywords, folder structure, and catalog cross-check
        fn = os.path.basename(p).lower()
        folder = os.path.dirname(p).lower()
        
        # Check if known mislabeled
        if any(mis in p for mis in [
            'HSB_Nili_Ravi_Bull-Parass_Mother_', # Mother = female
            'Pure_Nili_Ravi_Buffaloes_Gender_Female_', # Female
            'Nili_Ravi_Female_Calf_', # Female
            'Milking_Murrah_Buffalo_Female_750kg_', # Female
            'Murrah_Buffalo_-_Female_Breed_High_' # Female
        ]):
            # Correct to female
            audit_results['verified_females'].append(p)
            continue
            
        if any(k in fn for k in ['bull', 'sire', 'malebuffolo', 'male_bull', 'he_buffalo', 'ox']) and not any(k in fn for k in ['mother', 'dam', 'female', 'cow', 'dairy', 'milking']):
            audit_results['verified_males'].append(p)
        elif any(k in fn for k in ['cow', 'female', 'femalebuffolo', 'she_buffalo', 'dairy', 'milking', 'pregnant', 'dam', 'mother']):
            audit_results['verified_females'].append(p)
        elif 'male' in folder and 'female' not in folder:
            audit_results['verified_males'].append(p)
        elif 'female' in folder:
            audit_results['verified_females'].append(p)
        else:
            # Unlabeled in breed folder without gender annotation
            pass

    except Exception as e:
        continue

print("\n" + "=" * 80)
print("AUDIT RESULTS SUMMARY:")
print(f"Total candidates scanned:       {audit_results['total_candidates']}")
print(f"Quality check rejected:         {audit_results['quality_failed']}")
print(f"Non-single / unclear animals:   {audit_results['non_single_or_unclear']}")
print(f"Full body visibility failed:    {audit_results['full_body_failed']}")
print(f"Anatomical region occluded:     {audit_results['anatomy_occluded']}")
print(f"Species not Buffalo:            {audit_results['species_not_buffalo']}")
print(f"Duplicates / Near-duplicates:   {audit_results['duplicates_removed']}")
print(f"Verified Female Buffaloes:      {len(audit_results['verified_females'])}")
print(f"Verified Male Buffaloes:        {len(audit_results['verified_males'])}")
print("=" * 80)

# Save the audited dataset list
with open('models/verified_buffalo_gender_audit.json', 'w') as f:
    json.dump({
        'verified_females': audit_results['verified_females'],
        'verified_males': audit_results['verified_males'],
        'summary': {
            'total_candidates': audit_results['total_candidates'],
            'verified_females_count': len(audit_results['verified_females']),
            'verified_males_count': len(audit_results['verified_males']),
            'duplicates_removed': audit_results['duplicates_removed'],
            'quality_rejected': audit_results['quality_failed'],
            'occluded_rejected': audit_results['anatomy_occluded'],
            'full_body_rejected': audit_results['full_body_failed']
        }
    }, f, indent=2)

print("Saved verified audit to models/verified_buffalo_gender_audit.json")
