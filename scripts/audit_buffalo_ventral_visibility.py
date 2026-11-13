import os
import sys
import json
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, '.')
from ml_pipeline import pipeline

with open('models/buffalo_catalog_raw.json', 'r') as f:
    cat = json.load(f)

print(f"Auditing anatomical visibility for {len(cat)} buffaloes...")

audited_catalog = []
excluded_count = 0

os.makedirs('test_artifacts/buffalo_audit_ventral', exist_ok=True)

for i, a in enumerate(cat):
    p = a['primary']
    im = Image.open(p).convert('RGB')
    w, h = im.size
    
    # Run animal detection to get precise box
    cnt, dets, other, is_single, reason = pipeline.detect_and_count_animals(im, w, h)
    box = dets[0][2] if dets else None
    
    if box is not None:
        bx1, by1, bx2, by2 = box
        bw, bh = bx2 - bx1, by2 - by1
        y1 = int(by1 + 0.40 * bh)
        y2 = min(h, max(int(by1 + 0.98 * bh), int(by2 + 0.05 * bh), int(h * 0.94)))
        x1 = int(bx1 + 0.12 * bw)
        x2 = int(bx1 + 0.88 * bw)
    else:
        y1, y2 = int(h * 0.40), int(h * 0.94)
        x1, x2 = int(w * 0.12), int(w * 0.88)
        
    y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
    x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
    
    vc = im.crop((x1, y1, x2, y2))
    vc_cv = cv2.cvtColor(np.array(vc), cv2.COLOR_RGB2BGR)
    
    # Check if ventral crop has enough pixels and contrast
    vc_gray = cv2.cvtColor(vc_cv, cv2.COLOR_BGR2GRAY)
    vc_h, vc_w = vc_gray.shape
    if vc_h < 35 or vc_w < 50:
        excluded_count += 1
        continue
        
    # Check if submerged or obscured: if lower half of ventral crop is completely uniform (e.g. water/grass)
    lower_half = vc_gray[int(vc_h * 0.5):, :]
    std_lower = np.std(lower_half)
    if std_lower < 8.0:
        # Uniform occlusion (deep water/mud)
        excluded_count += 1
        continue
        
    # Check anatomical visibility via pipeline
    is_anat, anat_reason = pipeline.check_anatomical_visibility(im, cv2.imread(p), dets)
    
    # Save crop for inspection
    gid = a['base_id']
    save_p = f"test_artifacts/buffalo_audit_ventral/{gid}_{a['gender']}.jpg"
    vc.save(save_p)
    
    a['ventral_box'] = [x1, y1, x2, y2]
    a['anat_visible'] = is_anat
    a['anat_reason'] = anat_reason
    audited_catalog.append(a)

print(f"Auditing complete: {len(audited_catalog)} passed visibility checks, {excluded_count} excluded (occluded/small/submerged).")
f_passed = sum(1 for a in audited_catalog if a['gender'] == 'Female')
m_passed = sum(1 for a in audited_catalog if a['gender'] == 'Male')
print(f"Passed: Female={f_passed}, Male={m_passed}")

with open('models/buffalo_catalog_audited.json', 'w') as f:
    json.dump(audited_catalog, f, indent=2)
print("Saved models/buffalo_catalog_audited.json")
