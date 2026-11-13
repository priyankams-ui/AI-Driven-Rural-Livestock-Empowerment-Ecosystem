import os
import sys
import json
from PIL import Image

with open('models/buffalo_catalog_audited.json', 'r') as f:
    catalog = json.load(f)

print("=" * 80)
print("AUDITING EVERY CATALOG ENTRY FOR LABEL CORRUPTION AND ANATOMY")
print("=" * 80)

mislabeled_or_suspicious = []
for i, c in enumerate(catalog):
    p = c['primary']
    fn = os.path.basename(p).lower()
    gender = c['gender']
    
    # Check keyword conflicts in filename as an initial filter
    has_male_kw = any(k in fn for k in ['bull', 'sire', 'male', 'he_buffalo', 'ox'])
    has_female_kw = any(k in fn for k in ['cow', 'female', 'she_buffalo', 'dairy', 'milking', 'dam', 'mother', 'pregnant'])
    
    suspicious = False
    reasons = []
    
    if gender == 'Male' and has_female_kw:
        suspicious = True
        reasons.append(f"Labeled Male but filename has female keywords ({[k for k in ['cow', 'female', 'she_buffalo', 'dairy', 'milking', 'dam', 'mother', 'pregnant'] if k in fn]})")
        
    if gender == 'Female' and has_male_kw and not has_female_kw:
        suspicious = True
        reasons.append(f"Labeled Female but filename has male keywords ({[k for k in ['bull', 'sire', 'male', 'he_buffalo', 'ox'] if k in fn]})")
        
    if suspicious:
        mislabeled_or_suspicious.append((i, c['base_id'], gender, p, reasons))

print(f"\nTotal suspicious entries flagged for visual review: {len(mislabeled_or_suspicious)}")
for idx, bid, g, p, r in mislabeled_or_suspicious:
    print(f"[{idx:3d}] {bid} (Label: {g}) -> {p}\n      Reasons: {r}")
