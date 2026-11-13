import os
import sys
import json
import hashlib
from PIL import Image
import numpy as np

# Load catalog and verified audit
with open('models/buffalo_catalog_audited.json', 'r') as f:
    old_catalog = json.load(f)

with open('models/verified_buffalo_gender_audit.json', 'r') as f:
    audit_data = json.load(f)

# Correct mislabeled items in catalog:
# 1. "buffalo/nili-ravi/HSB_Nili_Ravi_Bull-Parass_Mother_ (1).jpg" -> Female (Mother)
# 2. "buffalo/nili-ravi/HSB_Nili_Ravi_Bull-Parass_Mother_ (3).jpg" -> Female (Mother)
# 3. "buffalo/nili-ravi/Nili_Ravi_Female_Calf_Niliravi.jpg" -> Female
# 4. "buffalo/nili-ravi/Nili_Ravi_Female_Calf_niliravi_.jpg" -> Female
# 5. "buffalo/nili-ravi/Pure_Nili_Ravi_Buffaloes_Gender_Female_ (2).jpg" -> Female
# 6. "buffalo/surti/Milking_Murrah_Buffalo_Female_750kg_.jpg" -> Female
# 7. "buffalo/murrah/Murrah_Buffalo_-_Female_Breed_High_.jpg" -> Female

# Build audited clean list
clean_catalog = []
seen_paths = set()

# Deduplicate with perceptual hash
def dhash(image, hash_size=8):
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(image.get_flattened_data(), dtype=float).reshape((hash_size, hash_size + 1))
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

seen_hashes = {}

# Process old catalog items first
mislabeled_females = {
    "buffalo/nili-ravi/HSB_Nili_Ravi_Bull-Parass_Mother_ (1).jpg",
    "buffalo/nili-ravi/HSB_Nili_Ravi_Bull-Parass_Mother_ (3).jpg",
    "buffalo/nili-ravi/Nili_Ravi_Female_Calf_Niliravi.jpg",
    "buffalo/nili-ravi/Nili_Ravi_Female_Calf_niliravi_.jpg",
    "buffalo/nili-ravi/Pure_Nili_Ravi_Buffaloes_Gender_Female_ (2).jpg",
    "buffalo/surti/Milking_Murrah_Buffalo_Female_750kg_.jpg",
    "buffalo/murrah/Murrah_Buffalo_-_Female_Breed_High_.jpg",
    "buffalo/murrah/Cows_Cattles_Ox_Buffalo_Sheep_Goat_.jpg" # remove ambiguous/multiple
}

for item in old_catalog:
    p = item['primary'].replace('\\', '/')
    if not os.path.exists(p):
        continue
    if 'cows_cattles_ox' in p.lower():
        continue # skip multi-animal
        
    gender = item['gender']
    if p in mislabeled_females or any(m in p for m in mislabeled_females):
        gender = 'Female' # correct label
        
    im = Image.open(p).convert('RGB')
    dh = dhash(im)
    if dh in seen_hashes:
        continue # duplicate
    seen_hashes[dh] = p
    
    clean_catalog.append({
        'base_id': f"Clean_Buff_{len(clean_catalog):03d}",
        'gender': gender,
        'primary': p,
        'breed': item.get('breed', 'buffalo')
    })

# Also add verified items from audit_data
for f_path in audit_data['verified_females']:
    p = f_path.replace('\\', '/')
    if not os.path.exists(p):
        continue
    im = Image.open(p).convert('RGB')
    dh = dhash(im)
    if dh in seen_hashes:
        continue
    seen_hashes[dh] = p
    clean_catalog.append({
        'base_id': f"Clean_Buff_{len(clean_catalog):03d}",
        'gender': 'Female',
        'primary': p,
        'breed': 'buffalo'
    })

for m_path in audit_data['verified_males']:
    p = m_path.replace('\\', '/')
    if not os.path.exists(p):
        continue
    im = Image.open(p).convert('RGB')
    dh = dhash(im)
    if dh in seen_hashes:
        continue
    seen_hashes[dh] = p
    clean_catalog.append({
        'base_id': f"Clean_Buff_{len(clean_catalog):03d}",
        'gender': 'Male',
        'primary': p,
        'breed': 'buffalo'
    })

females = [c for c in clean_catalog if c['gender'] == 'Female']
males = [c for c in clean_catalog if c['gender'] == 'Male']

print(f"Total Verified Clean Buffalo Catalog: {len(clean_catalog)}")
print(f"  - Verified Females: {len(females)}")
print(f"  - Verified Males:   {len(males)}")

with open('models/buffalo_catalog_verified_clean.json', 'w') as f:
    json.dump(clean_catalog, f, indent=2)

print("Saved clean catalog to models/buffalo_catalog_verified_clean.json")
