import os
import sys
import json
import hashlib
from PIL import Image
import numpy as np

print("=" * 80)
print("COMPREHENSIVE DATASET AUDIT FOR BUFFALO GENDER")
print("=" * 80)

# Check all images in buffalo/ and gender_dataset/buffalo/ and data/new_images/
catalog_path = 'models/buffalo_catalog_audited.json'
with open(catalog_path, 'r') as f:
    catalog = json.load(f)

print(f"Loaded existing catalog entries: {len(catalog)}")

# Compute perceptual hash (dHash) and MD5 for finding duplicates / near-duplicates
def dhash(image, hash_size=8):
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(image.getdata(), dtype=float).reshape((hash_size, hash_size + 1))
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def md5_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Analyze catalog
females = [c for c in catalog if c['gender'] == 'Female']
males = [c for c in catalog if c['gender'] == 'Male']

print(f"Existing Catalog: Female={len(females)}, Male={len(males)}, Total={len(catalog)}")

# Check duplicate hashes within catalog
seen_dhashes = {}
seen_md5s = {}
duplicate_groups = []

for c in catalog:
    p = c['primary']
    if not os.path.exists(p):
        print(f"Warning: primary path missing: {p}")
        continue
    img = Image.open(p)
    dh = dhash(img)
    m = md5_file(p)
    
    if dh in seen_dhashes:
        duplicate_groups.append((seen_dhashes[dh], c))
    else:
        seen_dhashes[dh] = c
    
    if m in seen_md5s:
        pass # exact file duplicate
    else:
        seen_md5s[m] = c

print(f"\nDuplicate / Near-duplicate pairs in catalog: {len(duplicate_groups)}")
for prev, curr in duplicate_groups[:10]:
    print(f"  - Group: {prev['base_id']} ({prev['gender']}, {prev['primary']}) <-> {curr['base_id']} ({curr['gender']}, {curr['primary']})")

# Let's also scan all files in buffalo/ to see what files exist, which ones have visible anatomy, which ones are males/females
all_buffalo_files = []
for root, dirs, files in os.walk('buffalo'):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            all_buffalo_files.append(os.path.join(root, f))

print(f"\nTotal image files in buffalo/ tree: {len(all_buffalo_files)}")
