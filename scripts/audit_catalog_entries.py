import os
import sys
import json
from PIL import Image

with open('models/buffalo_catalog_audited.json', 'r') as f:
    catalog = json.load(f)

print(f"Total catalog entries to audit: {len(catalog)}")

female_entries = [c for c in catalog if c['gender'] == 'Female']
male_entries = [c for c in catalog if c['gender'] == 'Male']

print(f"Female count: {len(female_entries)}")
print(f"Male count: {len(male_entries)}")

# Let's inspect the files and breeds
for i, c in enumerate(catalog):
    p = c['primary']
    exists = os.path.exists(p)
    sz = Image.open(p).size if exists else (0, 0)
    print(f"[{i:3d}] {c['base_id']:15s} | {c['gender']:6s} | {c['breed']:12s} | Size: {sz} | File: {p}")
