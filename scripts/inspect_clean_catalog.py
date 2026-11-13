import json
import os

with open('models/buffalo_catalog_verified_clean.json') as f:
    cat = json.load(f)

print(f"Total entries: {len(cat)}")
females = [c for c in cat if c['gender'] == 'Female']
males = [c for c in cat if c['gender'] == 'Male']
print(f"Females: {len(females)}, Males: {len(males)}")

for i, c in enumerate(cat):
    print(f"{i:3d}: {c['base_id']} | {c['gender']:6s} | {c['primary']}")
