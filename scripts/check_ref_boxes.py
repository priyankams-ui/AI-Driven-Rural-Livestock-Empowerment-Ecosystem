import json

with open('models/authoritative_buffalo_gender_catalog.json', 'r') as f:
    cat = json.load(f)

for c in cat:
    if 'data/new_images' in c['primary']:
        print(f"{c['primary']:35s} | {c['gender']:6s} | Box: {c.get('box')}")
