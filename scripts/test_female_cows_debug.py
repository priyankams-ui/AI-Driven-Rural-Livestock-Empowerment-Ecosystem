import sys
sys.path.insert(0, '.')
import glob
from ml_pipeline import pipeline

images = [
    'data/new_images/femalecow.jpg',
    'data/new_images/femalecow1.jpg',
    'data/new_images/femalecow2.jpg',
    'Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg',
    'Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg',
    'Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg',
    'Cattle Breeds/Jersey cattle/Jerseycattle0.jpg',
    'Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg',
]

print(f'Testing {len(images)} female cow images:')
for img in images:
    res = pipeline.predict_image(img)
    print(f"{img:55s} -> status={res.get('status')}, species={res.get('species')}, gender={res.get('gender')}, conf={res.get('gender_confidence')}, error={res.get('error')}")
