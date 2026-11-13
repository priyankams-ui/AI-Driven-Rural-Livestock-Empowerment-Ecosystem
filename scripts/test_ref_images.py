import os, sys
sys.path.insert(0, '.')
from ml_pipeline import pipeline

images = [
    'test_crop_female_brown_swiss.jpg',
    'test_crop_female_jersey.jpg',
    'test_crop_male_bull.jpg',
    'test_crop_male_steer.jpg',
    'data/new_images/femalecow.jpg',
    'data/new_images/femalecow1.jpg',
    'data/new_images/femalecow2.jpg',
    'data/new_images/malecow.jpg',
    'data/new_images/malecow1.jpg',
    'data/new_images/malecow2.jpg',
    'Cattle Breeds/Brown Swiss cattle/BrownSwisscattle101.jpg',
    'Cattle Breeds/Jersey cattle/Jerseycattle101.jpg',
    'Cattle Breeds/Jersey cattle/Jerseycattle103.jpg',
    'Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg',
    'Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg',
]

for img in images:
    if os.path.exists(img):
        res = pipeline.predict_image(img)
        print(f"{img:55s} -> status={res.get('status')}, species={res.get('species')}, gender={res.get('gender')}, conf={res.get('gender_confidence')}")
