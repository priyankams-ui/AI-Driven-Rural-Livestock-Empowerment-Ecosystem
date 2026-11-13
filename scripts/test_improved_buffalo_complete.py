import os
import sys
import json
from PIL import Image

sys.path.insert(0, '.')
from ml_pipeline import pipeline

print("=" * 100)
print("TESTING FULL PIPELINE ON BUFFALO REFERENCE IMAGES & SUITE")
print("=" * 100)

test_suite = [
    # Reference images from data/new_images/
    ('data/new_images/femalebuffolo.jpg', 'Female Buffalo 0', 'Female'),
    ('data/new_images/femalebuffolo1.jpg', 'Female Buffalo 1', 'Female'),
    ('data/new_images/femalebuffolo2.jpg', 'Female Buffalo 2 (White bg)', 'Female'),
    ('data/new_images/malebuffolo.jpg', 'Male Buffalo 0', 'Male'),
    ('data/new_images/malebuffolo1.jpg', 'Male Buffalo 1', 'Male'),
    ('data/new_images/malebuffolo2.jpg', 'Male Buffalo 2', 'Male'),
    ('data/new_images/malebuffolo3.jpg', 'Male Buffalo 3 (Heavy bull)', 'Male/Uncertain'),
    
    # Dataset images
    ('gender_dataset/buffalo/female/banni_female_0caae.jpg', 'Banni Female Buffalo', 'Female'),
    ('gender_dataset/buffalo/female/murrah_female_original.jpg', 'Murrah Female Buffalo', 'Female'),
    ('gender_dataset/buffalo/male/jafarabadi_bull_punjab.jpg', 'Jafarabadi Male Buffalo', 'Male'),
    
    # Non-livestock or Partial Buffalo Images
    ('test_artifacts/full_buffalo.jpg', 'Full Buffalo Test Sample', 'Female/Male/Uncertain'),
    ('test_artifacts/buffalo_head_only.jpg', 'Buffalo Head Only', 'Reject/Uncertain'),
    ('test_artifacts/blurry_buffalo.jpg', 'Blurry Buffalo', 'Reject/Uncertain'),
    ('test_artifacts/cropped_buffalo.jpg', 'Cropped Buffalo', 'Reject/Uncertain'),
    ('test_artifacts/buffalo_group.jpg', 'Buffalo Group', 'Reject/Multiple'),
    
    # Cattle images to verify cattle model is completely UNTOUCHED and WORKING
    ('data/new_images/femalecow.jpg', 'Female Cow 0', 'Female'),
    ('data/new_images/femalecow1.jpg', 'Female Cow 1', 'Female'),
    ('data/new_images/malecow.jpg', 'Male Cow 0', 'Male'),
    ('data/new_images/malecow1.jpg', 'Male Cow 1', 'Male'),
]

print(f"{'Description':35s} | {'Status':8s} | {'Species':8s} | {'Gender':10s} | {'Conf':5s} | {'Message'}")
print("-" * 100)

for p, desc, expected in test_suite:
    if not os.path.exists(p):
        print(f"{desc:35s} | MISSING FILE: {p}")
        continue
    res = pipeline.predict_image(p)
    status = res.get('status', 'error')
    species = res.get('species', 'Unknown')
    gender = res.get('gender', 'None')
    g_conf = res.get('gender_confidence', 0.0)
    msg = res.get('gender_message') or res.get('error', '')
    print(f"{desc:35s} | {status:8s} | {species:8s} | {str(gender):10s} | {g_conf:<5.2f} | {msg[:45]}")

print("=" * 100)
