import os
import sys

sys.path.insert(0, '.')
from ml_pipeline import pipeline

images = [
    'data/new_images/femalebuffolo.jpg',
    'data/new_images/femalebuffolo1.jpg',
    'data/new_images/femalebuffolo2.jpg',
    'data/new_images/malebuffolo.jpg',
    'data/new_images/malebuffolo1.jpg',
    'data/new_images/malebuffolo2.jpg',
    'data/new_images/malebuffolo3.jpg',
    'gender_dataset/buffalo/female/banni_female_0caae.jpg',
    'gender_dataset/buffalo/female/murrah_female_original.jpg',
    'gender_dataset/buffalo/male/banni_bull_0.jpg',
    'gender_dataset/buffalo/male/bhadawari_bull.jpg',
    'gender_dataset/buffalo/male/jafarabadi_bull_punjab.jpg',
    'test_artifacts/half_female_buffalo.jpg',
    'test_artifacts/half_male_buffalo.jpg',
    'test_artifacts/buffalo_head_only.jpg',
    'test_artifacts/blurry_buffalo.jpg',
    'test_artifacts/cropped_buffalo.jpg',
    'test_artifacts/full_buffalo.jpg'
]

print("=" * 100)
print(f"{'Image':40s} | {'Status':8s} | {'Species':8s} | {'Gender':10s} | {'Conf':5s} | {'Message'}")
print("=" * 100)
for img in images:
    if os.path.exists(img):
        res = pipeline.predict_image(img)
        print(f"{img:40s} | {str(res.get('status')):8s} | {str(res.get('species')):8s} | {str(res.get('gender')):10s} | {res.get('gender_confidence', 0.0):<5.2f} | {res.get('gender_message', '')[:40]}")
    else:
        print(f"{img:40s} | NOT FOUND")
