import sys
sys.path.insert(0, '.')
from ml_pipeline import pipeline

paths = [
    'data/new_images/femalecow.jpg',
    'data/new_images/malecow.jpg',
    'data/new_images/femalebuffolo2.jpg',
    'data/new_images/malebuffolo1.jpg',
    'gender_dataset/uncertain/cow_head_only.jpg',
    'gender_dataset/uncertain/buffalo_head_only.jpg',
    'gender_dataset/uncertain/calf_headshot_fence.jpg',
    'test_artifacts/half_female_cow.jpg',
    'test_artifacts/half_male_cow.jpg',
    'test_artifacts/half_female_buffalo.jpg',
    'test_artifacts/half_male_buffalo.jpg',
]

for p in paths:
    res = pipeline.predict_image(p)
    print(f"{p:45s} -> status={res.get('status')} sp={res.get('species')} g={res.get('gender')} err={res.get('error')}")
