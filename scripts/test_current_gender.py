import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline import LivestockMLPipeline

pipe = LivestockMLPipeline()
pipe.load_models()

samples = [
    ('Male Cow 0', 'data/new_images/malecow.jpg', 'Male'),
    ('Male Cow 1', 'data/new_images/malecow1.jpg', 'Male'),
    ('Male Cow 2', 'data/new_images/malecow2.jpg', 'Male'),
    ('Female Cow 0', 'data/new_images/femalecow.jpg', 'Female'),
    ('Female Cow 1', 'data/new_images/femalecow1.jpg', 'Female'),
    ('Female Cow 2', 'data/new_images/femalecow2.jpg', 'Female'),
    ('Male Buff 0', 'data/new_images/malebuffolo.jpg', 'Male'),
    ('Male Buff 1', 'data/new_images/malebuffolo1.jpg', 'Male'),
    ('Male Buff 2', 'data/new_images/malebuffolo2.jpg', 'Male'),
    ('Male Buff 3', 'data/new_images/malebuffolo3.jpg', 'Male'),
    ('Female Buff 0', 'data/new_images/femalebuffolo.jpg', 'Female'),
    ('Female Buff 1', 'data/new_images/femalebuffolo1.jpg', 'Female'),
    ('Female Buff 2', 'data/new_images/femalebuffolo2.jpg', 'Female'),
]

print("="*90)
print(f"{'Label':15s} | {'Exp':6s} | {'Predicted Gender':45s} | {'Conf':6s} | {'Match'}")
print("="*90)
for label, p, exp in samples:
    if os.path.exists(p):
        res = pipe.predict_image(p)
        gender = res.get('gender')
        conf = res.get('gender_confidence', 0.0)
        match = (gender == exp)
        print(f"{label:15s} | {exp:6s} | {str(gender):45s} | {conf:<6.4f} | {match}")
    else:
        print(f"{label:15s} | File not found: {p}")
