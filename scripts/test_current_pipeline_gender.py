import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_pipeline import pipeline

test_imgs = [
    ('data/new_images/malecow.jpg', 'Cow', 'Male'),
    ('data/new_images/malecow1.jpg', 'Cow', 'Male'),
    ('data/new_images/malecow2.jpg', 'Cow', 'Male'),
    ('data/new_images/femalecow.jpg', 'Cow', 'Female'),
    ('data/new_images/femalecow1.jpg', 'Cow', 'Female'),
    ('data/new_images/femalecow2.jpg', 'Cow', 'Female'),
    ('data/new_images/malebuffolo.jpg', 'Buffalo', 'Male'),
    ('data/new_images/malebuffolo1.jpg', 'Buffalo', 'Male'),
    ('data/new_images/malebuffolo3.jpg', 'Buffalo', 'Male'),
    ('buffalo/Jaffarabadi/Black_Jafarabadi_Bull_Semen_at_Best_.jpg', 'Buffalo', 'Male'),
    ('buffalo/murrah/Murrah_BullMurrah_Buffalo_BullsBull_.jpg', 'Buffalo', 'Male'),
    ('data/new_images/femalebuffolo.jpg', 'Buffalo', 'Female'),
    ('data/new_images/femalebuffolo1.jpg', 'Buffalo', 'Female'),
    ('buffalo/surti/Milking_Murrah_Buffalo_Female_750kg_.jpg', 'Buffalo', 'Female'),
    ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 'Buffalo', 'Female'),
    ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 'Buffalo', 'Female'),
    ('gender_dataset/uncertain/cow_head_only.jpg', 'Cow', 'Uncertain'),
    ('gender_dataset/uncertain/buffalo_head_only.jpg', 'Buffalo', 'Uncertain'),
]

for path, exp_sp, exp_g in test_imgs:
    if not os.path.exists(path):
        print(f"{path:55s} | FILE NOT FOUND")
        continue
    res = pipeline.predict_image(path)
    if 'error' in res or not res.get('is_valid', True):
        err = res.get('error', 'Rejected')
        status = res.get('gender_status', 'undetermined')
        p_g = res.get('gender', 'Uncertain')
        match = "OK" if exp_g == "Uncertain" and (p_g == "Uncertain" or not res.get('is_valid')) else "FAIL"
        print(f"{path:55s} | Exp: {exp_g:9s} | Got: {str(p_g):9s} | status: {status} | rejected: {err} | [{match}]")
    else:
        p_sp = res.get('species')
        p_g = res.get('gender')
        conf = res.get('gender_confidence')
        status = res.get('gender_status')
        match = "OK" if p_g == exp_g else "FAIL"
        print(f"{path:55s} | Exp: {exp_g:9s} | Got: {str(p_g):9s} (conf: {conf}) | [{match}]")
