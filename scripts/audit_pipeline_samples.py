import os, sys, pickle, torch
import numpy as np
from PIL import Image
sys.path.append(os.path.abspath('.'))
from ml_pipeline import LivestockMLPipeline

pipe = LivestockMLPipeline()
pipe.load_models()

sample_images = [
    # Male Cows
    ('Male Cow 0', 'data/new_images/malecow.jpg', 'Male'),
    ('Male Cow 1', 'data/new_images/malecow1.jpg', 'Male'),
    ('Male Cow 2', 'data/new_images/malecow2.jpg', 'Male'),
    ('Male Cow WebP', 'data/new_images/malecow2.webp', 'Male'),
    ('Male Bull Ayrshire', 'Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 'Male'),
    ('Male Bull Holstein', 'gender_dataset/cow/male/holstein_bull_male.jpg', 'Male'),
    ('Male Steer Holstein', 'gender_dataset/cow/male/holstein_steer_male.jpg', 'Male'),
    
    # Female Cows
    ('Female Cow 0', 'data/new_images/femalecow.jpg', 'Female'),
    ('Female Cow 1', 'data/new_images/femalecow1.jpg', 'Female'),
    ('Female Cow 2', 'data/new_images/femalecow2.jpg', 'Female'),
    ('Female Holstein', 'Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 'Female'),
    ('Female Jersey', 'Cattle Breeds/Jersey cattle/Jerseycattle0.jpg', 'Female'),
    ('Female Brown Swiss', 'Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg', 'Female'),
    ('Female Ayrshire', 'Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 'Female'),
    ('Female Red Dane', 'Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 'Female'),
    
    # Male Buffaloes
    ('Male Buff 0', 'data/new_images/malebuffolo.jpg', 'Male'),
    ('Male Buff 1', 'data/new_images/malebuffolo1.jpg', 'Male'),
    ('Male Buff 2', 'data/new_images/malebuffolo2.jpg', 'Male'),
    ('Male Buff 3', 'data/new_images/malebuffolo3.jpg', 'Male'),
    ('Male Banni Bull', 'buffalo/banni/baani-buffalo-bulls_0.jpg', 'Male'),
    ('Male Bhadawari Bull', 'buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg', 'Male'),
    ('Male Jafarabadi Bull', 'buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 'Male'),
    
    # Female Buffaloes
    ('Female Buff 0', 'data/new_images/femalebuffolo.jpg', 'Female'),
    ('Female Buff 1', 'data/new_images/femalebuffolo1.jpg', 'Female'),
    ('Female Buff 2', 'data/new_images/femalebuffolo2.jpg', 'Female'),
    ('Female Murrah', 'buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 'Female'),
    ('Female Banni', 'buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 'Female'),
    
    # Ambiguous / Head / Occluded cases
    ('Calf Headshot Fence', 'static/uploads/5da299733e69416d849ad4602bbd6459.jpg', 'Uncertain'),
    ('Cow Head Only', 'gender_dataset/uncertain/cow_head_only.jpg', 'Uncertain'),
    ('Buffalo Head Only', 'gender_dataset/uncertain/buffalo_head_only.jpg', 'Uncertain'),
]

print("="*100)
print(f"{'Label':24s} | {'Exp':8s} | {'Pred Gender':40s} | {'Conf':6s} | {'Status'}")
print("="*100)

for label, p, exp in sample_images:
    if not os.path.exists(p):
        print(f"{label:24s} | {exp:8s} | NOT FOUND: {p}")
        continue
    res = pipe.predict_image(p)
    gender = res.get('gender')
    conf = res.get('gender_confidence', 0.0)
    status = res.get('status')
    err = res.get('error', '')
    print(f"{label:24s} | {exp:8s} | {str(gender):40s} | {conf:<6.4f} | {status} {err[:20] if err else ''}")
