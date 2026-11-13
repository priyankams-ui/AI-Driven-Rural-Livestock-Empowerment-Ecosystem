import os, sys, glob
from PIL import Image

def audit_images():
    sample_categories = {
        'Male Cow': [
            'data/new_images/malecow.jpg',
            'data/new_images/malecow1.jpg',
            'data/new_images/malecow2.jpg',
            'data/new_images/malecow2.webp',
            'gender_dataset/cow/male/holstein_bull_male.jpg',
            'gender_dataset/cow/male/holstein_steer_male.jpg',
            'gender_dataset/cow/male/ayrshire_bull_4.jpg',
            'Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg',
        ],
        'Female Cow': [
            'data/new_images/femalecow.jpg',
            'data/new_images/femalecow1.jpg',
            'data/new_images/femalecow2.jpg',
            'gender_dataset/cow/female/ayrshire_female_1.jpg',
            'gender_dataset/cow/female/brownswiss_female_0.jpg',
            'gender_dataset/cow/female/holstein_female_1.jpg',
            'gender_dataset/cow/female/jersey_female_0.jpg',
            'gender_dataset/cow/female/reddane_female_0.jpg',
            'Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg',
            'Cattle Breeds/Jersey cattle/Jerseycattle0.jpg',
            'Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg',
            'Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg',
            'Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg',
        ],
        'Male Buffalo': [
            'data/new_images/malebuffolo.jpg',
            'data/new_images/malebuffolo1.jpg',
            'data/new_images/malebuffolo2.jpg',
            'data/new_images/malebuffolo3.jpg',
            'gender_dataset/buffalo/male/banni_bull_0.jpg',
            'gender_dataset/buffalo/male/bhadawari_bull.jpg',
            'gender_dataset/buffalo/male/jafarabadi_bull_punjab.jpg',
            'buffalo/banni/baani-buffalo-bulls_0.jpg',
            'buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg',
            'buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg',
        ],
        'Female Buffalo': [
            'data/new_images/femalebuffolo.jpg',
            'data/new_images/femalebuffolo1.jpg',
            'data/new_images/femalebuffolo2.jpg',
            'gender_dataset/buffalo/female/banni_female_0caae.jpg',
            'gender_dataset/buffalo/female/murrah_female_original.jpg',
            'buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg',
            'buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg',
        ],
        'Uncertain / Occluded / Ambiguous': [
            'gender_dataset/uncertain/buffalo_head_only.jpg',
            'gender_dataset/uncertain/calf_headshot_fence.jpg',
            'gender_dataset/uncertain/cow_head_only.jpg',
            'static/uploads/5da299733e69416d849ad4602bbd6459.jpg',
        ]
    }

    print("AUDITING IMAGE SAMPLES:")
    for cat, paths in sample_categories.items():
        print(f"\n=== Category: {cat} ===")
        for p in paths:
            if os.path.exists(p):
                im = Image.open(p)
                print(f"  {p:65s} | size: {im.size} | mode: {im.mode}")
            else:
                print(f"  {p:65s} | NOT FOUND")

if __name__ == '__main__':
    audit_images()
