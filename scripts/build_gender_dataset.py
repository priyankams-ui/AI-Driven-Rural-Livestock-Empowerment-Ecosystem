import os
import shutil
from PIL import Image

def setup_gender_dataset():
    base_dir = "gender_dataset"
    dirs = [
        os.path.join(base_dir, "cow", "female"),
        os.path.join(base_dir, "cow", "male"),
        os.path.join(base_dir, "buffalo", "female"),
        os.path.join(base_dir, "buffalo", "male"),
        os.path.join(base_dir, "uncertain")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 1. Verified Female Cows
    female_cows = [
        ('data/new_images/femalecow.jpg', 'femalecow.jpg'),
        ('data/new_images/femalecow1.jpg', 'femalecow1.jpg'),
        ('data/new_images/femalecow2.jpg', 'femalecow2.jpg'),
        ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 'holstein_female_1.jpg'),
        ('Cattle Breeds/Jersey cattle/Jerseycattle0.jpg', 'jersey_female_0.jpg'),
        ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg', 'brownswiss_female_0.jpg'),
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 'ayrshire_female_1.jpg'),
        ('Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 'reddane_female_0.jpg'),
    ]

    # 2. Verified Male Cows
    male_cows = [
        ('data/new_images/malecow.jpg', 'malecow.jpg'),
        ('data/new_images/malecow1.jpg', 'malecow1.jpg'),
        ('data/new_images/malecow2.jpg', 'malecow2.jpg'),
        ('static/uploads/ec95ac1e5e694f6eb9a39efa0151e85c.jpg', 'holstein_steer_male.jpg'),
        ('static/uploads/7fa0f07b81354ed68bac87e2704437c2.jpg', 'holstein_bull_male.jpg'),
        ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 'ayrshire_bull_4.jpg'),
    ]

    # 3. Verified Female Buffaloes
    female_buffaloes = [
        ('data/new_images/femalebuffolo.jpg', 'femalebuffolo.jpg'),
        ('data/new_images/femalebuffolo1.jpg', 'femalebuffolo1.jpg'),
        ('data/new_images/femalebuffolo2.jpg', 'femalebuffolo2.jpg'),
        ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 'murrah_female_original.jpg'),
        ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 'banni_female_0caae.jpg'),
    ]

    # 4. Verified Male Buffaloes
    male_buffaloes = [
        ('data/new_images/malebuffolo.jpg', 'malebuffolo.jpg'),
        ('data/new_images/malebuffolo1.jpg', 'malebuffolo1.jpg'),
        ('data/new_images/malebuffolo2.jpg', 'malebuffolo2.jpg'),
        ('data/new_images/malebuffolo3.jpg', 'malebuffolo3.jpg'),
        ('buffalo/banni/baani-buffalo-bulls_0.jpg', 'banni_bull_0.jpg'),
        ('buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg', 'bhadawari_bull.jpg'),
        ('buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 'jafarabadi_bull_punjab.jpg'),
    ]

    # 5. Uncertain (No ventral anatomical features / headshot / non-visible)
    uncertain_samples = [
        ('static/uploads/5da299733e69416d849ad4602bbd6459.jpg', 'calf_headshot_fence.jpg'),
        ('test_artifacts/cow_head_only.jpg', 'cow_head_only.jpg'),
        ('test_artifacts/buffalo_head_only.jpg', 'buffalo_head_only.jpg'),
    ]

    def copy_files(file_list, target_sub):
        target_path = os.path.join(base_dir, target_sub)
        copied = 0
        for src, name in file_list:
            if os.path.exists(src):
                dst = os.path.join(target_path, name)
                shutil.copyfile(src, dst)
                copied += 1
            else:
                print(f"Warning: source file not found: {src}")
        return copied

    c_fc = copy_files(female_cows, os.path.join("cow", "female"))
    c_mc = copy_files(male_cows, os.path.join("cow", "male"))
    c_fb = copy_files(female_buffaloes, os.path.join("buffalo", "female"))
    c_mb = copy_files(male_buffaloes, os.path.join("buffalo", "male"))
    c_un = copy_files(uncertain_samples, "uncertain")

    print("\n=== GENDER DATASET AUDIT REPORT ===")
    print(f"Female Cow     : {c_fc} images")
    print(f"Male Cow       : {c_mc} images")
    print(f"Female Buffalo : {c_fb} images")
    print(f"Male Buffalo   : {c_mb} images")
    print(f"Uncertain      : {c_un} images")
    print(f"Total Images   : {c_fc + c_mc + c_fb + c_mb + c_un}")
    print("===================================\n")

if __name__ == "__main__":
    setup_gender_dataset()
