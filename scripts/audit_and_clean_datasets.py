import os
import sys
import re
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image

CATTLE_DIR = "Cattle Breeds"
BUFFALO_DIR = "buffalo"
DUMMY_DIR = "dummy_unidentified"
OUTPUT_DIR = os.path.join("data", "cleaned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def get_base_image_group(filename, breed, species):
    """
    Group images belonging to the same animal / same original photograph
    (e.g., Ayrshirecattle0.jpg and Ayrshirecattle0_c.jpg, copy files, etc.)
    """
    stem = os.path.splitext(filename)[0]
    # Remove augment suffixes and copy suffixes
    cleaned_stem = re.sub(r'(_c|_f|_r|_flip|_crop|_rot\d*|_\d+|\s*-\s*Copy.*|\s*\(\d+\))$', '', stem, flags=re.IGNORECASE)
    cleaned_stem = re.sub(r'(_c|_f|_r|_flip|_crop|_rot\d*|_\d+)$', '', cleaned_stem, flags=re.IGNORECASE)
    return f"{species}_{breed}_{cleaned_stem}"

# Known irrelevant / noisy files to exclude from breed & gender
IRRELEVANT_PATTERNS = [
    r'african_buffalo', r'cape_buffalo', r'tattoos', r'fight_with_lions',
    r'spermatozoa', r'leather', r'brahmancattle', r'ankole', r'kwai_thai',
    r'dancing_a_hilarious', r'video', r'thumbnail', r'mat_', r'advertisement',
    r'sacrificed_during', r'dog_colostrum', r'colostrum'
]

def is_irrelevant(filename):
    fl = filename.lower()
    for pat in IRRELEVANT_PATTERNS:
        if re.search(pat, fl):
            return True
    return False

# Explicitly verified male specimens
VERIFIED_MALE_FILENAMES = {
    # Cattle Bulls
    'ayrshirecattle4.jpg', 'ayrshirecattle4_c.jpg',
    'reddanecattle4.jpg', 'reddanecattle4_c.jpg',
    # Buffalo Bulls
    '1280px-water_buffalo_bull-_near_mehsana-_gujarat-_india-_3.jpg',
    'baani-buffalo-bulls_0.jpg',
    'banni-male-buffalo-breed-360x240.png',
    'bhadavari_archives_-_baif_sire_directory.jpg',
    'best_india_breed_jafarabadi_bull_.jpg',
    'best_jafarabadi_bull_.jpg',
    'bhadawari-buffalo-bull_epashupalan.jpg',
    'big_jafarabadi_buffalo_bull_2023_sadeeq_.jpg',
    'black_jafarabadi_bull_semen_at_best_.jpg',
    'dairy_farm_with_jafarabadi_bull_.jpg',
    'jafarabadi_buffalo_bull_for_sales_4_.jpg',
    'jafarabadi_bull_at_badal_farm_punjab_.jpg',
    'jafarabadi_bull_from_gujarat_-_youtube.jpg',
    'jafarabadi_bull_from_gujarat_india_.jpg',
    'jafarabadi_bull_ss-abc_dams_yield_.jpg',
    'jaffrabadi_buffalo_bull.jpg',
    'jafrabadi_bull_-_meet_the_majestic_.jpg',
    'mashallah_jafrabi_buffalo_bull_.jpg',
    'shot_of_powerful_bull_at_cattle_.jpg',
    'some_good_jafrabadi_bulls.jpg',
    'surti_buffalo_black_jafarabadi_bull_at_.jpg',
    'tulsidas_-_baif_sire_directory.jpg',
    '1_yr_male_murrah_buffalo_calf_at_ (1).jpg',
    '1_yr_male_murrah_buffalo_calf_at_.jpg',
    'black_murrah_buffalo_bull_wholesaler_.jpg',
    'murrah_buffalo_bulls_to_be_used_for_.jpg',
    'murrah_buffalo_clone_1st_clone_male_.jpg',
    'murrah_buffalo_male_calf_at_3000000_.jpg',
    'murrah_bull_in_india_murrah_buffalo_is_ (1).jpg',
    'murrah_bull_in_india_murrah_buffalo_is_ (2).jpg',
    'murrah_bull_in_india_murrah_buffalo_is_ (3).jpg',
    'murrah_bull_in_india_murrah_buffalo_is_ (4).jpg',
    'murrah_bull_in_india_murrah_buffalo_is_ (5).jpg',
    'murrah_bull_in_india_murrah_buffalo_is_.jpg',
    'murrah_bull_yuvraj_in_india_murrah_.jpg',
    'thakya_-_baif_sire_directory.jpg',
    'buffalo_male_mili_ravi_complete_teeth_ (2).jpg',
    'buffalo_male_mili_ravi_complete_teeth_.jpg',
    'hsb_nili_ravi_bull-parass_mother_ (1).jpg',
    'hsb_nili_ravi_bull-parass_mother_ (2).jpg',
    'hsb_nili_ravi_bull-parass_mother_ (3).jpg',
    'hsb_nili_ravi_bull-parass_mother_.jpg',
    'neeli_ravi_buffalo_bull_ (2).jpg',
    'neeli_ravi_buffalo_bull_.jpg',
    'nili_ravi_bull_quality_full_niliravi_.jpg',
    'image_282x179_5.jpg'
}

# Verified adult dairy cows with clear udders and milk production traits
VERIFIED_FEMALE_PATTERNS = [
    r'ayrshirecattle0', r'ayrshirecattle1', r'ayrshirecattle3', r'ayrshirecattle5',
    r'ayrshirecattle10', r'ayrshirecattle11', r'ayrshirecattle12', r'ayrshirecattle14',
    r'ayrshirecattle15', r'ayrshirecattle16', r'ayrshirecattle20', r'ayrshirecattle21',
    r'brownswisscattle0', r'brownswisscattle1', r'brownswisscattle3', r'brownswisscattle4',
    r'brownswisscattle5', r'brownswisscattle8', r'brownswisscattle9', r'brownswisscattle10',
    r'holsteinfriesiancattle1', r'holsteinfriesiancattle2', r'holsteinfriesiancattle5',
    r'holsteinfriesiancattle8', r'holsteinfriesiancattle9', r'holsteinfriesiancattle10',
    r'holsteinfriesiancattle83', r'holsteinfriesiancattle100', r'holsteinfriesiancattle101',
    r'jerseycattle0', r'jerseycattle1', r'jerseycattle3', r'jerseycattle5',
    r'jerseycattle7', r'jerseycattle8', r'jerseycattle10', r'jerseycattle100',
    r'reddanecattle0', r'reddanecattle3', r'reddanecattle5', r'reddanecattle6',
    r'reddanecattle8', r'reddanecattle9', r'reddanecattle10', r'reddanecattle90',
    r'reddanecattle100', r'reddanecattle101',
    # Buffalo Milch Females
    r'milking_murrah', r'black_murrah_buffalovijay', r'murrah_buffalonand',
    r'murrah_buffalosuresh', r'mehsana_dairy', r'surti_dairy',
    r'dairy_farming_buffalo_pics', r'50_jafarabadi_buffalo_dairy',
    r'chandani_dairy', r'chaudhary_dairy_farm', r'0caae52ef510154bb215e99a7db35607'
]

def determine_gender(filename, species, is_usable):
    if not is_usable:
        return "Unknown", 0.0
    fl = filename.lower()
    if fl in VERIFIED_MALE_FILENAMES:
        return "Male", 0.98
    for pat in VERIFIED_FEMALE_PATTERNS:
        if re.search(pat, fl):
            return "Female", 0.98
    return "Unknown", 0.50

def audit_datasets():
    print("================================================================================")
    print("   STEP 1: RIGOROUS DATASET AUDIT & CLEANING                                    ")
    print("================================================================================")

    seen_hashes = {}
    records = []

    # 1. Scan Cattle
    cattle_breeds = sorted([d for d in os.listdir(CATTLE_DIR) if os.path.isdir(os.path.join(CATTLE_DIR, d))])
    for breed in cattle_breeds:
        folder = os.path.join(CATTLE_DIR, breed)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                continue
            fpath = os.path.join(folder, fname)
            file_hash = calculate_md5(fpath)
            
            is_usable = True
            flag_reason = "OK"
            
            if is_irrelevant(fname):
                is_usable = False
                flag_reason = "Irrelevant / Non-livestock"
            elif file_hash in seen_hashes:
                is_usable = False
                flag_reason = f"Duplicate Hash"
            else:
                seen_hashes[file_hash] = fpath
                
            try:
                with Image.open(fpath) as img:
                    w, h = img.size
                    if w < 64 or h < 64:
                        is_usable = False
                        flag_reason = "Low Resolution (<64px)"
            except Exception:
                is_usable = False
                flag_reason = "Corrupted File"
                
            base_group = get_base_image_group(fname, breed, "Cow")
            gender, g_conf = determine_gender(fname, "Cow", is_usable)
            
            records.append({
                'path': fpath,
                'filename': fname,
                'species': 'Cow',
                'breed': breed,
                'base_group': base_group,
                'gender': gender,
                'gender_confidence': g_conf,
                'is_usable': is_usable,
                'flag_reason': flag_reason
            })

    # 2. Scan Buffalo
    buffalo_breeds = sorted([d for d in os.listdir(BUFFALO_DIR) if os.path.isdir(os.path.join(BUFFALO_DIR, d))])
    for breed in buffalo_breeds:
        folder = os.path.join(BUFFALO_DIR, breed)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                continue
            fpath = os.path.join(folder, fname)
            file_hash = calculate_md5(fpath)
            
            is_usable = True
            flag_reason = "OK"
            
            if is_irrelevant(fname):
                is_usable = False
                flag_reason = "Irrelevant / Non-livestock"
            elif file_hash in seen_hashes:
                is_usable = False
                flag_reason = f"Duplicate Hash"
            else:
                seen_hashes[file_hash] = fpath
                
            try:
                with Image.open(fpath) as img:
                    w, h = img.size
                    if w < 64 or h < 64:
                        is_usable = False
                        flag_reason = "Low Resolution (<64px)"
            except Exception:
                is_usable = False
                flag_reason = "Corrupted File"
                
            base_group = get_base_image_group(fname, breed, "Buffalo")
            gender, g_conf = determine_gender(fname, "Buffalo", is_usable)
            
            records.append({
                'path': fpath,
                'filename': fname,
                'species': 'Buffalo',
                'breed': breed,
                'base_group': base_group,
                'gender': gender,
                'gender_confidence': g_conf,
                'is_usable': is_usable,
                'flag_reason': flag_reason
            })

    # 3. Scan Dummy/Unidentified (Negative / Non-livestock)
    if os.path.exists(DUMMY_DIR):
        for fname in sorted(os.listdir(DUMMY_DIR)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                continue
            fpath = os.path.join(DUMMY_DIR, fname)
            file_hash = calculate_md5(fpath)
            
            is_usable = True
            flag_reason = "OK"
            
            if file_hash in seen_hashes:
                is_usable = False
                flag_reason = "Duplicate Hash"
            else:
                seen_hashes[file_hash] = fpath
                
            try:
                with Image.open(fpath) as img:
                    w, h = img.size
                    if w < 64 or h < 64:
                        is_usable = False
                        flag_reason = "Low Resolution (<64px)"
            except Exception:
                is_usable = False
                flag_reason = "Corrupted File"
                
            records.append({
                'path': fpath,
                'filename': fname,
                'species': 'Unknown',
                'breed': 'Unknown',
                'base_group': f"Dummy_{fname}",
                'gender': 'Unknown',
                'gender_confidence': 0.0,
                'is_usable': is_usable,
                'flag_reason': flag_reason
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "cleaned_dataset_manifest.csv")
    df.to_csv(csv_path, index=False)
    print(f"Cleaned dataset manifest saved to: {csv_path}\n")

    print(f"Total images scanned: {len(df)}")
    print(f"Usable images: {len(df[df['is_usable']])}")
    print(f"Flagged images (duplicates, low-res, corrupt, irrelevant): {len(df[~df['is_usable']])}")

    # Summary by Gender
    usable_df = df[df['is_usable']]
    gender_counts = usable_df['gender'].value_counts()
    print("\n[VERIFIED GENDER DISTRIBUTION (USABLE LIVESTOCK)]")
    print(dict(gender_counts))

    return df

if __name__ == '__main__':
    audit_datasets()
