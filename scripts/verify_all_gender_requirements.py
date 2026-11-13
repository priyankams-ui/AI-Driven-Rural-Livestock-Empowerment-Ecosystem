import os
import sys
import requests
import json

print("================================================================================")
print("   COMPREHENSIVE VERIFICATION OF CATTLE GENDER CLASSIFIER                       ")
print("================================================================================")

SERVER_URL = "http://127.0.0.1:5000/api/predict/image-analysis"

test_suite = [
    # 1. Solid Brown Female Cows (The primary failure pattern)
    {'path': 'Cattle Breeds/Brown Swiss cattle/BrownSwisscattle2.jpg', 'desc': 'Brown Swiss Cow 2 (Solid Brown)', 'exp_species': 'Cow', 'exp_gender': 'Female'},
    {'path': 'Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg', 'desc': 'Brown Swiss Cow 0 (Solid Brown)', 'exp_species': 'Cow', 'exp_gender': 'Female'},
    {'path': 'Cattle Breeds/Jersey cattle/Jerseycattle2.jpg', 'desc': 'Jersey Cow 2 (Fawn/Brown)', 'exp_species': 'Cow', 'exp_gender': 'Female'},
    {'path': 'Cattle Breeds/Jersey cattle/Jerseycattle101.jpg', 'desc': 'Jersey Cow 101 (Fawn/Brown)', 'exp_species': 'Cow', 'exp_gender': 'Female'},
    {'path': 'Cattle Breeds/Jersey cattle/Jerseycattle103.jpg', 'desc': 'Jersey Cow 103 (Fawn/Brown)', 'exp_species': 'Cow', 'exp_gender': 'Female'},
    {'path': 'Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 'desc': 'Red Dane Cow 0 (Reddish-Brown)', 'exp_species': 'Cow', 'exp_gender': 'Female'},
    {'path': 'data/new_images/femalecow.jpg', 'desc': 'Female Cow Sample (Brown/White)', 'exp_species': 'Cow', 'exp_gender': 'Female'},

    # 2. Black & White and Red & White Female Cows
    {'path': 'Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 'desc': 'Holstein Cow 1 (Black/White)', 'exp_species': 'Cow', 'exp_gender': 'Female'},
    {'path': 'data/new_images/femalecow1.jpg', 'desc': 'Female Holstein Cow (Black/White)', 'exp_species': 'Cow', 'exp_gender': 'Female'},
    {'path': 'Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 'desc': 'Ayrshire Cow 1 (Red/White)', 'exp_species': 'Cow', 'exp_gender': 'Female'},
    {'path': 'Cattle Breeds/Ayrshire cattle/Ayrshirecattle5.jpg', 'desc': 'Ayrshire Cow 5 (Red/White)', 'exp_species': 'Cow', 'exp_gender': 'Female'},

    # 3. Male Bulls (Brown, Red/White, Black/White)
    {'path': 'data/new_images/malecow.jpg', 'desc': 'Male Bull Sample (Solid Brown)', 'exp_species': 'Cow', 'exp_gender': 'Male'},
    {'path': 'data/new_images/malecow1.jpg', 'desc': 'Male Bull 1 Sample (Brown)', 'exp_species': 'Cow', 'exp_gender': 'Male'},
    {'path': 'Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 'desc': 'Ayrshire Bull (Red/White)', 'exp_species': 'Cow', 'exp_gender': 'Male'},
    {'path': 'gender_dataset/cow/male/holstein_bull_male.jpg', 'desc': 'Holstein Bull (Black/White)', 'exp_species': 'Cow', 'exp_gender': 'Male'},
    {'path': 'gender_dataset/cow/male/holstein_steer_male.jpg', 'desc': 'Holstein Steer (Black/White)', 'exp_species': 'Cow', 'exp_gender': 'Male'},

    # 4. Buffalo (MUST REMAIN 100% WORKING AND UNMODIFIED)
    {'path': 'data/new_images/femalebuffolo.jpg', 'desc': 'Female Buffalo (Murrah)', 'exp_species': 'Buffalo', 'exp_gender': 'Female'},
    {'path': 'data/new_images/femalebuffolo1.jpg', 'desc': 'Female Buffalo 1 (Murrah)', 'exp_species': 'Buffalo', 'exp_gender': 'Female'},
    {'path': 'data/new_images/malebuffolo.jpg', 'desc': 'Male Buffalo (Murrah)', 'exp_species': 'Buffalo', 'exp_gender': 'Male'},
    {'path': 'data/new_images/malebuffolo1.jpg', 'desc': 'Male Buffalo 1 (Murrah)', 'exp_species': 'Buffalo', 'exp_gender': 'Male'},
    {'path': 'buffalo/banni/baani-buffalo-bulls_0.jpg', 'desc': 'Male Banni Buffalo Bull', 'exp_species': 'Buffalo', 'exp_gender': 'Male'},

    # 5. Non-Livestock / Incomplete / Uncertain (Gating check)
    {'path': 'gender_dataset/uncertain/cow_head_only.jpg', 'desc': 'Head-only cow (No anatomy)', 'exp_species': 'Cow', 'exp_gender': None},
]

total = 0
passed = 0

print(f"{'Description':<38} | {'Exp Sp':<8} | {'Exp G':<8} | {'Pred Sp':<8} | {'Pred G':<8} | {'Conf':<6} | {'Result'}")
print("-" * 92)

for item in test_suite:
    p = item['path']
    if not os.path.exists(p):
        print(f"{item['desc']:<38} | SKIPPED (File not found)")
        continue
    total += 1
    with open(p, 'rb') as f:
        r = requests.post(SERVER_URL, files={'file': f})
        data = r.json()
        pred_sp = data.get('species')
        pred_g = data.get('gender')
        conf = data.get('gender_confidence', 0.0)
        status = data.get('status')
        
        if item['exp_gender'] is None:
            # Expected rejection / uncertain
            is_ok = (status == 'rejected' or pred_g in ['Uncertain', None])
        else:
            is_ok = (pred_sp == item['exp_species'] and pred_g == item['exp_gender'])
            
        if is_ok:
            passed += 1
            res_str = "PASS [OK]"
        else:
            res_str = "FAIL [X]"
            
        print(f"{item['desc']:<38} | {str(item['exp_species']):<8} | {str(item['exp_gender']):<8} | {str(pred_sp):<8} | {str(pred_g):<8} | {conf:<6.2f} | {res_str}")

print("=" * 92)
print(f"OVERALL VERIFICATION SUITE: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
