import os
import sys
import shutil
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_pipeline import pipeline

print("=" * 80)
print("     COMPREHENSIVE INDEPENDENT EVALUATION ON UNSEEN DATA & REPEATABILITY")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. VERIFY CLASS MAPPING & PREPROCESSING
# ---------------------------------------------------------------------------
print("\n--- 1. Verification of Class Mapping & Model Loading ---")
print(f"Pipeline Gender Classes: {pipeline.gender_classes}")
print(f"Pipeline Gender Threshold: {pipeline.gender_threshold}")
print(f"Pipeline Min Margin: {getattr(pipeline, 'min_margin', 0.12)}")
print(f"Cattle Gender Model Loaded: {pipeline.cattle_gender_clf is not None}")
print(f"Buffalo Gender Model Loaded: {pipeline.buffalo_gender_clf is not None}")

# ---------------------------------------------------------------------------
# 2. DETERMINISM & REPEATABILITY TEST (Same image 10 times)
# ---------------------------------------------------------------------------
print("\n--- 2. Determinism Test (10 repeated runs on same image) ---")
test_repeat_img = 'data/new_images/malebuffolo.jpg'
repeat_results = []
for i in range(10):
    res = pipeline.predict_image(test_repeat_img)
    repeat_results.append((res.get('gender'), res.get('gender_confidence')))

genders = set(r[0] for r in repeat_results)
confs = [r[1] for r in repeat_results]
print(f"Repeated 10 runs on '{test_repeat_img}':")
print(f"  Unique Genders: {genders}")
print(f"  Confidence min={min(confs)}, max={max(confs)}, std={np.std(confs):.6f}")
assert len(genders) == 1, "Non-deterministic gender prediction!"
assert np.std(confs) < 1e-5, "Non-deterministic confidence values!"
print("  >>> DETERMINISM CONFIRMED: Identical output across all 10 runs.")

# ---------------------------------------------------------------------------
# 3. METADATA INDEPENDENCE TEST (Rename image, verify prediction unchanged)
# ---------------------------------------------------------------------------
print("\n--- 3. Metadata Independence Test (Zero reliance on filename/path) ---")
# Test a Male Buffalo renamed as 'female_cow_dairy_123.jpg'
temp_male_path = 'scripts/temp_female_cow_name.jpg'
shutil.copy('data/new_images/malebuffolo.jpg', temp_male_path)
try:
    res_orig = pipeline.predict_image('data/new_images/malebuffolo.jpg')
    res_fake = pipeline.predict_image(temp_male_path)
    print(f"Original file ('malebuffolo.jpg'): Gender={res_orig.get('gender')}, Conf={res_orig.get('gender_confidence')}")
    print(f"Renamed file ('temp_female_cow_name.jpg'): Gender={res_fake.get('gender')}, Conf={res_fake.get('gender_confidence')}")
    assert res_orig.get('gender') == res_fake.get('gender'), "Filename affected gender prediction!"
    assert abs(res_orig.get('gender_confidence') - res_fake.get('gender_confidence')) < 1e-4, "Confidence changed by filename!"
    print("  >>> METADATA INDEPENDENCE CONFIRMED: Filename has zero effect on prediction.")
finally:
    if os.path.exists(temp_male_path):
        os.remove(temp_male_path)

# Test a Female Buffalo renamed as 'heavy_breeding_bull.jpg'
temp_fem_path = 'scripts/temp_heavy_breeding_bull.jpg'
shutil.copy('buffalo/surti/Milking_Murrah_Buffalo_Female_750kg_.jpg', temp_fem_path)
try:
    res_orig_f = pipeline.predict_image('buffalo/surti/Milking_Murrah_Buffalo_Female_750kg_.jpg')
    res_fake_f = pipeline.predict_image(temp_fem_path)
    print(f"Original file ('Milking_Murrah...'): Gender={res_orig_f.get('gender')}, Conf={res_orig_f.get('gender_confidence')}")
    print(f"Renamed file ('temp_heavy_breeding_bull.jpg'): Gender={res_fake_f.get('gender')}, Conf={res_fake_f.get('gender_confidence')}")
    assert res_orig_f.get('gender') == res_fake_f.get('gender'), "Filename affected female prediction!"
    print("  >>> METADATA INDEPENDENCE CONFIRMED: Gender strictly comes from image content.")
finally:
    if os.path.exists(temp_fem_path):
        os.remove(temp_fem_path)

# ---------------------------------------------------------------------------
# 4. EVALUATION ON UNSEEN TEST DATASET
# ---------------------------------------------------------------------------
print("\n--- 4. Unseen Evaluation Dataset Across All 4 Categories & Uncertain ---")

unseen_test_catalog = [
    # A. Cow Male
    ('data/new_images/malecow2.webp', 'Cow', 'Male', 'Male Hereford Bull (WebP Unseen)'),
    ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle13.jpg', 'Cow', 'Male', 'Male Holstein Bull (Breeds Unseen)'),
    ('Cattle Breeds/Jersey cattle/Jerseycattle13.jpg', 'Cow', 'Male', 'Male Jersey Bull (Breeds Unseen)'),
    
    # B. Cow Female
    ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle1.jpg', 'Cow', 'Female', 'Female Brown Swiss Cow (Breeds Unseen)'),
    ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle0.jpg', 'Cow', 'Female', 'Female Holstein Cow (Breeds Unseen)'),
    ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle11.jpg', 'Cow', 'Female', 'Female Holstein Side View (Breeds Unseen)'),
    ('Cattle Breeds/Jersey cattle/Jerseycattle1.jpg', 'Cow', 'Female', 'Female Jersey Cow with Udder (Breeds Unseen)'),
    
    # C. Buffalo Male
    ('buffalo/Jaffarabadi/Jafarabadi_buffalo_bull_for_sales_4_.jpg', 'Buffalo', 'Male', 'Male Jafarabadi Bull (Unseen Test)'),
    ('buffalo/Jaffarabadi/Jafarabadi_bull_from_Gujarat_-_YouTube.jpg', 'Buffalo', 'Male', 'Male Jafarabadi Bull Gujarat (Unseen Test)'),
    ('buffalo/nagpuri/Thakya_-_BAIF_Sire_Directory.jpg', 'Buffalo', 'Male', 'Male Nagpuri Bull Thakya BAIF (Unseen Test)'),
    ('buffalo/nili-ravi/Neeli_Ravi_buffalo_bull_.jpg', 'Buffalo', 'Male', 'Male Nili-Ravi Bull (Unseen Test)'),
    
    # D. Buffalo Female
    ('buffalo/bhadwari/Mehsana_Dairy_Knowledge_Portal.jpg', 'Buffalo', 'Female', 'Female Mehsana Dairy Buffalo (Unseen Test)'),
    ('buffalo/bhadwari/Surti_Dairy_Knowledge_Portal.jpg', 'Buffalo', 'Female', 'Female Surti Dairy Buffalo (Unseen Test)'),
    ('buffalo/Jaffarabadi/Milking_Jafrabadi_Buffalo_Farm_.jpg', 'Buffalo', 'Female', 'Female Jafrabadi Dairy Buffalo (Unseen Test)'),
    ('buffalo/Jaffarabadi/Chandani_dairy_and_farming_-_My_.jpg', 'Buffalo', 'Female', 'Female Chandani Dairy Buffalo (Unseen Test)'),
    
    # E. Uncertain / Occluded / Non-visible Sex Anatomy
    ('gender_dataset/uncertain/cow_head_only.jpg', 'Cow', 'Uncertain', 'Cow Head Only (No Anatomy)'),
    ('gender_dataset/uncertain/buffalo_head_only.jpg', 'Buffalo', 'Uncertain', 'Buffalo Head Only (No Anatomy)'),
    ('gender_dataset/uncertain/calf_headshot_fence.jpg', 'Cow', 'Uncertain', 'Calf Face Behind Fence'),
    ('test_artifacts/half_male_cow.jpg', 'Cow', 'Uncertain', 'Severed Cow Half Body (No Anatomy)'),
]

results = []
y_true_all = []
y_pred_all = []

# Class-specific buckets
cat_metrics = {
    'Cow Male': {'total': 0, 'correct': 0, 'uncertain': 0, 'wrong': 0},
    'Cow Female': {'total': 0, 'correct': 0, 'uncertain': 0, 'wrong': 0},
    'Buffalo Male': {'total': 0, 'correct': 0, 'uncertain': 0, 'wrong': 0},
    'Buffalo Female': {'total': 0, 'correct': 0, 'uncertain': 0, 'wrong': 0},
    'Uncertain': {'total': 0, 'correct': 0, 'uncertain': 0, 'wrong': 0},
}

for path, species_exp, gender_exp, desc in unseen_test_catalog:
    if not os.path.exists(path):
        print(f"Skipping missing: {path}")
        continue
        
    res = pipeline.predict_image(path)
    pred_g = res.get('gender')
    if pred_g is None or not res.get('is_valid', True):
        pred_g = "Uncertain"
        
    conf = res.get('gender_confidence', 0.0)
    sp = res.get('species', 'Unknown')
    
    if gender_exp == 'Uncertain':
        bucket = 'Uncertain'
    else:
        bucket = f"{species_exp} {gender_exp}"
        
    cat_metrics[bucket]['total'] += 1
    
    if pred_g == gender_exp:
        cat_metrics[bucket]['correct'] += 1
        status = 'OK'
    elif pred_g == 'Uncertain':
        cat_metrics[bucket]['uncertain'] += 1
        status = 'UNCERTAIN'
    else:
        cat_metrics[bucket]['wrong'] += 1
        status = 'FAIL'
        
    results.append((path, desc, species_exp, gender_exp, pred_g, conf, status))
    y_true_all.append(gender_exp)
    y_pred_all.append(pred_g)
    
    print(f"  {desc:42s} | Exp: {gender_exp:9s} | Got: {pred_g:9s} (conf: {conf:.4f}) | [{status}]")

print("\n" + "=" * 80)
print("EVALUATION RESULTS BY CATEGORY")
print("=" * 80)
for cat, m in cat_metrics.items():
    tot = m['total']
    cor = m['correct']
    acc = (cor / tot * 100) if tot > 0 else 0.0
    print(f"  {cat:16s} | Total: {tot} | Correct: {cor} ({acc:.1f}%) | Uncertain: {m['uncertain']} | Wrong: {m['wrong']}")

# Filtered standard metrics on gender-identified animals (excluding Uncertain)
eval_gender_true = [y for y in y_true_all if y in ('Male', 'Female')]
eval_gender_pred = [p for y, p in zip(y_true_all, y_pred_all) if y in ('Male', 'Female')]

# Map Male=1, Female=0
map_dict = {'Female': 0, 'Male': 1, 'Uncertain': -1}
yt_num = [map_dict[y] for y in eval_gender_true]
yp_num = [map_dict[p] for p in eval_gender_pred]

print("\n" + "=" * 80)
print("CONFUSION MATRIX ON CONFIRMED LIVESTOCK GENDER (Female vs Male)")
print("=" * 80)
cm_gender = confusion_matrix([y for y in eval_gender_true], [p for p in eval_gender_pred], labels=['Female', 'Male'])
print(f"Labels: ['Female', 'Male']")
print(f"Matrix:\n{cm_gender}")
print(f"  True Female -> Predicted Female: {cm_gender[0,0]}")
print(f"  True Female -> Predicted Male:   {cm_gender[0,1]}")
print(f"  True Male   -> Predicted Female: {cm_gender[1,0]}")
print(f"  True Male   -> Predicted Male:   {cm_gender[1,1]}")

acc_ov = accuracy_score(eval_gender_true, eval_gender_pred)
prec_ov, rec_ov, f1_ov, _ = precision_recall_fscore_support(eval_gender_true, eval_gender_pred, average='weighted', zero_division=0)

print("\n" + "=" * 80)
print("OVERALL METRICS SUMMARY")
print("=" * 80)
print(f"Overall Accuracy:  {acc_ov * 100:.2f}%")
print(f"Overall Precision: {prec_ov * 100:.2f}%")
print(f"Overall Recall:    {rec_ov * 100:.2f}%")
print(f"Overall F1-Score:  {f1_ov * 100:.2f}%")
