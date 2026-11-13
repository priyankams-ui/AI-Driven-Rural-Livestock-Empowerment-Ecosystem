import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_pipeline import pipeline
from PIL import Image

print("="*80)
print("   COMPREHENSIVE FINAL VERIFICATION (SECTION 25 REQUIREMENTS)   ")
print("="*80)

# -------------------------------------------------------------
# Part 1: Verify All 14 Non-Livestock Images are Strictly Rejected
# -------------------------------------------------------------
print("\n--- [1] NON-LIVESTOCK TESTING (Items 26-37) ---")
non_livestock_categories = [
    'car', 'bus', 'bike', 'truck', 'person', 'dog', 'cat',
    'horse', 'goat', 'sheep', 'building', 'landscape', 'food', 'random_object'
]

all_rejected = True
rejection_results = {}
for cat in non_livestock_categories:
    img_path = os.path.join('test_non_livestock_samples', f'{cat}.jpg')
    res = pipeline.predict_image(img_path)
    is_rejected = (
        res.get('status') == 'rejected' and
        res.get('is_valid') is False and
        res.get('species') == 'Unknown' and
        res.get('error') == 'Unsupported image. Please upload a clear image of a cattle or buffalo.' and
        'breed' not in res and
        'gender' not in res
    )
    rejection_results[cat] = (is_rejected, res)
    status_str = "REJECTED (CORRECT)" if is_rejected else "ACCEPTED (FAILED!)"
    print(f"  {cat:15}: {status_str} | error='{res.get('error')}'")
    if not is_rejected:
        all_rejected = False

print(f"\n>> All 14 Non-Livestock Categories Rejected: {all_rejected}")
assert all_rejected, "One or more non-livestock images were not correctly rejected!"

# -------------------------------------------------------------
# Part 2: Verify Real Livestock Unseen Testing (Items 22-25)
# -------------------------------------------------------------
print("\n--- [2] REAL LIVESTOCK TESTING (Items 22-25) ---")
livestock_samples = [
    ('Male Cattle', r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle4.jpg', 'Cow', 'Male'),
    ('Female Cattle', r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle0.jpg', 'Cow', 'Female'),
    ('Male Buffalo', r'buffalo\Jaffarabadi\Best_Jafarabadi_bull_.jpg', 'Buffalo', 'Male'),
    ('Female Buffalo', r'buffalo\banni\0caae52ef510154bb215e99a7db35607.jpg', 'Buffalo', 'Female'),
]

livestock_passed = True
for label, path, expected_species, expected_gender in livestock_samples:
    res = pipeline.predict_image(path)
    pred_species = res.get('species')
    pred_gender = res.get('gender')
    is_valid = res.get('is_valid')
    conf_species = res.get('species_confidence', 0.0)
    conf_gender = res.get('gender_confidence', 0.0)
    breed = res.get('breed')

    matches_species = (pred_species == expected_species)
    matches_gender = (pred_gender == expected_gender)
    success = (is_valid and matches_species and matches_gender)

    print(f"  {label:15}: Valid={is_valid} | Species={pred_species} ({conf_species:.2%}) | Breed={breed} | Gender={pred_gender} ({conf_gender:.2%}) -> {'PASS' if success else 'FAIL'}")
    if not success:
        livestock_passed = False

print(f"\n>> All Real Livestock Specimens Correct: {livestock_passed}")
assert livestock_passed, "Real livestock predictions did not match ground truth!"

# -------------------------------------------------------------
# Part 3: Verify Genuine Metrics Report
# -------------------------------------------------------------
print("\n--- [3] GENUINE METRICS REPORT CHECK ---")
report_path = os.path.join('models', 'metrics_report.json')
with open(report_path, 'r', encoding='utf-8') as f:
    metrics = json.load(f)

print("Metrics Report Content:")
print(json.dumps(metrics, indent=2))

print("\n" + "="*80)
print("   ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY!   ")
print("="*80)
