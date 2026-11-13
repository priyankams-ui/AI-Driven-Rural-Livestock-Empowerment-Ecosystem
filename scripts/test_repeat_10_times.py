import sys
sys.path.insert(0, '.')
from ml_pipeline import pipeline

def test_10_times(image_path, label):
    print(f"\n--- Testing {label} ({image_path}) 10 times ---")
    results = []
    for i in range(1, 11):
        res = pipeline.predict_image(image_path)
        status = res.get('status')
        sp = res.get('species')
        sp_c = res.get('species_confidence')
        gender = res.get('gender')
        g_c = res.get('gender_confidence')
        err = res.get('error')
        results.append((status, sp, sp_c, gender, g_c, err))
        print(f"Run {i:2d}: status={status}, species={sp} ({sp_c}), gender={gender} ({g_c}), error={err}")
    
    first = results[0]
    all_same = all(r == first for r in results)
    print(f"Deterministic across all 10 runs: {all_same}")
    return all_same

test_10_times('data/new_images/femalecow.jpg', 'Female Cow')
test_10_times('data/new_images/malecow.jpg', 'Male Cow')
test_10_times('data/new_images/femalebuffolo2.jpg', 'Female Buffalo')
test_10_times('data/new_images/malebuffolo1.jpg', 'Male Buffalo')
