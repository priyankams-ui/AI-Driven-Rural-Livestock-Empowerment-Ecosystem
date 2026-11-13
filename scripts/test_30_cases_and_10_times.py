import os
import sys
sys.path.insert(0, '.')
from ml_pipeline import pipeline

test_cases = [
    (1, "Full real Cow", "data/new_images/femalecow.jpg", "ACCEPT"),
    (2, "Full real Buffalo", "data/new_images/femalebuffolo2.jpg", "ACCEPT"),
    (3, "Full Male Cow", "data/new_images/malecow.jpg", "ACCEPT"),
    (4, "Full Female Cow", "data/new_images/femalecow.jpg", "ACCEPT"),
    (5, "Full Male Buffalo", "data/new_images/malebuffolo.jpg", "ACCEPT"),
    (6, "Full Female Buffalo", "data/new_images/femalebuffolo2.jpg", "ACCEPT"),
    (7, "Cow face only", "test_artifacts/cow_head_only.jpg", "REJECT"),
    (8, "Buffalo face only", "test_artifacts/buffalo_head_only.jpg", "REJECT"),
    (9, "Half Cow", "test_artifacts/half_female_cow.jpg", "REJECT"),
    (10, "Half Buffalo", "test_artifacts/half_female_buffalo.jpg", "REJECT"),
    (11, "Cropped Cow", "test_artifacts/cropped_cow.jpg", "REJECT"),
    (12, "Cropped Buffalo", "test_artifacts/cropped_buffalo.jpg", "REJECT"),
    (13, "Real Google Cow", "test_artifacts/google_cow.jpg", "ACCEPT"),
    (14, "Real Google Buffalo", "test_artifacts/google_buffalo.jpg", "ACCEPT"),
    (15, "AI-generated Cow", "test_artifacts/ai_generated_samples/ai_cow.jpg", "REJECT"),
    (16, "AI-generated Buffalo", "test_artifacts/ai_generated_samples/ai_buffalo.jpg", "REJECT"),
    (17, "AI-generated Male Cow", "test_artifacts/ai_generated_samples/ai_male_cow.jpg", "REJECT"),
    (18, "AI-generated Female Cow", "test_artifacts/ai_generated_samples/ai_female_cow.jpg", "REJECT"),
    (19, "AI-generated Male Buffalo", "test_artifacts/ai_generated_samples/ai_male_buffalo.jpg", "REJECT"),
    (20, "AI-generated Female Buffalo", "test_artifacts/ai_generated_samples/ai_female_buffalo.jpg", "REJECT"),
    (21, "Goat", "test_non_livestock_samples/goat.jpg", "REJECT"),
    (22, "Sheep", "test_non_livestock_samples/sheep.jpg", "REJECT"),
    (23, "Dog", "test_non_livestock_samples/dog.jpg", "REJECT"),
    (24, "Cat", "test_non_livestock_samples/cat.jpg", "REJECT"),
    (25, "Horse", "test_non_livestock_samples/horse.jpg", "REJECT"),
    (26, "Human", "test_non_livestock_samples/person.jpg", "REJECT"),
    (27, "Car", "test_non_livestock_samples/car.jpg", "REJECT"),
    (28, "Bus", "test_non_livestock_samples/bus.jpg", "REJECT"),
    (29, "Chair", "test_non_livestock_samples/chair.jpg", "REJECT"),
    (30, "Random object", "test_non_livestock_samples/random_object.jpg", "REJECT"),
]

def run_30_tests():
    print("=" * 110)
    print("RUNNING ALL 30 REQUIRED TEST CASES (SECTION 16)")
    print("=" * 110)
    
    results = []
    all_passed = True
    
    for idx, name, path, expected in test_cases:
        if not os.path.exists(path):
            print(f"[{idx:2d}] {name:30s} -> FILE NOT FOUND: {path}")
            all_passed = False
            continue
            
        res = pipeline.predict_image(path)
        status = res.get('status')
        species = res.get('species')
        gender = res.get('gender')
        err = res.get('error')
        
        actual = "ACCEPT" if status == "success" else "REJECT"
        passed = (actual == expected)
        
        # Additional checks:
        # If REJECT, gender MUST NOT be assigned (must be None)
        if actual == "REJECT" and gender not in (None, "None"):
            passed = False
            err = f"LEAK! Gender was assigned: {gender}"
            
        if not passed:
            all_passed = False
            
        pass_str = "PASS" if passed else "FAIL"
        print(f"[{pass_str:4s}] Test {idx:2d}: {name:28s} | Expected: {expected:6s} | Actual: {actual:6s} | Species: {str(species):8s} | Gender: {str(gender):8s} | Error: {str(err)}")
        results.append((idx, name, expected, actual, species, gender, passed, err))
        
    print("=" * 110)
    print(f"30 TEST SUITE RESULT: {'ALL 30 PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 110)
    return all_passed

def run_10_times_test():
    print("\n" + "=" * 110)
    print("RUNNING 10-TIMES REPEAT TEST (SECTION 17)")
    print("=" * 110)
    
    ten_tests = [
        ("Face-only Cow", "test_artifacts/cow_head_only.jpg", "REJECT"),
        ("Face-only Buffalo", "test_artifacts/buffalo_head_only.jpg", "REJECT"),
        ("AI-generated Cow", "test_artifacts/ai_generated_samples/ai_cow.jpg", "REJECT"),
        ("AI-generated Buffalo", "test_artifacts/ai_generated_samples/ai_buffalo.jpg", "REJECT"),
    ]
    
    all_10_passed = True
    for label, path, expected in ten_tests:
        print(f"\n--- Testing {label} ({path}) 10 times ---")
        runs = []
        for i in range(1, 11):
            res = pipeline.predict_image(path)
            actual = "ACCEPT" if res.get('status') == 'success' else "REJECT"
            g = res.get('gender')
            sp = res.get('species')
            err = res.get('error')
            runs.append((actual, g, sp, err))
            print(f"  Run {i:2d}: decision={actual:6s} | gender={str(g):6s} | species={str(sp):8s} | error={err}")
            
        first = runs[0]
        deterministic = all(r == first for r in runs)
        matches_expected = (first[0] == expected and first[1] is None)
        passed = deterministic and matches_expected
        if not passed:
            all_10_passed = False
        print(f"  Result: {'PASS' if passed else 'FAIL'} (Deterministic: {deterministic}, Matches Expected: {matches_expected})")
        
    print("=" * 110)
    print(f"10-TIMES TEST SUITE RESULT: {'ALL PASSED' if all_10_passed else 'SOME FAILED'}")
    print("=" * 110)
    return all_10_passed

if __name__ == '__main__':
    ok30 = run_30_tests()
    ok10 = run_10_times_test()
    print(f"\nFINAL VERDICT: ALL TESTS PASSED = {ok30 and ok10}")
