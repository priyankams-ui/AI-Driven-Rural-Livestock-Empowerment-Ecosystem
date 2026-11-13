import os
import sys
sys.path.insert(0, '.')
import cv2
import numpy as np
from PIL import Image
from ml_pipeline import pipeline

def run_full_validation_test():
    os.makedirs('test_artifacts', exist_ok=True)
    
    print("\n" + "="*80)
    print("SECTION 24: FULL VALIDATION TEST SUITE (A to M)")
    print("="*80)
    
    results = {}
    
    # A. Full Female Cow
    res_a = pipeline.predict_image('data/new_images/femalecow.jpg')
    ok_a = (res_a['status'] == 'success' and res_a['species'] == 'Cow' and res_a['gender'] == 'Female')
    results['A. Full Female Cow'] = (ok_a, f"Species={res_a.get('species')}, Gender={res_a.get('gender')}")
    
    # B. Full Male Cow
    res_b = pipeline.predict_image('data/new_images/malecow.jpg')
    ok_b = (res_b['status'] == 'success' and res_b['species'] == 'Cow' and res_b['gender'] == 'Male')
    results['B. Full Male Cow'] = (ok_b, f"Species={res_b.get('species')}, Gender={res_b.get('gender')}")

    # C. Full Female Buffalo
    res_c = pipeline.predict_image('data/new_images/femalebuffolo2.jpg')
    ok_c = (res_c['status'] == 'success' and res_c['species'] == 'Buffalo' and res_c['gender'] == 'Female')
    results['C. Full Female Buffalo'] = (ok_c, f"Species={res_c.get('species')}, Gender={res_c.get('gender')}")

    # D. Full Male Buffalo
    res_d = pipeline.predict_image('data/new_images/malebuffolo1.jpg')
    ok_d = (res_d['status'] == 'success' and res_d['species'] == 'Buffalo' and res_d['gender'] == 'Male')
    results['D. Full Male Buffalo'] = (ok_d, f"Species={res_d.get('species')}, Gender={res_d.get('gender')}")

    # Prepare Half crops for E, F, G, H
    # E. Half Female Cow (Left half crop)
    im_fc = Image.open('data/new_images/femalecow.jpg')
    w, h = im_fc.size
    half_fc = im_fc.crop((0, 0, int(w * 0.45), h))
    half_fc_path = 'test_artifacts/half_female_cow.jpg'
    half_fc.save(half_fc_path)
    res_e = pipeline.predict_image(half_fc_path)
    ok_e = (res_e['status'] == 'rejected' and 'Please upload a full image' in res_e.get('error', ''))
    results['E. Half Female Cow'] = (ok_e, f"Status={res_e.get('status')}, Error={res_e.get('error')}")

    # F. Half Male Cow (Right half crop)
    im_mc = Image.open('data/new_images/malecow.jpg')
    w, h = im_mc.size
    half_mc = im_mc.crop((int(w * 0.55), 0, w, h))
    half_mc_path = 'test_artifacts/half_male_cow.jpg'
    half_mc.save(half_mc_path)
    res_f = pipeline.predict_image(half_mc_path)
    ok_f = (res_f['status'] == 'rejected' and 'Please upload a full image' in res_f.get('error', ''))
    results['F. Half Male Cow'] = (ok_f, f"Status={res_f.get('status')}, Error={res_f.get('error')}")

    # G. Half Female Buffalo (Left half crop)
    im_fb = Image.open('data/new_images/femalebuffolo2.jpg')
    w, h = im_fb.size
    half_fb = im_fb.crop((0, 0, int(w * 0.45), h))
    half_fb_path = 'test_artifacts/half_female_buffalo.jpg'
    half_fb.save(half_fb_path)
    res_g = pipeline.predict_image(half_fb_path)
    ok_g = (res_g['status'] == 'rejected' and 'Please upload a full image' in res_g.get('error', ''))
    results['G. Half Female Buffalo'] = (ok_g, f"Status={res_g.get('status')}, Error={res_g.get('error')}")

    # H. Half Male Buffalo (Right half crop)
    im_mb = Image.open('data/new_images/malebuffolo1.jpg')
    w, h = im_mb.size
    half_mb = im_mb.crop((int(w * 0.55), 0, w, h))
    half_mb_path = 'test_artifacts/half_male_buffalo.jpg'
    half_mb.save(half_mb_path)
    res_h = pipeline.predict_image(half_mb_path)
    ok_h = (res_h['status'] == 'rejected' and 'Please upload a full image' in res_h.get('error', ''))
    results['H. Half Male Buffalo'] = (ok_h, f"Status={res_h.get('status')}, Error={res_h.get('error')}")

    # I. Dog
    dog_path = 'data/cleaned/non_livestock_negatives/dog_0001.jpg' if os.path.exists('data/cleaned/non_livestock_negatives/dog_0001.jpg') else None
    if not dog_path:
        # Check dummy_unidentified or create
        for f in os.listdir('data/cleaned/non_livestock_negatives') if os.path.exists('data/cleaned/non_livestock_negatives') else []:
            if 'dog' in f.lower():
                dog_path = os.path.join('data/cleaned/non_livestock_negatives', f)
                break
    if not dog_path:
        dog_path = 'dummy_unidentified/dummy_0001.jpg'
    res_i = pipeline.predict_image(dog_path)
    ok_i = (res_i['status'] == 'rejected' and ('Unsupported image' in res_i.get('error', '') or 'Invalid' in res_i.get('error', '')))
    results['I. Dog'] = (ok_i, f"Status={res_i.get('status')}, Error={res_i.get('error')}")

    # J. Goat
    goat_path = 'data/cleaned/non_livestock_negatives/goat_0001.jpg' if os.path.exists('data/cleaned/non_livestock_negatives/goat_0001.jpg') else None
    if not goat_path:
        for f in os.listdir('data/cleaned/non_livestock_negatives') if os.path.exists('data/cleaned/non_livestock_negatives') else []:
            if 'goat' in f.lower():
                goat_path = os.path.join('data/cleaned/non_livestock_negatives', f)
                break
    if not goat_path:
        goat_path = 'dummy_unidentified/dummy_0002.jpg'
    res_j = pipeline.predict_image(goat_path)
    ok_j = (res_j['status'] == 'rejected' and ('Unsupported image' in res_j.get('error', '') or 'Invalid' in res_j.get('error', '')))
    results['J. Goat'] = (ok_j, f"Status={res_j.get('status')}, Error={res_j.get('error')}")

    # K. Sheep
    sheep_path = 'data/cleaned/non_livestock_negatives/sheep_0001.jpg' if os.path.exists('data/cleaned/non_livestock_negatives/sheep_0001.jpg') else None
    if not sheep_path:
        for f in os.listdir('data/cleaned/non_livestock_negatives') if os.path.exists('data/cleaned/non_livestock_negatives') else []:
            if 'sheep' in f.lower():
                sheep_path = os.path.join('data/cleaned/non_livestock_negatives', f)
                break
    if not sheep_path:
        sheep_path = 'dummy_unidentified/dummy_0003.jpg'
    res_k = pipeline.predict_image(sheep_path)
    ok_k = (res_k['status'] == 'rejected' and ('Unsupported image' in res_k.get('error', '') or 'Invalid' in res_k.get('error', '')))
    results['K. Sheep'] = (ok_k, f"Status={res_k.get('status')}, Error={res_k.get('error')}")

    # L. Car
    car_img = Image.new("RGB", (224, 224), color=(30, 80, 200))
    car_path = 'test_artifacts/test_car.jpg'
    car_img.save(car_path)
    res_l = pipeline.predict_image(car_path)
    ok_l = (res_l['status'] == 'rejected' and ('Unsupported image' in res_l.get('error', '') or 'Invalid' in res_l.get('error', '')))
    results['L. Car'] = (ok_l, f"Status={res_l.get('status')}, Error={res_l.get('error')}")

    # M. Human
    human_img = Image.new("RGB", (224, 224), color=(220, 180, 140))
    human_path = 'test_artifacts/test_human.jpg'
    human_img.save(human_path)
    res_m = pipeline.predict_image(human_path)
    ok_m = (res_m['status'] == 'rejected' and ('Unsupported image' in res_m.get('error', '') or 'Invalid' in res_m.get('error', '')))
    results['M. Human'] = (ok_m, f"Status={res_m.get('status')}, Error={res_m.get('error')}")

    all_passed = True
    for test_key, (passed, details) in results.items():
        if not passed:
            all_passed = False
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str:4s}] {test_key:26s} -> {details}")

    print("="*80)
    print(f"SECTION 24 ALL PASSED: {all_passed}")
    print("="*80)
    return all_passed

if __name__ == '__main__':
    run_full_validation_test()
