import os
import sys
sys.path.insert(0, '.')
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from ml_pipeline import pipeline

# 1. Species & Negative Validation
species_tests = [
    # Cows
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 'Cow'),
    ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 'Cow'),
    ('Cattle Breeds/Jersey cattle/Jerseycattle0.jpg', 'Cow'),
    ('data/new_images/malecow.jpg', 'Cow'),
    ('data/new_images/femalecow.jpg', 'Cow'),
    # Buffaloes
    ('buffalo/banni/baani-buffalo-bulls_0.jpg', 'Buffalo'),
    ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 'Buffalo'),
    ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 'Buffalo'),
    ('data/new_images/malebuffolo1.jpg', 'Buffalo'),
    ('data/new_images/femalebuffolo2.jpg', 'Buffalo'),
    # Other / Invalid
    ('test_artifacts/test_car.jpg', 'Rejected'),
    ('test_artifacts/test_object.jpg', 'Rejected'),
    ('data/cleaned/non_livestock_negatives/dog_1.jpg' if os.path.exists('data/cleaned/non_livestock_negatives/dog_1.jpg') else 'test_artifacts/test_car.jpg', 'Rejected'),
    ('data/cleaned/non_livestock_negatives/horse_1.jpg' if os.path.exists('data/cleaned/non_livestock_negatives/horse_1.jpg') else 'test_artifacts/test_object.jpg', 'Rejected'),
]

sp_true, sp_pred = [], []
for p, exp in species_tests:
    res = pipeline.predict_image(p)
    sp_true.append(exp)
    if res['status'] == 'rejected':
        sp_pred.append('Rejected')
    else:
        sp_pred.append(res['species'])

print('=== SPECIES METRICS ===')
labels_sp = ['Cow', 'Buffalo', 'Rejected']
prec_sp, rec_sp, f1_sp, sup_sp = precision_recall_fscore_support(sp_true, sp_pred, labels=labels_sp)
for i, l in enumerate(labels_sp):
    print(f'{l:10s} -> Precision: {prec_sp[i]*100:.2f}%, Recall: {rec_sp[i]*100:.2f}%, F1: {f1_sp[i]*100:.2f}%, Support: {sup_sp[i]}')

# 2. Full-Body vs Partial Image Validation
full_body_tests = [
    # Full Accept
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', True),
    ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', True),
    ('buffalo/banni/baani-buffalo-bulls_0.jpg', True),
    ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', True),
    ('data/new_images/malecow.jpg', True),
    ('data/new_images/femalecow.jpg', True),
    # Partial Reject
    ('test_artifacts/half_cow_front.jpg', False),
    ('test_artifacts/half_cow_rear.jpg', False),
    ('test_artifacts/half_buffalo.jpg', False),
    ('test_artifacts/cow_head_only.jpg', False),
    ('test_artifacts/buffalo_head_only.jpg', False),
    ('test_artifacts/cropped_cow.jpg', False),
]

fb_true, fb_pred = [], []
fb_errs = []
for p, exp_accept in full_body_tests:
    res = pipeline.predict_image(p)
    is_acc = (res['status'] == 'success')
    fb_true.append(exp_accept)
    fb_pred.append(is_acc)
    if not is_acc:
        fb_errs.append(res.get('error'))

print('\n=== FULL-BODY VALIDATION METRICS ===')
acc_fb = accuracy_score(fb_true, fb_pred)
print(f'Full-Body Acceptance/Rejection Accuracy: {acc_fb*100:.2f}%')
print(f'Full Image Acceptance Rate: {sum(1 for t, p in zip(fb_true, fb_pred) if t and p)} / {sum(1 for t in fb_true if t)} ({100*sum(1 for t, p in zip(fb_true, fb_pred) if t and p)/sum(1 for t in fb_true if t):.1f}%)')
print(f'Partial Image Rejection Rate: {sum(1 for t, p in zip(fb_true, fb_pred) if not t and not p)} / {sum(1 for t in fb_true if not t)} ({100*sum(1 for t, p in zip(fb_true, fb_pred) if not t and not p)/sum(1 for t in fb_true if not t):.1f}%)')
print(f'Rejection error message sample: \"{fb_errs[0]}\"')

# 3. Independent Gender Detection Metrics
print('\n=== INDEPENDENT GENDER METRICS ===')
gender_tests = [
    # Male Cow
    ('data/new_images/malecow2.webp', 'Male'),
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 'Male'),
    # Female Cow
    ('Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 'Female'),
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 'Female'),
    # Male Buffalo
    ('buffalo/banni/baani-buffalo-bulls_0.jpg', 'Male'),
    ('buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 'Male'),
    # Female Buffalo
    ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 'Female'),
    ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 'Female'),
    # Uncertain (Headshots / non-visible ventral anatomy)
    ('static/uploads/5da299733e69416d849ad4602bbd6459.jpg', 'Uncertain'),
    ('data/new_images/femalecow1.jpg', 'Uncertain'),
    ('data/new_images/malebuffolo.jpg', 'Uncertain'),
    ('data/new_images/femalebuffolo.jpg', 'Uncertain'),
]

g_true, g_pred = [], []
for p, exp in gender_tests:
    res = pipeline.predict_image(p)
    g_true.append(exp)
    pred_g = res.get('gender', 'Uncertain')
    g_pred.append(pred_g)

lbls_g = ['Male', 'Female', 'Uncertain']
prec_g, rec_g, f1_g, sup_g = precision_recall_fscore_support(g_true, g_pred, labels=lbls_g)
acc_g = accuracy_score(g_true, g_pred)
w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(g_true, g_pred, average='weighted')
cm_g = confusion_matrix(g_true, g_pred, labels=lbls_g)

print(f'Overall Accuracy   : {acc_g*100:.2f}%')
print(f'Weighted Precision : {w_prec*100:.2f}%')
print(f'Weighted Recall    : {w_rec*100:.2f}%')
print(f'Weighted F1-Score  : {w_f1*100:.2f}%')
for i, l in enumerate(lbls_g):
    print(f'{l:10s} -> Precision: {prec_g[i]*100:.2f}%, Recall: {rec_g[i]*100:.2f}%, F1: {f1_g[i]*100:.2f}%, Support: {sup_g[i]}')
print('Confusion Matrix (Rows=True, Cols=Pred [Male, Female, Uncertain]):')
print(cm_g)
