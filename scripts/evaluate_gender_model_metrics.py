import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from ml_pipeline import pipeline

# 4 Categories + Difficult / Insufficient anatomy cases
test_cases = [
    # A. Male Cow
    ('data/new_images/malecow.jpg', 'Male', 'Male Cow - Side view Zebu bull'),
    ('data/new_images/malecow1.jpg', 'Male', 'Male Cow - Brahman bull'),
    ('data/new_images/malecow2.jpg', 'Male', 'Male Cow - Hereford bull'),
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 'Male', 'Male Cow - Ayrshire bull'),
    
    # B. Female Cow
    ('data/new_images/femalecow.jpg', 'Female', 'Female Cow - Side view dairy cow'),
    ('data/new_images/femalecow1.jpg', 'Female', 'Female Cow - Jersey cow with udder'),
    ('data/new_images/femalecow2.jpg', 'Female', 'Female Cow - Large pasture cow'),
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 'Female', 'Female Cow - Ayrshire cow with udder'),
    ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg', 'Female', 'Female Cow - Brown Swiss with udder'),
    ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 'Female', 'Female Cow - Holstein with udder'),

    # C. Male Buffalo
    ('data/new_images/malebuffolo.jpg', 'Male', 'Male Buffalo - Murrah bull side profile'),
    ('data/new_images/malebuffolo1.jpg', 'Male', 'Male Buffalo - Large bull in pen'),
    ('data/new_images/malebuffolo2.jpg', 'Male', 'Male Buffalo - Field bull'),
    ('data/new_images/malebuffolo3.jpg', 'Male', 'Male Buffalo - Working bull'),
    ('buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 'Male', 'Male Buffalo - Jaffarabadi bull'),

    # D. Female Buffalo
    ('data/new_images/femalebuffolo.jpg', 'Female', 'Female Buffalo - Dairy buffalo with udder'),
    ('data/new_images/femalebuffolo1.jpg', 'Female', 'Female Buffalo - Murrah buffalo with udder'),
    ('data/new_images/femalebuffolo2.jpg', 'Female', 'Female Buffalo - Milking buffalo'),
    ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 'Female', 'Female Buffalo - Banni buffalo with udder'),
    ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 'Female', 'Female Buffalo - Murrah buffalo with udder'),

    # E. Difficult / Insufficient anatomical evidence cases (Should be rejected / undetermined)
    ('test_artifacts/half_cow_front.jpg', 'Uncertain', 'Partial Cow - Front half only (no ventral anatomy)'),
    ('test_artifacts/half_buffalo.jpg', 'Uncertain', 'Partial Buffalo - Front half only (no ventral anatomy)'),
    ('test_artifacts/blurry_cow.jpg', 'Uncertain', 'Blurry Cow - Severe blur obscuring features'),
    ('test_artifacts/blurry_buffalo.jpg', 'Uncertain', 'Blurry Buffalo - Severe blur obscuring features'),
]

print(f"Total evaluation cases: {len(test_cases)}")
y_true_binary, y_pred_binary = [], []
uncertain_count = 0
correct_determinations = 0
correct_rejections = 0

for p, exp, desc in test_cases:
    res = pipeline.predict_image(p)
    status = res.get('status')
    gender = res.get('gender')
    g_conf = res.get('gender_confidence', 0.0)
    
    if exp == 'Uncertain':
        if status == 'rejected' or gender == 'Uncertain' or not res.get('is_gender_confident'):
            correct_rejections += 1
            uncertain_count += 1
            res_str = 'CORRECT_REJECT/UNCERTAIN'
        else:
            res_str = f'WRONG_FORCED_DECISION ({gender})'
    else:
        if status == 'success' and gender == exp:
            correct_determinations += 1
            y_true_binary.append(0 if exp == 'Female' else 1)
            y_pred_binary.append(0 if gender == 'Female' else 1)
            res_str = 'CORRECT'
        elif gender == 'Uncertain' or status == 'rejected':
            uncertain_count += 1
            res_str = 'UNCERTAIN'
        else:
            y_true_binary.append(0 if exp == 'Female' else 1)
            y_pred_binary.append(0 if gender == 'Female' else 1)
            res_str = f'WRONG_GENDER ({gender})'
            
    print(f"{desc:55s} | Exp: {exp:9s} | Got: {str(gender):9s} ({g_conf:.2f}) | {res_str}")

acc = accuracy_score(y_true_binary, y_pred_binary)
prec, rec, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
prec_fem, rec_fem, f1_fem, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, labels=[0], average=None, zero_division=0)
prec_male, rec_male, f1_male, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, labels=[1], average=None, zero_division=0)
cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

male_count = sum(1 for _, e, _ in test_cases if e == 'Male')
female_count = sum(1 for _, e, _ in test_cases if e == 'Female')

print('\n' + '=' * 60)
print('EVALUATION METRICS SUMMARY (SECTION 10)')
print('=' * 60)
print(f"Total Test Images Analyzed     : {len(test_cases)}")
print(f"  * Male Images                : {male_count}")
print(f"  * Female Images              : {female_count}")
print(f"  * Uncertain / Rejected Images: {uncertain_count}")
print(f"  * Correct Validated Predictions: {correct_determinations}/{male_count + female_count}")
print(f"  * Correct Rejections/Uncertain: {correct_rejections}/{uncertain_count}")
print('-' * 60)
print(f"Overall Accuracy               : {acc * 100:.2f}%")
print(f"Overall Precision              : {prec * 100:.2f}%")
print(f"Overall Recall                 : {rec * 100:.2f}%")
print(f"Overall F1-Score               : {f1 * 100:.2f}%")
print(f"Female Class (0) -> Precision: {prec_fem[0]*100:.2f}%, Recall: {rec_fem[0]*100:.2f}%, F1: {f1_fem[0]*100:.2f}%")
print(f"Male Class (1)   -> Precision: {prec_male[0]*100:.2f}%, Recall: {rec_male[0]*100:.2f}%, F1: {f1_male[0]*100:.2f}%")
print(f"Confusion Matrix (Rows=True [Female, Male], Cols=Pred [Female, Male]):\n{cm}")
print('=' * 60)
