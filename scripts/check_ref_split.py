import json
import numpy as np

with open('models/authoritative_buffalo_gender_catalog.json', 'r') as f:
    catalog = json.load(f)

female_animals = [a for a in catalog if a['gender'] == 'Female']
male_animals = [a for a in catalog if a['gender'] == 'Male']

np.random.seed(42)
fem_entries = list(female_animals)
male_entries = list(male_animals)
np.random.shuffle(fem_entries)
np.random.shuffle(male_entries)

n_f_test = max(1, int(len(fem_entries) * 0.10))
n_f_val = max(1, int(len(fem_entries) * 0.10))
test_fem = fem_entries[:n_f_test]
val_fem = fem_entries[n_f_test:n_f_test + n_f_val]
train_fem = fem_entries[n_f_test + n_f_val:]

n_m_test = max(1, int(len(male_entries) * 0.10))
n_m_val = max(1, int(len(male_entries) * 0.10))
test_male = male_entries[:n_m_test]
val_male = male_entries[n_m_test:n_m_test + n_m_val]
train_male = male_entries[n_m_test + n_m_val:]

train_animals = train_fem + train_male
val_animals = val_fem + val_male
test_animals = test_fem + test_male

ref_paths = [
    'data/new_images/femalebuffolo.jpg',
    'data/new_images/femalebuffolo1.jpg',
    'data/new_images/femalebuffolo2.jpg',
    'data/new_images/malebuffolo.jpg',
    'data/new_images/malebuffolo1.jpg',
    'data/new_images/malebuffolo2.jpg',
    'data/new_images/malebuffolo3.jpg'
]

for rp in ref_paths:
    in_train = any(rp in a['primary'] for a in train_animals)
    in_val = any(rp in a['primary'] for a in val_animals)
    in_test = any(rp in a['primary'] for a in test_animals)
    print(f"{rp:35s} | Train: {in_train} | Val: {in_val} | Test: {in_test}")
