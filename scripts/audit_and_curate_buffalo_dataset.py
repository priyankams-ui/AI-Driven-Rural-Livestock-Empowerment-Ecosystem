import os
import sys
import hashlib
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

sys.path.insert(0, '.')
from ml_pipeline import pipeline

def md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def dhash(image, hash_size=8):
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(image)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def main():
    print("=" * 80)
    print("STEP 1: AUDITING ALL CANDIDATES IN BUFFALO REPO & GENDER DATASET")
    print("=" * 80)
    
    # Check gender_dataset/buffalo
    gd_females = [os.path.join('gender_dataset/buffalo/female', f) for f in os.listdir('gender_dataset/buffalo/female') if f.lower().endswith(('.jpg', '.png'))]
    gd_males = [os.path.join('gender_dataset/buffalo/male', f) for f in os.listdir('gender_dataset/buffalo/male') if f.lower().endswith(('.jpg', '.png'))]
    
    print(f"gender_dataset/buffalo: {len(gd_females)} females, {len(gd_males)} males")
    
    # Check new_images
    ni_females = [f'data/new_images/{f}' for f in ['femalebuffolo.jpg', 'femalebuffolo1.jpg', 'femalebuffolo2.jpg'] if os.path.exists(f'data/new_images/{f}')]
    ni_males = [f'data/new_images/{f}' for f in ['malebuffolo.jpg', 'malebuffolo1.jpg', 'malebuffolo2.jpg', 'malebuffolo3.jpg'] if os.path.exists(f'data/new_images/{f}')]
    print(f"data/new_images: {len(ni_females)} females, {len(ni_males)} males")
    
    # Candidate scan across breeds in buffalo/
    keywords_male = ['bull', 'male', 'sire', 'he_buffalo', 'breeding']
    keywords_female = ['female', 'cow', 'dairy', 'milking', 'she_buffalo', 'dam', 'mother', 'lactating', 'pregnant']
    
    cand_males = []
    cand_females = []
    for root, dirs, files in os.walk('buffalo'):
        for f in files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            p = os.path.join(root, f)
            fl = f.lower()
            if any(k in fl for k in keywords_male):
                cand_males.append(p)
            elif any(k in fl for k in keywords_female):
                cand_females.append(p)
                
    print(f"buffalo/ keyword candidates: {len(cand_females)} females, {len(cand_males)} males")

if __name__ == '__main__':
    main()
