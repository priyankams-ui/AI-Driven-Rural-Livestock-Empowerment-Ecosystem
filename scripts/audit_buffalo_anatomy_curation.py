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

def dhash(image, hash_size=8):
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(image)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def audit():
    print("=" * 80)
    print("DETAILED AUDIT OF BUFFALO CANDIDATES")
    print("=" * 80)
    
    keywords_male = ['bull', 'male', 'sire', 'he_buffalo', 'breeding']
    keywords_female = ['female', 'cow', 'dairy', 'milking', 'she_buffalo', 'dam', 'mother', 'lactating', 'pregnant']
    
    # Gather all candidate paths
    pool = []
    
    # Reference images
    for fn in ['femalebuffolo.jpg', 'femalebuffolo1.jpg', 'femalebuffolo2.jpg']:
        pool.append(('data/new_images/' + fn, 'Female', 'Reference'))
    for fn in ['malebuffolo.jpg', 'malebuffolo1.jpg', 'malebuffolo2.jpg', 'malebuffolo3.jpg']:
        pool.append(('data/new_images/' + fn, 'Male', 'Reference'))
        
    # gender_dataset/buffalo
    for fn in os.listdir('gender_dataset/buffalo/female'):
        if fn.lower().endswith(('.jpg', '.png')):
            pool.append((os.path.join('gender_dataset/buffalo/female', fn), 'Female', 'GenderDataset'))
    for fn in os.listdir('gender_dataset/buffalo/male'):
        if fn.lower().endswith(('.jpg', '.png')):
            pool.append((os.path.join('gender_dataset/buffalo/male', fn), 'Male', 'GenderDataset'))
            
    # buffalo/
    for root, dirs, files in os.walk('buffalo'):
        breed = os.path.basename(root)
        for f in files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            fl = f.lower()
            p = os.path.join(root, f)
            if any(k in fl for k in keywords_male):
                pool.append((p, 'Male', breed))
            elif any(k in fl for k in keywords_female):
                pool.append((p, 'Female', breed))
                
    print(f"Total raw candidates in pool: {len(pool)}")
    
    # Deduplicate by absolute file path
    unique_pool = []
    seen_paths = set()
    for p, g, b in pool:
        norm = os.path.normpath(os.path.abspath(p))
        if norm not in seen_paths and os.path.exists(p):
            seen_paths.add(norm)
            unique_pool.append((p, g, b))
    print(f"Unique file paths: {len(unique_pool)}")
    
    # Check hashes for duplicate / near-duplicate detection
    hash_to_files = {}
    valid_entries = []
    
    for p, g, b in unique_pool:
        try:
            im = Image.open(p).convert('RGB')
            w, h = im.size
            if w < 100 or h < 100:
                continue
            h_val = dhash(im)
            im_cv = cv2.imread(p)
            if im_cv is None:
                continue
            
            # Check pipeline single animal & quality
            gray = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)
            lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if lap < 25.0: # blurry
                continue
            
            cnt, cow_dets, other, is_single, r = pipeline.detect_and_count_animals(im, w, h)
            if cnt > 1: # multi animal
                continue
                
            # Check species
            img_tensor = pipeline.transform(im).unsqueeze(0).to(pipeline.device)
            with torch.no_grad():
                f_res = pipeline.resnet(img_tensor).cpu().numpy()
                f_mob = torch.flatten(pipeline.mobilenet_feat(img_tensor), 1).cpu().numpy()
                feat = np.hstack([f_res, f_mob])
            sp_pred = pipeline.species_clf.predict(feat)[0]
            sp_name = pipeline.rev_species_map.get(int(sp_pred), 'Unknown')
            if sp_name != 'Buffalo':
                continue
                
            valid_entries.append((p, g, b, h_val, w, h))
        except Exception as e:
            continue
            
    print(f"Valid single Buffalo images verified by detection & species pipeline: {len(valid_entries)}")
    females = [e for e in valid_entries if e[1] == 'Female']
    males = [e for e in valid_entries if e[1] == 'Male']
    print(f"  Female candidates: {len(females)}")
    print(f"  Male candidates:   {len(males)}")
    
    # Check duplicate hashes
    hashes = {}
    for p, g, b, h_val, w, h in valid_entries:
        hashes.setdefault(h_val, []).append((p, g, b))
    
    dups = {k: v for k, v in hashes.items() if len(v) > 1}
    print(f"  Hash clusters with multiple images (near/exact duplicates): {len(dups)}")
    for k, v in list(dups.items())[:5]:
        print(f"    Cluster {k}: {[x[0] for x in v]}")

if __name__ == '__main__':
    audit()
