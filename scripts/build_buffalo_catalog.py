import os
import sys
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

def main():
    print("Gathering and inspecting candidate buffaloes...")
    keywords_male = ['bull', 'male', 'sire', 'he_buffalo', 'breeding']
    keywords_female = ['female', 'cow', 'dairy', 'milking', 'she_buffalo', 'dam', 'mother', 'lactating', 'pregnant']
    
    pool = []
    # Reference images
    for fn in ['femalebuffolo.jpg', 'femalebuffolo1.jpg', 'femalebuffolo2.jpg']:
        pool.append(('data/new_images/' + fn, 'Female', 'Reference'))
    for fn in ['malebuffolo.jpg', 'malebuffolo1.jpg', 'malebuffolo2.jpg', 'malebuffolo3.jpg']:
        pool.append(('data/new_images/' + fn, 'Male', 'Reference'))
        
    # gender_dataset
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

    # Deduplicate paths
    seen_paths = set()
    unique_pool = []
    for p, g, b in pool:
        norm = os.path.normpath(os.path.abspath(p))
        if norm not in seen_paths and os.path.exists(p):
            seen_paths.add(norm)
            unique_pool.append((p.replace('\\', '/'), g, b))

    # Pipeline validation
    verified = []
    hash_map = {}
    for p, g, b in unique_pool:
        try:
            im = Image.open(p).convert('RGB')
            w, h = im.size
            if w < 100 or h < 100:
                continue
            im_cv = cv2.imread(p)
            if im_cv is None:
                continue
            gray = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)
            lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if lap < 25.0:
                continue
            cnt, cow_dets, other, is_single, r = pipeline.detect_and_count_animals(im, w, h)
            if cnt != 1:
                continue
            img_tensor = pipeline.transform(im).unsqueeze(0).to(pipeline.device)
            with torch.no_grad():
                f_res = pipeline.resnet(img_tensor).cpu().numpy()
                f_mob = torch.flatten(pipeline.mobilenet_feat(img_tensor), 1).cpu().numpy()
                feat = np.hstack([f_res, f_mob])
            sp_pred = pipeline.species_clf.predict(feat)[0]
            sp_name = pipeline.rev_species_map.get(int(sp_pred), 'Unknown')
            if sp_name != 'Buffalo':
                continue
                
            h_val = dhash(im)
            verified.append({
                'path': p,
                'gender': g,
                'breed': b,
                'hash': h_val,
                'w': w,
                'h': h
            })
            hash_map.setdefault(h_val, []).append(p)
        except Exception:
            continue
            
    print(f"Total verified single buffaloes: {len(verified)}")
    
    # Group by hash to identify duplicates / twins
    base_animals = []
    seen_hashes = set()
    for item in verified:
        h_val = item['hash']
        if h_val in seen_hashes:
            continue
        seen_hashes.add(h_val)
        all_with_hash = hash_map[h_val]
        primary = all_with_hash[0]
        twin = all_with_hash[1] if len(all_with_hash) > 1 else None
        base_animals.append({
            'base_id': f"Buff_{item['gender']}_{len(base_animals):03d}",
            'gender': item['gender'],
            'primary': primary,
            'twin': twin,
            'breed': item['breed'],
            'hash': h_val
        })
        
    print(f"Unique base animal clusters: {len(base_animals)}")
    f_count = sum(1 for a in base_animals if a['gender'] == 'Female')
    m_count = sum(1 for a in base_animals if a['gender'] == 'Male')
    print(f"  Female unique base animals: {f_count}")
    print(f"  Male unique base animals:   {m_count}")
    
    # Save base animals catalog
    import json
    with open('models/buffalo_catalog_raw.json', 'w') as f:
        json.dump(base_animals, f, indent=2)
    print("Saved catalog to models/buffalo_catalog_raw.json")

if __name__ == '__main__':
    main()
