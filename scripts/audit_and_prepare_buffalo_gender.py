import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline import pipeline

def inspect_candidates():
    print("=" * 70)
    print("SCANNING FOR GENUINE BUFFALO GENDER SAMPLES")
    print("=" * 70)
    
    males = []
    females = []
    for root, dirs, files in os.walk('buffalo'):
        for f in files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            fp = os.path.join(root, f)
            fl = f.lower()
            if any(k in fl for k in ['bull', 'male', 'sire', 'he_buffalo', 'breeding']):
                males.append(fp)
            elif any(k in fl for k in ['female', 'cow', 'dairy', 'milking', 'she_buffalo', 'dam', 'mother']):
                females.append(fp)
    
    print(f"Keyword search yielded: {len(males)} male candidates, {len(females)} female candidates.")
    
    def verify_list(file_list, target_gender):
        valid = []
        for p in file_list:
            try:
                im = Image.open(p).convert('RGB')
                w, h = im.size
                if w < 120 or h < 120:
                    continue
                im_cv = cv2.imread(p)
                if im_cv is None:
                    continue
                ok_q, _ = pipeline.check_image_quality(im, im_cv)
                if not ok_q:
                    continue
                
                cnt, cow_dets, other, is_single, r = pipeline.detect_and_count_animals(im, w, h)
                if cnt > 1:
                    continue # single animal only
                
                # Check species pipeline
                img_tensor = pipeline.transform(im).unsqueeze(0).to(pipeline.device)
                with torch.no_grad():
                    f_res = pipeline.resnet(img_tensor).cpu().numpy()
                    f_mob = torch.flatten(pipeline.mobilenet_feat(img_tensor), 1).cpu().numpy()
                    feat = np.hstack([f_res, f_mob])
                sp_pred = pipeline.species_clf.predict(feat)[0]
                sp_name = pipeline.rev_species_map.get(int(sp_pred), 'Unknown')
                if sp_name != 'Buffalo':
                    continue
                
                valid.append((p, w, h, os.path.basename(p)))
            except Exception as e:
                continue
        return valid

    valid_males = verify_list(males, 'Male')
    valid_females = verify_list(females, 'Female')
    print(f"Pipeline-verified single buffaloes: {len(valid_males)} males, {len(valid_females)} females")
    
    print("\nSample verified Male Buffalo images:")
    for p, w, h, fn in valid_males[:15]:
        print(f"  [M] {p} ({w}x{h})")
        
    print("\nSample verified Female Buffalo images:")
    for p, w, h, fn in valid_females[:15]:
        print(f"  [F] {p} ({w}x{h})")

if __name__ == '__main__':
    inspect_candidates()
