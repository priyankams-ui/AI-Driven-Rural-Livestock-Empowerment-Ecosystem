import sys
sys.path.insert(0, '.')
from PIL import Image
import torch
from ml_pipeline import pipeline

images = [
    ('test_artifacts/half_cow_front.jpg', False),
    ('test_artifacts/half_cow_rear.jpg', False),
    ('test_artifacts/half_buffalo.jpg', False),
    ('test_artifacts/cow_head_only.jpg', False),
    ('test_artifacts/buffalo_head_only.jpg', False),
    ('test_artifacts/cropped_cow.jpg', False),
    ('Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', True),
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', True),
    ('data/new_images/femalecow.jpg', True),
    ('data/new_images/femalecow1.jpg', True),
    ('data/new_images/femalecow2.jpg', True),
]

def check_full_body(b, w, h):
    bw, bh = b[2]-b[0], b[3]-b[1]
    aspect_ratio = bw / bh if bh > 0 else 0
    w_ratio = bw / w
    h_ratio = bh / h
    x1_norm = b[0] / w
    x2_norm = b[2] / w
    y1_norm = b[1] / h
    y2_norm = b[3] / h

    # Extreme close up
    if w_ratio > 0.98 and h_ratio > 0.98:
        return False, 'extreme_close_up'
    # Tiny / unclear animal
    if w_ratio < 0.20 or h_ratio < 0.20:
        return False, 'tiny_animal'
    # Severed / partial cuts at borders (e.g. half cow, head cut off)
    if w_ratio < 0.82:
        if (x1_norm <= 0.02 and x2_norm < 0.75) or (x2_norm >= 0.98 and x1_norm > 0.20):
            return False, 'border_truncated'
    if aspect_ratio < 1.05 and w_ratio < 0.82:
        if x1_norm <= 0.02 or x2_norm >= 0.98:
            return False, 'aspect_truncated'
    return True, 'full_body'

all_pass = True
for p, exp in images:
    im = Image.open(p)
    w, h = im.size
    ssd_batch = [pipeline.ssd_transform(im).to(pipeline.device)]
    with torch.no_grad():
        pred = pipeline.ssd_detector(ssd_batch)[0]
    boxes = pred['boxes'].cpu()
    labels = pred['labels'].cpu()
    scores = pred['scores'].cpu()
    cows = []
    for i in range(len(boxes)):
        sc = float(scores[i])
        if sc > 0.25:
            cat = pipeline.ssd_categories[labels[i]]
            if cat in ['cow', 'horse', 'sheep']:
                cows.append((cat, sc, boxes[i].tolist()))
    if not cows:
        res = (False, 'no_detection')
    else:
        best = max(cows, key=lambda x: x[1])
        res = check_full_body(best[2], w, h)
    match = (exp == res[0])
    if not match: all_pass = False
    print(f"{p:45s}: Exp={str(exp):5s}, Got={str(res[0]):5s} ({res[1]}), Match={match}")

print(f"\nAll tests pass: {all_pass}")
