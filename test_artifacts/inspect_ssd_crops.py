import os
import sys
sys.path.insert(0, '.')
import torch
from PIL import Image
from ml_pipeline import pipeline

files = [
    'test_artifacts/test_head_horns_landscape.jpg',
    'test_artifacts/test_front_half.jpg',
    'test_artifacts/test_rear_half.jpg',
    'test_artifacts/test_torso_only_no_head.jpg',
    'test_artifacts/test_m_head_horns.jpg',
    'test_artifacts/test_m_front_half.jpg',
    'test_artifacts/test_m_rear_half.jpg',
    'gender_dataset/uncertain/cow_head_only.jpg',
    'gender_dataset/uncertain/buffalo_head_only.jpg',
]

for p in files:
    im = Image.open(p).convert('RGB')
    w, h = im.size
    ssd_batch = [pipeline.ssd_transform(im).to(pipeline.device)]
    with torch.no_grad():
        pred = pipeline.ssd_detector(ssd_batch)[0]
    boxes = pred['boxes'].cpu()
    labels = pred['labels'].cpu()
    scores = pred['scores'].cpu()
    
    print(f"\n--- {p} (w={w}, h={h}) ---")
    for i in range(len(boxes)):
        sc = float(scores[i])
        if sc > 0.2:
            cat = pipeline.ssd_categories[labels[i]]
            b = [round(x, 1) for x in boxes[i].tolist()]
            bw, bh = b[2] - b[0], b[3] - b[1]
            ar = bw / bh if bh > 0 else 0
            w_rat = bw / w
            h_rat = bh / h
            print(f"  {cat:10s} sc={sc:.3f} box={b} bw={bw:.1f} bh={bh:.1f} ar={ar:.2f} w_ratio={w_rat:.2f} h_ratio={h_rat:.2f}")
