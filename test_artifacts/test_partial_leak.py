import sys
sys.path.insert(0, '.')
from PIL import Image
from ml_pipeline import pipeline

im = Image.open('data/new_images/femalecow.jpg')
w, h = im.size

crops = {
    'head_horns_landscape': im.crop((0, 0, int(w*0.7), int(h*0.6))),
    'front_half': im.crop((0, 0, int(w*0.6), h)),
    'rear_half': im.crop((int(w*0.4), 0, w, h)),
    'torso_only_no_head': im.crop((int(w*0.25), int(h*0.2), int(w*0.85), int(h*0.8))),
    'legs_only': im.crop((0, int(h*0.6), w, h)),
}

for name, crop in crops.items():
    path = f'test_artifacts/test_{name}.jpg'
    crop.save(path)
    res = pipeline.predict_image(path)
    print(f"{name:25s} -> status={res.get('status')} sp={res.get('species')} g={res.get('gender')} err={res.get('error')}")

# Also test on male cow
im_m = Image.open('data/new_images/malecow.jpg')
wm, hm = im_m.size
m_crops = {
    'm_head_horns': im_m.crop((0, 0, int(wm*0.55), int(hm*0.65))),
    'm_front_half': im_m.crop((0, 0, int(wm*0.65), hm)),
    'm_rear_half': im_m.crop((int(wm*0.35), 0, wm, hm)),
}
for name, crop in m_crops.items():
    path = f'test_artifacts/test_{name}.jpg'
    crop.save(path)
    res = pipeline.predict_image(path)
    print(f"{name:25s} -> status={res.get('status')} sp={res.get('species')} g={res.get('gender')} err={res.get('error')}")

# Also test on buffaloes
im_b = Image.open('data/new_images/malebuffolo1.jpg')
wb, hb = im_b.size
b_crops = {
    'b_head_horns': im_b.crop((0, 0, int(wb*0.6), int(hb*0.6))),
    'b_front_half': im_b.crop((0, 0, int(wb*0.65), hb)),
    'b_rear_half': im_b.crop((int(wb*0.35), 0, wb, hb)),
}
for name, crop in b_crops.items():
    path = f'test_artifacts/test_{name}.jpg'
    crop.save(path)
    res = pipeline.predict_image(path)
    print(f"{name:25s} -> status={res.get('status')} sp={res.get('species')} g={res.get('gender')} err={res.get('error')}")
