import os
import sys
from PIL import Image
import numpy as np

ref_images = [
    ('data/new_images/femalebuffolo.jpg', 'Female Buffalo 0'),
    ('data/new_images/femalebuffolo1.jpg', 'Female Buffalo 1'),
    ('data/new_images/femalebuffolo2.jpg', 'Female Buffalo 2'),
    ('data/new_images/malebuffolo.jpg', 'Male Buffalo 0'),
    ('data/new_images/malebuffolo1.jpg', 'Male Buffalo 1'),
    ('data/new_images/malebuffolo2.jpg', 'Male Buffalo 2'),
    ('data/new_images/malebuffolo3.jpg', 'Male Buffalo 3'),
]

print("=" * 80)
print("AUDITING REFERENCE BUFFALO IMAGES")
print("=" * 80)

for p, name in ref_images:
    if not os.path.exists(p):
        print(f"{name:20s}: File {p} not found")
        continue
    im = Image.open(p)
    w, h = im.size
    print(f"{name:20s} | Path: {p} | Size: {w}x{h} | Aspect: {w/h:.2f}")

