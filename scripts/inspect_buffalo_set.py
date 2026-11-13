import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, 'scripts')
from test_unified_gender_architectures import buffalo_females, buffalo_males

print(f"=== BUFFALO FEMALES ({len(buffalo_females)}) ===")
for i, p in enumerate(buffalo_females):
    im = Image.open(p)
    print(f"F{i+1:02d}: {p} | {im.size} | {im.mode}")

print(f"\n=== BUFFALO MALES ({len(buffalo_males)}) ===")
for i, p in enumerate(buffalo_males):
    im = Image.open(p)
    print(f"M{i+1:02d}: {p} | {im.size} | {im.mode}")
