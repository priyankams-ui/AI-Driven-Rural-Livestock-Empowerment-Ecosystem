import os, glob
from PIL import Image

def audit_dataset_images():
    base = "gender_dataset"
    print("="*90)
    print("DETAILED DATASET IMAGE AUDIT")
    print("="*90)
    for root, dirs, files in os.walk(base):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, base)
            try:
                im = Image.open(p)
                w, h = im.size
                print(f"{rel:45s} | size: {w:4d}x{h:<4d} | bytes: {os.path.getsize(p):6d}")
            except Exception as e:
                print(f"{rel:45s} | ERROR: {e}")

if __name__ == '__main__':
    audit_dataset_images()
