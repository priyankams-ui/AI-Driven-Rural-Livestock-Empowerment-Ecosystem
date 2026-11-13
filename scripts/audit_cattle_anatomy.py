import os, sys, glob
from PIL import Image
import numpy as np
import cv2

def inspect_cattle_images():
    print("=================================================================")
    print("INSPECTING CATTLE IMAGES FOR ANATOMICAL GROUND TRUTH")
    print("=================================================================")
    
    breeds = ['Ayrshire cattle', 'Brown Swiss cattle', 'Holstein Friesian cattle', 'Jersey cattle', 'Red Dane cattle']
    
    for b in breeds:
        folder = os.path.join('Cattle Breeds', b)
        files = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        print(f"\nBreed: {b} -> {len(files)} files")
        
        # Check first 10 files
        for f in files[:8]:
            try:
                im = Image.open(f)
                w, h = im.size
                print(f"  {os.path.basename(f):30s} | {w}x{h}")
            except Exception as e:
                print(f"  {f}: error {e}")

if __name__ == '__main__':
    inspect_cattle_images()
