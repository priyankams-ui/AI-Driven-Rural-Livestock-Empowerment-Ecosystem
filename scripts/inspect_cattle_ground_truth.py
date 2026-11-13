import os, sys, glob
from PIL import Image
import numpy as np

def inspect_breed_samples():
    breeds = ['Ayrshire cattle', 'Brown Swiss cattle', 'Holstein Friesian cattle', 'Jersey cattle', 'Red Dane cattle']
    
    # We want to check unique base animals
    for b in breeds:
        folder = os.path.join('Cattle Breeds', b)
        # get unique base files (without _c)
        bases = sorted([f for f in os.listdir(folder) if not f.endswith('_c.jpg') and f.endswith('.jpg')])
        print(f"\n{b}: {len(bases)} unique animals.")
        print(f"  First 10: {bases[:10]}")

if __name__ == '__main__':
    inspect_breed_samples()
