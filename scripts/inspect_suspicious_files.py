import os
import sys
import json
import hashlib
from PIL import Image
import numpy as np

# Let's inspect the 9 suspicious files and determine their actual anatomy
suspicious_files = [
    ("buffalo/Jaffarabadi/Dairy_Farm_with_Jafarabadi_Bull_.jpg", "Male"),
    ("buffalo/Jaffarabadi/Jafarabadi_Bull_SS-ABC_Dams_Yield_.jpg", "Male"),
    ("buffalo/murrah/Murrah_Buffalo_-_Female_Breed_High_.jpg", "Male"),
    ("buffalo/nili-ravi/HSB_Nili_Ravi_Bull-Parass_Mother_ (1).jpg", "Male"),
    ("buffalo/nili-ravi/HSB_Nili_Ravi_Bull-Parass_Mother_ (3).jpg", "Male"),
    ("buffalo/nili-ravi/Nili_Ravi_Female_Calf_Niliravi.jpg", "Male"),
    ("buffalo/nili-ravi/Nili_Ravi_Female_Calf_niliravi_.jpg", "Male"),
    ("buffalo/nili-ravi/Pure_Nili_Ravi_Buffaloes_Gender_Female_ (2).jpg", "Male"),
    ("buffalo/surti/Milking_Murrah_Buffalo_Female_750kg_.jpg", "Male"),
]

for p, current_label in suspicious_files:
    if os.path.exists(p):
        im = Image.open(p)
        print(f"File: {p}")
        print(f"  Current label in catalog: {current_label}")
        print(f"  Size: {im.size}")
