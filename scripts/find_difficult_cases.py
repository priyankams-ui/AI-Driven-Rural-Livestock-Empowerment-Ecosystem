import os, glob
from PIL import Image

def find_front_and_difficult_cases():
    # Let's inspect images in Cattle Breeds and buffalo
    print("Finding difficult and non-side cases:")
    # We can check images with aspect ratio < 1.0 (portrait/front view) or specific folders
    candidates = []
    for sp_dir in ['Cattle Breeds', 'buffalo']:
        for root, dirs, files in os.walk(sp_dir):
            for f in files:
                if f.endswith(('.jpg', '.webp', '.png')):
                    p = os.path.join(root, f)
                    try:
                        im = Image.open(p)
                        w, h = im.size
                        # aspect ratio close to 1 or < 1.0 often indicates front-facing pose
                        if 0.7 <= w/h <= 1.05 and w >= 200 and h >= 200:
                            candidates.append((p, w, h, w/h))
                    except:
                        pass
    print(f"Found {len(candidates)} portrait / square candidate images (potential front-facing/headshot).")
    for p, w, h, ar in candidates[:15]:
        print(f"  {p:65s} | size: {w}x{h} | AR: {ar:.2f}")

if __name__ == '__main__':
    find_front_and_difficult_cases()
