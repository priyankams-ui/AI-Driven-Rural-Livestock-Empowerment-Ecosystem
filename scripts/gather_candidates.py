import os, glob
from PIL import Image

def find_verified_images():
    # Let's inspect images in Cattle Breeds, buffalo, data/new_images
    print("Gathering candidate images...")
    
    # Let's inspect data/new_images first
    new_img_dir = "data/new_images"
    for f in sorted(os.listdir(new_img_dir)):
        if f.endswith(('.jpg', '.webp')):
            p = os.path.join(new_img_dir, f)
            im = Image.open(p)
            print(f"new_images: {f:20s} | size: {im.size}")

if __name__ == '__main__':
    find_verified_images()
