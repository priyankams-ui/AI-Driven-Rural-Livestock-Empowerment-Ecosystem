import os
import cv2
import numpy as np
from PIL import Image

os.makedirs('test_artifacts', exist_ok=True)

# 1. Two buffaloes
b1_path = 'buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg'
b2_path = 'buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg'
im_b1 = Image.open(b1_path).convert('RGB').resize((320, 320))
im_b2 = Image.open(b2_path).convert('RGB').resize((320, 320))
c_2b = Image.new('RGB', (640, 320))
c_2b.paste(im_b1, (0, 0))
c_2b.paste(im_b2, (320, 0))
c_2b.save('test_artifacts/two_buffaloes.jpg')

# 2. Cow + buffalo
c1_path = 'Cattle Breeds/Ayrshire cattle/Ayrshirecattle0.jpg'
im_c1 = Image.open(c1_path).convert('RGB').resize((320, 320))
c_cb = Image.new('RGB', (640, 320))
c_cb.paste(im_c1, (0, 0))
c_cb.paste(im_b1, (320, 0))
c_cb.save('test_artifacts/cow_and_buffalo.jpg')

# 3. Cattle group (4 cows in 2x2 grid)
c2_path = 'Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg'
c3_path = 'Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg'
c4_path = 'Cattle Breeds/Jersey cattle/Jerseycattle0.jpg'
im_c2 = Image.open(c2_path).convert('RGB').resize((300, 300))
im_c3 = Image.open(c3_path).convert('RGB').resize((300, 300))
im_c4 = Image.open(c4_path).convert('RGB').resize((300, 300))
grid_c = Image.new('RGB', (600, 600))
grid_c.paste(im_c1.resize((300, 300)), (0, 0))
grid_c.paste(im_c2, (300, 0))
grid_c.paste(im_c3, (0, 300))
grid_c.paste(im_c4, (300, 300))
grid_c.save('test_artifacts/cattle_group.jpg')

# 4. Buffalo group (4 buffaloes in 2x2 grid)
b3_path = 'buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg'
b4_path = 'buffalo/surti/surti_buffalo_female_at_best_price_.jpg'
if not os.path.exists(b4_path):
    b4_path = os.path.join('buffalo/surti', os.listdir('buffalo/surti')[0])
im_b3 = Image.open(b3_path).convert('RGB').resize((300, 300))
im_b4 = Image.open(b4_path).convert('RGB').resize((300, 300))
grid_b = Image.new('RGB', (600, 600))
grid_b.paste(im_b1.resize((300, 300)), (0, 0))
grid_b.paste(im_b2.resize((300, 300)), (300, 0))
grid_b.paste(im_b3, (0, 300))
grid_b.paste(im_b4, (300, 300))
grid_b.save('test_artifacts/buffalo_group.jpg')

# 5. Blurry cow
cv_c = cv2.imread(c1_path)
cv_c_blur = cv2.GaussianBlur(cv_c, (51, 51), 0)
cv2.imwrite('test_artifacts/blurry_cow.jpg', cv_c_blur)

# 6. Blurry buffalo
cv_b = cv2.imread(b1_path)
cv_b_blur = cv2.GaussianBlur(cv_b, (51, 51), 0)
cv2.imwrite('test_artifacts/blurry_buffalo.jpg', cv_b_blur)

print("Test failure-case sample images created successfully!")
