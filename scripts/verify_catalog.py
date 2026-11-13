import os

# Complete audited catalog of verified livestock gender images
# Class 0 = Female, Class 1 = Male

DATASET_CATALOG = [
    # =========================================================================
    # A. FEMALE COWS (Class 0: Visible udder/teats, dairy conformation)
    # =========================================================================
    ('data/new_images/femalecow.jpg', 0, 'Female Cow', 'Female Holstein-Friesian dairy cow with udder'),
    ('data/new_images/femalecow1.jpg', 0, 'Female Cow', 'Female Jersey dairy cow side view with udder'),
    ('data/new_images/femalecow2.jpg', 0, 'Female Cow', 'Female Red-and-White cow with visible udder'),
    ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle1.jpg', 0, 'Female Cow', 'Female Holstein cow with large dairy udder'),
    ('Cattle Breeds/Jersey cattle/Jerseycattle0.jpg', 0, 'Female Cow', 'Female Jersey cow with visible udder and teats'),
    ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle0.jpg', 0, 'Female Cow', 'Female Brown Swiss dairy cow with udder'),
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle1.jpg', 0, 'Female Cow', 'Female Ayrshire cow with prominent udder'),
    ('Cattle Breeds/Red Dane cattle/RedDanecattle0.jpg', 0, 'Female Cow', 'Female Red Dane cow with udder'),
    ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle0.jpg', 0, 'Female Cow', 'Female Holstein cow in field with udder'),
    ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle11.jpg', 0, 'Female Cow', 'Female Holstein cow side profile with udder'),
    ('Cattle Breeds/Jersey cattle/Jerseycattle1.jpg', 0, 'Female Cow', 'Female Jersey cow with udder'),
    ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle1.jpg', 0, 'Female Cow', 'Female Brown Swiss cow in pasture with udder'),
    
    # =========================================================================
    # B. MALE COWS (Class 1: Visible penile sheath / prepuce, muscular crest/neck, scrotum)
    # =========================================================================
    ('data/new_images/malecow.jpg', 1, 'Male Cow', 'Male Zebu bull with hump and pendulous penile sheath'),
    ('data/new_images/malecow1.jpg', 1, 'Male Cow', 'Male Brahman bull with prominent mid-ventral penile sheath'),
    ('data/new_images/malecow2.jpg', 1, 'Male Cow', 'Male Hereford bull with sheath and muscular neck'),
    ('data/new_images/malecow2.webp', 1, 'Male Cow', 'Male Hereford bull WebP format with sheath'),
    ('static/uploads/7fa0f07b81354ed68bac87e2704437c2.jpg', 1, 'Male Cow', 'Male Holstein bull with penile sheath'),
    ('static/uploads/ec95ac1e5e694f6eb9a39efa0151e85c.jpg', 1, 'Male Cow', 'Male Holstein steer with visible penile sheath'),
    ('Cattle Breeds/Ayrshire cattle/Ayrshirecattle4.jpg', 1, 'Male Cow', 'Male Ayrshire bull with penile sheath on belly'),
    ('Cattle Breeds/Holstein Friesian cattle/HolsteinFriesiancattle13.jpg', 1, 'Male Cow', 'Male Holstein bull in pen with penile sheath'),
    ('Cattle Breeds/Jersey cattle/Jerseycattle13.jpg', 1, 'Male Cow', 'Male Jersey bull with male muscular profile and sheath'),
    ('Cattle Breeds/Brown Swiss cattle/BrownSwisscattle13.jpg', 1, 'Male Cow', 'Male Brown Swiss bull with penile sheath'),

    # =========================================================================
    # C. FEMALE BUFFALOES (Class 0: Visible udder/teats, dairy conformation)
    # =========================================================================
    ('data/new_images/femalebuffolo.jpg', 0, 'Female Buffalo', 'Female buffalo with visible udder between hind legs'),
    ('data/new_images/femalebuffolo1.jpg', 0, 'Female Buffalo', 'Female dairy buffalo standing in farmyard with udder'),
    ('data/new_images/femalebuffolo2.jpg', 0, 'Female Buffalo', 'Female buffalo isolated profile with prominent udder'),
    ('buffalo/murrah/Female_Black_Original_Murrah_Buffalo_.jpg', 0, 'Female Buffalo', 'Female Murrah buffalo with prominent dairy udder'),
    ('buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg', 0, 'Female Buffalo', 'Female Banni buffalo with udder'),
    ('buffalo/murrah/131_Murrah_Buffalo_for_Dairy_Farming_.jpg', 0, 'Female Buffalo', 'Female Murrah dairy buffalo with teats'),
    ('buffalo/banni/8c0d96d2994e6378e90e66eb1f324cba.jpg', 0, 'Female Buffalo', 'Female Banni buffalo with visible udder in field'),
    ('buffalo/surti/Milking_Murrah_Buffalo_Female_750kg_.jpg', 0, 'Female Buffalo', 'Female milking buffalo with large udder'),

    # =========================================================================
    # D. MALE BUFFALOES (Class 1: Visible penile sheath / prepuce, massive crest, scrotum)
    # =========================================================================
    ('data/new_images/malebuffolo.jpg', 1, 'Male Buffalo', 'Male buffalo bull side profile with visible sheath'),
    ('data/new_images/malebuffolo1.jpg', 1, 'Male Buffalo', 'Male buffalo bull with pendulous penile sheath on belly'),
    ('data/new_images/malebuffolo2.jpg', 1, 'Male Buffalo', 'Male buffalo bull with visible male anatomy'),
    ('data/new_images/malebuffolo3.jpg', 1, 'Male Buffalo', 'Male buffalo bull in open field with clear penile sheath'),
    ('buffalo/Jaffarabadi/Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 1, 'Male Buffalo', 'Male Jafarabadi breeding bull with sheath'),
    ('buffalo/Jaffarabadi/Bhadawari-buffalo-bull_epashupalan.jpg', 1, 'Male Buffalo', 'Male Bhadawari bull with sheath and muscular build'),
    ('buffalo/banni/baani-buffalo-bulls_0.jpg', 1, 'Male Buffalo', 'Male Banni bull with visible penile sheath'),
    ('buffalo/murrah/Black_Murrah_Buffalo_Bull_Wholesaler_.jpg', 1, 'Male Buffalo', 'Male Murrah breeding bull with penile sheath'),
    ('buffalo/Jaffarabadi/JAFFRABADI_BUFFALO_BULL.jpg', 1, 'Male Buffalo', 'Male Jaffarabadi buffalo bull with sheath'),
    ('buffalo/nili-ravi/Neeli_Ravi_buffalo_bull_.jpg', 1, 'Male Buffalo', 'Male Nili-Ravi bull with penile sheath'),

    # =========================================================================
    # E. UNCERTAIN / INSUFFICIENT ANATOMICAL EVIDENCE
    # (Non-visible anatomy, headshots, occluded belly, front-facing poses)
    # =========================================================================
    ('gender_dataset/uncertain/cow_head_only.jpg', 2, 'Uncertain', 'Cow head only - zero ventral sex anatomy'),
    ('gender_dataset/uncertain/buffalo_head_only.jpg', 2, 'Uncertain', 'Buffalo head only - zero ventral sex anatomy'),
    ('gender_dataset/uncertain/calf_headshot_fence.jpg', 2, 'Uncertain', 'Calf face behind fence - zero ventral sex anatomy'),
    ('static/uploads/5da299733e69416d849ad4602bbd6459.jpg', 2, 'Uncertain', 'Calf headshot - no visible sex anatomy'),
]

print(f"Total cataloged images: {len(DATASET_CATALOG)}")
missing = [p for p, _, _, _ in DATASET_CATALOG if not os.path.exists(p)]
if missing:
    print(f"WARNING: {len(missing)} missing files: {missing}")
else:
    print("ALL cataloged files verified present on disk!")
