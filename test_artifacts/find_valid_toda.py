import os
from ml_pipeline import pipeline

toda_dir = 'buffalo/toda'
found = []
for fname in os.listdir(toda_dir):
    p = os.path.join(toda_dir, fname)
    if not p.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    res = pipeline.predict_image(p)
    if res['status'] == 'success' and res['species'] == 'Buffalo' and res['breed'] == 'toda':
        print(f"FOUND VALID TODA: {p} -> Breed Conf: {res['breed_confidence']:.2f}")
        found.append(p)
        if len(found) >= 3:
            break

print("Done. Found:", found)
