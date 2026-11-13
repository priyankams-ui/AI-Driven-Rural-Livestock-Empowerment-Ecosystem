import sys, glob
sys.path.insert(0, '.')
from ml_pipeline import pipeline

print("Testing cow images in data/new_images:")
for f in sorted(glob.glob("data/new_images/*cow*.*")):
    res = pipeline.predict_image(f)
    print(f"{f:35s} -> status={res.get('status')}, species={res.get('species')}, gender={res.get('gender')}, conf={res.get('gender_confidence')}, err={res.get('error')}")
