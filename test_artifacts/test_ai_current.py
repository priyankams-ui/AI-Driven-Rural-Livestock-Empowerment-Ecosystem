import sys
sys.path.insert(0, '.')
from ml_pipeline import pipeline
import os

ai_dir = 'test_artifacts/ai_generated_samples'
for f in os.listdir(ai_dir):
    p = os.path.join(ai_dir, f)
    res = pipeline.predict_image(p)
    print(f"{f:22s} -> status={res.get('status')} sp={res.get('species')} g={res.get('gender')} err={res.get('error')}")
