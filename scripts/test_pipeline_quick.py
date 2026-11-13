import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_pipeline import pipeline

for i in [1, 5, 10, 50, 100]:
    p = os.path.join('dummy_unidentified', f'dummy_{i:04d}.jpg')
    if os.path.exists(p):
        res = pipeline.predict_image(p)
        print(f"dummy_{i:04d}: status={res.get('status')}, species={res.get('species')}, is_valid={res.get('is_valid')}, error={res.get('error')}")
