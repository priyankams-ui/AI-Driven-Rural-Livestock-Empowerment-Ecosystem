import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_pipeline import pipeline

sample_breeds = ['banni', 'Chhattisgarhi', 'Jaffarabadi', 'murrah', 'nili-ravi', 'pandharpuri', 'surti', 'toda']
for b in sample_breeds:
    folder = os.path.join('buffalo', b)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))][:2]
    for f in files:
        p = os.path.join(folder, f)
        res = pipeline.predict_image(p)
        pred_b = str(res.get('breed') or 'N/A')
        print(f"True={b:<15} | PredBreed={pred_b:<15} | Conf={res.get('breed_confidence', 0.0):.2f} | Species={res.get('species')} | Gender={res.get('gender')}")


