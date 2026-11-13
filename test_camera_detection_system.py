import os
import io
import unittest
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import app

class TestCameraAndDetectionSystem(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_artifacts')
        self.samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_non_livestock_samples')

    def test_genuine_cow_accepted(self):
        cow_path = os.path.join(self.test_dir, 'full_cow.jpg')
        if not os.path.exists(cow_path):
            self.skipTest("full_cow.jpg not found")
        with open(cow_path, 'rb') as f:
            res = self.app.post('/api/predict/image-analysis', data={'file': (f, 'live_cow.jpg', 'image/jpeg')})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get('is_valid'))
        self.assertEqual(data.get('species'), 'Cow')
        print(f"[Verified Cow] Species: {data.get('species')}, Breed: {data.get('breed')}, Gender: {data.get('gender')}")

    def test_genuine_buffalo_accepted(self):
        buff_path = os.path.join(self.test_dir, 'full_buffalo.jpg')
        if not os.path.exists(buff_path):
            self.skipTest("full_buffalo.jpg not found")
        with open(buff_path, 'rb') as f:
            res = self.app.post('/api/predict/image-analysis', data={'file': (f, 'live_buffalo.jpg', 'image/jpeg')})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get('is_valid'))
        self.assertEqual(data.get('species'), 'Buffalo')
        print(f"[Verified Buffalo] Species: {data.get('species')}, Breed: {data.get('breed')}, Gender: {data.get('gender')}")

    def test_reject_human_image(self):
        human_path = os.path.join(self.test_dir, 'test_human.jpg')
        if not os.path.exists(human_path):
            human_path = os.path.join(self.samples_dir, 'person.jpg')
        with open(human_path, 'rb') as f:
            res = self.app.post('/api/predict/image-analysis', data={'file': (f, 'human_capture.jpg', 'image/jpeg')})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data.get('is_valid'))
        self.assertEqual(data.get('error'), 'No valid cow or buffalo detected. Please capture a clear livestock image.')
        print(f"[Verified Human Rejection] Error: {data.get('error')}")

    def test_reject_vehicle_and_object_images(self):
        for fname in ['car.jpg', 'dog.jpg', 'horse.jpg', 'sheep.jpg', 'chair.jpg']:
            p = os.path.join(self.samples_dir, fname)
            if not os.path.exists(p):
                continue
            with open(p, 'rb') as f:
                res = self.app.post('/api/predict/image-analysis', data={'file': (f, f"capture_{fname}", 'image/jpeg')})
            self.assertEqual(res.status_code, 400)
            data = res.get_json()
            self.assertFalse(data.get('is_valid'))
            self.assertEqual(data.get('error'), 'No valid cow or buffalo detected. Please capture a clear livestock image.')
            print(f"[Verified Rejection for {fname:12s}] Error: {data.get('error')}")

if __name__ == '__main__':
    unittest.main()
