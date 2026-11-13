import os
import unittest
from PIL import Image
import numpy as np
from ml_pipeline import pipeline

class TestRealWorld19Cases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 80)
        print("   RUNNING REAL-WORLD 19 TEST CASES (Section 11)   ")
        print("=" * 80)
        cls.results = []

    def record_case(self, case_id, name, path, expected_action, result):
        is_pass = False
        status = result.get('status')
        err = result.get('error', '')
        species = result.get('species')
        breed = result.get('breed')
        gender = result.get('gender')
        g_conf = result.get('gender_confidence')

        if expected_action == 'ACCEPT':
            is_pass = (status == 'success' and result.get('is_valid') is True and species in ['Cow', 'Buffalo'])
            notes = f"Species: {species} | Breed: {breed} ({result.get('breed_confidence'):.2f}) | Gender: {gender} ({g_conf:.2f})"
        elif expected_action == 'REJECT_MULTIPLE':
            is_pass = (status == 'rejected' and 'Multiple animals' in err)
            notes = f"Error: '{err}'"
        elif expected_action == 'REJECT_PARTIAL':
            is_pass = (status == 'rejected' and ('Animal is not fully visible' in err or 'full' in err.lower()))
            notes = f"Error: '{err}'"
        elif expected_action == 'REJECT_QUALITY':
            is_pass = (status == 'rejected' and ('blurry' in err.lower() or 'quality' in err.lower() or 'dark' in err.lower()))
            notes = f"Error: '{err}'"
        elif expected_action == 'REJECT_AI':
            is_pass = (status == 'rejected' and ('AI-generated' in err or 'real' in err.lower()))
            notes = f"Error: '{err}'"
        elif expected_action == 'REJECT_INVALID':
            is_pass = (status == 'rejected' and ('Invalid' in err or 'No cow' in err or 'unrelated' in err.lower() or 'clear' in err.lower()))
            notes = f"Error: '{err}'"
        elif expected_action == 'ACCEPT_OR_UNDETERMINED_GENDER':
            is_pass = (status == 'success' and result.get('is_valid') is True)
            notes = f"Species: {species} | Breed: {breed} | Gender: {gender}"

        self.results.append({
            'case_id': case_id,
            'name': name,
            'action': expected_action,
            'passed': is_pass,
            'notes': notes
        })

    def test_01_one_clear_cow(self):
        """Case 1: One clear cow -> Must accept"""
        p = 'data/new_images/femalecow.jpg'
        res = pipeline.predict_image(p)
        self.record_case(1, "One clear cow", p, "ACCEPT", res)
        self.assertEqual(res['status'], 'success')
        self.assertEqual(res['species'], 'Cow')

    def test_02_one_clear_buffalo(self):
        """Case 2: One clear buffalo -> Must accept"""
        p = 'buffalo/banni/0caae52ef510154bb215e99a7db35607.jpg'
        res = pipeline.predict_image(p)
        self.record_case(2, "One clear buffalo", p, "ACCEPT", res)
        self.assertEqual(res['status'], 'success')
        self.assertEqual(res['species'], 'Buffalo')

    def test_03_two_cows(self):
        """Case 3: Two cows -> Must reject"""
        p = 'test_artifacts/two_cows.jpg'
        res = pipeline.predict_image(p)
        self.record_case(3, "Two cows", p, "REJECT_MULTIPLE", res)
        self.assertEqual(res['status'], 'rejected')
        self.assertIn("Multiple animals detected", res['error'])

    def test_04_two_buffaloes(self):
        """Case 4: Two buffaloes -> Must reject"""
        p = 'test_artifacts/two_buffaloes.jpg'
        res = pipeline.predict_image(p)
        self.record_case(4, "Two buffaloes", p, "REJECT_MULTIPLE", res)
        self.assertEqual(res['status'], 'rejected')
        self.assertIn("Multiple animals detected", res['error'])

    def test_05_cow_and_buffalo(self):
        """Case 5: Cow + buffalo -> Must reject"""
        p = 'test_artifacts/cow_and_buffalo.jpg'
        res = pipeline.predict_image(p)
        self.record_case(5, "Cow + buffalo", p, "REJECT_MULTIPLE", res)
        self.assertEqual(res['status'], 'rejected')
        self.assertIn("Multiple animals detected", res['error'])

    def test_06_large_cattle_group(self):
        """Case 6: Large cattle group -> Must reject"""
        p = 'test_artifacts/cattle_group.jpg'
        res = pipeline.predict_image(p)
        self.record_case(6, "Large cattle group", p, "REJECT_MULTIPLE", res)
        self.assertEqual(res['status'], 'rejected')
        self.assertIn("Multiple animals detected", res['error'])

    def test_07_large_buffalo_group(self):
        """Case 7: Large buffalo group -> Must reject"""
        p = 'test_artifacts/buffalo_group.jpg'
        res = pipeline.predict_image(p)
        self.record_case(7, "Large buffalo group", p, "REJECT_MULTIPLE", res)
        self.assertEqual(res['status'], 'rejected')
        self.assertIn("Multiple animals detected", res['error'])

    def test_08_partial_cow(self):
        """Case 8: Partial cow -> Must reject"""
        p = 'test_artifacts/half_cow_front.jpg'
        res = pipeline.predict_image(p)
        self.record_case(8, "Partial cow", p, "REJECT_PARTIAL", res)
        self.assertEqual(res['status'], 'rejected')
        self.assertIn("Animal is not fully visible", res['error'])

    def test_09_partial_buffalo(self):
        """Case 9: Partial buffalo -> Must reject"""
        p = 'test_artifacts/half_buffalo.jpg'
        res = pipeline.predict_image(p)
        self.record_case(9, "Partial buffalo", p, "REJECT_PARTIAL", res)
        self.assertEqual(res['status'], 'rejected')
        self.assertIn("Animal is not fully visible", res['error'])

    def test_10_blurry_cow(self):
        """Case 10: Blurry cow -> Must reject"""
        p = 'test_artifacts/blurry_cow.jpg'
        res = pipeline.predict_image(p)
        self.record_case(10, "Blurry cow", p, "REJECT_QUALITY", res)
        self.assertEqual(res['status'], 'rejected')
        self.assertIn("blurry", res['error'].lower())

    def test_11_blurry_buffalo(self):
        """Case 11: Blurry buffalo -> Must reject"""
        p = 'test_artifacts/blurry_buffalo.jpg'
        res = pipeline.predict_image(p)
        self.record_case(11, "Blurry buffalo", p, "REJECT_QUALITY", res)
        self.assertEqual(res['status'], 'rejected')
        self.assertIn("blurry", res['error'].lower())

    def test_12_unrelated_animal(self):
        """Case 12: Unrelated animal -> Must reject"""
        p = 'test_artifacts/test_car.jpg'
        res = pipeline.predict_image(p)
        self.record_case(12, "Unrelated animal / vehicle", p, "REJECT_INVALID", res)
        self.assertEqual(res['status'], 'rejected')

    def test_13_no_animal(self):
        """Case 13: No animal -> Must reject"""
        p = 'test_artifacts/test_object.jpg'
        res = pipeline.predict_image(p)
        self.record_case(13, "No animal (object)", p, "REJECT_INVALID", res)
        self.assertEqual(res['status'], 'rejected')

    def test_14_ai_generated_cow(self):
        """Case 14: AI-generated cow image -> Must reject"""
        p = 'test_artifacts/ai_generated_samples/ai_cow.jpg'
        res = pipeline.predict_image(p)
        self.record_case(14, "AI-generated cow image", p, "REJECT_AI", res)
        self.assertEqual(res['status'], 'rejected')
        self.assertIn("AI-generated", res['error'])

    def test_15_ai_generated_buffalo(self):
        """Case 15: AI-generated buffalo image -> Must reject"""
        p = 'test_artifacts/ai_generated_samples/ai_buffalo.jpg'
        res = pipeline.predict_image(p)
        self.record_case(15, "AI-generated buffalo image", p, "REJECT_AI", res)
        self.assertEqual(res['status'], 'rejected')
        self.assertIn("AI-generated", res['error'])

    def test_16_difficult_rare_cow_breed(self):
        """Case 16: Difficult/rare cow breed -> Must accept and predict"""
        p = 'Cattle Breeds/Red Dane cattle/RedDanecattle10.jpg'
        res = pipeline.predict_image(p)
        self.record_case(16, "Rare cow breed (Red Dane)", p, "ACCEPT", res)
        self.assertEqual(res['status'], 'success')
        self.assertEqual(res['species'], 'Cow')
        self.assertEqual(res['breed'], 'Red Dane cattle')

    def test_17_difficult_rare_buffalo_breed(self):
        """Case 17: Difficult/rare buffalo breed -> Must accept and predict"""
        p = 'buffalo/toda/image_163x220_68.jpg'
        res = pipeline.predict_image(p)
        self.record_case(17, "Rare buffalo breed (toda)", p, "ACCEPT", res)
        self.assertEqual(res['status'], 'success')
        self.assertEqual(res['species'], 'Buffalo')
        self.assertEqual(res['breed'], 'toda')

    def test_18_difficult_male_image(self):
        """Case 18: Difficult male image -> Must accept with reliable gender or undetermined"""
        p = 'data/new_images/malecow.jpg'
        res = pipeline.predict_image(p)
        self.record_case(18, "Difficult male image", p, "ACCEPT", res)
        self.assertEqual(res['status'], 'success')
        self.assertEqual(res['gender'], 'Male')

    def test_19_difficult_female_image(self):
        """Case 19: Difficult female image -> Must accept with reliable gender or undetermined"""
        p = 'data/new_images/femalecow.jpg'
        res = pipeline.predict_image(p)
        self.record_case(19, "Difficult female image", p, "ACCEPT", res)
        self.assertEqual(res['status'], 'success')
        self.assertEqual(res['gender'], 'Female')

    @classmethod
    def tearDownClass(cls):
        print("\n" + "=" * 90)
        print(f"{'#':<3} | {'Test Case Description':<35} | {'Expected':<15} | {'Status':<6} | {'Details'}")
        print("-" * 90)
        all_passed = True
        for r in cls.results:
            st = "PASS" if r['passed'] else "FAIL"
            if not r['passed']:
                all_passed = False
            print(f"{r['case_id']:<3} | {r['name']:<35} | {r['action']:<15} | {st:<6} | {r['notes']}")
        print("=" * 90)
        total = len(cls.results)
        passed = sum(1 for r in cls.results if r['passed'])
        print(f"SUMMARY: {passed}/{total} Test Cases Passed ({passed/total*100:.1f}%)")
        print("=" * 90 + "\n")

if __name__ == '__main__':
    unittest.main()
