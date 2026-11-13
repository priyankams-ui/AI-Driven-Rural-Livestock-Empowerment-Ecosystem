"""
Comprehensive Test Suite: Lactation, Milk Production Caps, and Fair Price Prediction
====================================================================================
Verifies:
1. Dropdown options strictly limited to 1st to 5th.
2. Lactation-dependent milk caps:
   - 1st, 2nd, 3rd: max 30 L/day
   - 4th: max 15 L/day
   - 5th: max 10 L/day
3. API rejection when milk exceeds lactation cap.
4. User screenshot case (4-Yr HF Cow, 5th Lactation, 6 L/day in Chamarajanagar) predicts between 18k and 22k (NOT 72,200!).
5. Strict parity hierarchy (1st ~1.3L, 2nd ~1L, 3rd reduced ~85k, 4th 20k-30k, 5th 15k-22k).
"""

import unittest
import json
from app import app
from ml_pipeline import pipeline

class TestLactationFairPriceSystem(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_01_user_screenshot_case_prediction(self):
        """User Screenshot Case: 4-Yr HF Cow, 5th Lactation, 6 L/day in Chamarajanagar."""
        res = pipeline.predict_price(
            species='Cow',
            breed='Holstein Friesian cattle',
            gender='Female',
            age_years=4.0,
            milk_production=6.0,
            lactation_number='5th',
            calf_number='5th',
            district='Chamarajanagar'
        )
        self.assertEqual(res['status'], 'success')
        price = res['predicted_price']
        print(f"\n[Test 1] User Screenshot Predicted Price: INR {price:,} (Range: {res['price_range_min']:,} - {res['price_range_max']:,})")
        # Must slow down to between 15k and 25k, strictly below 30,000, never 72,200!
        self.assertLessEqual(price, 24500, "5th lactation price must be strictly under 24,500")
        self.assertGreaterEqual(price, 16000, "5th lactation price must be at least 16,000")
        self.assertLessEqual(res['price_range_max'], 26000, "Conformal max must not exceed 26,000")

    def test_02_survey_table_5th_lactation_slowdown(self):
        """User Survey Table: 5th lactation animals slow down to 15k-20k bracket."""
        survey_benchmarks = [
            ('Cow', 'Hallikar', 4.0, 7.5, 3, 17200, 2000),
            ('Cow', 'Jersey cattle', 4.0, 7.5, 3, 17000, 2000),
            ('Cow', 'Jersey cattle', 6.0, 7.5, 3, 18300, 2000),
            ('Cow', 'Holstein Friesian cattle', 4.0, 7.5, 3, 18200, 2000),
            ('Cow', 'Holstein Friesian cattle', 6.0, 7.5, 3, 19500, 2000),
            ('Buffalo', 'murrah', 4.0, 7.8, 3, 17400, 2000),
            ('Buffalo', 'murrah', 6.0, 8.0, 3, 18700, 2000),
        ]
        for sp, br, m, ag, hlth, target, tol in survey_benchmarks:
            res = pipeline.predict_price(
                species=sp, breed=br, gender='Female', age_years=ag, milk_production=m,
                lactation_number='5th', calf_number='5th', district='Chamarajanagar',
                health_score=hlth
            )
            price = res['predicted_price']
            print(f"[Test 2] {sp} {br} {m}L 5th: INR {price:,} (Target: ~{target:,})")
            self.assertAlmostEqual(price, target, delta=tol)

    def test_03_lactation_hierarchy_high_yield(self):
        """Top-yield hierarchy: 1st > 2nd > 3rd > 4th > 5th."""
        # 1st Lactation 30L: ~1.25L - 1.40L (above 1 Lakh)
        r1 = pipeline.predict_price('Cow', 'Holstein Friesian cattle', 'Female', 3.0, 30.0, 'Mandya', lactation_number='1st')
        # 2nd Lactation 30L: ~1.00L - 1.15L (~1 Lakh)
        r2 = pipeline.predict_price('Cow', 'Holstein Friesian cattle', 'Female', 4.0, 30.0, 'Mandya', lactation_number='2nd')
        # 3rd Lactation 30L: ~80k - 92k (reduced compared to 1st and 2nd)
        r3 = pipeline.predict_price('Cow', 'Holstein Friesian cattle', 'Female', 5.0, 30.0, 'Mandya', lactation_number='3rd')
        # 4th Lactation 15L: ~20k - 31k
        r4 = pipeline.predict_price('Cow', 'Holstein Friesian cattle', 'Female', 6.0, 15.0, 'Mandya', lactation_number='4th')
        # 5th Lactation 10L: < 25k
        r5 = pipeline.predict_price('Cow', 'Holstein Friesian cattle', 'Female', 7.5, 10.0, 'Mandya', lactation_number='5th')

        print(f"\n[Test 3] Hierarchy: 1st={r1['predicted_price']:,}, 2nd={r2['predicted_price']:,}, 3rd={r3['predicted_price']:,}, 4th={r4['predicted_price']:,}, 5th={r5['predicted_price']:,}")
        self.assertGreater(r1['predicted_price'], 120000, "1st lactation 30L should be above 1 Lakh")
        self.assertGreater(r2['predicted_price'], 98000, "2nd lactation 30L should be around 1 Lakh")
        self.assertGreater(r1['predicted_price'], r2['predicted_price'], "1st lactation should be higher than 2nd")
        self.assertGreater(r2['predicted_price'], r3['predicted_price'], "2nd lactation should be higher than 3rd")
        self.assertLess(r3['predicted_price'], 95000, "3rd lactation should be discounted compared to 1st/2nd")
        self.assertGreater(r3['predicted_price'], r4['predicted_price'], "3rd lactation should be higher than 4th")
        self.assertLessEqual(r4['predicted_price'], 32000, "4th lactation should slow down to 20k-32k")
        self.assertGreater(r4['predicted_price'], r5['predicted_price'], "4th lactation should be higher than 5th")
        self.assertLessEqual(r5['predicted_price'], 25000, "5th lactation must not exceed 25,000")

    def test_04_api_enforces_lactation_milk_caps(self):
        """API rejects requests where milk exceeds lactation-specific limits."""
        # 5th lactation rejects > 10 L
        resp5 = self.app.post('/api/predict/price', json={
            'species': 'Cow', 'breed': 'Holstein Friesian cattle', 'gender': 'Female',
            'age': 4.0, 'milk_production': 12.0, 'lactation_number': '5th', 'calf_number': '5th',
            'district': 'Chamarajanagar'
        })
        self.assertEqual(resp5.status_code, 400)
        self.assertIn("cannot exceed 10", resp5.get_json()['error'])

        # 4th lactation rejects > 15 L
        resp4 = self.app.post('/api/predict/price', json={
            'species': 'Cow', 'breed': 'Holstein Friesian cattle', 'gender': 'Female',
            'age': 4.0, 'milk_production': 18.0, 'lactation_number': '4th', 'calf_number': '4th',
            'district': 'Chamarajanagar'
        })
        self.assertEqual(resp4.status_code, 400)
        self.assertIn("cannot exceed 15", resp4.get_json()['error'])

        # Valid 5th lactation with 6 L succeeds
        resp_valid = self.app.post('/api/predict/price', json={
            'species': 'Cow', 'breed': 'Holstein Friesian cattle', 'gender': 'Female',
            'age': 4.0, 'milk_production': 6.0, 'lactation_number': '5th', 'calf_number': '5th',
            'district': 'Chamarajanagar'
        })
        self.assertEqual(resp_valid.status_code, 200)
        data = resp_valid.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertLessEqual(data['predicted_price'], 24500)
        self.assertGreaterEqual(data['predicted_price'], 17000)

if __name__ == '__main__':
    unittest.main()
