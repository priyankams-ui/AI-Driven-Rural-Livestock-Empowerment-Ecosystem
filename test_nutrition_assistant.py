import sys
import os
import unittest
import json

# Ensure parent and current directory are on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app
from database import init_db, get_db
from services.nutrition_advisor import LivestockNutritionAdvisor, nutrition_advisor

class TestLivestockNutritionAssistant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_advisor_presets(self):
        """Verify species and dynamic breed lists from service and API."""
        response = self.client.get('/api/buyer/nutrition/presets')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'success')
        cattle = data.get('cattle_breeds', [])
        buffalo = data.get('buffalo_breeds', [])
        species = data.get('species', [])
        
        self.assertIn('Cow', species)
        self.assertIn('Buffalo', species)
        self.assertTrue(any('Jersey' in b for b in cattle))
        self.assertTrue(any('Holstein' in b for b in cattle))
        self.assertTrue(any('Murrah' in b for b in buffalo))
        self.assertTrue(any('Jaffarabadi' in b for b in buffalo))
        print("[PASS] Test 1: Presets API returns verified Cow and Buffalo breeds.")

    def test_02_scenario_1_jersey_female_10_liters(self):
        """Scenario 1: Jersey cow – Female – 10 litres of milk per day."""
        payload = {
            "session_id": "test_scenario_1_jersey",
            "message": "Give me a daily feeding plan for my Jersey cow giving 10 litres of milk per day.",
            "animal_context": {
                "species": "Cow",
                "breed": "Jersey cattle",
                "gender": "Female",
                "age_years": 4.0,
                "milk_production": 10.0,
                "is_pregnant": 0,
                "purpose": "Milk Production"
            }
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        reply = data.get('reply')
        
        # Verify ICAR 400g/L ratio, Jersey fat mention, Fresh vs Dry Matter distinction
        self.assertIn("Jersey", reply)
        self.assertTrue("400" in reply or "0.4" in reply or "5.5 kg" in reply)
        self.assertTrue("dry matter" in reply.lower() or "dmi" in reply.lower())
        self.assertTrue("green fodder" in reply.lower() or "hasiru hullu" in reply.lower())
        self.assertTrue("ragi" in reply.lower() or "straw" in reply.lower())
        # Cites authentic knowledge sources
        self.assertTrue("icar" in reply.lower() or "nddb" in reply.lower())
        # Ensure zero male/bull terminology
        self.assertNotIn("breeding bull", reply.lower())
        print("[PASS] Scenario 1: Jersey cow - Female - 10 L/day verified.")

    def test_03_scenario_2_hf_female_15_liters(self):
        """Scenario 2: Holstein Friesian cow – Female – 15 litres of milk per day."""
        payload = {
            "session_id": "test_scenario_2_hf",
            "message": "What is the daily feeding recommendation for my HF cow producing 15 litres milk?",
            "animal_context": {
                "species": "Cow",
                "breed": "Holstein Friesian cattle",
                "gender": "Female",
                "age_years": 4.5,
                "milk_production": 15.0,
                "is_pregnant": 0,
                "purpose": "Milk Production"
            }
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        reply = data.get('reply')
        
        # HF is large frame (600 kg): check 400g/L ratio, Dry Matter, sodium bicarbonate buffer, heat stress
        self.assertIn("Holstein Friesian", reply)
        self.assertTrue("dry matter" in reply.lower() or "dmi" in reply.lower())
        self.assertTrue("concentrate" in reply.lower())
        self.assertTrue("buffer" in reply.lower() or "bicarbonate" in reply.lower() or "baking soda" in reply.lower() or "acidosis" in reply.lower())
        print("[PASS] Scenario 2: Holstein Friesian cow - Female - 15 L/day verified.")

    def test_04_scenario_3_murrah_female_8_liters(self):
        """Scenario 3: Murrah buffalo – Female – 8 litres of milk per day."""
        payload = {
            "session_id": "test_scenario_3_murrah_8l",
            "message": "What is the recommended feed ration for my Murrah buffalo producing 8 litres of milk per day?",
            "animal_context": {
                "species": "Buffalo",
                "breed": "Murrah",
                "gender": "Female",
                "age_years": 5.0,
                "milk_production": 8.0,
                "is_pregnant": 0,
                "purpose": "Milk Production"
            }
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        reply = data.get('reply')
        
        # Murrah buffalo: 500g concentrate per liter (due to 7-8.5% fat), 2.0 kg maintenance, superior fiber digestion
        self.assertIn("Murrah", reply)
        self.assertTrue("500" in reply or "0.5" in reply or "6.0 kg" in reply)
        self.assertTrue("dry matter" in reply.lower() or "dmi" in reply.lower())
        self.assertTrue("straw" in reply.lower() or "roughage" in reply.lower())
        print("[PASS] Scenario 3: Murrah buffalo - Female - 8 L/day verified.")

    def test_05_scenario_4_murrah_buffalo_male(self):
        """Scenario 4: Murrah buffalo – Male."""
        payload = {
            "session_id": "test_scenario_4_murrah_male",
            "message": "What should I feed my adult Murrah breeding bull?",
            "animal_context": {
                "species": "Buffalo",
                "breed": "Murrah",
                "gender": "Male",
                "age_years": 3.5,
                "milk_production": 0.0,
                "is_pregnant": 0,
                "purpose": "Breeding"
            }
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        reply = data.get('reply')
        
        # Male buffalo: breeding/draft ration, muscle tone, reproductive stamina
        # STRICTLY ZERO LACTATION CLAIMS, ZERO MILKING ROUTINE
        self.assertTrue("male" in reply.lower() or "bull" in reply.lower())
        self.assertNotIn("per liter of milk produced", reply.lower())
        self.assertNotIn("milking routine", reply.lower())
        self.assertNotIn("lactation curve", reply.lower())
        print("[PASS] Scenario 4: Murrah buffalo - Male verified (Zero lactation claims).")

    def test_06_scenario_5_cow_male(self):
        """Scenario 5: Cow – Male (Draft bullock / working ox)."""
        payload = {
            "session_id": "test_scenario_5_cow_male",
            "message": "What is the feeding schedule and feed quantity for my working draft bullock?",
            "animal_context": {
                "species": "Cow",
                "breed": "Hallikar",
                "gender": "Male",
                "age_years": 4.0,
                "milk_production": 0.0,
                "is_pregnant": 0,
                "purpose": "Draft / Heavy Work"
            }
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        reply = data.get('reply')
        
        # Male cow: work stamina, muscle maintenance, no lactation
        self.assertTrue("male" in reply.lower() or "bull" in reply.lower() or "work" in reply.lower())
        self.assertNotIn("per liter of milk", reply.lower())
        self.assertNotIn("udder", reply.lower())
        print("[PASS] Scenario 5: Cow - Male draft bullock verified.")

    def test_07_scenario_6_pregnant_cow(self):
        """Scenario 6: Pregnant cow (Late gestation)."""
        payload = {
            "session_id": "test_scenario_6_pregnant",
            "message": "My cow is 8 months pregnant. What special nutrition should I provide?",
            "animal_context": {
                "species": "Cow",
                "breed": "Holstein Friesian cattle",
                "gender": "Female",
                "age_years": 4.0,
                "milk_production": 0.0,
                "is_pregnant": 1,
                "pregnancy_month": 8,
                "purpose": "Pregnancy"
            }
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        reply = data.get('reply')
        
        # Pregnant cow: steaming-up, dry period, calcium restriction pre-calving to prevent Milk Fever
        self.assertTrue("steaming" in reply.lower() or "pregnant" in reply.lower() or "gestation" in reply.lower())
        self.assertTrue("milk fever" in reply.lower() or "calcium" in reply.lower() or "hypocalcemia" in reply.lower())
        print("[PASS] Scenario 6: Pregnant cow & Milk Fever prevention verified.")

    def test_08_scenario_7_young_calf(self):
        """Scenario 7: Young calf (<1 year old)."""
        payload = {
            "session_id": "test_scenario_7_calf",
            "message": "What is the feeding schedule and diet for a young 2-month-old calf?",
            "animal_context": {
                "species": "Cow",
                "breed": "Jersey cattle",
                "gender": "Female",
                "age_years": 0.2,
                "milk_production": 0.0,
                "is_pregnant": 0,
                "purpose": "Growth"
            }
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        reply = data.get('reply')
        
        # Calf: colostrum in first 2 hours, calf starter (20-22% CP), rumen papillae development
        self.assertTrue("colostrum" in reply.lower())
        self.assertTrue("calf starter" in reply.lower() or "starter" in reply.lower())
        self.assertTrue("rumen" in reply.lower() or "papillae" in reply.lower() or "protein" in reply.lower())
        print("[PASS] Scenario 7: Young calf nutrition and rumen development verified.")

    def test_09_scenario_8_missing_information_clarification(self):
        """Scenario 8: Animal with missing age or milk-production information."""
        # Case A: User introduces a Jersey cow without stating milk production
        payload = {
            "session_id": "test_missing_info_session",
            "message": "I have a 4-year-old Jersey cow. What should I feed her?",
            "animal_context": {
                "species": "Cow",
                "breed": "Jersey cattle",
                "gender": "Female",
                "age_years": 4.0,
                "milk_production": None,  # Explicitly MISSING
                "is_pregnant": 0
            }
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        reply = data.get('reply')
        
        # Chatbot MUST NOT assume 10L. It MUST ask whether she is producing milk and how many litres!
        self.assertTrue("producing milk" in reply.lower() or "how many litres" in reply.lower() or "lactating" in reply.lower())
        self.assertEqual(data.get('intent'), 'clarification_needed')
        print("[PASS] Scenario 8: Clarification asked when milk production information is missing.")

    def test_10_scenario_9_summer_water_requirements(self):
        """Scenario 9: Buyer asking about summer water requirements."""
        payload = {
            "session_id": "test_scenario_9_water",
            "message": "How many liters of water should I provide daily in hot summer weather, and how many times a day?",
            "animal_context": {
                "species": "Cow",
                "breed": "Jersey cattle",
                "gender": "Female",
                "age_years": 4.0,
                "milk_production": 12.0,
                "is_pregnant": 0,
                "purpose": "Milk Production"
            }
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        reply = data.get('reply')
        
        # Must mandate continuous 24/7 ad-libitum access (never only 2-3 times/day)
        self.assertTrue("24" in reply or "ad-libitum" in reply.lower() or "freely" in reply.lower() or "continuous" in reply.lower())
        # Must give quantitative estimate (baseline + milk yield + summer increase)
        self.assertTrue("summer" in reply.lower() or "hot" in reply.lower())
        self.assertTrue("litres" in reply.lower() or "liters" in reply.lower())
        # Warning signs of inadequate water
        self.assertTrue("warning" in reply.lower() or "muzzle" in reply.lower() or "skin" in reply.lower() or "drop" in reply.lower())
        print("[PASS] Scenario 9: Summer water management & 24/7 ad-libitum access verified.")

    def test_11_scenario_10_poisonous_or_moldy_feed(self):
        """Scenario 10: Buyer asking about poisonous or moldy feed."""
        payload = {
            "session_id": "test_scenario_10_toxic",
            "message": "Which foods, plants, or moldy feeds are poisonous and harmful to cows?",
            "animal_context": {
                "species": "Cow",
                "breed": "Indigenous",
                "gender": "Female",
                "age_years": 3.0,
                "milk_production": 4.0,
                "is_pregnant": 0
            }
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        reply = data.get('reply')
        
        # Moldy fodder aflatoxin, immature sorghum HCN, acidosis, Lantana
        self.assertTrue("aflatoxin" in reply.lower() or "mold" in reply.lower())
        self.assertTrue("hcn" in reply.lower() or "prussic" in reply.lower() or "sorghum" in reply.lower())
        self.assertTrue("acidosis" in reply.lower() or "grain" in reply.lower())
        self.assertTrue("lantana" in reply.lower())
        # Structured format: Why it is harmful + safer alternative
        self.assertTrue("harmful" in reply.lower())
        self.assertTrue("safer" in reply.lower() or "alternative" in reply.lower() or "practice" in reply.lower())
        print("[PASS] Scenario 10: Poisonous and moldy feed warnings & safer alternatives verified.")

    def test_12_scenario_11_exact_daily_feeding_quantities(self):
        """Scenario 11: Buyer asking for exact daily feeding quantities (Fresh vs Dry Matter distinction)."""
        payload = {
            "session_id": "test_scenario_11_quantities",
            "message": "How many kilograms of green fodder, dry fodder, and concentrate should I provide daily for my 10-liter cow?",
            "animal_context": {
                "species": "Cow",
                "breed": "Jersey cattle",
                "gender": "Female",
                "age_years": 4.0,
                "milk_production": 10.0,
                "is_pregnant": 0,
                "purpose": "Milk Production"
            }
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        reply = data.get('reply')
        
        # Must clearly distinguish fresh fodder weight from dry matter
        self.assertTrue("fresh" in reply.lower())
        self.assertTrue("dry matter" in reply.lower() or "dm" in reply)
        # Explains when an accurate ration requires live body weight and feed analysis
        self.assertTrue("body weight" in reply.lower())
        self.assertTrue("icar" in reply.lower() or "nddb" in reply.lower())
        print("[PASS] Scenario 11: Exact daily quantities & Fresh vs Dry Matter distinction verified.")

    def test_13_scenario_12_how_to_improve_milk_yield(self):
        """Scenario 12: Buyer asking how to improve milk yield."""
        payload = {
            "session_id": "test_scenario_12_milk_yield",
            "message": "How can I increase milk production and butterfat in my dairy cow?",
            "animal_context": {
                "species": "Cow",
                "breed": "Holstein Friesian cattle",
                "gender": "Female",
                "age_years": 4.0,
                "milk_production": 14.0,
                "is_pregnant": 0,
                "purpose": "Milk Production"
            }
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        reply = data.get('reply')
        
        # Must explain genetics, balanced nutrition, water, feeding consistency
        self.assertTrue("genetics" in reply.lower())
        self.assertTrue("water" in reply.lower())
        self.assertTrue("mineral" in reply.lower() or "buffer" in reply.lower())
        # Never promise magical litre increase; warns against excessive concentrate
        self.assertTrue("acidosis" in reply.lower() or "excessive concentrate" in reply.lower() or "sara" in reply.lower() or "balanced" in reply.lower())
        print("[PASS] Scenario 12: Milk yield optimization & genetic reality verified.")

    def test_14_multi_turn_conversation_flow(self):
        """Multi-turn conversation flow: Intro -> Clarification -> Answer -> Morning feeding."""
        session_id = "test_multiturn_flow"
        
        # Turn 1: Buyer introduces animal without milk yield
        res1 = self.client.post('/api/buyer/nutrition/chat', json={
            "session_id": session_id,
            "message": "I have a 4-year-old Jersey cow.",
            "animal_context": {"species": "Cow", "breed": "Jersey cattle", "gender": "Female", "age_years": 4.0, "milk_production": None}
        })
        self.assertEqual(res1.status_code, 200)
        reply1 = res1.get_json().get('reply')
        self.assertTrue("producing milk" in reply1.lower() or "how many litres" in reply1.lower())

        # Turn 2: Buyer replies: "10 litres"
        res2 = self.client.post('/api/buyer/nutrition/chat', json={
            "session_id": session_id,
            "message": "10 litres.",
            "animal_context": None
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.get_json()
        reply2 = data2.get('reply')
        # Chatbot acknowledges 4-year-old lactating Jersey cow giving 10 litres and notes body weight & feed analysis
        self.assertTrue("jersey" in reply2.lower())
        self.assertTrue("10" in reply2)
        self.assertTrue("dry matter" in reply2.lower() or "body weight" in reply2.lower())

        # Turn 3: Buyer asks follow-up: "What should I feed her in the morning?"
        res3 = self.client.post('/api/buyer/nutrition/chat', json={
            "session_id": session_id,
            "message": "What should I feed her in the morning?",
            "animal_context": None
        })
        self.assertEqual(res3.status_code, 200)
        reply3 = res3.get_json().get('reply')
        self.assertTrue("morning" in reply3.lower())
        self.assertTrue("concentrate" in reply3.lower())
        self.assertTrue("green fodder" in reply3.lower() or "water" in reply3.lower())
        print("[PASS] Scenario 13: Multi-turn conversational clarification and context retention verified.")

    def test_15_emergency_medical_referral(self):
        """Emergency escalation for severe bloating and toxic weed ingestion."""
        payload = {
            "session_id": "test_emergency_bloat",
            "message": "My cow ate poison weed and is suffering from severe acute bloat with frothy saliva. What medicine should I give?",
            "animal_context": {"species": "Cow", "breed": "Jersey cattle", "gender": "Female"}
        }
        res = self.client.post('/api/buyer/nutrition/chat', json=payload)
        self.assertEqual(res.status_code, 200)
        reply = res.get_json().get('reply')
        
        # Must give strong veterinary medical emergency warning, stop feeding, call AHVS/veterinarian
        self.assertTrue("emergency" in reply.lower())
        self.assertTrue("veterinarian" in reply.lower() or "ahvs" in reply.lower() or "doctor" in reply.lower())
        self.assertTrue("stop" in reply.lower())
        print("[PASS] Scenario 14: Emergency veterinary referral verified.")

    def test_16_marketplace_listing_context(self):
        """Verify fetching animal context from a marketplace listing."""
        conn = get_db()
        row = conn.execute("SELECT id FROM listings LIMIT 1").fetchone()
        conn.close()
        if row:
            listing_id = row['id']
            res = self.client.get(f'/api/buyer/nutrition/listing/{listing_id}')
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data.get('status'), 'success')
            ctx = data.get('animal_context', {})
            self.assertIn('species', ctx)
            self.assertIn('breed', ctx)
            print(f"[PASS] Scenario 15: Marketplace listing #{listing_id} context fetched successfully.")
        else:
            print("[SKIP] Scenario 15: No listings found in database.")

if __name__ == '__main__':
    unittest.main()
