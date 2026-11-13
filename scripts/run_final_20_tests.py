import os
import sys
import unittest
import json
from dotenv import load_dotenv

# Ensure environment is loaded
load_dotenv()

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
import notification_service
from services import email_service
from database import get_db, init_db, seed_single_admin, reset_user_and_transaction_data, hash_password
from ml_pipeline import pipeline

CAPTURED_EMAILS = []

orig_send_email = email_service.send_email
def intercepted_send_email(to_email, subject, html_body, plain_body=None):
    CAPTURED_EMAILS.append({
        'to': to_email.strip().lower(),
        'subject': subject,
        'plain': plain_body or '',
        'html': html_body or ''
    })
    return True, "Email dispatched successfully."

email_service.send_email = intercepted_send_email
notification_service.send_email = intercepted_send_email

class FinalTwentyAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        init_db()
        seed_single_admin()
        reset_user_and_transaction_data()
        cls.client = app.test_client()

        # Seed Verified Farmer and Verified Buyer
        conn = get_db()
        cursor = conn.cursor()
        pwd_hash = hash_password("SecurePass@2026")

        cursor.execute('''
            INSERT INTO users (name, email, username, phone, password_hash, password, role, email_verified, status)
            VALUES ('Ramesh Farmer', 'ramesh.farmer.audit@agri.in', 'ramesh_farmer', '+919876543211', ?, '[BCRYPT_PROTECTED]', 'farmer', 1, 'Active')
        ''', (pwd_hash,))
        cls.farmer_id = cursor.lastrowid
        cls.farmer_email = 'ramesh.farmer.audit@agri.in'

        cursor.execute('''
            INSERT INTO users (name, email, username, phone, password_hash, password, role, email_verified, status)
            VALUES ('Suresh Buyer', 'suresh.buyer.audit@dairy.in', 'suresh_buyer', '+919876543222', ?, '[BCRYPT_PROTECTED]', 'buyer', 1, 'Active')
        ''', (pwd_hash,))
        cls.buyer_id = cursor.lastrowid
        cls.buyer_email = 'suresh.buyer.audit@dairy.in'

        conn.commit()
        conn.close()

    def setUp(self):
        CAPTURED_EMAILS.clear()

    # -------------------------------------------------------------
    # IMAGE VALIDATION & GENDER TESTS (TEST 1 to TEST 10)
    # -------------------------------------------------------------

    def test_01_one_clear_cow_accepted(self):
        """TEST 1: One clear cow image -> accepted."""
        img_path = 'test_artifacts/full_cow.jpg'
        res = pipeline.predict_image(img_path)
        self.assertEqual(res['status'], 'success', f"Expected clear cow to be accepted. Got: {res}")
        self.assertEqual(res['species'], 'Cow')
        self.assertTrue(res['is_valid'])

    def test_02_one_clear_buffalo_accepted(self):
        """TEST 2: One clear buffalo image -> accepted."""
        img_path = 'test_artifacts/full_buffalo.jpg'
        res = pipeline.predict_image(img_path)
        self.assertEqual(res['status'], 'success', f"Expected clear buffalo to be accepted. Got: {res}")
        self.assertEqual(res['species'], 'Buffalo')
        self.assertTrue(res['is_valid'])

    def test_03_two_cows_rejected(self):
        """TEST 3: Two cows -> rejected."""
        img_path = 'test_artifacts/two_cows.jpg'
        res = pipeline.predict_image(img_path)
        self.assertEqual(res['status'], 'rejected')
        self.assertFalse(res['is_valid'])
        self.assertIn("Multiple animals detected", res['error'])
        self.assertIn("exactly one cow or buffalo", res['error'])

    def test_04_cow_and_buffalo_rejected(self):
        """TEST 4: Cow + buffalo -> rejected."""
        img_path = 'test_artifacts/cow_and_buffalo.jpg'
        res = pipeline.predict_image(img_path)
        self.assertEqual(res['status'], 'rejected')
        self.assertFalse(res['is_valid'])
        self.assertIn("Multiple animals detected", res['error'])

    def test_05_herd_group_rejected(self):
        """TEST 5: Herd/group -> rejected."""
        img_path = 'test_artifacts/cattle_group.jpg'
        res = pipeline.predict_image(img_path)
        self.assertEqual(res['status'], 'rejected')
        self.assertFalse(res['is_valid'])
        self.assertIn("Multiple animals detected", res['error'])

    def test_06_head_only_image_rejected(self):
        """TEST 6: Head-only image -> rejected."""
        img_path = 'test_artifacts/cow_head_only.jpg'
        res = pipeline.predict_image(img_path)
        self.assertEqual(res['status'], 'rejected')
        self.assertFalse(res['is_valid'])
        self.assertIn("not fully visible", res['error'])

    def test_07_leg_only_image_rejected(self):
        """TEST 7: Leg-only image -> rejected."""
        img_path = 'test_artifacts/test_legs_only.jpg'
        res = pipeline.predict_image(img_path)
        self.assertEqual(res['status'], 'rejected')
        self.assertFalse(res['is_valid'])

    def test_08_blurry_unrelated_image_rejected(self):
        """TEST 8: Blurry/unrelated image -> rejected."""
        img_path = 'test_artifacts/blurry_cow.jpg'
        res = pipeline.predict_image(img_path)
        self.assertEqual(res['status'], 'rejected')
        self.assertFalse(res['is_valid'])
        self.assertIn("blurry", res['error'].lower())

    def test_09_clearly_labeled_female_gender(self):
        """TEST 9: Clearly labeled female test image -> verify gender."""
        img_path = 'data/new_images/femalecow.jpg'
        res = pipeline.predict_image(img_path)
        self.assertEqual(res['status'], 'success')
        self.assertEqual(res['gender'], 'Female')
        self.assertGreaterEqual(res['gender_confidence'], 0.65)

    def test_10_clearly_labeled_male_gender(self):
        """TEST 10: Clearly labeled male test image -> verify gender."""
        img_path = 'data/new_images/malecow.jpg'
        res = pipeline.predict_image(img_path)
        self.assertEqual(res['status'], 'success')
        self.assertEqual(res['gender'], 'Male')
        self.assertGreaterEqual(res['gender_confidence'], 0.65)

    # -------------------------------------------------------------
    # PURCHASE & EMAIL TRANSACTION TESTS (TEST 11 to TEST 20)
    # -------------------------------------------------------------

    def test_11_buyer_purchases_valid_livestock(self):
        """TEST 11: Buyer purchases valid livestock -> purchase succeeds with valid FKs."""
        # 1. Farmer creates listing
        res_list = self.client.post('/api/listings', json={
            'farmer_id': self.farmer_id,
            'species': 'Cow',
            'breed': 'Hallikar',
            'gender': 'Female',
            'age_years': 4.0,
            'milk_production': 10.0,
            'district': 'Mandya',
            'taluk': 'Maddur',
            'village': 'Kadur',
            'address': 'Main Road Farm',
            'pincode': '571428',
            'image_url': '/static/uploads/audit_cow_11.jpg',
            'predicted_price': 65000,
            'asking_price': 68000
        })
        self.assertEqual(res_list.status_code, 200, res_list.get_json())
        listing_id = res_list.get_json()['listing_id']

        # 2. Buyer purchases listing
        res_buy = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': self.buyer_id,
            'delivery_district': 'Mandya',
            'delivery_taluk': 'Maddur',
            'delivery_village': 'Kadur',
            'delivery_address': 'Buyer Dairy Farm',
            'delivery_pincode': '571428'
        })
        self.assertEqual(res_buy.status_code, 200, res_buy.get_json())
        data = res_buy.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['summary']['order_status'], 'CONFIRMED')
        order_id = data['order_id']

        # Verify DB foreign keys and listing status
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM listings WHERE id = ?", (listing_id,))
        self.assertEqual(cursor.fetchone()['status'], 'SOLD_OUT')

        cursor.execute("SELECT listing_id, buyer_id, farmer_id, total_amount FROM purchases WHERE id = ?", (order_id,))
        p_row = cursor.fetchone()
        self.assertEqual(p_row['listing_id'], listing_id)
        self.assertEqual(p_row['buyer_id'], self.buyer_id)
        self.assertEqual(p_row['farmer_id'], self.farmer_id)
        self.assertEqual(p_row['total_amount'], 68000)
        conn.close()

    def test_12_buyer_purchases_cow_listing(self):
        """TEST 12: Buyer purchases Cow -> purchase succeeds."""
        res_list = self.client.post('/api/listings', json={
            'farmer_id': self.farmer_id,
            'species': 'Cow',
            'breed': 'Amritmahal',
            'gender': 'Female',
            'age_years': 3.5,
            'milk_production': 8.0,
            'district': 'Mysuru',
            'taluk': 'Hunsur',
            'village': 'Bilikere',
            'address': 'Farm 12',
            'pincode': '571103',
            'image_url': '/static/uploads/audit_cow_12.jpg',
            'asking_price': 72000
        })
        listing_id = res_list.get_json()['listing_id']

        res_buy = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': self.buyer_id,
            'delivery_district': 'Mysuru',
            'delivery_address': 'Mysuru Dairy Center',
            'delivery_pincode': '570001'
        })
        self.assertEqual(res_buy.status_code, 200)
        self.assertEqual(res_buy.get_json()['status'], 'success')

    def test_13_buyer_purchases_buffalo_listing(self):
        """TEST 13: Buyer purchases Buffalo -> purchase succeeds."""
        res_list = self.client.post('/api/listings', json={
            'farmer_id': self.farmer_id,
            'species': 'Buffalo',
            'breed': 'Murrah',
            'gender': 'Female',
            'age_years': 5.0,
            'milk_production': 16.0,
            'district': 'Dharwad',
            'taluk': 'Hubballi',
            'village': 'Unkal',
            'address': 'Buffalo Farm 13',
            'pincode': '580031',
            'image_url': '/static/uploads/audit_buffalo_13.jpg',
            'asking_price': 95000
        })
        listing_id = res_list.get_json()['listing_id']

        res_buy = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': self.buyer_id,
            'delivery_district': 'Dharwad',
            'delivery_address': 'Hubli Dairy',
            'delivery_pincode': '580031'
        })
        self.assertEqual(res_buy.status_code, 200)
        self.assertEqual(res_buy.get_json()['status'], 'success')

    def test_14_buyer_receives_purchase_confirmation_email(self):
        """TEST 14: Successful purchase -> buyer receives purchase confirmation email."""
        res_list = self.client.post('/api/listings', json={
            'farmer_id': self.farmer_id,
            'species': 'Cow',
            'breed': 'Gir',
            'gender': 'Female',
            'age_years': 4.0,
            'milk_production': 14.0,
            'district': 'Tumakuru',
            'taluk': 'Tiptur',
            'address': 'Gir Farm',
            'pincode': '572201',
            'image_url': '/static/uploads/audit_cow_14.jpg',
            'asking_price': 85000
        })
        listing_id = res_list.get_json()['listing_id']
        CAPTURED_EMAILS.clear()

        res_buy = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': self.buyer_id,
            'delivery_district': 'Tumakuru',
            'delivery_address': 'Buyer House 14',
            'delivery_pincode': '572201'
        })
        self.assertEqual(res_buy.status_code, 200)

        # Buyer confirmation email check
        buyer_emails = [e for e in CAPTURED_EMAILS if e['to'] == self.buyer_email and e['subject'] == 'Livestock Purchase Confirmed']
        self.assertTrue(len(buyer_emails) >= 1, "Buyer must receive 'Livestock Purchase Confirmed' email")
        b_body = buyer_emails[0]['plain']
        self.assertIn("Your livestock purchase has been successfully confirmed.", b_body)
        self.assertIn("Gir", b_body)
        self.assertIn("85,000", b_body)

    def test_15_farmer_receives_purchase_notification_email(self):
        """TEST 15: Successful purchase -> farmer receives purchase notification email."""
        res_list = self.client.post('/api/listings', json={
            'farmer_id': self.farmer_id,
            'species': 'Buffalo',
            'breed': 'Surti',
            'gender': 'Female',
            'age_years': 4.5,
            'milk_production': 11.0,
            'district': 'Belagavi',
            'taluk': 'Gokak',
            'address': 'Surti Buffalo Center',
            'pincode': '591307',
            'image_url': '/static/uploads/audit_buff_15.jpg',
            'asking_price': 76000
        })
        listing_id = res_list.get_json()['listing_id']
        CAPTURED_EMAILS.clear()

        res_buy = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': self.buyer_id,
            'delivery_district': 'Belagavi',
            'delivery_address': 'Belagavi Market',
            'delivery_pincode': '590001'
        })
        self.assertEqual(res_buy.status_code, 200)

        # Farmer purchase notification email check
        farmer_emails = [e for e in CAPTURED_EMAILS if e['to'] == self.farmer_email and e['subject'] == 'Your Livestock Has Been Purchased']
        self.assertTrue(len(farmer_emails) >= 1, "Farmer must receive 'Your Livestock Has Been Purchased' email")
        f_body = farmer_emails[0]['plain']
        self.assertIn("Your livestock has been purchased successfully.", f_body)
        self.assertIn("76,000", f_body)

    def test_16_farmer_receives_publication_confirmation_email(self):
        """TEST 16: Farmer publishes listing -> farmer receives publication confirmation email."""
        CAPTURED_EMAILS.clear()
        res_list = self.client.post('/api/listings', json={
            'farmer_id': self.farmer_id,
            'species': 'Cow',
            'breed': 'Deoni',
            'gender': 'Female',
            'age_years': 4.0,
            'milk_production': 9.5,
            'district': 'Bidar',
            'taluk': 'Bhalki',
            'village': 'Morambi',
            'address': 'Farm 16',
            'pincode': '585328',
            'image_url': '/static/uploads/audit_cow_16.jpg',
            'asking_price': 58000
        })
        self.assertEqual(res_list.status_code, 200)
        listing_id = res_list.get_json()['listing_id']

        pub_emails = [e for e in CAPTURED_EMAILS if e['to'] == self.farmer_email and 'Published' in e['subject']]
        self.assertTrue(len(pub_emails) >= 1, "Farmer must receive listing published confirmation email")
        pub_body = pub_emails[0]['plain']
        self.assertIn(f"#{listing_id}", pub_body)
        self.assertIn("Deoni", pub_body)
        self.assertIn("58,000", pub_body)

    def test_17_purchase_cancellation_emails(self):
        """TEST 17: Purchase cancellation -> buyer and farmer receive appropriate cancellation emails."""
        # Create listing and purchase
        res_list = self.client.post('/api/listings', json={
            'farmer_id': self.farmer_id,
            'species': 'Cow',
            'breed': 'Hallikar',
            'gender': 'Female',
            'age_years': 3.0,
            'milk_production': 10.0,
            'district': 'Mandya',
            'image_url': '/static/uploads/audit_cow_17.jpg',
            'asking_price': 60000
        })
        listing_id = res_list.get_json()['listing_id']
        res_buy = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': self.buyer_id,
            'delivery_district': 'Mandya',
            'delivery_address': 'Farm 17'
        })
        order_id = res_buy.get_json()['order_id']
        CAPTURED_EMAILS.clear()

        # Cancel purchase
        res_cancel = self.client.post(f'/api/orders/{order_id}/cancel', json={
            'buyer_id': self.buyer_id,
            'cancellation_reason': 'Changed Mind',
            'cancellation_details': 'Purchased another cattle locally'
        })
        self.assertEqual(res_cancel.status_code, 200)
        self.assertEqual(res_cancel.get_json()['order_status'], 'CANCELLED')

        # Verify listing returned to Available
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM listings WHERE id = ?", (listing_id,))
        self.assertEqual(cursor.fetchone()['status'], 'Available')
        conn.close()

        # Verify cancellation emails sent to both buyer and farmer
        b_cancels = [e for e in CAPTURED_EMAILS if e['to'] == self.buyer_email and 'Cancelled' in e['subject']]
        f_cancels = [e for e in CAPTURED_EMAILS if e['to'] == self.farmer_email and 'Cancelled' in e['subject']]
        self.assertTrue(len(b_cancels) >= 1, "Buyer must receive cancellation email")
        self.assertTrue(len(f_cancels) >= 1, "Farmer must receive cancellation email")

    def test_18_purchase_rejection_emails(self):
        """TEST 18: Purchase rejection -> appropriate buyer/farmer notifications are sent."""
        res_list = self.client.post('/api/listings', json={
            'farmer_id': self.farmer_id,
            'species': 'Buffalo',
            'breed': 'Jaffarabadi',
            'gender': 'Female',
            'age_years': 4.0,
            'milk_production': 15.0,
            'district': 'Belagavi',
            'image_url': '/static/uploads/audit_buff_18.jpg',
            'asking_price': 88000
        })
        listing_id = res_list.get_json()['listing_id']
        res_buy = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': self.buyer_id,
            'delivery_district': 'Belagavi',
            'delivery_address': 'Farm 18'
        })
        order_id = res_buy.get_json()['order_id']
        CAPTURED_EMAILS.clear()

        # Reject delivery
        res_reject = self.client.post(f'/api/orders/{order_id}/reject-delivery', json={
            'buyer_id': self.buyer_id,
            'rejection_reason': 'Health issue noticed upon delivery inspection'
        })
        self.assertEqual(res_reject.status_code, 200)
        self.assertEqual(res_reject.get_json()['delivery_status'], 'REJECTED')

        # Verify listing returned to Available
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM listings WHERE id = ?", (listing_id,))
        self.assertEqual(cursor.fetchone()['status'], 'Available')
        conn.close()

        # Verify rejection emails sent to both buyer and farmer
        b_rejs = [e for e in CAPTURED_EMAILS if e['to'] == self.buyer_email and 'Rejected' in e['subject']]
        f_rejs = [e for e in CAPTURED_EMAILS if e['to'] == self.farmer_email and 'Rejected' in e['subject']]
        self.assertTrue(len(b_rejs) >= 1, "Buyer must receive rejection notification")
        self.assertTrue(len(f_rejs) >= 1, "Farmer must receive rejection notification")

    def test_19_failed_db_transaction_does_not_mark_sold(self):
        """TEST 19: Failed database transaction -> no SOLD_OUT status and no false purchase-success email."""
        # Create an available listing
        res_list = self.client.post('/api/listings', json={
            'farmer_id': self.farmer_id,
            'species': 'Cow',
            'breed': 'Khillari',
            'gender': 'Female',
            'age_years': 4.0,
            'milk_production': 8.0,
            'district': 'Vijayapura',
            'image_url': '/static/uploads/audit_cow_19.jpg',
            'asking_price': 50000
        })
        listing_id = res_list.get_json()['listing_id']
        CAPTURED_EMAILS.clear()

        # Attempt purchase with NON-EXISTENT buyer ID (FK violation)
        invalid_buyer_id = 999999
        res_fail = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': invalid_buyer_id,
            'delivery_district': 'Vijayapura',
            'delivery_address': 'Invalid Address'
        })
        self.assertIn(res_fail.status_code, [404, 400, 500])

        # Verify listing is NOT marked SOLD_OUT
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM listings WHERE id = ?", (listing_id,))
        status = cursor.fetchone()['status']
        self.assertEqual(status, 'Available', "Failed transaction must NOT mark listing SOLD_OUT")

        # Verify purchases table has NO record for this listing
        cursor.execute("SELECT id FROM purchases WHERE listing_id = ?", (listing_id,))
        self.assertIsNone(cursor.fetchone(), "Failed transaction must NOT create a purchase record")
        conn.close()

        # Verify NO purchase confirmation emails were dispatched
        purchase_emails = [e for e in CAPTURED_EMAILS if 'Purchase Confirmed' in e['subject'] or 'Has Been Purchased' in e['subject']]
        self.assertEqual(len(purchase_emails), 0, "No purchase emails should be sent upon transaction failure")

    def test_20_agentmail_429_graceful_handling(self):
        """TEST 20: AgentMail 429 -> application shows friendly message and does not duplicate transactions."""
        # 1. Test 429 on Registration OTP endpoint
        def mock_send_email_429(to_email, subject, html_body, plain_body=None):
            return False, "Email service recipient limit has temporarily been reached. Please try again later."

        saved_send = email_service.send_email
        email_service.send_email = mock_send_email_429
        notification_service.send_email = mock_send_email_429

        try:
            # Request Registration OTP with simulated 429 recipient limit
            res_otp = self.client.post('/api/auth/register-request-otp', json={
                'name': 'Rate Limit Test User',
                'email': 'newuser.rate.limited@example.com',
                'phone': '+919876543299',
                'password': 'Password@123',
                'confirm_password': 'Password@123',
                'role': 'buyer'
            })
            self.assertIn(res_otp.status_code, [400, 429])
            err_data = res_otp.get_json()
            self.assertIn("Email service recipient limit has temporarily been reached. Please try again later.", err_data['error'])

            # 2. Test 429 on Purchase: Purchase should succeed in DB without error or duplication
            res_list = self.client.post('/api/listings', json={
                'farmer_id': self.farmer_id,
                'species': 'Cow',
                'breed': 'Malnad Gidda',
                'gender': 'Female',
                'age_years': 3.0,
                'milk_production': 4.0,
                'district': 'Shivamogga',
                'image_url': '/static/uploads/audit_cow_20.jpg',
                'asking_price': 35000
            })
            listing_id = res_list.get_json()['listing_id']

            res_buy = self.client.post(f'/api/listings/{listing_id}/purchase', json={
                'buyer_id': self.buyer_id,
                'delivery_district': 'Shivamogga',
                'delivery_address': 'Shimoga Dairy'
            })
            self.assertEqual(res_buy.status_code, 200)
            data = res_buy.get_json()
            self.assertEqual(data['status'], 'success')
            self.assertIn("Purchase confirmed successfully", data['message'])
            self.assertIn("Email service recipient limit has temporarily been reached", data['message'])

            # Verify only 1 purchase record exists
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM purchases WHERE listing_id = ?", (listing_id,))
            cnt = cursor.fetchone()[0]
            self.assertEqual(cnt, 1, "There must be exactly one purchase record created")
            conn.close()

        finally:
            email_service.send_email = intercepted_send_email
            notification_service.send_email = intercepted_send_email

if __name__ == '__main__':
    unittest.main()
