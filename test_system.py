import os
import io
import sys
import uuid
import unittest
import json
import datetime
import app as app_module
from app import app
import notification_service
from database import get_db, init_db, seed_single_admin, reset_user_and_transaction_data

from services import email_service

# In-memory interceptor for OTPs dispatched during test execution
CAPTURED_OTPS = {}

orig_send_email = email_service.send_email
def intercepted_send_email(to_email, subject, html_body, plain_body=None):
    cfg = email_service.get_agentmail_config()
    if cfg['api_key'] and 'your_agentmail' not in cfg['api_key'].lower() and 'placeholder' not in cfg['api_key'].lower():
        return orig_send_email(to_email, subject, html_body, plain_body)
    return True, "Email dispatched successfully."

email_service.send_email = intercepted_send_email
notification_service.send_email = intercepted_send_email

orig_send_admin_otp = notification_service.send_admin_otp_email
def intercepted_send_admin_otp(admin_email, otp_code):
    CAPTURED_OTPS[admin_email.strip().lower()] = otp_code
    return orig_send_admin_otp(admin_email, otp_code)
notification_service.send_admin_otp_email = intercepted_send_admin_otp
app_module.send_admin_otp_email = intercepted_send_admin_otp

orig_send_user_otp = notification_service.send_user_registration_otp_email
def intercepted_send_user_otp(user_email, user_name, otp_code, role):
    CAPTURED_OTPS[user_email.strip().lower()] = otp_code
    return orig_send_user_otp(user_email, user_name, otp_code, role)
notification_service.send_user_registration_otp_email = intercepted_send_user_otp
app_module.send_user_registration_otp_email = intercepted_send_user_otp


class TestPashuDhanSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        init_db()
        seed_single_admin()
        reset_user_and_transaction_data()
        cls.client = app.test_client()

    # -------------------------------------------------------------
    # 1. STRICT ADMIN AUTHENTICATION (2-FACTOR OTP)
    # -------------------------------------------------------------
    def test_01_admin_login_step1_and_step2_flow(self):
        """Verify strict admin Rakshitha M (rakshitham556@gmail.com) logs in via 2-factor OTP."""
        admin_email = 'rakshitham556@gmail.com'
        admin_pass = 'Rakshi@2026'

        # Step 1: Valid credentials -> triggers OTP
        res1 = self.client.post('/api/admin/login', json={
            'email': admin_email,
            'password': admin_pass
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.get_json()
        self.assertEqual(data1.get('status'), 'otp_required')
        self.assertEqual(data1.get('email'), admin_email)

        # Verify OTP was dispatched to authorized admin email
        self.assertIn(admin_email, CAPTURED_OTPS)
        otp = CAPTURED_OTPS[admin_email]
        self.assertEqual(len(otp), 6)

        # Step 2: Invalid OTP rejection
        res_fail_otp = self.client.post('/api/admin/verify-otp', json={
            'email': admin_email,
            'otp': '000000'
        })
        self.assertEqual(res_fail_otp.status_code, 400)
        self.assertIn('Invalid OTP', res_fail_otp.get_json()['error'])

        # Step 2: Valid OTP verification
        res2 = self.client.post('/api/admin/verify-otp', json={
            'email': admin_email,
            'otp': otp
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.get_json()
        self.assertEqual(data2['user']['role'], 'admin')
        self.assertEqual(data2['user']['name'], 'Rakshitha M')
        self.assertEqual(data2['user']['email'], admin_email)
        self.assertNotIn('password_hash', data2['user'])

        # Re-use of same OTP must fail (single-use)
        res_reuse = self.client.post('/api/admin/verify-otp', json={
            'email': admin_email,
            'otp': otp
        })
        self.assertEqual(res_reuse.status_code, 400)

    def test_02_admin_login_security_rejections(self):
        """Verify wrong password, wrong email, and second admin creation attempts are blocked."""
        # Wrong password
        res_wrong_pw = self.client.post('/api/admin/login', json={
            'email': 'rakshitham556@gmail.com',
            'password': 'WrongPassword123'
        })
        self.assertEqual(res_wrong_pw.status_code, 401)

        # Unauthorized email
        res_wrong_email = self.client.post('/api/admin/login', json={
            'email': 'intruder@domain.com',
            'password': 'Rakshi@2026'
        })
        self.assertEqual(res_wrong_email.status_code, 401)
        self.assertIn('Invalid administrator credentials', res_wrong_email.get_json()['error'])

        # Second admin setup endpoint permanently rejected
        res_setup = self.client.post('/api/admin/setup', json={
            'name': 'Second Admin',
            'email': 'admin2@gmail.com',
            'password': 'Password@123'
        })
        self.assertIn(res_setup.status_code, [403, 404])

    # -------------------------------------------------------------
    # 2. FARMER & BUYER REGISTRATION (EMAIL OTP & PHONE VALIDATION)
    # -------------------------------------------------------------
    def test_03_farmer_registration_with_otp(self):
        """Verify Farmer registration requests OTP, validates Indian phone, and activates account."""
        farmer_email = f"farmer_{uuid.uuid4().hex[:6]}@gmail.com"
        farmer_phone = "+919845123456"

        # 1. Invalid phone number rejection
        res_bad_phone = self.client.post('/api/auth/register-request-otp', json={
            'role': 'farmer',
            'name': 'Basavaraju Gowda',
            'email': farmer_email,
            'phone': '123456', # Invalid
            'password': 'Password@123',
            'confirm_password': 'Password@123',
            'district': 'Mandya',
            'taluk': 'Maddur',
            'village': 'KM Doddi',
            'address': 'Main Dairy Cross',
            'pincode': '571428'
        })
        self.assertEqual(res_bad_phone.status_code, 400)
        self.assertIn('Invalid Indian phone number', res_bad_phone.get_json()['error'])

        # 2. Valid request OTP
        res_req = self.client.post('/api/auth/register-request-otp', json={
            'role': 'farmer',
            'name': 'Basavaraju Gowda',
            'email': farmer_email,
            'phone': farmer_phone,
            'password': 'Password@123',
            'confirm_password': 'Password@123',
            'district': 'Mandya',
            'taluk': 'Maddur',
            'village': 'KM Doddi',
            'address': 'Main Dairy Cross',
            'pincode': '571428'
        })
        self.assertEqual(res_req.status_code, 200)
        data_req = res_req.get_json()
        self.assertEqual(data_req['status'], 'otp_sent')

        # 3. Retrieve captured OTP and verify
        self.assertIn(farmer_email, CAPTURED_OTPS)
        otp = CAPTURED_OTPS[farmer_email]

        res_verify = self.client.post('/api/auth/register-verify-otp', json={
            'email': farmer_email,
            'otp': otp,
            'name': 'Basavaraju Gowda',
            'phone': farmer_phone,
            'password': 'Password@123',
            'role': 'farmer',
            'district': 'Mandya',
            'taluk': 'Maddur',
            'village': 'KM Doddi',
            'address': 'Main Dairy Cross',
            'pincode': '571428'
        })
        self.assertEqual(res_verify.status_code, 200)
        data_user = res_verify.get_json()
        self.assertEqual(data_user.get('status'), 'success')
        self.assertEqual(data_user['user']['role'], 'farmer')

        # 4. Normal password login works after activation
        res_login = self.client.post('/api/auth/login', json={
            'identifier': farmer_email,
            'password': 'Password@123',
            'role': 'farmer'
        })
        self.assertEqual(res_login.status_code, 200)
        self.assertEqual(res_login.get_json()['user']['email'], farmer_email)

    def test_04_buyer_registration_and_duplicate_block(self):
        """Verify Buyer registration and ensure duplicate email or phone is blocked."""
        buyer_email = f"buyer_{uuid.uuid4().hex[:6]}@gmail.com"
        buyer_phone = "+919108112233"

        res_req = self.client.post('/api/auth/register-request-otp', json={
            'role': 'buyer',
            'name': 'Suresh Kumar',
            'email': buyer_email,
            'phone': buyer_phone,
            'password': 'BuyerPass@2026',
            'confirm_password': 'BuyerPass@2026',
            'district': 'Bengaluru Urban',
            'taluk': 'Yelahanka',
            'village': 'Jala',
            'address': 'Flat 204, Green Heights',
            'pincode': '560064'
        })
        self.assertEqual(res_req.status_code, 200)
        otp = CAPTURED_OTPS[buyer_email]

        res_verify = self.client.post('/api/auth/register-verify-otp', json={
            'email': buyer_email,
            'otp': otp,
            'name': 'Suresh Kumar',
            'phone': buyer_phone,
            'password': 'BuyerPass@2026',
            'role': 'buyer',
            'district': 'Bengaluru Urban',
            'taluk': 'Yelahanka',
            'village': 'Jala',
            'address': 'Flat 204, Green Heights',
            'pincode': '560064'
        })
        self.assertEqual(res_verify.status_code, 200)

        # Duplicate email block
        res_dup = self.client.post('/api/auth/register-request-otp', json={
            'role': 'buyer',
            'name': 'Duplicate Person',
            'email': buyer_email,
            'phone': '+919988776655',
            'password': 'SomePassword1',
            'confirm_password': 'SomePassword1',
            'district': 'Mysuru',
            'taluk': 'Mysuru',
            'village': 'Hootagalli',
            'address': 'Ring Road',
            'pincode': '570018'
        })
        self.assertEqual(res_dup.status_code, 400)
        self.assertIn('already exists', res_dup.get_json()['error'])

    # -------------------------------------------------------------
    # 3. KARNATAKA LOCATION MASTER DATA
    # -------------------------------------------------------------
    def test_05_karnataka_location_hierarchy(self):
        """Verify all 31 Karnataka districts and cascading taluks and villages exist."""
        res = self.client.get('/api/locations/districts')
        self.assertEqual(res.status_code, 200)
        districts = res.get_json()
        self.assertEqual(len(districts), 31)

        # Verify Mandya and Bengaluru Urban exist
        district_names = [d['name'] for d in districts]
        self.assertIn('Mandya', district_names)
        self.assertIn('Bengaluru Urban', district_names)

        # Taluks for Mandya
        mandya = next(d for d in districts if d['name'] == 'Mandya')
        res_taluks = self.client.get(f'/api/locations/districts/{mandya["id"]}/taluks')
        self.assertEqual(res_taluks.status_code, 200)
        taluks = res_taluks.get_json()
        self.assertGreater(len(taluks), 0)

        # Villages for first taluk
        first_taluk = taluks[0]
        res_villages = self.client.get(f'/api/locations/taluks/{first_taluk["id"]}/villages?limit=5')
        self.assertEqual(res_villages.status_code, 200)
        self.assertGreater(len(res_villages.get_json()), 0)

    # -------------------------------------------------------------
    # 4. ML VISION: NON-LIVESTOCK REJECTION & GENDER DETECTION
    # -------------------------------------------------------------
    def test_06_non_livestock_rejection_message(self):
        """Verify non-livestock image returns exact required rejection message."""
        dummy_dir = 'dummy_unidentified'
        if os.path.exists(dummy_dir) and os.listdir(dummy_dir):
            sample_dummy = os.path.join(dummy_dir, os.listdir(dummy_dir)[0])
            with open(sample_dummy, 'rb') as f:
                res = self.client.post('/api/predict/image-analysis', data={
                    'file': (f, 'dummy.jpg')
                }, content_type='multipart/form-data')
                data = res.get_json()
                self.assertEqual(data.get('is_valid'), False)
                self.assertEqual(data.get('error'), 'Unsupported image. Please upload a clear image of a cattle or buffalo.')

    def test_07_livestock_identification_and_gender(self):
        """Verify cattle image identifies species, breed, and automatic gender classification."""
        sample_dir = os.path.join('static', 'sample_cows')
        if os.path.exists(sample_dir) and os.listdir(sample_dir):
            sample_cow = os.path.join(sample_dir, os.listdir(sample_dir)[0])
            with open(sample_cow, 'rb') as f:
                res = self.client.post('/api/predict/image-analysis', data={
                    'file': (f, 'cow.jpg')
                }, content_type='multipart/form-data')
                data = res.get_json()
                self.assertEqual(data.get('is_valid'), True)
                self.assertIn(data.get('species'), ['Cow', 'Buffalo'])
                self.assertIn('breed', data)
                self.assertIn('species_confidence', data)
                self.assertIn('breed_confidence', data)
                self.assertIn(data.get('gender'), ['Male', 'Female'])
                self.assertIn('gender_confidence', data)

    # -------------------------------------------------------------
    # 5. STRICT GENDER & PREGNANCY PRICING LOGIC
    # -------------------------------------------------------------
    def test_08_male_and_female_pricing_logic(self):
        """Verify male pricing enforces zero milk & no pregnancy, while female allows pregnancy bonus."""
        # Male livestock: milk must be 0, pregnancy is N/A
        res_male = self.client.post('/api/predict/price', json={
            'species': 'Cow',
            'breed': 'Amritmahal cattle',
            'gender': 'Male',
            'age_years': 4.0,
            'district': 'Mandya'
        })
        self.assertEqual(res_male.status_code, 200)
        data_male = res_male.get_json()
        self.assertEqual(data_male['valuation_factors']['gender'], 'Male')
        self.assertIn('0 L/day', str(data_male['valuation_factors']['milk_production']))
        self.assertEqual(data_male['valuation_factors']['is_pregnant'], 'Not Applicable')

        # Female livestock non-pregnant
        res_female_np = self.client.post('/api/predict/price', json={
            'species': 'Cow',
            'breed': 'Holstein Friesian cattle',
            'gender': 'Female',
            'pregnancy': 'No',
            'milk_production': 20.0,
            'age_years': 4.0,
            'district': 'Mandya'
        })
        self.assertEqual(res_female_np.status_code, 200)
        data_female_np = res_female_np.get_json()

        # Female livestock pregnant (Month 6)
        res_female_preg = self.client.post('/api/predict/price', json={
            'species': 'Cow',
            'breed': 'Holstein Friesian cattle',
            'gender': 'Female',
            'pregnancy': 'Yes',
            'pregnancy_month': 6,
            'milk_production': 20.0,
            'age_years': 4.0,
            'district': 'Mandya'
        })
        self.assertEqual(res_female_preg.status_code, 200)
        data_female_preg = res_female_preg.get_json()

        # Pregnant female valuation must exceed non-pregnant female
        self.assertGreater(data_female_preg['predicted_price'], data_female_np['predicted_price'])

    # -------------------------------------------------------------
    # 6. ZERO DELIVERY CHARGES & COD PURCHASE WORKFLOW
    # -------------------------------------------------------------
    def test_09_zero_delivery_charges_and_order_confirmation(self):
        """Verify delivery charge is strictly 0 and purchase amount equals livestock price."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, phone FROM users WHERE role = 'farmer' LIMIT 1")
        farmer = cursor.fetchone()
        cursor.execute("SELECT id, name, phone FROM users WHERE role = 'buyer' LIMIT 1")
        buyer = cursor.fetchone()
        conn.close()

        # 1. Publish Listing
        pub_res = self.client.post('/api/listings', json={
            'farmer_id': farmer['id'],
            'farmer_name': farmer['name'],
            'farmer_phone': farmer['phone'],
            'image_url': f'/static/sample_cows/sample_{uuid.uuid4().hex[:6]}.jpg',
            'species': 'Cow',
            'breed': 'Holstein Friesian cattle',
            'gender': 'Female',
            'health_score': 5,
            'health_text': 'Excellent',
            'age_years': 3.5,
            'milk_production': 18.0,
            'district': 'Mandya',
            'taluk': 'Maddur',
            'village': 'KM Doddi',
            'address': 'Main Dairy Cross',
            'pincode': '571428',
            'predicted_price': 80000,
            'asking_price': 78000
        })
        self.assertEqual(pub_res.status_code, 200)
        listing_id = pub_res.get_json()['listing_id']

        # 2. Buy Now via Cash on Delivery (Zero Delivery Charge)
        buy_res = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': buyer['id'],
            'buyer_name': buyer['name'],
            'buyer_phone': buyer['phone'],
            'delivery_district': 'Bidar',
            'delivery_taluk': 'Bidar',
            'delivery_address': 'Main Market Road',
            'delivery_pincode': '585401'
        })
        self.assertEqual(buy_res.status_code, 200)
        buy_data = buy_res.get_json()
        self.assertEqual(buy_data['summary']['delivery_charge'], 0)
        self.assertEqual(buy_data['summary']['total_amount'], 78000)
        self.assertEqual(buy_data['summary']['order_status'], 'CONFIRMED')
        self.assertIsNone(buy_data['summary']['delivery_status'])

    # -------------------------------------------------------------
    # 7. ORDER LIFECYCLE: CANCELLATION, REJECTION, PURCHASED (PROOF), REVIEWS
    # -------------------------------------------------------------
    def test_10_order_lifecycle_cancellation_flow(self):
        """Verify order cancellation restores listing to Available and requires reason."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, phone FROM users WHERE role = 'farmer' LIMIT 1")
        farmer = cursor.fetchone()
        cursor.execute("SELECT id, name, phone FROM users WHERE role = 'buyer' LIMIT 1")
        buyer = cursor.fetchone()
        conn.close()

        # Create listing and order
        pub_res = self.client.post('/api/listings', json={
            'farmer_id': farmer['id'],
            'farmer_name': farmer['name'],
            'farmer_phone': farmer['phone'],
            'image_url': f'/static/sample_cows/sample_{uuid.uuid4().hex[:6]}.jpg',
            'species': 'Cow',
            'breed': 'Jersey cattle',
            'gender': 'Female',
            'age_years': 4.0,
            'milk_production': 14.0,
            'district': 'Mandya',
            'asking_price': 65000
        })
        self.assertEqual(pub_res.status_code, 200)
        listing_id = pub_res.get_json()['listing_id']

        buy_res = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': buyer['id'],
            'buyer_name': buyer['name'],
            'buyer_phone': buyer['phone'],
            'delivery_district': 'Bengaluru Urban',
            'delivery_address': 'MG Road',
            'delivery_pincode': '560001'
        })
        self.assertEqual(buy_res.status_code, 200)
        order_id = buy_res.get_json()['order_id']

        # Cancel without reason fails
        cancel_fail = self.client.post(f'/api/orders/{order_id}/cancel', json={
            'user_id': buyer['id'],
            'cancellation_reason': ''
        })
        self.assertEqual(cancel_fail.status_code, 400)

        # Cancel with reason succeeds
        cancel_ok = self.client.post(f'/api/orders/{order_id}/cancel', json={
            'user_id': buyer['id'],
            'cancellation_reason': 'Changed mind'
        })
        self.assertEqual(cancel_ok.status_code, 200)

        # Verify listing status restored to Available
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM listings WHERE id = ?", (listing_id,))
        listing_row = cursor.fetchone()
        self.assertEqual(listing_row['status'], 'Available')

        # Verify order status is CANCELLED
        cursor.execute("SELECT order_status, cancellation_reason FROM purchases WHERE id = ?", (order_id,))
        purch_row = cursor.fetchone()
        self.assertEqual(purch_row['order_status'], 'CANCELLED')
        self.assertEqual(purch_row['cancellation_reason'], 'Changed mind')
        conn.close()

    def test_11_order_lifecycle_rejection_flow(self):
        """Verify delivery rejection requires reason and restores listing to Available."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, phone FROM users WHERE role = 'farmer' LIMIT 1")
        farmer = cursor.fetchone()
        cursor.execute("SELECT id, name, phone FROM users WHERE role = 'buyer' LIMIT 1")
        buyer = cursor.fetchone()
        conn.close()

        pub_res = self.client.post('/api/listings', json={
            'farmer_id': farmer['id'],
            'farmer_name': farmer['name'],
            'farmer_phone': farmer['phone'],
            'image_url': f'/static/sample_cows/sample_{uuid.uuid4().hex[:6]}.jpg',
            'species': 'Cow',
            'breed': 'Red Dane cattle',
            'gender': 'Female',
            'age_years': 4.0,
            'milk_production': 16.0,
            'district': 'Mandya',
            'asking_price': 70000
        })
        self.assertEqual(pub_res.status_code, 200)
        listing_id = pub_res.get_json()['listing_id']

        buy_res = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': buyer['id'],
            'buyer_name': buyer['name'],
            'buyer_phone': buyer['phone'],
            'delivery_district': 'Mysuru',
            'delivery_address': 'Saraswathipuram',
            'delivery_pincode': '570009'
        })
        self.assertEqual(buy_res.status_code, 200)
        order_id = buy_res.get_json()['order_id']

        # Reject delivery
        rej_res = self.client.post(f'/api/orders/{order_id}/reject-delivery', json={
            'user_id': buyer['id'],
            'rejection_reason': 'Animal does not match listing description'
        })
        self.assertEqual(rej_res.status_code, 200)

        # Verify delivery status is REJECTED
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT delivery_status, rejection_reason FROM purchases WHERE id = ?", (order_id,))
        purch_row = cursor.fetchone()
        self.assertEqual(purch_row['delivery_status'], 'REJECTED')
        self.assertEqual(purch_row['rejection_reason'], 'Animal does not match listing description')

        # Verify listing returned to Available
        cursor.execute("SELECT status FROM listings WHERE id = ?", (listing_id,))
        listing_row = cursor.fetchone()
        self.assertEqual(listing_row['status'], 'Available')
        conn.close()

    def test_12_order_lifecycle_mark_purchased_with_proof_and_review(self):
        """Verify marking order as PURCHASED strictly requires proof photo, sets DELIVERED, and allows 1-5 star review."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, phone FROM users WHERE role = 'farmer' LIMIT 1")
        farmer = cursor.fetchone()
        cursor.execute("SELECT id, name, phone FROM users WHERE role = 'buyer' LIMIT 1")
        buyer = cursor.fetchone()
        conn.close()

        pub_res = self.client.post('/api/listings', json={
            'farmer_id': farmer['id'],
            'farmer_name': farmer['name'],
            'farmer_phone': farmer['phone'],
            'image_url': f'/static/sample_cows/sample_{uuid.uuid4().hex[:6]}.jpg',
            'species': 'Buffalo',
            'breed': 'murrah',
            'gender': 'Female',
            'age_years': 4.0,
            'milk_production': 12.0,
            'district': 'Mandya',
            'asking_price': 85000
        })
        self.assertEqual(pub_res.status_code, 200)
        listing_id = pub_res.get_json()['listing_id']

        buy_res = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': buyer['id'],
            'buyer_name': buyer['name'],
            'buyer_phone': buyer['phone'],
            'delivery_district': 'Bengaluru Urban',
            'delivery_address': 'Jayanagar 4th Block',
            'delivery_pincode': '560011'
        })
        self.assertEqual(buy_res.status_code, 200)
        order_id = buy_res.get_json()['order_id']

        # 1. Mark as purchased without proof photo must fail
        mark_fail = self.client.post(f'/api/orders/{order_id}/mark-purchased', data={
            'user_id': buyer['id']
        })
        self.assertEqual(mark_fail.status_code, 400)
        self.assertIn('proof photo', mark_fail.get_json()['error'].lower())

        # 2. Mark as purchased with proof photo succeeds
        fake_photo = (io.BytesIO(b"fake image data"), "delivery_proof.jpg")
        mark_ok = self.client.post(f'/api/orders/{order_id}/mark-purchased', data={
            'user_id': buyer['id'],
            'proof_photo': fake_photo
        }, content_type='multipart/form-data')
        self.assertEqual(mark_ok.status_code, 200)

        # Verify order_status is PURCHASED and delivery_status is DELIVERED
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT order_status, delivery_status, proof_photo_url FROM purchases WHERE id = ?", (order_id,))
        purch_row = cursor.fetchone()
        self.assertEqual(purch_row['order_status'], 'PURCHASED')
        self.assertEqual(purch_row['delivery_status'], 'DELIVERED')
        self.assertTrue(bool(purch_row['proof_photo_url']))
        conn.close()

        # 3. Submit Rating & Review (1 to 5 stars)
        # Invalid rating > 5 rejected
        rev_fail = self.client.post(f'/api/orders/{order_id}/review', json={
            'buyer_id': buyer['id'],
            'rating': 6,
            'review': 'Super animal'
        })
        self.assertEqual(rev_fail.status_code, 400)

        # Valid 5-star rating accepted
        rev_ok = self.client.post(f'/api/orders/{order_id}/review', json={
            'buyer_id': buyer['id'],
            'rating': 5,
            'review': 'Excellent health, high milk yield, smooth doorstep delivery!'
        })
        self.assertEqual(rev_ok.status_code, 200)

        # Verify review recorded in DB
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT rating, review FROM purchases WHERE id = ?", (order_id,))
        rec_row = cursor.fetchone()
        self.assertEqual(rec_row['rating'], 5)
        self.assertEqual(rec_row['review'], 'Excellent health, high milk yield, smooth doorstep delivery!')
        conn.close()

    # -------------------------------------------------------------
    # 8. 15-DAY SOLD-OUT REMOVAL RULE & HISTORY PROTECTION
    # -------------------------------------------------------------
    def test_13_fifteen_day_sold_out_retention(self):
        """Verify sold-out listing removal is blocked before 15 days, allowed after 15 days, and history remains intact."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE role = 'admin'")
        admin_id = cursor.fetchone()['id']
        cursor.execute("SELECT id, name, phone FROM users WHERE role = 'farmer' LIMIT 1")
        farmer = cursor.fetchone()
        cursor.execute("SELECT id, name, phone FROM users WHERE role = 'buyer' LIMIT 1")
        buyer = cursor.fetchone()
        conn.close()

        pub_res = self.client.post('/api/listings', json={
            'farmer_id': farmer['id'],
            'farmer_name': farmer['name'],
            'farmer_phone': farmer['phone'],
            'image_url': f'/static/sample_cows/sample_{uuid.uuid4().hex[:6]}.jpg',
            'species': 'Cow',
            'breed': 'Brown Swiss cattle',
            'gender': 'Female',
            'age_years': 3.5,
            'milk_production': 19.0,
            'district': 'Mandya',
            'asking_price': 82000
        })
        self.assertEqual(pub_res.status_code, 200)
        listing_id = pub_res.get_json()['listing_id']

        buy_res = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': buyer['id'],
            'buyer_name': buyer['name'],
            'buyer_phone': buyer['phone'],
            'delivery_district': 'Bengaluru Urban',
            'delivery_address': 'Whitefield',
            'delivery_pincode': '560066'
        })
        self.assertEqual(buy_res.status_code, 200)

        # Attempt removal right away (0 days elapsed) -> BLOCKED
        rem_early = self.client.post(f'/api/admin/listings/{listing_id}/remove-soldout', json={
            'admin_id': admin_id
        })
        self.assertEqual(rem_early.status_code, 400)
        self.assertIn('Cannot remove listing before 15 days', rem_early.get_json()['error'])

        # Simulate 16 days elapsed
        sixteen_days_ago = (datetime.datetime.now() - datetime.timedelta(days=16)).strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE purchases SET purchased_at = ? WHERE listing_id = ?", (sixteen_days_ago, listing_id))
        conn.commit()
        conn.close()

        # Removal on day 16 -> MUST SUCCEED
        rem_success = self.client.post(f'/api/admin/listings/{listing_id}/remove-soldout', json={
            'admin_id': admin_id
        })
        self.assertEqual(rem_success.status_code, 200)
        self.assertEqual(rem_success.get_json()['status'], 'REMOVED')

        # Verify purchase history for buyer still survives
        buyer_purch = self.client.get(f'/api/buyer/{buyer["id"]}/purchases').get_json()
        item = next((p for p in buyer_purch if p['listing_id'] == listing_id), None)
        self.assertIsNotNone(item)
        self.assertEqual(item['livestock_price'], 82000)

        # Verify listing history for farmer shows REMOVED
        farmer_hist = self.client.get(f'/api/farmer/{farmer["id"]}/history').get_json()
        f_item = next((h for h in farmer_hist if h['id'] == listing_id), None)
        self.assertIsNotNone(f_item)
        self.assertEqual(f_item['status'], 'REMOVED')


if __name__ == '__main__':
    unittest.main()
