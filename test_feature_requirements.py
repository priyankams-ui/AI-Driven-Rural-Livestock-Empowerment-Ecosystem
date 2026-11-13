import unittest
import os
import re
import json
import io
import datetime
from dotenv import load_dotenv

load_dotenv()

from app import app
import notification_service
from services import email_service
from database import get_db, init_db, seed_single_admin, reset_user_and_transaction_data, hash_password

CAPTURED_EMAILS = []

orig_send_email = email_service.send_email
def intercepted_send_email(to_email, subject, html_body, plain_body=None):
    CAPTURED_EMAILS.append({
        'to': to_email.strip().lower(),
        'subject': subject,
        'plain': plain_body or '',
        'html': html_body or ''
    })
    cfg = email_service.get_agentmail_config()
    if cfg['api_key'] and 'your_agentmail' not in cfg['api_key'].lower() and 'placeholder' not in cfg['api_key'].lower():
        try:
            ok, msg = orig_send_email(to_email, subject, html_body, plain_body)
            if not ok and any(k in str(msg).lower() for k in ['suppressed', 'blocked', 'bounced', 'authentication failed', 'connecterror', 'getaddrinfo failed', 'network']):
                return True, "Email intercepted for testing."
            return ok, msg
        except Exception:
            return True, "Email intercepted for testing."
    return True, "Email dispatched successfully."

email_service.send_email = intercepted_send_email
notification_service.send_email = intercepted_send_email

class TestAllRequestedFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        init_db()
        seed_single_admin()
        reset_user_and_transaction_data()
        cls.client = app.test_client()

    def setUp(self):
        CAPTURED_EMAILS.clear()
        email_service.send_email = intercepted_send_email
        notification_service.send_email = intercepted_send_email

    # =========================================================================
    # FEATURE 1: FORGOT PASSWORD FOR ADMIN, FARMER, AND BUYER
    # =========================================================================

    def test_01_forgot_password_admin_flow(self):
        """Admin forgot password: request OTP -> verify OTP -> reset -> login with new password."""
        admin_email = "rakshitham556@gmail.com"

        # 1. Request password reset OTP
        res = self.client.post('/api/auth/forgot-password/request-otp', json={'email': admin_email})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()['status'], 'otp_sent')

        # Check OTP email captured
        otp_emails = [e for e in CAPTURED_EMAILS if e['to'] == admin_email and 'Password Reset' in e['subject']]
        self.assertTrue(len(otp_emails) >= 1)
        otp_match = re.search(r'is:\s*\n\n(\d{6})', otp_emails[-1]['plain'])
        self.assertIsNotNone(otp_match)
        otp_code = otp_match.group(1)

        # 2. Verify OTP
        res_v = self.client.post('/api/auth/forgot-password/verify-otp', json={
            'email': admin_email,
            'otp': otp_code
        })
        self.assertEqual(res_v.status_code, 200)
        reset_token = res_v.get_json().get('reset_token')
        self.assertIsNotNone(reset_token)

        # 3. Reset password
        new_pass = "Rakshitha@Admin2026!"
        res_reset = self.client.post('/api/auth/forgot-password/reset', json={
            'email': admin_email,
            'reset_token': reset_token,
            'new_password': new_pass,
            'confirm_password': new_pass
        })
        self.assertEqual(res_reset.status_code, 200)
        self.assertEqual(res_reset.get_json()['status'], 'success')

        # 4. Verify login with old password fails
        res_old_login = self.client.post('/api/admin/login', json={
            'email': admin_email,
            'password': 'old_wrong_password'
        })
        self.assertEqual(res_old_login.status_code, 401)

        # 5. Verify login with new password succeeds (initiates Admin 2FA OTP)
        res_new_login = self.client.post('/api/admin/login', json={
            'email': admin_email,
            'password': new_pass
        })
        self.assertEqual(res_new_login.status_code, 200)
        self.assertEqual(res_new_login.get_json()['status'], 'otp_required')

    def test_02_forgot_password_farmer_flow(self):
        """Farmer forgot password: request OTP -> verify OTP -> reset -> login with new password."""
        farmer_email = "farmer.verify.fp@gmail.com"
        # Seed farmer account
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            INSERT INTO users (name, email, username, phone, password_hash, password, role, email_verified, status)
            VALUES ('Basava Gowda', ?, 'basava_fp', '+919845331122', ?, '[BCRYPT_PROTECTED]', 'farmer', 1, 'Active')
        """, (farmer_email, hash_password("OldFarmerPass@123")))
        conn.commit()
        conn.close()

        # 1. Request OTP
        res = self.client.post('/api/auth/forgot-password/request-otp', json={'email': farmer_email})
        self.assertEqual(res.status_code, 200)

        # 2. Extract OTP
        otp_emails = [e for e in CAPTURED_EMAILS if e['to'] == farmer_email and 'Password Reset' in e['subject']]
        self.assertTrue(len(otp_emails) >= 1)
        otp_code = re.search(r'is:\s*\n\n(\d{6})', otp_emails[-1]['plain']).group(1)

        # 3. Verify OTP
        res_v = self.client.post('/api/auth/forgot-password/verify-otp', json={
            'email': farmer_email,
            'otp': otp_code
        })
        self.assertEqual(res_v.status_code, 200)
        reset_token = res_v.get_json().get('reset_token')

        # 4. Reset password
        new_pass = "NewFarmerPass@2026"
        res_reset = self.client.post('/api/auth/forgot-password/reset', json={
            'email': farmer_email,
            'reset_token': reset_token,
            'new_password': new_pass,
            'confirm_password': new_pass
        })
        self.assertEqual(res_reset.status_code, 200)

        # 5. Farmer login with new password succeeds
        res_login = self.client.post('/api/auth/login', json={
            'email': farmer_email,
            'password': new_pass
        })
        self.assertEqual(res_login.status_code, 200)
        self.assertEqual(res_login.get_json()['user']['role'], 'farmer')

    def test_03_forgot_password_buyer_flow(self):
        """Buyer forgot password: request OTP -> verify OTP -> reset -> login with new password."""
        buyer_email = "buyer.verify.fp@gmail.com"
        # Seed buyer account
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            INSERT INTO users (name, email, username, phone, password_hash, password, role, email_verified, status)
            VALUES ('Suresh Kumar', ?, 'suresh_fp', '+919845332244', ?, '[BCRYPT_PROTECTED]', 'buyer', 1, 'Active')
        """, (buyer_email, hash_password("OldBuyerPass@123")))
        conn.commit()
        conn.close()

        # 1. Request OTP
        res = self.client.post('/api/auth/forgot-password/request-otp', json={'email': buyer_email})
        self.assertEqual(res.status_code, 200)

        # 2. Extract OTP
        otp_emails = [e for e in CAPTURED_EMAILS if e['to'] == buyer_email and 'Password Reset' in e['subject']]
        self.assertTrue(len(otp_emails) >= 1)
        otp_code = re.search(r'is:\s*\n\n(\d{6})', otp_emails[-1]['plain']).group(1)

        # 3. Verify OTP
        res_v = self.client.post('/api/auth/forgot-password/verify-otp', json={
            'email': buyer_email,
            'otp': otp_code
        })
        self.assertEqual(res_v.status_code, 200)
        reset_token = res_v.get_json().get('reset_token')

        # 4. Reset password
        new_pass = "NewBuyerPass@2026"
        res_reset = self.client.post('/api/auth/forgot-password/reset', json={
            'email': buyer_email,
            'reset_token': reset_token,
            'new_password': new_pass,
            'confirm_password': new_pass
        })
        self.assertEqual(res_reset.status_code, 200)

        # 5. Buyer login with new password succeeds
        res_login = self.client.post('/api/auth/login', json={
            'email': buyer_email,
            'password': new_pass
        })
        self.assertEqual(res_login.status_code, 200)
        self.assertEqual(res_login.get_json()['user']['role'], 'buyer')

    def test_04_forgot_password_edge_cases(self):
        """Test forgot password validations: unregistered email, wrong OTP, password mismatch, short password."""
        # A. Unregistered email: returns generic success message without leaking account status
        res_unknown = self.client.post('/api/auth/forgot-password/request-otp', json={'email': 'unregistered.person@gmail.com'})
        self.assertEqual(res_unknown.status_code, 200)
        self.assertIn('If an account exists', res_unknown.get_json()['message'])

        # B. Wrong OTP
        res_wrong_otp = self.client.post('/api/auth/forgot-password/verify-otp', json={
            'email': 'rakshitham556@gmail.com',
            'otp': '999999'
        })
        self.assertEqual(res_wrong_otp.status_code, 400)

        # C. Password mismatch
        res_mismatch = self.client.post('/api/auth/forgot-password/reset', json={
            'email': 'rakshitham556@gmail.com',
            'reset_token': 'fake_token',
            'new_password': 'Password123',
            'confirm_password': 'Password456'
        })
        self.assertEqual(res_mismatch.status_code, 400)
        self.assertIn('do not match', res_mismatch.get_json()['error'])

        # D. Password too short (<6 characters)
        res_short = self.client.post('/api/auth/forgot-password/reset', json={
            'email': 'rakshitham556@gmail.com',
            'reset_token': 'fake_token',
            'new_password': '123',
            'confirm_password': '123'
        })
        self.assertEqual(res_short.status_code, 400)
        self.assertIn('at least 6 characters', res_short.get_json()['error'])

    # =========================================================================
    # FEATURE 2: LIVESTOCK SPECIES VALIDATION (COW OR BUFFALO ONLY)
    # =========================================================================

    def test_05_species_validation_cow_accepted(self):
        """Valid Cow image must be accepted by /api/predict/image-analysis."""
        cow_path = r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle0.jpg'
        self.assertTrue(os.path.exists(cow_path))

        with open(cow_path, 'rb') as img_f:
            data = {'file': (io.BytesIO(img_f.read()), 'cow.jpg', 'image/jpeg')}
            res = self.client.post('/api/predict/image-analysis', data=data, content_type='multipart/form-data')

        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertEqual(d.get('status'), 'success')
        self.assertTrue(d.get('is_valid'))
        self.assertEqual(d.get('species'), 'Cow')

    def test_06_species_validation_buffalo_accepted(self):
        """Valid Buffalo image must be accepted by /api/predict/image-analysis."""
        buff_path = r'buffalo\banni\0caae52ef510154bb215e99a7db35607.jpg'
        self.assertTrue(os.path.exists(buff_path))

        with open(buff_path, 'rb') as img_f:
            data = {'file': (io.BytesIO(img_f.read()), 'buffalo.jpg', 'image/jpeg')}
            res = self.client.post('/api/predict/image-analysis', data=data, content_type='multipart/form-data')

        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertEqual(d.get('status'), 'success')
        self.assertTrue(d.get('is_valid'))
        self.assertEqual(d.get('species'), 'Buffalo')

    def test_07_species_validation_non_livestock_rejected_exact_message(self):
        """Non-livestock / unsupported species image must be rejected with exact error message."""
        dummy_path = r'dummy_unidentified\dummy_0001.jpg'
        self.assertTrue(os.path.exists(dummy_path))

        with open(dummy_path, 'rb') as img_f:
            data = {'file': (io.BytesIO(img_f.read()), 'dummy.jpg', 'image/jpeg')}
            res = self.client.post('/api/predict/image-analysis', data=data, content_type='multipart/form-data')

        self.assertEqual(res.status_code, 400)
        d = res.get_json()
        self.assertEqual(d.get('status'), 'rejected')
        self.assertFalse(d.get('is_valid'))
        # Exact requirement message:
        expected_msg = "Unsupported image. Please upload a clear image of a cattle or buffalo."
        self.assertEqual(d.get('error'), expected_msg)

    def test_08_rejected_species_cannot_create_listing(self):
        """Farmer cannot create listing with rejected/unsupported species (e.g. Goat, Sheep, Dog, etc.)."""
        payload = {
            'farmer_id': 1,
            'farmer_name': 'Test Farmer',
            'farmer_phone': '+919845112233',
            'image_url': '/static/uploads/dummy.jpg',
            'species': 'Goat',
            'breed': 'Boer',
            'gender': 'Female',
            'district': 'Mysuru',
            'asking_price': 15000
        }
        res = self.client.post('/api/listings', json=payload)
        self.assertEqual(res.status_code, 400)
        expected_msg = "Unsupported image. Please upload a clear image of a cattle or buffalo."
        self.assertEqual(res.get_json().get('error'), expected_msg)

    # =========================================================================
    # FEATURE 3: AUTOMATIC GENDER DETECTION
    # =========================================================================

    def test_09_automatic_gender_detection_and_storage(self):
        """Gender is automatically detected from image, returned to client, and stored with listing."""
        # 1. Analyze cow image
        cow_path = r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle0.jpg'
        with open(cow_path, 'rb') as img_f:
            data = {'file': (io.BytesIO(img_f.read()), 'cow.jpg', 'image/jpeg')}
            res = self.client.post('/api/predict/image-analysis', data=data, content_type='multipart/form-data')

        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        detected_gender = d.get('gender')
        self.assertIn(detected_gender, ['Male', 'Female'])

        # 2. Create listing using the automatically detected gender (no manual dropdown selection)
        listing_payload = {
            'farmer_id': 1,
            'farmer_name': 'Basava Gowda',
            'farmer_phone': '+919845331122',
            'image_url': d.get('image_url'),
            'species': d.get('species'),
            'breed': d.get('breed'),
            'gender': detected_gender, # Automatically assigned from AI detection
            'age_years': 3.5,
            'milk_production': 0.0 if detected_gender == 'Male' else 12.0,
            'district': 'Mysuru',
            'taluk': 'Mysuru',
            'village': 'Hebbal',
            'asking_price': 75000
        }
        res_list = self.client.post('/api/listings', json=listing_payload)
        self.assertEqual(res_list.status_code, 200)
        created_listing = res_list.get_json()
        self.assertEqual(created_listing.get('status'), 'success')
        listing_id = created_listing.get('listing_id')

        # 3. Verify in database
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT gender FROM listings WHERE id = ?", (listing_id,))
        row = c.fetchone()
        conn.close()
        self.assertEqual(row['gender'], detected_gender)

    def test_09b_gender_prediction_male_and_female_cow_and_buffalo(self):
        """Verify model accurately detects Male and Female for both Cow and Buffalo, and does NOT always return Female."""
        samples = [
            ('Male Cow', r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle4.jpg', 'Cow', 'Male'),
            ('Female Cow', r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle1.jpg', 'Cow', 'Female'),
            ('Male Buffalo', r'buffalo\Jaffarabadi\Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 'Buffalo', 'Male'),
            ('Female Buffalo', r'buffalo\banni\0caae52ef510154bb215e99a7db35607.jpg', 'Buffalo', 'Female'),
        ]
        genders_predicted = set()
        for label, path, expected_species, expected_gender in samples:
            self.assertTrue(os.path.exists(path), f"Path {path} must exist")
            with open(path, 'rb') as img_f:
                data = {'file': (io.BytesIO(img_f.read()), os.path.basename(path), 'image/jpeg')}
                res = self.client.post('/api/predict/image-analysis', data=data, content_type='multipart/form-data')
            self.assertEqual(res.status_code, 200)
            d = res.get_json()
            self.assertEqual(d.get('status'), 'success')
            self.assertEqual(d.get('species'), expected_species)
            self.assertEqual(d.get('gender'), expected_gender, f"{label} expected {expected_gender} but got {d.get('gender')}")
            genders_predicted.add(d.get('gender'))

        # Verify that both Male and Female exist in predictions (not always Female)
        self.assertEqual(genders_predicted, {'Male', 'Female'})

    # =========================================================================
    # FEATURE 4: PURCHASE CONFIRMATION WITHOUT PURCHASE OTP + DUAL EMAILS
    # =========================================================================

    def test_10_purchase_confirmation_without_otp_and_both_emails_dispatched(self):
        """
        Buyer confirms purchase directly without OTP.
        1. Confirmation email is sent to BUYER with subject 'Livestock Purchase Confirmed'.
        2. Notification email is sent to FARMER with subject 'Your Livestock Has Been Purchased'.
        """
        conn = get_db()
        c = conn.cursor()

        # 1. Create farmer user
        farmer_email = "farmer.direct.test@gmail.com"
        c.execute("""
            INSERT INTO users (name, email, username, phone, password_hash, password, role, email_verified, status)
            VALUES ('Basava Farmer', ?, 'farmer_direct', '+919845887711', 'hash', '[BCRYPT_PROTECTED]', 'farmer', 1, 'Active')
        """, (farmer_email,))
        farmer_id = c.lastrowid

        # 2. Create available listing for this farmer
        c.execute("""
            INSERT INTO listings (
                farmer_id, farmer_name, farmer_phone, species, breed, gender, sex,
                health_score, health_text,
                age_years, milk_production, milk_yield_liters, is_pregnant, pregnancy_month,
                district, taluk, village, address, pincode,
                predicted_price, asking_price, image_url, status
            ) VALUES (
                ?, 'Basava Farmer', '+919845887711', 'Cow', 'Holstein Friesian cattle', 'Female', 'Female',
                5, 'Good',
                4.0, 14.5, 14.5, 0, 0,
                'Dharwad', 'Hubballi', 'Vidyanagar', 'Main Road', '580021',
                80000, 80000, '/static/uploads/cow_test.jpg', 'Available'
            )
        """, (farmer_id,))
        listing_id = c.lastrowid

        # Create buyer
        buyer_email = "direct.buyer@gmail.com"
        c.execute("""
            INSERT INTO users (name, email, username, phone, password_hash, password, role, email_verified, status)
            VALUES ('Direct Buyer', ?, 'direct_b', '+919845887722', ?, '[BCRYPT_PROTECTED]', 'buyer', 1, 'Active')
        """, (buyer_email, hash_password("BuyerPassword@123")))
        buyer_id = c.lastrowid
        conn.commit()
        conn.close()

        # Buyer directly purchases listing WITHOUT providing any OTP parameter
        res = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': buyer_id,
            'delivery_district': 'Dharwad',
            'delivery_address': 'Koppikar Road, Hubballi'
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data.get('summary', {}).get('order_status'), 'CONFIRMED')
        self.assertEqual(data.get('summary', {}).get('delivery_charge'), 0)
        self.assertEqual(data.get('summary', {}).get('total_amount'), 80000)

        # Verify BUYER confirmation email was sent via AgentMail
        buyer_emails = [e for e in CAPTURED_EMAILS if e['to'] == buyer_email and e['subject'] == 'Livestock Purchase Confirmed']
        self.assertTrue(len(buyer_emails) >= 1, "Buyer confirmation email must be dispatched")
        buyer_mail = buyer_emails[-1]
        self.assertIn("Hello Direct Buyer,", buyer_mail['plain'])
        self.assertIn("Livestock: Holstein Friesian cattle", buyer_mail['plain'])
        self.assertIn("Species: Cow", buyer_mail['plain'])
        self.assertIn("Gender: Female", buyer_mail['plain'])
        self.assertIn("Price: ₹80,000", buyer_mail['plain'])

        # Verify FARMER notification email was sent via AgentMail to the listing farmer
        farmer_emails = [e for e in CAPTURED_EMAILS if e['to'] == farmer_email and e['subject'] == 'Your Livestock Has Been Purchased']
        self.assertTrue(len(farmer_emails) >= 1, "Farmer notification email must be dispatched")
        farmer_mail = farmer_emails[-1]
        self.assertIn("Hello Basava Farmer,", farmer_mail['plain'])
        self.assertIn("Livestock: Holstein Friesian cattle", farmer_mail['plain'])
        self.assertIn("Species: Cow", farmer_mail['plain'])
        self.assertIn("Gender: Female", farmer_mail['plain'])
        self.assertIn("Purchase Price: ₹80,000", farmer_mail['plain'])
        self.assertIn("The livestock listed by you has been successfully purchased by a buyer.", farmer_mail['plain'])

        # Verify listing is marked as SOLD_OUT and second purchase attempt is rejected
        res_repeat = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': buyer_id,
            'delivery_district': 'Dharwad',
            'delivery_address': 'Another Road'
        })
        self.assertEqual(res_repeat.status_code, 400)
        self.assertIn('no longer available', res_repeat.get_json().get('error', '').lower())


if __name__ == '__main__':
    unittest.main()
