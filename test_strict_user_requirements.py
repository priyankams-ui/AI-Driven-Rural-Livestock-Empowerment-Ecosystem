import unittest
import os
import re
import io
import json
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

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
    cfg = email_service.get_agentmail_config()
    if cfg['api_key'] and 'your_agentmail' not in cfg['api_key'].lower() and 'placeholder' not in cfg['api_key'].lower():
        try:
            ok, msg = orig_send_email(to_email, subject, html_body, plain_body)
            return True, msg
        except Exception as e:
            return True, f"Mocked send due to: {e}"
    return True, "Email dispatched successfully."

email_service.send_email = intercepted_send_email
notification_service.send_email = intercepted_send_email

class TestStrictUserRequirements(unittest.TestCase):
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
    # 1. ACCEPT ONLY COW OR BUFFALO IMAGES (REJECT ALL OTHERS)
    # =========================================================================

    def test_01_cow_image_accepted(self):
        """Cow image must be accepted by AI species detection."""
        cow_path = r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle0.jpg'
        self.assertTrue(os.path.exists(cow_path))
        with open(cow_path, 'rb') as f:
            res = self.client.post('/api/predict/image-analysis',
                                   data={'file': (io.BytesIO(f.read()), 'cow.jpg', 'image/jpeg')},
                                   content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertEqual(d.get('status'), 'success')
        self.assertTrue(d.get('is_valid'))
        self.assertEqual(d.get('species'), 'Cow')

    def test_02_buffalo_image_accepted(self):
        """Buffalo image must be accepted by AI species detection."""
        buff_path = r'buffalo\banni\0caae52ef510154bb215e99a7db35607.jpg'
        self.assertTrue(os.path.exists(buff_path))
        with open(buff_path, 'rb') as f:
            res = self.client.post('/api/predict/image-analysis',
                                   data={'file': (io.BytesIO(f.read()), 'buff.jpg', 'image/jpeg')},
                                   content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertEqual(d.get('status'), 'success')
        self.assertTrue(d.get('is_valid'))
        self.assertEqual(d.get('species'), 'Buffalo')

    def test_03_non_livestock_and_unclear_images_rejected(self):
        """
        Verify rejection of non-livestock images:
        Goat, Sheep, Dog, Cat, Human, Vehicle, Building, Food, Random object, Unclear image.
        Must return exact error: 'Unsupported image. Please upload a clear image of a cattle or buffalo.'
        """
        expected_msg = "Unsupported image. Please upload a clear image of a cattle or buffalo."

        # 1. Dummy unidentified objects from dataset
        for i in range(1, 6):
            d_path = os.path.join('dummy_unidentified', f'dummy_{i:04d}.jpg')
            if os.path.exists(d_path):
                with open(d_path, 'rb') as f:
                    res = self.client.post('/api/predict/image-analysis',
                                           data={'file': (io.BytesIO(f.read()), 'dummy.jpg', 'image/jpeg')},
                                           content_type='multipart/form-data')
                self.assertEqual(res.status_code, 400)
                d = res.get_json()
                self.assertEqual(d.get('status'), 'rejected')
                self.assertFalse(d.get('is_valid'))
                self.assertIn("clear image of a", d.get('error'))

        # 2. Synthetic test images: Blank/solid (unclear), Random noise (random object/food)
        blank_img = Image.new('RGB', (224, 224), color=(200, 200, 200))
        buf = io.BytesIO()
        blank_img.save(buf, format='JPEG')
        buf.seek(0)
        res_blank = self.client.post('/api/predict/image-analysis',
                                     data={'file': (buf, 'blank.jpg', 'image/jpeg')},
                                     content_type='multipart/form-data')
        self.assertEqual(res_blank.status_code, 400)
        self.assertIn("clear image of a", res_blank.get_json().get('error'))

        noise_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise_arr)
        buf_noise = io.BytesIO()
        noise_img.save(buf_noise, format='JPEG')
        buf_noise.seek(0)
        res_noise = self.client.post('/api/predict/image-analysis',
                                     data={'file': (buf_noise, 'noise.jpg', 'image/jpeg')},
                                     content_type='multipart/form-data')
        self.assertEqual(res_noise.status_code, 400)
        self.assertIn("clear image of a", res_noise.get_json().get('error'))

    def test_04_rejected_image_cannot_create_listing(self):
        """Rejected image/species cannot create listing."""
        expected_msg = "Unsupported image. Please upload a clear image of a cattle or buffalo."
        unsupported = ['Goat', 'Sheep', 'Dog', 'Cat', 'Horse', 'Bird', 'Human', 'Vehicle', 'Building', 'Food', 'Random', 'Unknown']
        for unsp in unsupported:
            payload = {
                'farmer_id': 1,
                'farmer_name': 'Test Farmer',
                'farmer_phone': '+919845112233',
                'image_url': f'/static/uploads/{unsp.lower()}.jpg',
                'species': unsp,
                'breed': 'Unknown',
                'gender': 'Female',
                'district': 'Mandya',
                'asking_price': 20000
            }
            res = self.client.post('/api/listings', json=payload)
            self.assertEqual(res.status_code, 400)
            self.assertEqual(res.get_json().get('error'), expected_msg)

    # =========================================================================
    # 2. CORRECT GENDER DETECTION (MALE VS FEMALE)
    # =========================================================================

    def test_05_gender_detection_not_always_female(self):
        """
        Verify AI gender detection returns BOTH Male and Female.
        Test actual images:
        1. Male Cow
        2. Female Cow
        3. Male Buffalo
        4. Female Buffalo
        """
        samples = [
            ('Male Cow', r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle4.jpg', 'Cow', 'Male'),
            ('Female Cow', r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle1.jpg', 'Cow', 'Female'),
            ('Male Buffalo', r'buffalo\Jaffarabadi\Jafarabadi_Bull_at_Badal_Farm_Punjab_.jpg', 'Buffalo', 'Male'),
            ('Female Buffalo', r'buffalo\banni\0caae52ef510154bb215e99a7db35607.jpg', 'Buffalo', 'Female'),
        ]
        results = {}
        for label, path, exp_species, exp_gender in samples:
            self.assertTrue(os.path.exists(path), f"File {path} must exist")
            with open(path, 'rb') as f:
                res = self.client.post('/api/predict/image-analysis',
                                       data={'file': (io.BytesIO(f.read()), os.path.basename(path), 'image/jpeg')},
                                       content_type='multipart/form-data')
            self.assertEqual(res.status_code, 200)
            d = res.get_json()
            self.assertEqual(d.get('status'), 'success')
            self.assertEqual(d.get('species'), exp_species)
            self.assertEqual(d.get('gender'), exp_gender, f"{label} expected {exp_gender} but got {d.get('gender')}")
            results[label] = d.get('gender')

        # Assert system produces both Male and Female
        self.assertIn('Male', results.values())
        self.assertIn('Female', results.values())
        self.assertEqual(results['Male Cow'], 'Male')
        self.assertEqual(results['Female Cow'], 'Female')
        self.assertEqual(results['Male Buffalo'], 'Male')
        self.assertEqual(results['Female Buffalo'], 'Female')

    def test_06_male_and_female_gender_persisted_to_database(self):
        """Verify detected Male and Female genders are correctly persisted into database listing."""
        # 1. Male Cow Listing
        male_cow_path = r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle4.jpg'
        with open(male_cow_path, 'rb') as f:
            res_male = self.client.post('/api/predict/image-analysis',
                                        data={'file': (io.BytesIO(f.read()), 'mcow.jpg', 'image/jpeg')},
                                        content_type='multipart/form-data')
        d_male = res_male.get_json()
        self.assertEqual(d_male.get('gender'), 'Male')

        res_post_male = self.client.post('/api/listings', json={
            'farmer_id': 1,
            'farmer_name': 'Ramesh Farmer',
            'farmer_phone': '+919845112233',
            'image_url': d_male.get('image_url'),
            'species': d_male.get('species'),
            'breed': d_male.get('breed'),
            'gender': d_male.get('gender'),
            'age_years': 4.0,
            'milk_production': 0.0,
            'district': 'Mandya',
            'taluk': 'Maddur',
            'village': 'KM Doddi',
            'asking_price': 85000
        })
        self.assertEqual(res_post_male.status_code, 200)
        male_listing_id = res_post_male.get_json().get('listing_id')

        # 2. Female Cow Listing
        female_cow_path = r'Cattle Breeds\Ayrshire cattle\Ayrshirecattle1.jpg'
        with open(female_cow_path, 'rb') as f:
            res_female = self.client.post('/api/predict/image-analysis',
                                          data={'file': (io.BytesIO(f.read()), 'fcow.jpg', 'image/jpeg')},
                                          content_type='multipart/form-data')
        d_female = res_female.get_json()
        self.assertEqual(d_female.get('gender'), 'Female')

        res_post_female = self.client.post('/api/listings', json={
            'farmer_id': 1,
            'farmer_name': 'Ramesh Farmer',
            'farmer_phone': '+919845112233',
            'image_url': d_female.get('image_url'),
            'species': d_female.get('species'),
            'breed': d_female.get('breed'),
            'gender': d_female.get('gender'),
            'age_years': 3.5,
            'milk_production': 14.0,
            'district': 'Mandya',
            'taluk': 'Maddur',
            'village': 'KM Doddi',
            'asking_price': 90000
        })
        self.assertEqual(res_post_female.status_code, 200)
        female_listing_id = res_post_female.get_json().get('listing_id')

        # Check DB
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT gender, milk_production FROM listings WHERE id = ?", (male_listing_id,))
        m_row = c.fetchone()
        self.assertEqual(m_row['gender'], 'Male')
        self.assertEqual(m_row['milk_production'], 0.0)

        c.execute("SELECT gender, milk_production FROM listings WHERE id = ?", (female_listing_id,))
        f_row = c.fetchone()
        self.assertEqual(f_row['gender'], 'Female')
        self.assertEqual(f_row['milk_production'], 14.0)
        conn.close()

    # =========================================================================
    # 3. BUYER PURCHASE — NO PURCHASE OTP + DUAL EMAILS (BUYER & FARMER)
    # =========================================================================

    def test_07_purchase_flow_no_otp_and_both_emails(self):
        """
        Purchase flow:
        - Buyer selects livestock
        - Clicks Confirm Purchase (NO PURCHASE OTP)
        - Purchase recorded successfully
        - Buyer receives confirmation email ('Livestock Purchase Confirmed')
        - Farmer receives purchase notification email ('Your Livestock Has Been Purchased')
        - Both contain correct fields
        """
        conn = get_db()
        c = conn.cursor()

        # Seed farmer account
        farmer_email = "farmer.notification.target@gmail.com"
        farmer_name = "Suresh Gowda"
        c.execute("""
            INSERT INTO users (name, email, username, phone, password_hash, password, role, email_verified, status)
            VALUES (?, ?, 'suresh_farmer', '+919845778899', 'hash', '[BCRYPT_PROTECTED]', 'farmer', 1, 'Active')
        """, (farmer_name, farmer_email))
        farmer_id = c.lastrowid

        # Seed listing
        c.execute("""
            INSERT INTO listings (
                farmer_id, farmer_name, farmer_phone, species, breed, gender, sex,
                health_score, health_text,
                age_years, milk_production, milk_yield_liters, is_pregnant, pregnancy_month,
                district, taluk, village, address, pincode,
                predicted_price, asking_price, image_url, status
            ) VALUES (
                ?, ?, '+919845778899', 'Buffalo', 'Murrah', 'Female', 'Female',
                5, 'Good',
                4.5, 16.0, 16.0, 0, 0,
                'Belagavi', 'Gokak', 'Konur', 'Farm Plot 12', '591307',
                95000, 95000, '/static/uploads/murrah.jpg', 'Available'
            )
        """, (farmer_id, farmer_name))
        listing_id = c.lastrowid

        # Seed buyer account
        buyer_email = "buyer.verified.client@gmail.com"
        buyer_name = "Manjunath K"
        c.execute("""
            INSERT INTO users (name, email, username, phone, password_hash, password, role, email_verified, status)
            VALUES (?, ?, 'manju_buyer', '+919845119900', ?, '[BCRYPT_PROTECTED]', 'buyer', 1, 'Active')
        """, (buyer_name, buyer_email, hash_password("BuyerPassword@2026")))
        buyer_id = c.lastrowid
        conn.commit()
        conn.close()

        # Login as buyer
        res_login = self.client.post('/api/auth/login', json={
            'identifier': buyer_email,
            'password': 'BuyerPassword@2026',
            'role': 'buyer'
        })
        self.assertEqual(res_login.status_code, 200)

        # Confirm purchase with NO OTP
        res_purchase = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': buyer_id,
            'delivery_district': 'Belagavi',
            'delivery_address': 'Main Market Road, Belagavi'
        })
        self.assertEqual(res_purchase.status_code, 200)
        p_data = res_purchase.get_json()
        self.assertEqual(p_data.get('status'), 'success')
        self.assertEqual(p_data.get('summary', {}).get('order_status'), 'CONFIRMED')
        self.assertEqual(p_data.get('summary', {}).get('total_amount'), 95000)

        # 1. Verify Buyer Confirmation Email
        b_emails = [e for e in CAPTURED_EMAILS if e['to'] == buyer_email and e['subject'] == 'Livestock Purchase Confirmed']
        self.assertTrue(len(b_emails) >= 1, "Buyer must receive confirmation email")
        b_plain = b_emails[-1]['plain']
        self.assertIn(f"Hello {buyer_name},", b_plain)
        self.assertIn("Your livestock purchase has been successfully confirmed.", b_plain)
        self.assertIn("Livestock: Murrah", b_plain)
        self.assertIn("Species: Buffalo", b_plain)
        self.assertIn("Gender: Female", b_plain)
        self.assertIn("Price: ₹95,000", b_plain)
        self.assertIn("Thank you for using the Livestock Marketplace.", b_plain)

        # 2. Verify Farmer Purchase Notification Email
        f_emails = [e for e in CAPTURED_EMAILS if e['to'] == farmer_email and e['subject'] == 'Your Livestock Has Been Purchased']
        self.assertTrue(len(f_emails) >= 1, "Farmer must receive purchase notification email")
        f_plain = f_emails[-1]['plain']
        self.assertIn(f"Hello {farmer_name},", f_plain)
        self.assertIn("Your livestock has been purchased successfully.", f_plain)
        self.assertIn("Livestock: Murrah", f_plain)
        self.assertIn("Species: Buffalo", f_plain)
        self.assertIn("Gender: Female", f_plain)
        self.assertIn("Purchase Price: ₹95,000", f_plain)
        self.assertIn("The livestock listed by you has been successfully purchased by a buyer.", f_plain)
        self.assertIn("Thank you for using the Livestock Marketplace.", f_plain)

    # =========================================================================
    # 4. REGISTRATION AND FORGOT PASSWORD OTP STILL WORK
    # =========================================================================

    def test_08_registration_otp_works(self):
        """Verify farmer and buyer registration OTP flow still works normally."""
        # 1. Farmer Registration OTP Request
        res_f = self.client.post('/api/auth/register-request-otp', json={
            'name': 'New Farmer Test',
            'email': 'new.farmer.test.reg@gmail.com',
            'phone': '+919845223344',
            'password': 'FarmerPass@2026',
            'confirm_password': 'FarmerPass@2026',
            'role': 'farmer',
            'district': 'Mysuru',
            'taluk': 'Mysuru',
            'village': 'Hebbal',
            'address': 'Main Road',
            'pincode': '570016'
        })
        self.assertEqual(res_f.status_code, 200)
        self.assertEqual(res_f.get_json().get('status'), 'otp_sent')

        f_otps = [e for e in CAPTURED_EMAILS if e['to'] == 'new.farmer.test.reg@gmail.com' and 'Verification' in e['subject']]
        self.assertTrue(len(f_otps) >= 1)
        f_otp_code = re.search(r'is:\s*\n\n(\d{6})', f_otps[-1]['plain']).group(1)

        # 2. Farmer Registration OTP Verify
        res_f_verify = self.client.post('/api/auth/register-verify-otp', json={
            'email': 'new.farmer.test.reg@gmail.com',
            'otp': f_otp_code
        })
        self.assertEqual(res_f_verify.status_code, 200)
        self.assertEqual(res_f_verify.get_json().get('status'), 'success')

    def test_09_forgot_password_otp_works(self):
        """Verify forgot-password OTP flow still works normally."""
        target_email = "buyer.verified.client@gmail.com"
        res_req = self.client.post('/api/auth/forgot-password/request-otp', json={'email': target_email})
        self.assertEqual(res_req.status_code, 200)
        self.assertEqual(res_req.get_json().get('status'), 'otp_sent')

        fp_otps = [e for e in CAPTURED_EMAILS if e['to'] == target_email and 'Password Reset' in e['subject']]
        self.assertTrue(len(fp_otps) >= 1)
        fp_code = re.search(r'is:\s*\n\n(\d{6})', fp_otps[-1]['plain']).group(1)

        res_ver = self.client.post('/api/auth/forgot-password/verify-otp', json={
            'email': target_email,
            'otp': fp_code
        })
        self.assertEqual(res_ver.status_code, 200)
        token = res_ver.get_json().get('reset_token')

        res_rst = self.client.post('/api/auth/forgot-password/reset', json={
            'email': target_email,
            'reset_token': token,
            'new_password': 'UpdatedPass@2026',
            'confirm_password': 'UpdatedPass@2026'
        })
        self.assertEqual(res_rst.status_code, 200)
        self.assertEqual(res_rst.get_json().get('status'), 'success')


if __name__ == '__main__':
    unittest.main()
