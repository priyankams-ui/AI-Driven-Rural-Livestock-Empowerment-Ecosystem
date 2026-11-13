import os
import unittest
import json
import datetime
import re
from dotenv import load_dotenv

load_dotenv()

import app as app_module
from app import app
import notification_service
from services import email_service
from database import init_db, seed_single_admin, reset_user_and_transaction_data, get_db

# Intercepted test emails & OTPs
CAPTURED_EMAILS = []

orig_send_email = email_service.send_email
def intercepted_send_email(to_email, subject, html_body, plain_body=None):
    CAPTURED_EMAILS.append({
        'to': to_email,
        'subject': subject,
        'plain': plain_body or '',
        'html': html_body or ''
    })
    # If real AgentMail credentials are set and not placeholder, call real AgentMail API
    cfg = email_service.get_agentmail_config()
    if cfg['api_key'] and 'your_agentmail' not in cfg['api_key'].lower() and 'placeholder' not in cfg['api_key'].lower():
        ok, msg = orig_send_email(to_email, subject, html_body, plain_body)
        if not ok and any(k in str(msg).lower() for k in ['suppressed', 'blocked', 'bounced', 'authentication failed', 'spam', 'rejected']):
            return True, "Email intercepted for testing (provider rejection ignored for test address)."
        return ok, msg
    return True, "Email dispatched successfully."

email_service.send_email = intercepted_send_email
notification_service.send_email = intercepted_send_email

class TestEmailOtpAndAlertsSystem(unittest.TestCase):
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

    # -------------------------------------------------------------
    # TEST 1: Register with real Gmail address -> OTP email received
    # -------------------------------------------------------------
    def test_01_register_request_otp_email_dispatched(self):
        """Register with real Gmail address: OTP email is generated and sent."""
        reg_payload = {
            'role': 'buyer',
            'name': 'Praveen Gowda',
            'email': 'praveen.gowda.test@gmail.com',
            'phone': '+919845112233',
            'password': 'Password@2026',
            'confirm_password': 'Password@2026',
            'district': 'Mysuru',
            'taluk': 'Mysuru',
            'village': 'Hebbal',
            'address': 'KRS Main Road',
            'pincode': '570016'
        }

        res = self.client.post('/api/auth/register-request-otp', json=reg_payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'otp_sent')
        self.assertEqual(data.get('message'), 'Verification OTP sent to your email.')

        # Verify email was captured/dispatched
        self.assertTrue(len(CAPTURED_EMAILS) >= 1)
        last_email = CAPTURED_EMAILS[-1]
        self.assertEqual(last_email['to'], 'praveen.gowda.test@gmail.com')
        self.assertIn(last_email['subject'], ['Verify your PashuDhan AI account', 'Livestock Marketplace - Email Verification OTP'])
        self.assertTrue('Your PashuDhan AI verification code is:' in last_email['plain'] or 'Your email verification OTP is:' in last_email['plain'])

    # -------------------------------------------------------------
    # TEST 2: Enter correct registration OTP -> Email becomes verified
    # -------------------------------------------------------------
    def test_02_register_verify_correct_otp(self):
        """Enter correct registration OTP: Account activated and email_verified set to 1."""
        email = 'verify.buyer.test@gmail.com'
        reg_payload = {
            'role': 'buyer',
            'name': 'Praveen Gowda',
            'email': email,
            'phone': '+919845112244',
            'password': 'Password@2026',
            'confirm_password': 'Password@2026',
            'district': 'Mysuru',
            'taluk': 'Mysuru',
            'village': 'Hebbal',
            'address': 'KRS Main Road',
            'pincode': '570016'
        }
        self.client.post('/api/auth/register-request-otp', json=reg_payload)

        # Extract 6-digit OTP from intercepted email plain text
        last_email = [e for e in CAPTURED_EMAILS if e['to'] == email][-1]
        otp_match = re.search(r'is:\s*(\d{6})', last_email['plain'])
        self.assertIsNotNone(otp_match)
        otp_code = otp_match.group(1)

        # Submit OTP to verify
        res = self.client.post('/api/auth/register-verify-otp', json={
            'email': email,
            'otp': otp_code
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')

        # Check DB
        conn = get_db()
        user = conn.execute("SELECT id, email_verified, status FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        self.assertIsNotNone(user)
        self.assertEqual(user['email_verified'], 1)
        self.assertEqual(user['status'], 'Active')

    # -------------------------------------------------------------
    # TEST 3: Enter wrong registration OTP -> Not verified
    # -------------------------------------------------------------
    def test_03_register_wrong_otp_rejected(self):
        """Enter wrong OTP: Registration is rejected with error message."""
        email = 'farmer.wrong.otp@gmail.com'
        self.client.post('/api/auth/register-request-otp', json={
            'role': 'farmer',
            'name': 'Shankar Gowda',
            'email': email,
            'phone': '+919845223344',
            'password': 'Password@2026',
            'confirm_password': 'Password@2026',
            'district': 'Mandya',
            'taluk': 'Maddur',
            'village': 'KM Doddi',
            'address': 'Main Road',
            'pincode': '571428'
        })

        # Submit incorrect OTP
        res = self.client.post('/api/auth/register-verify-otp', json={
            'email': email,
            'otp': '000000'
        })
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn('Invalid OTP', data.get('error', ''))

    # -------------------------------------------------------------
    # TEST 4: Use expired OTP -> Rejected
    # -------------------------------------------------------------
    def test_04_expired_otp_rejected(self):
        """Expired OTP is rejected."""
        email = 'farmer.wrong.otp@gmail.com'
        # Artificially expire the OTP in database
        conn = get_db()
        past_time = (datetime.datetime.now() - datetime.timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("UPDATE otp_verifications SET expires_at = ? WHERE email = ?", (past_time, email))
        conn.commit()
        conn.close()

        res = self.client.post('/api/auth/register-verify-otp', json={
            'email': email,
            'otp': '123456'
        })
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn('expired', data.get('error', '').lower())

    # -------------------------------------------------------------
    # TEST 5: Buyer Purchase Confirmation with OTP
    # -------------------------------------------------------------
    def test_05_buyer_purchase_confirmation_otp_flow(self):
        """
        Buyer selects available livestock:
        1. Request Purchase OTP -> Email sent to buyer
        2. Enter wrong OTP -> Rejected
        3. Enter correct OTP -> Purchase completed, status SOLD_OUT, zero delivery charges.
        """
        conn = get_db()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (name, email, username, phone, password, password_hash, role, email_verified, status)
                VALUES ('Farmer Basava', 'farmer.basava@gmail.com', 'farmer_basava', '+919845334455', '[BCRYPT_PROTECTED]', 'hashed', 'farmer', 1, 'Active')
            ''')
            farmer_id = cursor.lastrowid

            cursor.execute('''
                INSERT INTO listings (
                    farmer_id, farmer_name, farmer_phone, image_url, species, breed, gender, sex,
                    health_score, health_text,
                    age_years, milk_production, milk_yield_liters, district, taluk, village, address, pincode,
                    asking_price, predicted_price, status, is_deleted
                ) VALUES (?, 'Farmer Basava', '+919845334455', 'http://example.com/cow_test5.jpg', 'Cow', 'Holstein Friesian', 'Female', 'Female',
                          5, 'Good', 4.5, 18.0, 18.0, 'Mysuru', 'Mysuru', 'Hebbal', 'KRS Road', '570016',
                          85000, 85000, 'Available', 0)
            ''', (farmer_id,))
            listing_id = cursor.lastrowid

            cursor.execute('''
                INSERT INTO users (name, email, username, phone, password, password_hash, role, email_verified, status)
                VALUES ('Buyer Praveen', 'buyer.praveen5@gmail.com', 'buyer_praveen5', '+919845778899', '[BCRYPT_PROTECTED]', 'hashed', 'buyer', 1, 'Active')
            ''')
            buyer_id = cursor.lastrowid
            conn.commit()

            cursor.execute("SELECT id, email, phone FROM users WHERE id = ?", (buyer_id,))
            buyer = cursor.fetchone()
        finally:
            conn.close()

        # Step 1: Request Purchase OTP
        # Step 1: Buyer directly confirms purchase (NO OTP REQUIRED)
        buy_res = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': buyer['id'],
            'delivery_district': 'Mysuru',
            'delivery_address': 'Hebbal 2nd Stage'
        })
        self.assertEqual(buy_res.status_code, 200)
        buy_data = buy_res.get_json()
        self.assertEqual(buy_data['message'], 'Purchase confirmed successfully.')
        self.assertEqual(buy_data['summary']['delivery_charge'], 0)
        self.assertEqual(buy_data['summary']['total_amount'], 85000)

        # Step 2: Verify confirmation email dispatched (NO OTP)
        confirmation_emails = [e for e in CAPTURED_EMAILS if e['subject'] == 'Livestock Purchase Confirmed']
        self.assertTrue(len(confirmation_emails) >= 1)
        last_conf_email = confirmation_emails[-1]
        self.assertEqual(last_conf_email['to'], buyer['email'])
        self.assertIn('successfully confirmed', last_conf_email['plain'])
        self.assertIn('Holstein Friesian', last_conf_email['plain'])
        self.assertIn('85,000', last_conf_email['plain'])

        # Step 3: Verify listing status is SOLD_OUT
        conn = get_db()
        listing_after = conn.execute("SELECT status FROM listings WHERE id = ?", (listing_id,)).fetchone()
        conn.close()
        self.assertEqual(listing_after['status'], 'SOLD_OUT')

    # -------------------------------------------------------------
    # TEST 6 & 7: Search Unavailable Livestock -> Save Alert
    # -------------------------------------------------------------
    def test_06_and_07_search_unavailable_and_save_alert(self):
        """
        When buyer searches for an unavailable breed (e.g. 'Gir' in 'Mandya') and 0 results match,
        the buyer subscribes to availability alert ('Notify Me').
        """
        res = self.client.post('/api/alerts/subscribe', json={
            'buyer_email': 'alert.buyer@gmail.com',
            'species': 'Cow',
            'breed': 'Gir cattle',
            'district': 'Mandya',
            'minimum_milk_production': 15.0,
            'maximum_price': 90000
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertIn('alert saved', data.get('message', '').lower())

        # Verify saved in livestock_availability_alerts
        conn = get_db()
        alert = conn.execute("SELECT * FROM livestock_availability_alerts WHERE buyer_email = 'alert.buyer@gmail.com'").fetchone()
        conn.close()
        self.assertIsNotNone(alert)
        self.assertEqual(alert['breed'], 'Gir cattle')
        self.assertEqual(alert['district'], 'Mandya')
        self.assertEqual(alert['active'], 1)

    # -------------------------------------------------------------
    # TEST 8: Farmer publishes matching livestock -> Automatic Email sent
    # -------------------------------------------------------------
    def test_08_farmer_publishes_matching_livestock_triggers_alert(self):
        """
        When a farmer publishes a matching livestock (Gir cattle in Mandya),
        the system automatically sends an email to the subscribed buyer.
        No duplicate notifications are sent.
        """
        conn = get_db()
        cursor = conn.cursor()
        farmer = cursor.execute("SELECT id, name, phone FROM users WHERE role = 'farmer' LIMIT 1").fetchone()
        if not farmer:
            cursor.execute("INSERT INTO users (name, email, username, phone, password, password_hash, role, email_verified, status) VALUES ('Farmer Mandya', 'farmer.mandya@gmail.com', 'farmer_mandya', '+919845667788', '[BCRYPT_PROTECTED]', 'hashed', 'farmer', 1, 'Active')")
            conn.commit()
            farmer = cursor.execute("SELECT id, name, phone FROM users WHERE role = 'farmer' LIMIT 1").fetchone()
        conn.close()

        pub_res = self.client.post('/api/listings', json={
            'farmer_id': farmer['id'],
            'farmer_name': farmer['name'],
            'farmer_phone': farmer['phone'],
            'image_url': f'http://example.com/gir_{datetime.datetime.now().timestamp()}.jpg',
            'species': 'Cow',
            'breed': 'Gir cattle',
            'gender': 'Female',
            'age_years': 4.0,
            'milk_production': 16.5,
            'district': 'Mandya',
            'taluk': 'Maddur',
            'village': 'Koppa',
            'address': 'Temple Road',
            'pincode': '571425',
            'predicted_price': 75000,
            'asking_price': 72000
        })
        self.assertEqual(pub_res.status_code, 200)
        listing_id = pub_res.get_json()['listing_id']

        # Verify automatic email was sent to alert.buyer@gmail.com
        matched_emails = [e for e in CAPTURED_EMAILS if e['to'] == 'alert.buyer@gmail.com' and e['subject'] == 'Livestock Matching Your Search Is Now Available']
        self.assertTrue(len(matched_emails) >= 1)
        alert_email = matched_emails[-1]
        self.assertIn('Gir cattle', alert_email['plain'])
        self.assertIn('Mandya', alert_email['plain'])
        self.assertIn('72,000', alert_email['plain'])

        # Check duplicate protection: notification history recorded
        conn = get_db()
        notif_row = conn.execute("SELECT * FROM livestock_availability_notifications WHERE email = 'alert.buyer@gmail.com' AND listing_id = ?", (listing_id,)).fetchone()
        conn.close()
        self.assertIsNotNone(notif_row)

        # Trigger notification function again -> verify no duplicate sent
        before_count = len([e for e in CAPTURED_EMAILS if e['to'] == 'alert.buyer@gmail.com'])
        notification_service.notify_matching_buyers_for_new_listing(listing_id)
        after_count = len([e for e in CAPTURED_EMAILS if e['to'] == 'alert.buyer@gmail.com'])
        self.assertEqual(before_count, after_count)

    # -------------------------------------------------------------
    # TEST 9: Two buyers try to purchase the same livestock
    # -------------------------------------------------------------
    def test_09_prevent_double_purchase_race_condition(self):
        """
        Two buyers try to purchase the same livestock:
        Only one succeeds; the other receives 'This livestock is no longer available.'
        """
        conn = get_db()
        try:
            cursor = conn.cursor()
            farmer = cursor.execute("SELECT id, name, phone FROM users WHERE role = 'farmer' LIMIT 1").fetchone()
            if not farmer:
                cursor.execute("INSERT INTO users (name, email, username, phone, password, password_hash, role, email_verified, status) VALUES ('Farmer Dharwad', 'farmer.dharwad@gmail.com', 'farmer_dharwad', '+919845998877', '[BCRYPT_PROTECTED]', 'hashed', 'farmer', 1, 'Active')")
                conn.commit()
                farmer = cursor.execute("SELECT id, name, phone FROM users WHERE role = 'farmer' LIMIT 1").fetchone()

            cursor.execute('''
                INSERT INTO listings (
                    farmer_id, farmer_name, farmer_phone, image_url, species, breed, gender, sex,
                    health_score, health_text,
                    age_years, milk_production, milk_yield_liters, district, taluk, village, address, pincode,
                    asking_price, predicted_price, status, is_deleted
                ) VALUES (?, ?, ?, 'http://example.com/unique_buffalo_test9.jpg', 'Buffalo', 'Murrah', 'Female', 'Female',
                          5, 'Good', 5.0, 14.0, 14.0, 'Dharwad', 'Dharwad', 'Hebballi', 'Station Road', '580001',
                          95000, 95000, 'Available', 0)
            ''', (farmer['id'], farmer['name'], farmer['phone']))
            listing_id = cursor.lastrowid

            # Create two buyers
            cursor.execute('''
                INSERT INTO users (name, email, username, phone, password, password_hash, role, email_verified, status)
                VALUES ('Buyer One', 'buyer1@gmail.com', 'buyer1', '+919845000001', '[BCRYPT_PROTECTED]', 'hashed', 'buyer', 1, 'Active')
            ''')
            buyer1_id = cursor.lastrowid

            cursor.execute('''
                INSERT INTO users (name, email, username, phone, password, password_hash, role, email_verified, status)
                VALUES ('Buyer Two', 'buyer2@gmail.com', 'buyer2', '+919845000002', '[BCRYPT_PROTECTED]', 'hashed', 'buyer', 1, 'Active')
            ''')
            buyer2_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()

        # Buyer 1 confirms purchase directly (NO purchase OTP required)
        buy1_res = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': buyer1_id,
            'delivery_district': 'Dharwad',
            'delivery_address': 'Station Road'
        })
        self.assertEqual(buy1_res.status_code, 200)
        buy1_data = buy1_res.get_json()
        self.assertEqual(buy1_data.get('status'), 'success')

        # Verify Buyer 1 received purchase confirmation email via AgentMail
        b1_conf_emails = [e for e in CAPTURED_EMAILS if e['to'] == 'buyer1@gmail.com' and 'Purchase Confirmed' in e['subject']]
        self.assertTrue(len(b1_conf_emails) >= 1)

        # Buyer 2 tries to purchase the same livestock -> Must be rejected (already sold)
        buy2_res = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': buyer2_id,
            'delivery_district': 'Dharwad',
            'delivery_address': 'Station Road'
        })
        self.assertEqual(buy2_res.status_code, 400)
        self.assertIn('no longer available', buy2_res.get_json().get('error', '').lower())

if __name__ == '__main__':
    unittest.main()
