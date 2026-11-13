import unittest
import uuid
import app as app_module
from app import app
import notification_service
from database import get_db, init_db, seed_single_admin, reset_user_and_transaction_data

from services import email_service

CAPTURED_OTPS = {}

orig_send_email = email_service.send_email
def intercepted_send_email(to_email, subject, html_body, plain_body=None):
    cfg = email_service.get_agentmail_config()
    if cfg['api_key'] and 'your_agentmail' not in cfg['api_key'].lower() and 'placeholder' not in cfg['api_key'].lower():
        return orig_send_email(to_email, subject, html_body, plain_body)
    return True, "Email dispatched successfully."

email_service.send_email = intercepted_send_email
notification_service.send_email = intercepted_send_email

CAPTURED_PURCHASE_OTPS = {}

orig_send_purchase_otp = notification_service.send_buyer_purchase_otp_email
def intercepted_send_purchase_otp(buyer_email, otp_code, livestock):
    CAPTURED_PURCHASE_OTPS[buyer_email.strip().lower()] = otp_code
    return orig_send_purchase_otp(buyer_email, otp_code, livestock)
notification_service.send_buyer_purchase_otp_email = intercepted_send_purchase_otp
app_module.send_buyer_purchase_otp_email = intercepted_send_purchase_otp

orig_send_user_otp = notification_service.send_user_registration_otp_email
def intercepted_send_user_otp(user_email, user_name, otp_code, role):
    CAPTURED_OTPS[user_email.strip().lower()] = otp_code
    return orig_send_user_otp(user_email, user_name, otp_code, role)
notification_service.send_user_registration_otp_email = intercepted_send_user_otp
app_module.send_user_registration_otp_email = intercepted_send_user_otp


class TestFreshMarketplaceFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        seed_single_admin()
        reset_user_and_transaction_data()
        cls.client = app.test_client()

    def test_01_farmer_registration_and_dashboard(self):
        """TEST 1: Create a farmer via OTP -> login -> verify farmer dashboard works."""
        email = 'shankar.gowda@gmail.com'
        phone = '+919845012345'
        
        # 1. Request OTP
        res_req = self.client.post('/api/auth/register-request-otp', json={
            'role': 'farmer',
            'name': 'Shankar Gowda',
            'email': email,
            'phone': phone,
            'password': 'farmer_secret_123',
            'confirm_password': 'farmer_secret_123',
            'district': 'Mysuru',
            'taluk': 'Nanjangud',
            'village': 'Debur',
            'address': 'Farmhouse 12',
            'pincode': '571301'
        })
        self.assertEqual(res_req.status_code, 200)

        # 2. Verify OTP & Activate Account
        otp = CAPTURED_OTPS[email]
        res_verify = self.client.post('/api/auth/register-verify-otp', json={
            'email': email,
            'otp': otp,
            'role': 'farmer',
            'name': 'Shankar Gowda',
            'phone': phone,
            'password': 'farmer_secret_123',
            'district': 'Mysuru',
            'taluk': 'Nanjangud',
            'village': 'Debur',
            'address': 'Farmhouse 12',
            'pincode': '571301'
        })
        self.assertEqual(res_verify.status_code, 200)
        farmer_data = res_verify.get_json()['user']
        farmer_id = farmer_data['id']
        self.assertEqual(farmer_data['role'], 'farmer')

        # 3. Login as farmer
        res_login = self.client.post('/api/auth/login', json={
            'identifier': email,
            'password': 'farmer_secret_123',
            'role': 'farmer'
        })
        self.assertEqual(res_login.status_code, 200)
        self.assertEqual(res_login.get_json()['user']['name'], 'Shankar Gowda')

        # 4. Verify farmer dashboard APIs return empty active listings and history
        res_listings = self.client.get(f'/api/farmer/{farmer_id}/listings')
        self.assertEqual(res_listings.status_code, 200)
        self.assertEqual(len(res_listings.get_json()), 0)

        res_history = self.client.get(f'/api/farmer/{farmer_id}/history')
        self.assertEqual(res_history.status_code, 200)
        self.assertEqual(len(res_history.get_json()), 0)

        print("\n[TEST 1 PASSED] Create a farmer via OTP -> login -> verify farmer dashboard works.")

    def test_02_buyer_registration_and_dashboard(self):
        """TEST 2: Create a buyer via OTP -> login -> verify buyer dashboard works."""
        email = 'ramesh.patel@gmail.com'
        phone = '+919108112233'

        # 1. Request OTP
        res_req = self.client.post('/api/auth/register-request-otp', json={
            'role': 'buyer',
            'name': 'Ramesh Patel',
            'email': email,
            'phone': phone,
            'password': 'buyer_secret_123',
            'confirm_password': 'buyer_secret_123',
            'district': 'Mysuru',
            'taluk': 'Mysuru',
            'village': 'Hebbal',
            'address': 'Plot 45',
            'pincode': '570016'
        })
        self.assertEqual(res_req.status_code, 200)

        # 2. Verify OTP & Activate
        otp = CAPTURED_OTPS[email]
        res_verify = self.client.post('/api/auth/register-verify-otp', json={
            'email': email,
            'otp': otp,
            'role': 'buyer',
            'name': 'Ramesh Patel',
            'phone': phone,
            'password': 'buyer_secret_123',
            'district': 'Mysuru',
            'taluk': 'Mysuru',
            'village': 'Hebbal',
            'address': 'Plot 45',
            'pincode': '570016'
        })
        self.assertEqual(res_verify.status_code, 200)
        buyer_data = res_verify.get_json()['user']
        buyer_id = buyer_data['id']
        self.assertEqual(buyer_data['role'], 'buyer')

        # 3. Login as buyer
        res_login = self.client.post('/api/auth/login', json={
            'identifier': email,
            'password': 'buyer_secret_123',
            'role': 'buyer'
        })
        self.assertEqual(res_login.status_code, 200)
        self.assertEqual(res_login.get_json()['user']['name'], 'Ramesh Patel')

        # 4. Verify buyer dashboard purchases is empty
        res_purch = self.client.get(f'/api/buyer/{buyer_id}/purchases')
        self.assertEqual(res_purch.status_code, 200)
        self.assertEqual(len(res_purch.get_json()), 0)

        print("[TEST 2 PASSED] Create a buyer via OTP -> login -> verify buyer dashboard works.")

    def test_03_farmer_creates_livestock_buyer_views_and_buys(self):
        """TEST 3: Farmer creates livestock -> buyer views it -> purchase flow works with ZERO delivery fee."""
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT id, name, phone, email FROM users WHERE email = 'shankar.gowda@gmail.com'")
        farmer = c.fetchone()
        c.execute("SELECT id, name, phone, email FROM users WHERE email = 'ramesh.patel@gmail.com'")
        buyer = c.fetchone()
        conn.close()

        farmer_id = farmer['id']
        buyer_id = buyer['id']

        # Farmer publishes livestock
        pub_res = self.client.post('/api/listings', json={
            'farmer_id': farmer_id,
            'farmer_name': farmer['name'],
            'farmer_phone': farmer['phone'],
            'image_url': f'/static/uploads/cow_sample_{uuid.uuid4().hex[:6]}.jpg',
            'species': 'Cow',
            'breed': 'Holstein Friesian cattle',
            'gender': 'Female',
            'age_years': 4.0,
            'milk_production': 20.0,
            'district': 'Mysuru',
            'taluk': 'Nanjangud',
            'village': 'Debur',
            'address': 'Farmhouse 12',
            'pincode': '571301',
            'predicted_price': 85000,
            'asking_price': 88000
        })
        self.assertEqual(pub_res.status_code, 200)
        listing_id = pub_res.get_json()['listing_id']

        # Buyer browses active listings
        res_browse = self.client.get('/api/listings?status=Available')
        self.assertEqual(res_browse.status_code, 200)
        listings = res_browse.get_json()
        self.assertEqual(len(listings), 1)
        self.assertEqual(listings[0]['id'], listing_id)

        # Buyer requests Purchase OTP
        req_res = self.client.post(f'/api/listings/{listing_id}/purchase-request-otp', json={'buyer_id': buyer_id})
        self.assertEqual(req_res.status_code, 200)
        purchase_otp = CAPTURED_PURCHASE_OTPS.get(buyer['email'].strip().lower())

        # Buyer purchases listing under Cash on Delivery with ZERO delivery fee
        purch_res = self.client.post(f'/api/listings/{listing_id}/purchase', json={
            'buyer_id': buyer_id,
            'buyer_name': buyer['name'],
            'buyer_phone': buyer['phone'],
            'delivery_district': 'Mysuru',
            'delivery_taluk': 'Mysuru',
            'delivery_village': 'Hebbal',
            'delivery_address': 'Plot 45',
            'delivery_pincode': '570016',
            'otp': purchase_otp
        })
        self.assertEqual(purch_res.status_code, 200)
        p_data = purch_res.get_json()
        self.assertEqual(p_data['status'], 'success')
        self.assertEqual(p_data['summary']['order_status'], 'CONFIRMED')
        self.assertEqual(p_data['summary']['delivery_charge'], 0) # Strictly ZERO delivery charge
        self.assertEqual(p_data['summary']['total_amount'], 88000)

        # Verify listing is no longer Available in marketplace
        res_after = self.client.get('/api/listings?status=Available')
        self.assertEqual(len(res_after.get_json()), 0)

        # Verify buyer purchase history has recorded the transaction
        res_buyer_history = self.client.get(f'/api/buyer/{buyer_id}/purchases')
        self.assertEqual(res_buyer_history.status_code, 200)
        self.assertEqual(len(res_buyer_history.get_json()), 1)
        self.assertEqual(res_buyer_history.get_json()[0]['delivery_charge'], 0)
        self.assertEqual(res_buyer_history.get_json()[0]['total_amount'], 88000)

        print("[TEST 3 PASSED] Farmer creates livestock -> buyer views it -> purchase flow works with ZERO delivery fee.")

    def test_04_zero_delivery_charges_across_karnataka(self):
        """TEST 4: Zero delivery charges policy: COD total is strictly livestock asking price."""
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM purchases LIMIT 1")
        purch = c.fetchone()
        conn.close()

        self.assertEqual(purch['delivery_charge'], 0)
        self.assertEqual(purch['total_amount'], purch['livestock_price'])
        print("[TEST 4 PASSED] Zero delivery charges policy: Total amount strictly equals asking price.")

    def test_05_restart_persistence_no_demo_users(self):
        """TEST 5: Restart backend -> only genuine users and single administrator exist."""
        init_db()
        seed_single_admin()

        conn = get_db()
        c = conn.cursor()
        # Single administrator Rakshitha M
        c.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        self.assertEqual(c.fetchone()[0], 1)
        c.execute("SELECT name, email FROM users WHERE role = 'admin'")
        admin = c.fetchone()
        self.assertEqual(admin['name'], 'Rakshitha M')
        self.assertEqual(admin['email'], 'rakshitham556@gmail.com')

        # Real users registered during test
        c.execute("SELECT COUNT(*) FROM users WHERE role = 'farmer'")
        self.assertEqual(c.fetchone()[0], 1)
        c.execute("SELECT COUNT(*) FROM users WHERE role = 'buyer'")
        self.assertEqual(c.fetchone()[0], 1)
        c.execute("SELECT COUNT(*) FROM listings")
        self.assertEqual(c.fetchone()[0], 1)
        c.execute("SELECT COUNT(*) FROM purchases")
        self.assertEqual(c.fetchone()[0], 1)
        conn.close()

        print("[TEST 5 PASSED] Restart backend -> single admin 'Rakshitha M' and real users preserved without demo data.")


if __name__ == '__main__':
    unittest.main()
