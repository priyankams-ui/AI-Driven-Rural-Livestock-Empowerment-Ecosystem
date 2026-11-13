import os
import sys
import unittest
import uuid
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

load_dotenv()

from app import app
from notification_service import get_db, hash_otp
import services.email_service as email_service
import notification_service

class TestRegistrationOTPFlow(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        self.orig_send_email = email_service.send_email
        self.orig_notif_send = notification_service.send_email

    def tearDown(self):
        email_service.send_email = self.orig_send_email
        notification_service.send_email = self.orig_notif_send
        self.app_context.pop()

    def test_exact_email_binding_three_recipients(self):
        """
        SECTION 10:
        Test with three different addresses (A, B, C).
        Confirm AgentMail recipient equals EXACT entered recipient email.
        """
        emails = [
            f"user_a_{uuid.uuid4().hex[:6]}@gmail.com",
            f"user_b_{uuid.uuid4().hex[:6]}@yahoo.com",
            f"user_c_{uuid.uuid4().hex[:6]}@outlook.com"
        ]

        for idx, email in enumerate(emails, 1):
            with patch("services.email_service.send_email") as mock_send:
                mock_send.return_value = (True, "Email dispatched successfully.")
                notification_service.send_email = mock_send

                payload = {
                    "role": "farmer",
                    "name": f"Test Farmer {idx}",
                    "phone": f"+9198{idx:02d}123456",
                    "email": email,
                    "password": "Password@123",
                    "confirm_password": "Password@123",
                    "district": "Mandya",
                    "taluk": "Maddur",
                    "village": "KM Doddi",
                    "address": "Farm House",
                    "pincode": "571428"
                }

                res = self.client.post("/api/auth/register-request-otp", json=payload)
                self.assertEqual(res.status_code, 200, f"Failed for {email}: {res.get_json()}")
                data = res.get_json()

                # Verify API response does NOT contain dev_otp or plaintext OTP
                self.assertNotIn("dev_otp", data)
                self.assertNotIn("otp", data)
                self.assertIn("Verification code sent to", data.get("message", ""))
                self.assertIn(email, data.get("message", ""))
                self.assertEqual(data.get("email"), email)

                # Confirm mock_send was called with EXACT recipient
                mock_send.assert_called_once()
                args, kwargs = mock_send.call_args
                recipient_sent = args[0]
                self.assertEqual(recipient_sent, email, f"Recipient mismatch! Expected {email}, got {recipient_sent}")

                # Confirm recipient is NOT sender inbox
                inbox_id = os.getenv("AGENTMAIL_INBOX_ID")
                if inbox_id:
                    self.assertNotEqual(recipient_sent, inbox_id)

    def test_edit_email_and_cross_email_otp_rejection(self):
        """
        SECTION 10 & SECTION 4:
        A -> request OTP
        B -> edit registration email -> request OTP again
        The second OTP must go ONLY to B.
        Verify that OTP for A cannot verify registration for B.
        """
        email_a = f"alice_{uuid.uuid4().hex[:6]}@example.com"
        email_b = f"bob_{uuid.uuid4().hex[:6]}@example.com"
        phone_a = f"+9191{uuid.uuid4().int % 100000000:08d}"
        phone_b = f"+9192{uuid.uuid4().int % 100000000:08d}"

        captured_otps = {}

        def fake_send(to, subject, html, text):
            import re
            m = re.search(r'\b(\d{6})\b', text or html or '')
            if m:
                captured_otps[to.lower()] = m.group(1)
            return True, "Email dispatched successfully."

        email_service.send_email = fake_send
        notification_service.send_email = fake_send

        # Step 1: Request OTP for Email A
        payload_a = {
            "role": "farmer",
            "name": "Alice Farmer",
            "phone": phone_a,
            "email": email_a,
            "password": "Password@123",
            "confirm_password": "Password@123",
            "district": "Mandya",
            "taluk": "Maddur",
            "village": "KM Doddi",
            "address": "Farm 1",
            "pincode": "571428"
        }
        res_a = self.client.post("/api/auth/register-request-otp", json=payload_a)
        self.assertEqual(res_a.status_code, 200, res_a.get_json())
        otp_a = captured_otps.get(email_a)
        self.assertIsNotNone(otp_a)
        self.assertEqual(len(otp_a), 6)

        # Step 2: User edits registration email to Email B and requests OTP again
        payload_b = dict(payload_a)
        payload_b["email"] = email_b
        payload_b["phone"] = phone_b
        payload_b["name"] = "Bob Farmer"

        res_b = self.client.post("/api/auth/register-request-otp", json=payload_b)
        self.assertEqual(res_b.status_code, 200, res_b.get_json())
        otp_b = captured_otps.get(email_b)
        self.assertIsNotNone(otp_b)
        self.assertEqual(len(otp_b), 6)

        # Confirm OTP for Email A cannot verify Email B
        verify_res_cross = self.client.post("/api/auth/register-verify-otp", json={
            "email": email_b,
            "otp": otp_a,
            "role": "farmer",
            "name": "Bob Farmer",
            "phone": phone_b,
            "password": "Password@123",
            "district": "Mandya",
            "taluk": "Maddur",
            "village": "KM Doddi",
            "address": "Farm 1",
            "pincode": "571428"
        })
        self.assertEqual(verify_res_cross.status_code, 400)
        self.assertIn("Invalid", verify_res_cross.get_json().get("error", ""))

        # Confirm OTP for Email B DOES verify Email B
        verify_res_b = self.client.post("/api/auth/register-verify-otp", json={
            "email": email_b,
            "otp": otp_b,
            "role": "farmer",
            "name": "Bob Farmer",
            "phone": phone_b,
            "password": "Password@123",
            "district": "Mandya",
            "taluk": "Maddur",
            "village": "KM Doddi",
            "address": "Farm 1",
            "pincode": "571428"
        })
        self.assertEqual(verify_res_b.status_code, 200, verify_res_b.get_json())
        self.assertEqual(verify_res_b.get_json().get("status"), "success")

    def test_farmer_and_buyer_registration_flows(self):
        """
        SECTION 8:
        Farmer registration AND Buyer registration both work and preserve their roles.
        """
        captured_otps = {}

        def fake_send(to, subject, html, text):
            import re
            m = re.search(r'\b(\d{6})\b', text or html or '')
            if m:
                captured_otps[to.lower()] = m.group(1)
            return True, "Email dispatched successfully."

        email_service.send_email = fake_send
        notification_service.send_email = fake_send

        # Test Farmer
        f_email = f"farmer_{uuid.uuid4().hex[:6]}@agri.in"
        f_phone = f"+9193{uuid.uuid4().int % 100000000:08d}"
        f_res = self.client.post("/api/auth/register-request-otp", json={
            "role": "farmer",
            "name": "Kishan Farmer",
            "phone": f_phone,
            "email": f_email,
            "password": "Password@123",
            "confirm_password": "Password@123",
            "district": "Mandya",
            "taluk": "Maddur",
            "village": "KM Doddi",
            "address": "Field 10",
            "pincode": "571428"
        })
        self.assertEqual(f_res.status_code, 200, f_res.get_json())
        f_otp = captured_otps.get(f_email)

        f_v_res = self.client.post("/api/auth/register-verify-otp", json={
            "email": f_email,
            "otp": f_otp,
            "role": "farmer",
            "name": "Kishan Farmer",
            "phone": f_phone,
            "password": "Password@123"
        })
        self.assertEqual(f_v_res.status_code, 200, f_v_res.get_json())
        self.assertEqual(f_v_res.get_json()["user"]["role"], "farmer")

        # Test Buyer
        b_email = f"buyer_{uuid.uuid4().hex[:6]}@dairy.org"
        b_phone = f"+9194{uuid.uuid4().int % 100000000:08d}"
        b_res = self.client.post("/api/auth/register-request-otp", json={
            "role": "buyer",
            "name": "Suresh Buyer",
            "phone": b_phone,
            "email": b_email,
            "password": "Password@123",
            "confirm_password": "Password@123",
            "district": "Mandya",
            "taluk": "Maddur",
            "village": "KM Doddi",
            "address": "Shop 5",
            "pincode": "571428"
        })
        self.assertEqual(b_res.status_code, 200, b_res.get_json())
        b_otp = captured_otps.get(b_email)

        b_v_res = self.client.post("/api/auth/register-verify-otp", json={
            "email": b_email,
            "otp": b_otp,
            "role": "buyer",
            "name": "Suresh Buyer",
            "phone": b_phone,
            "password": "Password@123"
        })
        self.assertEqual(b_v_res.status_code, 200, b_v_res.get_json())
        self.assertEqual(b_v_res.get_json()["user"]["role"], "buyer")

    def test_purchase_otp_uses_verified_buyer_email(self):
        """
        SECTION 9:
        Purchase OTP sends strictly to the verified buyer's email.
        """
        conn = get_db()
        cursor = conn.cursor()
        b_email = f"verified_buyer_{uuid.uuid4().hex[:6]}@domain.com"
        b_phone = f"+9195{uuid.uuid4().int % 100000000:08d}"
        cursor.execute("""
            INSERT INTO users (name, username, password, email, phone, password_hash, role, email_verified, status)
            VALUES (?, ?, 'secret', ?, ?, 'hash123', 'buyer', 1, 'active')
        """, ("Verified Buyer", b_email, b_email, b_phone))
        buyer_id = cursor.lastrowid

        cursor.execute("SELECT id FROM listings WHERE status = 'Available' AND is_deleted = 0 LIMIT 1")
        existing = cursor.fetchone()
        if existing:
            listing_id = existing['id']
        else:
            cursor.execute("""
                INSERT INTO listings (
                    farmer_id, farmer_name, farmer_phone, image_url, species, breed, sex,
                    health_score, health_text, age_years, milk_yield_liters, district,
                    taluk, village, address, predicted_price, asking_price, status, is_deleted
                ) VALUES (
                    1, 'Test Farmer', '+919876543210', 'https://example.com/cow.jpg', 'Cow', 'Hallikar', 'Female',
                    95, 'Healthy', 3, 10, 'Mandya',
                    'Maddur', 'KM Doddi', 'Farm 1', 50000, 50000, 'Available', 0
                )
            """)
            listing_id = cursor.lastrowid
        conn.commit()
        conn.close()

        mock_send = MagicMock(return_value=(True, "Email dispatched successfully."))
        email_service.send_email = mock_send
        notification_service.send_email = mock_send

        res = self.client.post(f"/api/listings/{listing_id}/purchase-request-otp", json={
            "buyer_id": buyer_id
        })
        self.assertEqual(res.status_code, 200, res.get_json())
        data = res.get_json()
        self.assertNotIn("dev_otp", data)
        self.assertEqual(data.get("email"), b_email)

        mock_send.assert_called_once()
        args, _ = mock_send.call_args
        recipient = args[0]
        self.assertEqual(recipient, b_email)

    def test_registration_rate_limiting(self):
        """
        Verify existing rate limiting (HTTP 429) remains enforced.
        """
        email = f"cooldown_{uuid.uuid4().hex[:6]}@example.com"
        phone = f"+9196{uuid.uuid4().int % 100000000:08d}"
        payload = {
            "role": "farmer",
            "name": "Cooldown Test",
            "phone": phone,
            "email": email,
            "password": "Password@123",
            "confirm_password": "Password@123",
            "district": "Mandya",
            "taluk": "Maddur",
            "village": "KM Doddi",
            "address": "Farm 1",
            "pincode": "571428"
        }

        mock_send = MagicMock(return_value=(True, "Email dispatched successfully."))
        email_service.send_email = mock_send
        notification_service.send_email = mock_send

        # 1st request -> 200
        res1 = self.client.post("/api/auth/register-request-otp", json=payload)
        self.assertEqual(res1.status_code, 200)

        # Immediate 2nd request -> 429 Rate limited
        res2 = self.client.post("/api/auth/register-request-otp", json=payload)
        self.assertEqual(res2.status_code, 429)
        self.assertIn("wait", res2.get_json().get("error", "").lower())

if __name__ == "__main__":
    unittest.main()
