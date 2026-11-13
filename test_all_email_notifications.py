import unittest
from unittest.mock import patch, MagicMock
import json
import sqlite3
import app
from services.email_service import (
    send_buyer_registration_success_email,
    send_order_confirmed_buyer_email,
    send_order_confirmed_farmer_email,
    send_purchase_completed_buyer_email,
    send_purchase_completed_farmer_email,
    send_order_rejected_buyer_email,
    send_order_rejected_farmer_email,
    send_order_cancelled_buyer_email,
    send_order_cancelled_farmer_email
)

class TestEmailNotifications(unittest.TestCase):
    def setUp(self):
        self.app = app.app.test_client()
        self.app.testing = True

        self.sample_livestock = {
            'id': 101,
            'listing_id': 101,
            'species': 'Cattle',
            'breed': 'Gir',
            'gender': 'Female',
            'age_years': 4,
            'weight': '420 kg',
            'milk_production': 14.5,
            'district': 'Mysuru',
            'taluk': 'Nanjangud',
            'village': 'Hullahalli',
            'asking_price': 55000,
            'farmer_name': 'Ramesh Kumar',
            'farmer_phone': '+919876543210'
        }

        self.sample_order = {
            'order_id': 202,
            'id': 202,
            'listing_id': 101,
            'total_amount': 55000,
            'livestock_price': 55000,
            'delivery_address': 'Plot 45, Main Road',
            'delivery_district': 'Bengaluru Rural',
            'delivery_taluk': 'Devanahalli',
            'delivery_village': 'Bidaluru',
            'delivery_pincode': '562110',
            'buyer_name': 'Ananya Rao',
            'buyer_phone': '+919876543211',
            'farmer_name': 'Ramesh Kumar',
            'farmer_phone': '+919876543210'
        }

    # 1. Buyer Registration Success Email
    @patch('services.email_service.send_email')
    def test_buyer_registration_success_email(self, mock_send):
        mock_send.return_value = (True, "Dispatched")
        ok, msg = send_buyer_registration_success_email("ananya@example.com", "Ananya Rao")
        self.assertTrue(ok)
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        to_email, subject, html_body, plain_body = args[0], args[1], args[2], args[3]
        
        self.assertEqual(to_email, "ananya@example.com")
        self.assertEqual(subject, "LivestockIQ - Successfully Registered as Buyer")
        self.assertIn("Congratulations! You have successfully registered as a buyer on LivestockIQ.", plain_body)
        self.assertIn("Dear Ananya Rao", plain_body)
        self.assertIn("LivestockIQ", plain_body)
        self.assertNotIn("PashuDhan", plain_body)
        self.assertNotIn("Pashudhan", plain_body)
        print("[PASS] Scenario 1: Buyer Registration Success Email Passed")

    # 2A. Order Confirmed - Buyer Email
    @patch('services.email_service.send_email')
    def test_order_confirmed_buyer_email(self, mock_send):
        mock_send.return_value = (True, "Dispatched")
        ok, msg = send_order_confirmed_buyer_email("ananya@example.com", self.sample_livestock, "Ananya Rao", self.sample_order)
        self.assertTrue(ok)
        args = mock_send.call_args[0]
        to_email, subject, html_body, plain_body = args[0], args[1], args[2], args[3]

        self.assertEqual(to_email, "ananya@example.com")
        self.assertEqual(subject, "LivestockIQ - Order Successfully Confirmed")
        self.assertIn("You successfully confirmed the order for the following livestock on LivestockIQ.", plain_body)
        self.assertIn("Livestock ID: #101", plain_body)
        self.assertIn("Species: Cattle", plain_body)
        self.assertIn("Breed: Gir", plain_body)
        self.assertIn("Gender: Female", plain_body)
        self.assertIn("Age: 4 years", plain_body)
        self.assertIn("Price: ₹55,000", plain_body)
        self.assertIn("Order ID: #202", plain_body)
        self.assertNotIn("PashuDhan", plain_body)
        print("[PASS] Scenario 2A: Order Confirmed Buyer Email Passed")

    # 2B. Order Confirmed - Farmer Email
    @patch('services.email_service.send_email')
    def test_order_confirmed_farmer_email(self, mock_send):
        mock_send.return_value = (True, "Dispatched")
        ok, msg = send_order_confirmed_farmer_email("ramesh@example.com", self.sample_livestock, "Ramesh Kumar", "Ananya Rao", self.sample_order)
        self.assertTrue(ok)
        args = mock_send.call_args[0]
        to_email, subject, html_body, plain_body = args[0], args[1], args[2], args[3]

        self.assertEqual(to_email, "ramesh@example.com")
        self.assertEqual(subject, "LivestockIQ - Your Livestock Has Been Confirmed")
        self.assertIn("Your livestock has been confirmed by Ananya Rao on LivestockIQ.", plain_body)
        self.assertIn("Livestock ID: #101", plain_body)
        self.assertIn("Buyer: Ananya Rao", plain_body)
        self.assertNotIn("PashuDhan", plain_body)
        print("[PASS] Scenario 2B: Order Confirmed Farmer Email Passed")

    # 3A. Purchase Completed - Buyer Email
    @patch('services.email_service.send_email')
    def test_purchase_completed_buyer_email(self, mock_send):
        mock_send.return_value = (True, "Dispatched")
        ok, msg = send_purchase_completed_buyer_email("ananya@example.com", self.sample_livestock, "Ananya Rao", self.sample_order)
        self.assertTrue(ok)
        args = mock_send.call_args[0]
        to_email, subject, html_body, plain_body = args[0], args[1], args[2], args[3]

        self.assertEqual(to_email, "ananya@example.com")
        self.assertEqual(subject, "LivestockIQ - Livestock Successfully Purchased")
        self.assertIn("You successfully purchased the following livestock on LivestockIQ.", plain_body)
        self.assertIn("Livestock ID: #101", plain_body)
        self.assertIn("Farmer: Ramesh Kumar", plain_body)
        self.assertIn("Your purchase is complete.", plain_body)
        self.assertNotIn("PashuDhan", plain_body)
        print("[PASS] Scenario 3A: Purchase Completed Buyer Email Passed")

    # 3B. Purchase Completed - Farmer Email
    @patch('services.email_service.send_email')
    def test_purchase_completed_farmer_email(self, mock_send):
        mock_send.return_value = (True, "Dispatched")
        ok, msg = send_purchase_completed_farmer_email("ramesh@example.com", self.sample_livestock, "Ramesh Kumar", "Ananya Rao", self.sample_order)
        self.assertTrue(ok)
        args = mock_send.call_args[0]
        to_email, subject, html_body, plain_body = args[0], args[1], args[2], args[3]

        self.assertEqual(to_email, "ramesh@example.com")
        self.assertEqual(subject, "LivestockIQ - Your Livestock Has Been Successfully Purchased")
        self.assertIn("Your livestock is successfully purchased by Ananya Rao on LivestockIQ.", plain_body)
        self.assertIn("The purchase for your livestock has been completed successfully.", plain_body)
        self.assertNotIn("PashuDhan", plain_body)
        print("[PASS] Scenario 3B: Purchase Completed Farmer Email Passed")

    # 4A. Delivery Rejected - Buyer Email
    @patch('services.email_service.send_email')
    def test_delivery_rejected_buyer_email(self, mock_send):
        mock_send.return_value = (True, "Dispatched")
        reason = "Physical health does not match health certificate."
        ok, msg = send_order_rejected_buyer_email("ananya@example.com", self.sample_livestock, "Ananya Rao", self.sample_order, reason)
        self.assertTrue(ok)
        args = mock_send.call_args[0]
        to_email, subject, html_body, plain_body = args[0], args[1], args[2], args[3]

        self.assertEqual(to_email, "ananya@example.com")
        self.assertEqual(subject, "LivestockIQ - Delivery Rejected")
        self.assertIn("You rejected the delivery of the following livestock on LivestockIQ.", plain_body)
        self.assertIn("Reason for Rejection:\nPhysical health does not match health certificate.", plain_body)
        self.assertIn("Your rejection has been recorded.", plain_body)
        self.assertNotIn("PashuDhan", plain_body)
        print("[PASS] Scenario 4A: Delivery Rejected Buyer Email Passed")

    # 4B. Delivery Rejected - Farmer Email
    @patch('services.email_service.send_email')
    def test_delivery_rejected_farmer_email(self, mock_send):
        mock_send.return_value = (True, "Dispatched")
        reason = "Physical health does not match health certificate."
        ok, msg = send_order_rejected_farmer_email("ramesh@example.com", self.sample_livestock, "Ramesh Kumar", "Ananya Rao", self.sample_order, reason)
        self.assertTrue(ok)
        args = mock_send.call_args[0]
        to_email, subject, html_body, plain_body = args[0], args[1], args[2], args[3]

        self.assertEqual(to_email, "ramesh@example.com")
        self.assertEqual(subject, "LivestockIQ - Your Livestock Delivery Has Been Rejected")
        self.assertIn("Your livestock delivery is rejected by Ananya Rao on LivestockIQ.", plain_body)
        self.assertIn("Reason for Rejection:\nPhysical health does not match health certificate.", plain_body)
        self.assertIn("Please review your order details on LivestockIQ.", plain_body)
        self.assertNotIn("PashuDhan", plain_body)
        print("[PASS] Scenario 4B: Delivery Rejected Farmer Email Passed")

    # 5A. Order Cancelled - Buyer Email
    @patch('services.email_service.send_email')
    def test_order_cancelled_buyer_email(self, mock_send):
        mock_send.return_value = (True, "Dispatched")
        reason = "Found an alternative local seller."
        ok, msg = send_order_cancelled_buyer_email("ananya@example.com", self.sample_livestock, "Ananya Rao", self.sample_order, reason)
        self.assertTrue(ok)
        args = mock_send.call_args[0]
        to_email, subject, html_body, plain_body = args[0], args[1], args[2], args[3]

        self.assertEqual(to_email, "ananya@example.com")
        self.assertEqual(subject, "LivestockIQ - Your Order Has Been Cancelled")
        self.assertIn("You cancelled your order for the following livestock on LivestockIQ.", plain_body)
        self.assertIn("Reason for Cancellation:\nFound an alternative local seller.", plain_body)
        self.assertIn("Your cancellation has been processed.", plain_body)
        self.assertNotIn("PashuDhan", plain_body)
        print("[PASS] Scenario 5A: Order Cancelled Buyer Email Passed")

    # 5B. Order Cancelled - Farmer Email
    @patch('services.email_service.send_email')
    def test_order_cancelled_farmer_email(self, mock_send):
        mock_send.return_value = (True, "Dispatched")
        reason = "Found an alternative local seller."
        ok, msg = send_order_cancelled_farmer_email("ramesh@example.com", self.sample_livestock, "Ramesh Kumar", "Ananya Rao", self.sample_order, reason)
        self.assertTrue(ok)
        args = mock_send.call_args[0]
        to_email, subject, html_body, plain_body = args[0], args[1], args[2], args[3]

        self.assertEqual(to_email, "ramesh@example.com")
        self.assertEqual(subject, "LivestockIQ - Livestock Order Has Been Cancelled")
        self.assertIn("Your livestock has been cancelled by Ananya Rao on LivestockIQ.", plain_body)
        self.assertIn("Reason for Cancellation:\nFound an alternative local seller.", plain_body)
        self.assertIn("The livestock listing has been restored to the marketplace.", plain_body)
        self.assertNotIn("PashuDhan", plain_body)
        print("[PASS] Scenario 5B: Order Cancelled Farmer Email Passed")

if __name__ == '__main__':
    unittest.main()
