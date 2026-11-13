import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

def test_flow(role, name, email, phone):
    print(f"\n--- Testing {role.capitalize()} Registration OTP Request ---")
    url = "http://127.0.0.1:5000/api/auth/register-request-otp"
    payload = {
        "role": role,
        "name": name,
        "email": email,
        "phone": phone,
        "password": "Password@123",
        "confirm_password": "Password@123",
        "district": "Bengaluru Urban",
        "taluk": "Bengaluru North",
        "village": "Yelahanka",
        "address": "Main Street",
        "pincode": "560064"
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        res = urllib.request.urlopen(req)
        print("HTTP Status:", res.code)
        print("Response JSON:", res.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print("HTTP Status:", e.code)
        print("Response JSON:", body)
        data = json.loads(body)
        print(f"User-facing error in toast: \"{data.get('error')}\"")
        assert "authentication failed" not in data.get('error', '').lower(), "FAILED: still says authentication failed!"
        assert "spam" in data.get('error', '').lower() or "rejected" in data.get('error', '').lower(), "FAILED: spam not mentioned!"
        print("=> Error classification verified: Properly identified as spam rejection, NOT authentication failure.")

if __name__ == "__main__":
    test_flow("farmer", "Farmer Basavaraj", "farmer.basavaraj.test@gmail.com", "+919876500001")
    test_flow("buyer", "Buyer Chandrashekhar", "buyer.chandrashekhar.test@gmail.com", "+919876500002")
