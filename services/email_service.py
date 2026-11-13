import os
import re
from dotenv import load_dotenv
from agentmail import AgentMail
from agentmail.core.api_error import ApiError

# Load local environment variables from .env
load_dotenv()

def get_agentmail_config() -> dict:
    """
    Retrieve AgentMail configuration strictly from environment variables.
    Never hardcode or expose API keys.
    """
    api_key = (os.getenv("AGENTMAIL_API_KEY") or "").strip()
    inbox_id = (os.getenv("AGENTMAIL_INBOX_ID") or "").strip()
    return {
        'api_key': api_key,
        'inbox_id': inbox_id
    }

# Backward compatibility alias
get_resend_config = get_agentmail_config

def is_agentmail_configured() -> tuple[bool, str]:
    """
    Verify if AgentMail credentials are configured and not placeholders.
    Returns (True, "ok") or (False, reason).
    """
    cfg = get_agentmail_config()
    api_key = cfg['api_key']
    inbox_id = cfg['inbox_id']

    if not api_key or 'your_agentmail' in api_key.lower() or 'placeholder' in api_key.lower():
        return False, "missing_api_key"
    if not inbox_id or 'your_agentmail' in inbox_id.lower() or 'placeholder' in inbox_id.lower():
        return False, "missing_inbox_id"
    return True, "ok"

def send_email(to_email: str, subject: str, html_body: str, plain_body: str = None, text_body: str = None) -> tuple[bool, str]:
    """
    Send an email via official AgentMail Transactional Email API.
    Returns (True, success_msg) on success, or (False, user_friendly_msg) on failure.
    Never exposes API keys, passwords, or plain OTPs in logs.
    """
    text_content = text_body or plain_body or ""

    if not to_email or '@' not in to_email:
        return False, "Invalid recipient email address: address is required."

    clean_to = to_email.strip().lower()

    # Validate standard RFC email format
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', clean_to):
        return False, "Invalid recipient email format: please provide a valid email address."

    # Extract safe recipient domain for logging (NEVER log full sensitive email if private or secrets)
    recipient_domain = clean_to.split('@')[-1]

    # Verify AgentMail credentials
    is_ready, reason = is_agentmail_configured()
    cfg = get_agentmail_config()
    api_key = cfg['api_key']
    inbox_id = cfg['inbox_id']

    if not is_ready:
        if reason == "missing_api_key":
            print(f"[AgentMail Warning] AGENTMAIL_API_KEY is not configured in .env. Target Domain: @{recipient_domain} | Subject: {subject}")
            return False, "Email service configuration error: AGENTMAIL_API_KEY is missing in .env."
        elif reason == "missing_inbox_id":
            print(f"[AgentMail Warning] AGENTMAIL_INBOX_ID is not configured in .env. Target Domain: @{recipient_domain} | Subject: {subject}")
            return False, "Email service configuration error: AGENTMAIL_INBOX_ID is missing in .env."

    try:
        client = AgentMail(api_key=api_key)

        print(f"[AgentMail Dispatch] Dispatching email to domain: @{recipient_domain} via inbox {inbox_id} | Subject: {subject}")

        response = client.inboxes.messages.send(
            inbox_id=inbox_id,
            to=clean_to,
            subject=subject,
            html=html_body or "",
            text=text_content
        )

        message_id = None
        if hasattr(response, 'message_id'):
            message_id = response.message_id
        elif isinstance(response, dict):
            message_id = response.get('message_id')

        if message_id:
            print(f"[AgentMail Success] Dispatched message to domain: @{recipient_domain} (Message ID: {message_id}) | Subject: {subject}")
            return True, f"Email dispatched successfully. Message ID: {message_id}"
        else:
            print(f"[AgentMail Success] Dispatched message to domain: @{recipient_domain} | Subject: {subject}")
            return True, "Email dispatched successfully."

    except ApiError as e:
        status_code = getattr(e, 'status_code', None)
        err_body = getattr(e, 'body', None)
        err_msg = ""
        if hasattr(err_body, 'message') and err_body.message:
            err_msg = str(err_body.message)
        elif isinstance(err_body, dict) and 'message' in err_body:
            err_msg = str(err_body['message'])
        else:
            err_msg = str(err_body or e)

        print(f"[AgentMail Error] API Exception (HTTP {status_code}) for domain @{recipient_domain}: {err_msg}")

        err_lower = err_msg.lower()

        if 'spam' in err_lower or 'rejected' in err_lower or 'high-confidence spam' in err_lower:
            return False, "Verification email was rejected by the email service's spam protection. Please try again later."
        elif 'suppressed' in err_lower or 'bounced' in err_lower or 'blocked' in err_lower:
            return False, f"Recipient email address is blocked or suppressed by the email service: {err_msg}"
        elif 'permission' in err_lower or 'forbidden' in err_lower or 'not allowed' in err_lower or 'unauthorized inbox' in err_lower:
            return False, "Email service does not have permission to send messages."
        elif status_code == 401 or 'unauthorized' in err_lower or 'invalid api key' in err_lower or 'authentication' in err_lower:
            return False, "Email service authentication failed."
        elif status_code == 429 or 'rate limit' in err_lower or 'recipient limit' in err_lower:
            return False, "Email service recipient limit has temporarily been reached. Please try again later."
        elif status_code == 404:
            return False, f"AgentMail inbox not found (HTTP 404). Please verify AGENTMAIL_INBOX_ID: {inbox_id}."
        elif status_code == 422:
            return False, f"Email service rejected message (HTTP 422): {err_msg}."
        elif status_code and status_code >= 500:
            return False, f"Email service server error (HTTP {status_code}). Please try again shortly."
        elif status_code == 403:
            return False, f"Email service request forbidden (HTTP 403): {err_msg}."
        else:
            return False, f"Email service API error ({status_code or 'unknown'}): {err_msg}."

    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e)
        print(f"[AgentMail Error] Network/Client exception ({err_type}) for domain @{recipient_domain}: {err_msg}")
        return False, f"Email delivery failed ({err_type}): {err_msg}."


# =========================================================
# HELPER: EXTRACT STANDARDIZED LIVESTOCK & ORDER DETAILS
# =========================================================
def _extract_livestock_info(livestock: dict = None, order_details: dict = None) -> dict:
    livestock = livestock or {}
    order_details = order_details or {}

    listing_id = livestock.get('id') or livestock.get('listing_id') or order_details.get('listing_id') or 'N/A'
    species = livestock.get('species') or order_details.get('species') or 'N/A'
    breed = livestock.get('breed') or order_details.get('breed') or 'N/A'
    gender = livestock.get('gender') or livestock.get('sex') or order_details.get('gender') or 'N/A'
    
    age_val = livestock.get('age_years') or livestock.get('age') or order_details.get('age_years')
    age = f"{age_val} years" if age_val is not None and str(age_val).strip() not in ['', 'N/A'] else 'N/A'
    
    weight_val = livestock.get('weight') or livestock.get('weight_kg') or order_details.get('weight')
    weight = f"{weight_val} kg" if weight_val is not None and str(weight_val).strip() not in ['', 'N/A'] else (str(weight_val) if weight_val is not None else 'N/A')

    milk_val = livestock.get('milk_production') or livestock.get('milk_yield_liters') or order_details.get('milk_production')
    milk = f"{milk_val} L/day" if milk_val is not None and str(milk_val).strip() not in ['', 'N/A'] else 'N/A'

    village = livestock.get('village') or order_details.get('delivery_village') or ''
    taluk = livestock.get('taluk') or order_details.get('delivery_taluk') or ''
    district = livestock.get('district') or order_details.get('delivery_district') or ''
    location = ", ".join(filter(bool, [village, taluk, district])) or livestock.get('address') or order_details.get('delivery_address') or 'N/A'

    price = order_details.get('total_amount') or order_details.get('livestock_price') or livestock.get('asking_price') or livestock.get('predicted_price') or livestock.get('price', 0)
    try:
        price_formatted = f"{int(price):,}"
    except Exception:
        price_formatted = str(price)

    order_id = order_details.get('order_id') or order_details.get('id') or 'N/A'
    farmer_name = livestock.get('farmer_name') or order_details.get('farmer_name') or 'Farmer'
    farmer_phone = livestock.get('farmer_phone') or order_details.get('farmer_phone') or ''
    buyer_name = order_details.get('buyer_name') or 'Buyer'
    buyer_phone = order_details.get('buyer_phone') or ''
    order_date = order_details.get('confirmed_at') or order_details.get('purchased_at') or order_details.get('created_at') or 'N/A'

    return {
        'listing_id': listing_id,
        'species': species,
        'breed': breed,
        'gender': gender,
        'age': age,
        'weight': weight,
        'milk': milk,
        'location': location,
        'price': price,
        'price_formatted': price_formatted,
        'order_id': order_id,
        'farmer_name': farmer_name,
        'farmer_phone': farmer_phone,
        'buyer_name': buyer_name,
        'buyer_phone': buyer_phone,
        'order_date': order_date
    }


# =========================================================
# 1. AUTHENTICATION & OTP EMAILS
# =========================================================
def send_verification_otp(email: str, otp: str, user_name: str = "User", role: str = "user") -> tuple[bool, str]:
    """
    Send Registration Email Verification OTP using AgentMail.
    Identical AgentMail email service for BOTH Farmer and Buyer.
    Subject: Verify your LivestockIQ account
    """
    subject = "Verify your LivestockIQ account"
    plain_body = f"""Hello,

Your LivestockIQ verification code is: {otp}

This code is valid for 5 minutes and can be used only once.

If you did not request this verification code, you can ignore this email.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Verify your LivestockIQ account</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <p>Hello,</p>
    <p>Your LivestockIQ verification code is: <strong>{otp}</strong></p>
    <p>This code is valid for 5 minutes and can be used only once.</p>
    <p>If you did not request this verification code, you can ignore this email.</p>
    <p>Regards,<br>LivestockIQ Team</p>
</body>
</html>"""
    return send_email(email, subject, html_body, plain_body)

def send_password_reset_otp(email: str, otp: str, user_name: str = "User", role: str = "user") -> tuple[bool, str]:
    """
    Send Password Reset OTP using AgentMail.
    Applicable for Admin, Farmer, and Buyer roles.
    Subject: Verify your LivestockIQ password reset
    """
    subject = "Verify your LivestockIQ password reset"
    plain_body = f"""Hello,

Your LivestockIQ verification code for password reset is: {otp}

This code is valid for 5 minutes and can be used only once.

If you did not request a password reset, please ignore this email.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Verify your LivestockIQ password reset</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <p>Hello,</p>
    <p>Your LivestockIQ verification code for password reset is: <strong>{otp}</strong></p>
    <p>This code is valid for 5 minutes and can be used only once.</p>
    <p>If you did not request a password reset, please ignore this email.</p>
    <p>Regards,<br>LivestockIQ Team</p>
</body>
</html>"""
    return send_email(email, subject, html_body, plain_body)

def send_purchase_otp(email: str, otp: str, livestock: dict) -> tuple[bool, str]:
    """
    Send Buyer Purchase Confirmation OTP using AgentMail.
    Subject: Verify your LivestockIQ livestock purchase
    """
    breed = livestock.get('breed', 'Livestock')
    price = livestock.get('asking_price', 0)
    try:
        price_formatted = f"{int(price):,}"
    except Exception:
        price_formatted = str(price)

    subject = "Verify your LivestockIQ livestock purchase"
    plain_body = f"""Hello,

Your LivestockIQ verification code to confirm the livestock purchase ({breed}, Rs. {price_formatted}) is: {otp}

This code is valid for 5 minutes and can be used only once.

If you did not request this purchase, please ignore this email.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Verify your LivestockIQ livestock purchase</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <p>Hello,</p>
    <p>Your LivestockIQ verification code to confirm the livestock purchase (<strong>{breed}</strong>, Rs. {price_formatted}) is: <strong>{otp}</strong></p>
    <p>This code is valid for 5 minutes and can be used only once.</p>
    <p>If you did not request this purchase, please ignore this email.</p>
    <p>Regards,<br>LivestockIQ Team</p>
</body>
</html>"""
    return send_email(email, subject, html_body, plain_body)


# =========================================================
# 2. BUYER SUCCESSFUL REGISTRATION EMAIL
# =========================================================
def send_buyer_registration_success_email(buyer_email: str, buyer_name: str = "Buyer") -> tuple[bool, str]:
    """
    Immediately sent after buyer completes registration & verification.
    Subject: LivestockIQ - Successfully Registered as Buyer
    """
    greeting_name = buyer_name.strip() if buyer_name and buyer_name.strip() != "" else "Buyer"
    subject = "LivestockIQ - Successfully Registered as Buyer"

    plain_body = f"""Dear {greeting_name},

Congratulations! You have successfully registered as a buyer on LivestockIQ.

You can now browse available livestock, set availability alerts, and place orders directly on the platform.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LivestockIQ - Successfully Registered as Buyer</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <div style="max-width: 580px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden;">
        <div style="background: linear-gradient(135deg, #15803d, #166534); padding: 24px 28px; text-align: center;">
            <h2 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700;">LivestockIQ</h2>
            <p style="color: #dcfce7; margin: 6px 0 0 0; font-size: 14px;">Buyer Registration Successful</p>
        </div>
        <div style="padding: 28px 28px 24px 28px; color: #1e293b;">
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">Dear {greeting_name},</p>
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">
                <strong>Congratulations! You have successfully registered as a buyer on LivestockIQ.</strong>
            </p>
            <p style="font-size: 14px; line-height: 1.6; color: #475569;">
                You can now explore the livestock marketplace, set custom availability alerts, and securely purchase verified livestock.
            </p>
            <p style="font-size: 14px; color: #475569; line-height: 1.5; margin-top: 24px;">
                Regards,<br><strong>LivestockIQ Team</strong>
            </p>
        </div>
    </div>
</body>
</html>"""
    return send_email(buyer_email, subject, html_body, plain_body)


# =========================================================
# 3. BUYER CONFIRMS LIVESTOCK ORDER (BUY NOW)
# =========================================================
def send_order_confirmed_buyer_email(buyer_email: str, livestock: dict, buyer_name: str = "Buyer", order_details: dict = None) -> tuple[bool, str]:
    """
    Sent to Buyer upon clicking Buy Now / Confirming Order.
    Subject: LivestockIQ - Order Successfully Confirmed
    """
    info = _extract_livestock_info(livestock, order_details)
    greeting_name = buyer_name.strip() if buyer_name and buyer_name.strip() != "" else info['buyer_name']
    farmer_disp = info['farmer_name']
    if info['farmer_phone']:
        farmer_disp += f" ({info['farmer_phone']})"

    subject = "LivestockIQ - Order Successfully Confirmed"
    plain_body = f"""Dear {greeting_name},

You successfully confirmed the order for the following livestock on LivestockIQ.

Livestock Details:
--------------------------------
Livestock ID: #{info['listing_id']}
Species: {info['species']}
Breed: {info['breed']}
Gender: {info['gender']}
Age: {info['age']}
Weight: {info['weight']}
Milk Production: {info['milk']}
Location: {info['location']}
Price: ₹{info['price_formatted']}
Farmer: {farmer_disp}
Order ID: #{info['order_id']}
--------------------------------

Your order has been successfully confirmed.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LivestockIQ - Order Successfully Confirmed</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <div style="max-width: 580px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden;">
        <div style="background: linear-gradient(135deg, #15803d, #166534); padding: 24px 28px; text-align: center;">
            <h2 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700;">LivestockIQ</h2>
            <p style="color: #dcfce7; margin: 6px 0 0 0; font-size: 14px;">Order Confirmation</p>
        </div>
        <div style="padding: 28px 28px 24px 28px; color: #1e293b;">
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">Dear {greeting_name},</p>
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">
                You successfully confirmed the order for the following livestock on LivestockIQ.
            </p>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin: 20px 0;">
                <h4 style="margin: 0 0 12px 0; color: #0f172a; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">Livestock Details:</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px; color: #334155; line-height: 1.8;">
                    <tr><td style="width: 40%; color: #64748b;"><strong>Livestock ID:</strong></td><td>#{info['listing_id']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Species:</strong></td><td>{info['species']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Breed:</strong></td><td>{info['breed']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Gender:</strong></td><td>{info['gender']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Age:</strong></td><td>{info['age']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Weight:</strong></td><td>{info['weight']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Milk Production:</strong></td><td>{info['milk']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Location:</strong></td><td>{info['location']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Price:</strong></td><td>&#8377;{info['price_formatted']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Farmer:</strong></td><td>{farmer_disp}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Order ID:</strong></td><td>#{info['order_id']}</td></tr>
                </table>
            </div>
            <p style="font-size: 14px; line-height: 1.6; color: #334155;">
                Your order has been successfully confirmed.
            </p>
            <p style="font-size: 14px; color: #475569; line-height: 1.5; margin-top: 20px;">
                Regards,<br><strong>LivestockIQ Team</strong>
            </p>
        </div>
    </div>
</body>
</html>"""
    return send_email(buyer_email, subject, html_body, plain_body)

# Backward-compatibility alias
send_purchase_confirmation_email = send_order_confirmed_buyer_email

def send_order_confirmed_farmer_email(farmer_email: str, livestock: dict, farmer_name: str = "Farmer", buyer_name: str = "Buyer", order_details: dict = None) -> tuple[bool, str]:
    """
    Sent to Farmer upon Buyer confirming order.
    Subject: LivestockIQ - Your Livestock Has Been Confirmed
    """
    info = _extract_livestock_info(livestock, order_details)
    greeting_name = farmer_name.strip() if farmer_name and farmer_name.strip() != "" else info['farmer_name']
    actual_buyer = buyer_name.strip() if buyer_name and buyer_name.strip() != "" else info['buyer_name']
    buyer_disp = actual_buyer
    if info['buyer_phone']:
        buyer_disp += f" ({info['buyer_phone']})"

    subject = "LivestockIQ - Your Livestock Has Been Confirmed"
    plain_body = f"""Dear {greeting_name},

Your livestock has been confirmed by {actual_buyer} on LivestockIQ.

Livestock Details:
--------------------------------
Livestock ID: #{info['listing_id']}
Species: {info['species']}
Breed: {info['breed']}
Gender: {info['gender']}
Age: {info['age']}
Weight: {info['weight']}
Milk Production: {info['milk']}
Location: {info['location']}
Price: ₹{info['price_formatted']}
Buyer: {buyer_disp}
Order ID: #{info['order_id']}
--------------------------------

Please prepare for the next steps according to your order process.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LivestockIQ - Your Livestock Has Been Confirmed</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <div style="max-width: 580px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden;">
        <div style="background: linear-gradient(135deg, #1e3a8a, #1d4ed8); padding: 24px 28px; text-align: center;">
            <h2 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700;">LivestockIQ</h2>
            <p style="color: #dbeafe; margin: 6px 0 0 0; font-size: 14px;">Livestock Confirmation Notification</p>
        </div>
        <div style="padding: 28px 28px 24px 28px; color: #1e293b;">
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">Dear {greeting_name},</p>
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">
                Your livestock has been confirmed by <strong>{actual_buyer}</strong> on LivestockIQ.
            </p>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin: 20px 0;">
                <h4 style="margin: 0 0 12px 0; color: #0f172a; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">Livestock Details:</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px; color: #334155; line-height: 1.8;">
                    <tr><td style="width: 40%; color: #64748b;"><strong>Livestock ID:</strong></td><td>#{info['listing_id']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Species:</strong></td><td>{info['species']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Breed:</strong></td><td>{info['breed']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Gender:</strong></td><td>{info['gender']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Age:</strong></td><td>{info['age']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Weight:</strong></td><td>{info['weight']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Milk Production:</strong></td><td>{info['milk']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Location:</strong></td><td>{info['location']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Price:</strong></td><td>&#8377;{info['price_formatted']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Buyer:</strong></td><td>{buyer_disp}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Order ID:</strong></td><td>#{info['order_id']}</td></tr>
                </table>
            </div>
            <p style="font-size: 14px; line-height: 1.6; color: #334155;">
                Please prepare for the next steps according to your order process.
            </p>
            <p style="font-size: 14px; color: #475569; line-height: 1.5; margin-top: 20px;">
                Regards,<br><strong>LivestockIQ Team</strong>
            </p>
        </div>
    </div>
</body>
</html>"""
    return send_email(farmer_email, subject, html_body, plain_body)

# Backward-compatibility alias
def send_farmer_purchase_notification_email(farmer_email: str, livestock: dict, farmer_name: str = "Farmer", order_details: dict = None) -> tuple[bool, str]:
    buyer_name = (order_details or {}).get('buyer_name', 'Buyer')
    return send_order_confirmed_farmer_email(farmer_email, livestock, farmer_name, buyer_name, order_details)


# =========================================================
# 4. BUYER MARKS LIVESTOCK AS PURCHASED (PROOF UPLOAD / COMPLETE)
# =========================================================
def send_purchase_completed_buyer_email(buyer_email: str, livestock: dict, buyer_name: str = "Buyer", order_details: dict = None) -> tuple[bool, str]:
    """
    Sent to Buyer when livestock is marked as PURCHASED / proof uploaded.
    Subject: LivestockIQ - Livestock Successfully Purchased
    """
    info = _extract_livestock_info(livestock, order_details)
    greeting_name = buyer_name.strip() if buyer_name and buyer_name.strip() != "" else info['buyer_name']
    farmer_disp = info['farmer_name']
    if info['farmer_phone']:
        farmer_disp += f" ({info['farmer_phone']})"

    subject = "LivestockIQ - Livestock Successfully Purchased"
    plain_body = f"""Dear {greeting_name},

You successfully purchased the following livestock on LivestockIQ.

Livestock Details:
--------------------------------
Livestock ID: #{info['listing_id']}
Species: {info['species']}
Breed: {info['breed']}
Gender: {info['gender']}
Age: {info['age']}
Weight: {info['weight']}
Milk Production: {info['milk']}
Location: {info['location']}
Price: ₹{info['price_formatted']}
Farmer: {farmer_disp}
Order ID: #{info['order_id']}
--------------------------------

Your purchase is complete.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LivestockIQ - Livestock Successfully Purchased</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <div style="max-width: 580px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden;">
        <div style="background: linear-gradient(135deg, #15803d, #166534); padding: 24px 28px; text-align: center;">
            <h2 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700;">LivestockIQ</h2>
            <p style="color: #dcfce7; margin: 6px 0 0 0; font-size: 14px;">Purchase Completed</p>
        </div>
        <div style="padding: 28px 28px 24px 28px; color: #1e293b;">
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">Dear {greeting_name},</p>
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">
                You successfully purchased the following livestock on LivestockIQ.
            </p>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin: 20px 0;">
                <h4 style="margin: 0 0 12px 0; color: #0f172a; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">Livestock Details:</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px; color: #334155; line-height: 1.8;">
                    <tr><td style="width: 40%; color: #64748b;"><strong>Livestock ID:</strong></td><td>#{info['listing_id']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Species:</strong></td><td>{info['species']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Breed:</strong></td><td>{info['breed']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Gender:</strong></td><td>{info['gender']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Age:</strong></td><td>{info['age']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Weight:</strong></td><td>{info['weight']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Milk Production:</strong></td><td>{info['milk']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Location:</strong></td><td>{info['location']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Price:</strong></td><td>&#8377;{info['price_formatted']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Farmer:</strong></td><td>{farmer_disp}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Order ID:</strong></td><td>#{info['order_id']}</td></tr>
                </table>
            </div>
            <p style="font-size: 14px; line-height: 1.6; color: #334155;">
                Your purchase is complete.
            </p>
            <p style="font-size: 14px; color: #475569; line-height: 1.5; margin-top: 20px;">
                Regards,<br><strong>LivestockIQ Team</strong>
            </p>
        </div>
    </div>
</body>
</html>"""
    return send_email(buyer_email, subject, html_body, plain_body)

def send_purchase_completed_farmer_email(farmer_email: str, livestock: dict, farmer_name: str = "Farmer", buyer_name: str = "Buyer", order_details: dict = None) -> tuple[bool, str]:
    """
    Sent to Farmer when buyer marks livestock as purchased / uploads proof.
    Subject: LivestockIQ - Your Livestock Has Been Successfully Purchased
    """
    info = _extract_livestock_info(livestock, order_details)
    greeting_name = farmer_name.strip() if farmer_name and farmer_name.strip() != "" else info['farmer_name']
    actual_buyer = buyer_name.strip() if buyer_name and buyer_name.strip() != "" else info['buyer_name']
    buyer_disp = actual_buyer
    if info['buyer_phone']:
        buyer_disp += f" ({info['buyer_phone']})"

    subject = "LivestockIQ - Your Livestock Has Been Successfully Purchased"
    plain_body = f"""Dear {greeting_name},

Your livestock is successfully purchased by {actual_buyer} on LivestockIQ.

Livestock Details:
--------------------------------
Livestock ID: #{info['listing_id']}
Species: {info['species']}
Breed: {info['breed']}
Gender: {info['gender']}
Age: {info['age']}
Weight: {info['weight']}
Milk Production: {info['milk']}
Location: {info['location']}
Price: ₹{info['price_formatted']}
Buyer: {buyer_disp}
Order ID: #{info['order_id']}
--------------------------------

The purchase for your livestock has been completed successfully.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LivestockIQ - Your Livestock Has Been Successfully Purchased</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <div style="max-width: 580px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden;">
        <div style="background: linear-gradient(135deg, #1e3a8a, #1d4ed8); padding: 24px 28px; text-align: center;">
            <h2 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700;">LivestockIQ</h2>
            <p style="color: #dbeafe; margin: 6px 0 0 0; font-size: 14px;">Purchase Completed</p>
        </div>
        <div style="padding: 28px 28px 24px 28px; color: #1e293b;">
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">Dear {greeting_name},</p>
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">
                Your livestock is successfully purchased by <strong>{actual_buyer}</strong> on LivestockIQ.
            </p>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin: 20px 0;">
                <h4 style="margin: 0 0 12px 0; color: #0f172a; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">Livestock Details:</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px; color: #334155; line-height: 1.8;">
                    <tr><td style="width: 40%; color: #64748b;"><strong>Livestock ID:</strong></td><td>#{info['listing_id']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Species:</strong></td><td>{info['species']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Breed:</strong></td><td>{info['breed']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Gender:</strong></td><td>{info['gender']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Age:</strong></td><td>{info['age']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Weight:</strong></td><td>{info['weight']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Milk Production:</strong></td><td>{info['milk']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Location:</strong></td><td>{info['location']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Price:</strong></td><td>&#8377;{info['price_formatted']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Buyer:</strong></td><td>{buyer_disp}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Order ID:</strong></td><td>#{info['order_id']}</td></tr>
                </table>
            </div>
            <p style="font-size: 14px; line-height: 1.6; color: #334155;">
                The purchase for your livestock has been completed successfully.
            </p>
            <p style="font-size: 14px; color: #475569; line-height: 1.5; margin-top: 20px;">
                Regards,<br><strong>LivestockIQ Team</strong>
            </p>
        </div>
    </div>
</body>
</html>"""
    return send_email(farmer_email, subject, html_body, plain_body)


# =========================================================
# 5. BUYER REJECTS DELIVERY
# =========================================================
def send_order_rejected_buyer_email(buyer_email: str, livestock: dict, buyer_name: str = "Buyer", order_details: dict = None, reason: str = "") -> tuple[bool, str]:
    """
    Sent to Buyer when they reject livestock delivery.
    Subject: LivestockIQ - Delivery Rejected
    """
    info = _extract_livestock_info(livestock, order_details)
    greeting_name = buyer_name.strip() if buyer_name and buyer_name.strip() != "" else info['buyer_name']
    rejection_reason = reason.strip() if reason and reason.strip() != "" else "Delivery rejected by buyer."
    farmer_disp = info['farmer_name']
    if info['farmer_phone']:
        farmer_disp += f" ({info['farmer_phone']})"

    subject = "LivestockIQ - Delivery Rejected"
    plain_body = f"""Dear {greeting_name},

You rejected the delivery of the following livestock on LivestockIQ.

Livestock Details:
--------------------------------
Livestock ID: #{info['listing_id']}
Species: {info['species']}
Breed: {info['breed']}
Gender: {info['gender']}
Age: {info['age']}
Weight: {info['weight']}
Milk Production: {info['milk']}
Location: {info['location']}
Price: ₹{info['price_formatted']}
Farmer: {farmer_disp}
Order ID: #{info['order_id']}
--------------------------------

Reason for Rejection:
{rejection_reason}

Your rejection has been recorded.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LivestockIQ - Delivery Rejected</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <div style="max-width: 580px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden;">
        <div style="background: linear-gradient(135deg, #d97706, #b45309); padding: 24px 28px; text-align: center;">
            <h2 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700;">LivestockIQ</h2>
            <p style="color: #fef3c7; margin: 6px 0 0 0; font-size: 14px;">Delivery Rejection Recorded</p>
        </div>
        <div style="padding: 28px 28px 24px 28px; color: #1e293b;">
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">Dear {greeting_name},</p>
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">
                You rejected the delivery of the following livestock on LivestockIQ.
            </p>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin: 20px 0;">
                <h4 style="margin: 0 0 12px 0; color: #0f172a; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">Livestock Details:</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px; color: #334155; line-height: 1.8;">
                    <tr><td style="width: 40%; color: #64748b;"><strong>Livestock ID:</strong></td><td>#{info['listing_id']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Species:</strong></td><td>{info['species']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Breed:</strong></td><td>{info['breed']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Gender:</strong></td><td>{info['gender']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Age:</strong></td><td>{info['age']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Weight:</strong></td><td>{info['weight']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Milk Production:</strong></td><td>{info['milk']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Location:</strong></td><td>{info['location']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Price:</strong></td><td>&#8377;{info['price_formatted']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Farmer:</strong></td><td>{farmer_disp}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Order ID:</strong></td><td>#{info['order_id']}</td></tr>
                </table>
            </div>
            <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 14px 18px; margin: 16px 0;">
                <strong style="color: #92400e; font-size: 14px;">Reason for Rejection:</strong>
                <p style="margin: 6px 0 0 0; color: #78350f; font-size: 14px;">{rejection_reason}</p>
            </div>
            <p style="font-size: 14px; line-height: 1.6; color: #334155;">
                Your rejection has been recorded.
            </p>
            <p style="font-size: 14px; color: #475569; line-height: 1.5; margin-top: 20px;">
                Regards,<br><strong>LivestockIQ Team</strong>
            </p>
        </div>
    </div>
</body>
</html>"""
    return send_email(buyer_email, subject, html_body, plain_body)

def send_order_rejected_farmer_email(farmer_email: str, livestock: dict, farmer_name: str = "Farmer", buyer_name: str = "Buyer", order_details: dict = None, reason: str = "") -> tuple[bool, str]:
    """
    Sent to Farmer when buyer rejects delivery.
    Subject: LivestockIQ - Your Livestock Delivery Has Been Rejected
    """
    info = _extract_livestock_info(livestock, order_details)
    greeting_name = farmer_name.strip() if farmer_name and farmer_name.strip() != "" else info['farmer_name']
    actual_buyer = buyer_name.strip() if buyer_name and buyer_name.strip() != "" else info['buyer_name']
    rejection_reason = reason.strip() if reason and reason.strip() != "" else "Delivery rejected by buyer."
    buyer_disp = actual_buyer
    if info['buyer_phone']:
        buyer_disp += f" ({info['buyer_phone']})"

    subject = "LivestockIQ - Your Livestock Delivery Has Been Rejected"
    plain_body = f"""Dear {greeting_name},

Your livestock delivery is rejected by {actual_buyer} on LivestockIQ.

Livestock Details:
--------------------------------
Livestock ID: #{info['listing_id']}
Species: {info['species']}
Breed: {info['breed']}
Gender: {info['gender']}
Age: {info['age']}
Weight: {info['weight']}
Milk Production: {info['milk']}
Location: {info['location']}
Price: ₹{info['price_formatted']}
Buyer: {buyer_disp}
Order ID: #{info['order_id']}
--------------------------------

Reason for Rejection:
{rejection_reason}

Please review your order details on LivestockIQ.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LivestockIQ - Your Livestock Delivery Has Been Rejected</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <div style="max-width: 580px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden;">
        <div style="background: linear-gradient(135deg, #d97706, #b45309); padding: 24px 28px; text-align: center;">
            <h2 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700;">LivestockIQ</h2>
            <p style="color: #fef3c7; margin: 6px 0 0 0; font-size: 14px;">Delivery Rejection Notification</p>
        </div>
        <div style="padding: 28px 28px 24px 28px; color: #1e293b;">
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">Dear {greeting_name},</p>
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">
                Your livestock delivery is rejected by <strong>{actual_buyer}</strong> on LivestockIQ.
            </p>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin: 20px 0;">
                <h4 style="margin: 0 0 12px 0; color: #0f172a; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">Livestock Details:</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px; color: #334155; line-height: 1.8;">
                    <tr><td style="width: 40%; color: #64748b;"><strong>Livestock ID:</strong></td><td>#{info['listing_id']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Species:</strong></td><td>{info['species']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Breed:</strong></td><td>{info['breed']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Gender:</strong></td><td>{info['gender']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Age:</strong></td><td>{info['age']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Weight:</strong></td><td>{info['weight']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Milk Production:</strong></td><td>{info['milk']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Location:</strong></td><td>{info['location']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Price:</strong></td><td>&#8377;{info['price_formatted']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Buyer:</strong></td><td>{buyer_disp}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Order ID:</strong></td><td>#{info['order_id']}</td></tr>
                </table>
            </div>
            <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 14px 18px; margin: 16px 0;">
                <strong style="color: #92400e; font-size: 14px;">Reason for Rejection:</strong>
                <p style="margin: 6px 0 0 0; color: #78350f; font-size: 14px;">{rejection_reason}</p>
            </div>
            <p style="font-size: 14px; line-height: 1.6; color: #334155;">
                Please review your order details on LivestockIQ.
            </p>
            <p style="font-size: 14px; color: #475569; line-height: 1.5; margin-top: 20px;">
                Regards,<br><strong>LivestockIQ Team</strong>
            </p>
        </div>
    </div>
</body>
</html>"""
    return send_email(farmer_email, subject, html_body, plain_body)

# Backward-compatibility alias
def send_purchase_rejection_email(recipient_email: str, recipient_role: str, order: dict, reason: str = "") -> tuple[bool, str]:
    if recipient_role.lower() == 'buyer':
        return send_order_rejected_buyer_email(recipient_email, order, order.get('buyer_name', 'Buyer'), order, reason)
    else:
        return send_order_rejected_farmer_email(recipient_email, order, order.get('farmer_name', 'Farmer'), order.get('buyer_name', 'Buyer'), order, reason)


# =========================================================
# 6. BUYER CANCELS ORDER
# =========================================================
def send_order_cancelled_buyer_email(buyer_email: str, livestock: dict, buyer_name: str = "Buyer", order_details: dict = None, reason: str = "") -> tuple[bool, str]:
    """
    Sent to Buyer when they cancel order.
    Subject: LivestockIQ - Your Order Has Been Cancelled
    """
    info = _extract_livestock_info(livestock, order_details)
    greeting_name = buyer_name.strip() if buyer_name and buyer_name.strip() != "" else info['buyer_name']
    cancel_reason = reason.strip() if reason and reason.strip() != "" else "Order cancelled by buyer."
    farmer_disp = info['farmer_name']
    if info['farmer_phone']:
        farmer_disp += f" ({info['farmer_phone']})"

    subject = "LivestockIQ - Your Order Has Been Cancelled"
    plain_body = f"""Dear {greeting_name},

You cancelled your order for the following livestock on LivestockIQ.

Livestock Details:
--------------------------------
Livestock ID: #{info['listing_id']}
Species: {info['species']}
Breed: {info['breed']}
Gender: {info['gender']}
Age: {info['age']}
Weight: {info['weight']}
Milk Production: {info['milk']}
Location: {info['location']}
Price: ₹{info['price_formatted']}
Farmer: {farmer_disp}
Order ID: #{info['order_id']}
--------------------------------

Reason for Cancellation:
{cancel_reason}

Your cancellation has been processed.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LivestockIQ - Your Order Has Been Cancelled</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <div style="max-width: 580px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden;">
        <div style="background: linear-gradient(135deg, #dc2626, #b91c1c); padding: 24px 28px; text-align: center;">
            <h2 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700;">LivestockIQ</h2>
            <p style="color: #fee2e2; margin: 6px 0 0 0; font-size: 14px;">Order Cancellation Confirmation</p>
        </div>
        <div style="padding: 28px 28px 24px 28px; color: #1e293b;">
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">Dear {greeting_name},</p>
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">
                You cancelled your order for the following livestock on LivestockIQ.
            </p>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin: 20px 0;">
                <h4 style="margin: 0 0 12px 0; color: #0f172a; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">Livestock Details:</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px; color: #334155; line-height: 1.8;">
                    <tr><td style="width: 40%; color: #64748b;"><strong>Livestock ID:</strong></td><td>#{info['listing_id']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Species:</strong></td><td>{info['species']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Breed:</strong></td><td>{info['breed']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Gender:</strong></td><td>{info['gender']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Age:</strong></td><td>{info['age']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Weight:</strong></td><td>{info['weight']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Milk Production:</strong></td><td>{info['milk']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Location:</strong></td><td>{info['location']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Price:</strong></td><td>&#8377;{info['price_formatted']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Farmer:</strong></td><td>{farmer_disp}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Order ID:</strong></td><td>#{info['order_id']}</td></tr>
                </table>
            </div>
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 14px 18px; margin: 16px 0;">
                <strong style="color: #991b1b; font-size: 14px;">Reason for Cancellation:</strong>
                <p style="margin: 6px 0 0 0; color: #7f1d1d; font-size: 14px;">{cancel_reason}</p>
            </div>
            <p style="font-size: 14px; line-height: 1.6; color: #334155;">
                Your cancellation has been processed.
            </p>
            <p style="font-size: 14px; color: #475569; line-height: 1.5; margin-top: 20px;">
                Regards,<br><strong>LivestockIQ Team</strong>
            </p>
        </div>
    </div>
</body>
</html>"""
    return send_email(buyer_email, subject, html_body, plain_body)

def send_order_cancelled_farmer_email(farmer_email: str, livestock: dict, farmer_name: str = "Farmer", buyer_name: str = "Buyer", order_details: dict = None, reason: str = "") -> tuple[bool, str]:
    """
    Sent to Farmer when buyer cancels order.
    Subject: LivestockIQ - Livestock Order Has Been Cancelled
    """
    info = _extract_livestock_info(livestock, order_details)
    greeting_name = farmer_name.strip() if farmer_name and farmer_name.strip() != "" else info['farmer_name']
    actual_buyer = buyer_name.strip() if buyer_name and buyer_name.strip() != "" else info['buyer_name']
    cancel_reason = reason.strip() if reason and reason.strip() != "" else "Order cancelled by buyer."
    buyer_disp = actual_buyer
    if info['buyer_phone']:
        buyer_disp += f" ({info['buyer_phone']})"

    subject = "LivestockIQ - Livestock Order Has Been Cancelled"
    plain_body = f"""Dear {greeting_name},

Your livestock has been cancelled by {actual_buyer} on LivestockIQ.

Livestock Details:
--------------------------------
Livestock ID: #{info['listing_id']}
Species: {info['species']}
Breed: {info['breed']}
Gender: {info['gender']}
Age: {info['age']}
Weight: {info['weight']}
Milk Production: {info['milk']}
Location: {info['location']}
Price: ₹{info['price_formatted']}
Buyer: {buyer_disp}
Order ID: #{info['order_id']}
--------------------------------

Reason for Cancellation:
{cancel_reason}

The livestock listing has been restored to the marketplace.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LivestockIQ - Livestock Order Has Been Cancelled</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <div style="max-width: 580px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden;">
        <div style="background: linear-gradient(135deg, #dc2626, #b91c1c); padding: 24px 28px; text-align: center;">
            <h2 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700;">LivestockIQ</h2>
            <p style="color: #fee2e2; margin: 6px 0 0 0; font-size: 14px;">Order Cancellation Notification</p>
        </div>
        <div style="padding: 28px 28px 24px 28px; color: #1e293b;">
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">Dear {greeting_name},</p>
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">
                Your livestock has been cancelled by <strong>{actual_buyer}</strong> on LivestockIQ.
            </p>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin: 20px 0;">
                <h4 style="margin: 0 0 12px 0; color: #0f172a; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">Livestock Details:</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px; color: #334155; line-height: 1.8;">
                    <tr><td style="width: 40%; color: #64748b;"><strong>Livestock ID:</strong></td><td>#{info['listing_id']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Species:</strong></td><td>{info['species']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Breed:</strong></td><td>{info['breed']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Gender:</strong></td><td>{info['gender']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Age:</strong></td><td>{info['age']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Weight:</strong></td><td>{info['weight']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Milk Production:</strong></td><td>{info['milk']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Location:</strong></td><td>{info['location']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Price:</strong></td><td>&#8377;{info['price_formatted']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Buyer:</strong></td><td>{buyer_disp}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Order ID:</strong></td><td>#{info['order_id']}</td></tr>
                </table>
            </div>
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 14px 18px; margin: 16px 0;">
                <strong style="color: #991b1b; font-size: 14px;">Reason for Cancellation:</strong>
                <p style="margin: 6px 0 0 0; color: #7f1d1d; font-size: 14px;">{cancel_reason}</p>
            </div>
            <p style="font-size: 14px; line-height: 1.6; color: #334155;">
                The livestock listing has been restored to the marketplace.
            </p>
            <p style="font-size: 14px; color: #475569; line-height: 1.5; margin-top: 20px;">
                Regards,<br><strong>LivestockIQ Team</strong>
            </p>
        </div>
    </div>
</body>
</html>"""
    return send_email(farmer_email, subject, html_body, plain_body)

# Backward-compatibility alias
def send_purchase_cancellation_email(recipient_email: str, recipient_role: str, order: dict, reason: str = "", explanation: str = "") -> tuple[bool, str]:
    full_reason = f"{reason}" + (f" ({explanation})" if explanation else "")
    if recipient_role.lower() == 'buyer':
        return send_order_cancelled_buyer_email(recipient_email, order, order.get('buyer_name', 'Buyer'), order, full_reason)
    else:
        return send_order_cancelled_farmer_email(recipient_email, order, order.get('farmer_name', 'Farmer'), order.get('buyer_name', 'Buyer'), order, full_reason)


# =========================================================
# 7. FARMER PUBLICATION & DELETION EMAILS
# =========================================================
def send_farmer_publication_confirmation_email(farmer_email: str, listing: dict, farmer_name: str = "Farmer") -> tuple[bool, str]:
    """
    Send Confirmation Email to Farmer when their livestock listing is published via AgentMail.
    Subject: Livestock Listing Published Successfully
    """
    info = _extract_livestock_info(listing)
    greeting_name = farmer_name.strip() if farmer_name and farmer_name.strip() != "" else info['farmer_name']

    subject = "Livestock Listing Published Successfully"
    plain_body = f"""Hello {greeting_name},

Your livestock listing has been published successfully on LivestockIQ.

Listing ID: #{info['listing_id']}
Livestock: {info['breed']}
Species: {info['species']}
Gender: {info['gender']}
Age: {info['age']}
Milk Production: {info['milk']}
Asking Price: ₹{info['price_formatted']}
Location: {info['location']}

Interested buyers in your district will be notified.

Thank you for using LivestockIQ."""

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 580px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <div style="background: linear-gradient(135deg, #059669, #047857); padding: 24px 28px; text-align: center;">
            <h2 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700; letter-spacing: 0.5px;">LivestockIQ</h2>
            <p style="color: #d1fae5; margin: 6px 0 0 0; font-size: 13px;">Listing Publication Confirmation</p>
        </div>
        <div style="padding: 28px 28px 24px 28px; color: #1e293b;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="display: inline-block; background: #d1fae5; color: #047857; padding: 8px 16px; border-radius: 20px; font-weight: 700; font-size: 15px;">
                    &#10004; Listing Published
                </div>
            </div>

            <p style="font-size: 15px; line-height: 1.6; color: #334155;">Hello {greeting_name},</p>
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">
                Your livestock listing has been successfully published to the marketplace.
            </p>

            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin: 20px 0;">
                <div style="font-size: 15px; color: #334155; line-height: 1.8;">
                    <div><strong>Listing ID:</strong> #{info['listing_id']}</div>
                    <div><strong>Livestock:</strong> {info['breed']}</div>
                    <div><strong>Species:</strong> {info['species']}</div>
                    <div><strong>Gender:</strong> {info['gender']}</div>
                    <div><strong>Age:</strong> {info['age']}</div>
                    <div><strong>Milk Production:</strong> {info['milk']}</div>
                    <div><strong>Asking Price:</strong> &#8377;{info['price_formatted']}</div>
                    <div><strong>Location:</strong> {info['location']}</div>
                </div>
            </div>

            <p style="font-size: 14px; line-height: 1.6; color: #334155;">
                Interested buyers in your district will be notified automatically.
            </p>
            <p style="font-size: 14px; color: #475569; line-height: 1.5; margin-top: 20px;">Best regards,<br><strong>LivestockIQ Team</strong></p>
        </div>
    </div>
    """
    return send_email(farmer_email, subject, html_body, plain_body)

def send_farmer_livestock_deletion_email(farmer_email: str, livestock: dict, farmer_name: str = "Farmer") -> tuple[bool, str]:
    """
    Send Notification Email to Farmer when their published livestock listing is deleted/removed via AgentMail.
    Subject: LivestockIQ - Your Published Livestock Has Been Removed
    """
    info = _extract_livestock_info(livestock)
    created_at = livestock.get('created_at', 'N/A')
    greeting_name = farmer_name.strip() if farmer_name and farmer_name.strip() != "" else info['farmer_name']

    subject = "LivestockIQ - Your Published Livestock Has Been Removed"
    plain_body = f"""Dear {greeting_name},

You removed your published livestock from LivestockIQ.

Livestock Details:
--------------------------------
Livestock ID: #{info['listing_id']}
Species: {info['species']}
Breed: {info['breed']}
Gender: {info['gender']}
Age: {info['age']}
Weight: {info['weight']}
Milk Production: {info['milk']}
Location: {info['location']}
Price: ₹{info['price_formatted']}
Published Date: {created_at}
--------------------------------

Your livestock listing has been successfully removed from LivestockIQ.

Regards,
LivestockIQ Team"""

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 580px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); padding: 24px 28px; text-align: center;">
            <h2 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700; letter-spacing: 0.5px;">LivestockIQ</h2>
            <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 13px;">Livestock Removal Notification</p>
        </div>
        <div style="padding: 28px 28px 24px 28px; color: #1e293b;">
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">Dear {greeting_name},</p>
            <p style="font-size: 15px; line-height: 1.6; color: #334155;">
                You removed your published livestock from LivestockIQ.
            </p>

            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin: 20px 0;">
                <h4 style="margin: 0 0 12px 0; color: #0f172a; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">Livestock Details:</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px; color: #334155; line-height: 1.8;">
                    <tr><td style="width: 40%; color: #64748b;"><strong>Livestock ID:</strong></td><td>#{info['listing_id']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Species:</strong></td><td>{info['species']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Breed:</strong></td><td>{info['breed']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Gender:</strong></td><td>{info['gender']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Age:</strong></td><td>{info['age']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Weight:</strong></td><td>{info['weight']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Milk Production:</strong></td><td>{info['milk']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Location:</strong></td><td>{info['location']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Price:</strong></td><td>&#8377;{info['price_formatted']}</td></tr>
                    <tr><td style="color: #64748b;"><strong>Published Date:</strong></td><td>{created_at}</td></tr>
                </table>
            </div>

            <p style="font-size: 14px; line-height: 1.6; color: #334155;">
                Your livestock listing has been successfully removed from LivestockIQ.
            </p>
            <p style="font-size: 14px; color: #475569; line-height: 1.5; margin-top: 20px;">Regards,<br><strong>LivestockIQ Team</strong></p>
        </div>
    </div>
    """
    return send_email(farmer_email, subject, html_body, plain_body)


# =========================================================
# 8. BUYER AVAILABILITY ALERT NOTIFICATION EMAIL
# =========================================================
def send_livestock_available_notification(email: str, livestock: dict) -> tuple[bool, str]:
    """
    Send Automatic Notification to Buyer via AgentMail when matching livestock is listed by a farmer.
    Subject: Livestock Matching Your Search Is Now Available
    """
    info = _extract_livestock_info(livestock)

    subject = "Livestock Matching Your Search Is Now Available"
    plain_body = f"""Dear Buyer,

A livestock matching your saved search is now available on LivestockIQ.

Breed: {info['breed']}
Location: {info['location']}
Age: {info['age']}
Milk Production: {info['milk']}
Price: Rs. {info['price_formatted']}

You can now open LivestockIQ to view the listing.

Thank you,
LivestockIQ Team"""

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 580px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <div style="background: linear-gradient(135deg, #059669, #065f46); padding: 24px 28px; text-align: center;">
            <h2 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700; letter-spacing: 0.5px;">LivestockIQ</h2>
            <p style="color: #a7f3d0; margin: 6px 0 0 0; font-size: 13px;">Livestock Availability Alert</p>
        </div>
        <div style="padding: 28px 28px 24px 28px; color: #1e293b;">
            <p style="font-size: 14px; line-height: 1.6; color: #334155;">Dear Buyer,</p>
            <p style="font-size: 14px; line-height: 1.6; color: #334155;">A livestock matching your saved search criteria has just become available on LivestockIQ!</p>

            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 18px; margin: 20px 0;">
                <table style="width: 100%; border-collapse: collapse; font-size: 14px; color: #1e293b;">
                    <tr>
                        <td style="padding: 6px 0; color: #64748b; width: 40%;"><strong>Breed:</strong></td>
                        <td style="padding: 6px 0; font-weight: 600; color: #065f46;">{info['breed']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #64748b;"><strong>Location:</strong></td>
                        <td style="padding: 6px 0;">{info['location']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #64748b;"><strong>Age:</strong></td>
                        <td style="padding: 6px 0;">{info['age']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #64748b;"><strong>Milk Production:</strong></td>
                        <td style="padding: 6px 0;">{info['milk']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #64748b;"><strong>Price:</strong></td>
                        <td style="padding: 6px 0; font-size: 16px; font-weight: 700; color: #15803d;">&#8377;{info['price_formatted']}</td>
                    </tr>
                </table>
            </div>

            <p style="font-size: 14px; line-height: 1.6; color: #334155;">You can now open LivestockIQ and view the listing to confirm your purchase.</p>

            <p style="font-size: 14px; color: #475569; line-height: 1.5; margin-top: 20px;">Thank you,<br><strong>LivestockIQ Team</strong></p>
        </div>
    </div>
    """
    return send_email(email, subject, html_body, plain_body)
