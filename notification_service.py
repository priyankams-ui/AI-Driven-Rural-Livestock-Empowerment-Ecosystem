import os
import re
import hmac
import hashlib
import secrets
import datetime
from dotenv import load_dotenv

# Ensure local environment variables are loaded
load_dotenv()

from database import get_db
from services.email_service import (
    send_email,
    send_verification_otp,
    send_purchase_otp,
    send_password_reset_otp,
    send_buyer_registration_success_email,
    send_order_confirmed_buyer_email,
    send_order_confirmed_farmer_email,
    send_purchase_completed_buyer_email,
    send_purchase_completed_farmer_email,
    send_order_rejected_buyer_email,
    send_order_rejected_farmer_email,
    send_order_cancelled_buyer_email,
    send_order_cancelled_farmer_email,
    send_purchase_confirmation_email,
    send_farmer_purchase_notification_email,
    send_farmer_publication_confirmation_email,
    send_purchase_cancellation_email,
    send_purchase_rejection_email,
    send_livestock_available_notification,
    send_farmer_livestock_deletion_email
)


MAX_OTP_ATTEMPTS = 5
OTP_VALIDITY_MINUTES = 5
OTP_RESEND_COOLDOWN_SECONDS = 60

def is_valid_indian_phone(phone: str) -> bool:
    """Validate Indian phone number: +91 followed by exactly 10 digits starting with 6-9."""
    if not phone:
        return False
    clean = phone.strip()
    return bool(re.match(r'^\+91[6-9]\d{9}$', clean))

def normalize_indian_phone(phone: str) -> str:
    """Normalize phone to standard +91XXXXXXXXXX format."""
    clean = re.sub(r'[\s\-]', '', str(phone).strip())
    if clean.startswith('+91') and len(clean) == 13:
        return clean
    if clean.startswith('91') and len(clean) == 12:
        return f"+{clean}"
    if len(clean) == 10 and clean[0] in '6789':
        return f"+91{clean}"
    return clean

def is_valid_email_format(email: str) -> bool:
    """Validate standard RFC email format."""
    if not email:
        return False
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email.strip()))

def is_valid_pincode(pincode: str) -> bool:
    """Validate 6-digit Indian PIN code."""
    if not pincode:
        return False
    return bool(re.match(r'^[1-9]\d{5}$', pincode.strip()))

def generate_secure_otp(length: int = 6) -> str:
    """Generate a cryptographically secure numeric OTP."""
    digits = [str(secrets.randbelow(10)) for _ in range(length)]
    return "".join(digits)

def hash_otp(otp: str) -> str:
    """Hash OTP using SHA-256 for secure storage."""
    salt = os.environ.get('SECRET_KEY', 'pashudhan_otp_salt_2026')
    return hmac.new(salt.encode('utf-8'), otp.encode('utf-8'), hashlib.sha256).hexdigest()

def create_and_store_otp(email: str, purpose: str, user_id: int = None, listing_id: int = None) -> tuple[str, bool, str]:
    """
    Generate, hash, and store a new OTP in the database.
    Enforces cooldown and single-active-OTP rule.
    Supports user_id and listing_id for purchase verification.
    Returns: (otp_plain, success, error_message)
    """
    clean_email = email.strip().lower()
    conn = get_db()
    cursor = conn.cursor()

    # Check resend cooldown
    if listing_id:
        cursor.execute('''
            SELECT created_at FROM otp_verifications 
            WHERE email = ? AND purpose = ? AND listing_id = ?
            ORDER BY id DESC LIMIT 1
        ''', (clean_email, purpose, listing_id))
    else:
        cursor.execute('''
            SELECT created_at FROM otp_verifications 
            WHERE email = ? AND purpose = ? 
            ORDER BY id DESC LIMIT 1
        ''', (clean_email, purpose))
    
    last_row = cursor.fetchone()
    now = datetime.datetime.now()

    if last_row:
        try:
            last_ts = datetime.datetime.strptime(str(last_row['created_at']).replace('T', ' ').split('.')[0], "%Y-%m-%d %H:%M:%S")
            diff_secs = (now - last_ts).total_seconds()
            if diff_secs < OTP_RESEND_COOLDOWN_SECONDS:
                wait_time = int(OTP_RESEND_COOLDOWN_SECONDS - diff_secs)
                conn.close()
                return "", False, f"Please wait {wait_time} second(s) before requesting another OTP."
        except Exception:
            pass

    # Invalidate previous OTPs for this email, purpose, and listing_id
    if listing_id:
        cursor.execute("DELETE FROM otp_verifications WHERE email = ? AND purpose = ? AND listing_id = ?", (clean_email, purpose, listing_id))
    else:
        cursor.execute("DELETE FROM otp_verifications WHERE email = ? AND purpose = ?", (clean_email, purpose))

    otp_plain = generate_secure_otp(6)
    hashed = hash_otp(otp_plain)
    expires_at = (now + datetime.timedelta(minutes=OTP_VALIDITY_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
    created_at = now.strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        INSERT INTO otp_verifications (user_id, listing_id, email, otp_hash, purpose, attempts, expires_at, created_at)
        VALUES (?, ?, ?, ?, ?, 0, ?, ?)
    ''', (user_id, listing_id, clean_email, hashed, purpose, expires_at, created_at))

    conn.commit()
    conn.close()
    return otp_plain, True, ""

def verify_submitted_otp(email: str, submitted_otp: str, purpose: str, listing_id: int = None) -> tuple[bool, str]:
    """
    Verify submitted OTP against stored hash.
    Enforces single-use, expiry, and attempt rate limiting.
    Supports listing_id for purchase verification.
    """
    clean_email = email.strip().lower()
    clean_otp = submitted_otp.strip()

    conn = get_db()
    cursor = conn.cursor()

    if listing_id:
        cursor.execute('''
            SELECT id, otp_hash, attempts, expires_at 
            FROM otp_verifications 
            WHERE email = ? AND purpose = ? AND listing_id = ?
            ORDER BY id DESC LIMIT 1
        ''', (clean_email, purpose, listing_id))
    else:
        cursor.execute('''
            SELECT id, otp_hash, attempts, expires_at 
            FROM otp_verifications 
            WHERE email = ? AND purpose = ? 
            ORDER BY id DESC LIMIT 1
        ''', (clean_email, purpose))

    row = cursor.fetchone()

    if not row:
        conn.close()
        return False, "No active OTP found. Please request a new OTP."

    otp_id = row['id']
    stored_hash = row['otp_hash']
    attempts = row['attempts'] or 0
    expires_at_str = row['expires_at']

    now = datetime.datetime.now()
    try:
        expires_at = datetime.datetime.strptime(str(expires_at_str).replace('T', ' ').split('.')[0], "%Y-%m-%d %H:%M:%S")
        if now > expires_at:
            cursor.execute("DELETE FROM otp_verifications WHERE id = ?", (otp_id,))
            conn.commit()
            conn.close()
            return False, "OTP expired. Please request a new OTP."
    except Exception:
        pass

    if attempts >= MAX_OTP_ATTEMPTS:
        cursor.execute("DELETE FROM otp_verifications WHERE id = ?", (otp_id,))
        conn.commit()
        conn.close()
        return False, "Too many incorrect attempts. This OTP has been invalidated. Please request a new OTP."

    submitted_hash = hash_otp(clean_otp)
    if not hmac.compare_digest(stored_hash, submitted_hash):
        new_attempts = attempts + 1
        remaining = MAX_OTP_ATTEMPTS - new_attempts
        if remaining <= 0:
            cursor.execute("DELETE FROM otp_verifications WHERE id = ?", (otp_id,))
            conn.commit()
            conn.close()
            return False, "Invalid OTP. Maximum attempts reached. Please request a new OTP."
        else:
            cursor.execute("UPDATE otp_verifications SET attempts = ? WHERE id = ?", (new_attempts, otp_id))
            conn.commit()
            conn.close()
            return False, f"Invalid OTP. Please try again. ({remaining} attempt(s) remaining)"

    # Success: Invalidate OTP immediately to ensure single-use
    cursor.execute("DELETE FROM otp_verifications WHERE id = ?", (otp_id,))
    conn.commit()
    conn.close()
    return True, "OTP verified successfully."

def send_email_notification(to_email: str, subject: str, html_body: str, plain_body: str = None) -> tuple[bool, str]:
    """
    Delegate directly to services.email_service.send_email (AgentMail API).
    """
    return send_email(to_email, subject, html_body, plain_body)

# Legacy alias
send_smtp_email = send_email_notification

def send_admin_otp_email(admin_email: str, otp_code: str) -> tuple[bool, str]:
    """Send single-use Admin OTP email strictly to the authorized administrator."""
    subject = "Verify your LivestockIQ administrator login"
    plain_body = f"""Hello Rakshitha M,

Your LivestockIQ admin verification code is: {otp_code}

This code is valid for 5 minutes and can be used only once.

If you did not request this verification code, you can ignore this email.

Regards,
LivestockIQ Team"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Verify your LivestockIQ administrator login</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 20px;">
    <p>Hello Rakshitha M,</p>
    <p>Your LivestockIQ admin verification code is: <strong>{otp_code}</strong></p>
    <p>This code is valid for 5 minutes and can be used only once.</p>
    <p>If you did not request this verification code, you can ignore this email.</p>
    <p>Regards,<br>LivestockIQ Team</p>
</body>
</html>"""
    return send_email(admin_email, subject, html_body, plain_body)

def send_user_registration_otp_email(user_email: str, user_name: str, otp_code: str, role: str) -> tuple[bool, str]:
    """Send single-use account verification OTP email for Farmer / Buyer registration."""
    return send_verification_otp(user_email, otp_code, user_name, role)

def send_password_reset_otp_email(user_email: str, otp_code: str, user_name: str = "User", role: str = "User") -> tuple[bool, str]:
    """Send single-use password reset OTP email for Admin, Farmer, or Buyer."""
    return send_password_reset_otp(user_email, otp_code, user_name, role)

def send_buyer_purchase_confirmation_email(buyer_email: str, livestock: dict, buyer_name: str = "Buyer", order_details: dict = None) -> tuple[bool, str]:
    """Send purchase confirmation email to buyer (NO OTP)."""
    return send_order_confirmed_buyer_email(buyer_email, livestock, buyer_name, order_details)

def send_farmer_purchase_notification_email(farmer_email: str, livestock: dict, farmer_name: str = "Farmer", order_details: dict = None) -> tuple[bool, str]:
    """Send livestock purchased notification email to farmer."""
    buyer_name = (order_details or {}).get('buyer_name', 'Buyer')
    return send_order_confirmed_farmer_email(farmer_email, livestock, farmer_name, buyer_name, order_details)

def send_farmer_publication_email(farmer_email: str, listing: dict, farmer_name: str = "Farmer") -> tuple[bool, str]:
    """Send listing publication confirmation email to farmer."""
    return send_farmer_publication_confirmation_email(farmer_email, listing, farmer_name)

def send_farmer_livestock_deletion_email(farmer_email: str, livestock: dict, farmer_name: str = "Farmer") -> tuple[bool, str]:
    """Send livestock deletion notification email to farmer with LivestockIQ branding."""
    from services.email_service import send_farmer_livestock_deletion_email as _svc_deletion_email
    return _svc_deletion_email(farmer_email, livestock, farmer_name)

def send_order_cancellation_emails(buyer_email: str, farmer_email: str, order: dict, reason: str = "", explanation: str = "") -> dict:
    """Send cancellation notifications to both buyer and farmer."""
    b_ok, b_msg = (False, "")
    f_ok, f_msg = (False, "")
    full_reason = f"{reason}" + (f" ({explanation})" if explanation else "")
    buyer_name = order.get('buyer_name', 'Buyer')
    farmer_name = order.get('farmer_name', 'Farmer')

    if buyer_email and is_valid_email_format(buyer_email):
        b_ok, b_msg = send_order_cancelled_buyer_email(buyer_email, order, buyer_name, order, full_reason)
    if farmer_email and is_valid_email_format(farmer_email):
        f_ok, f_msg = send_order_cancelled_farmer_email(farmer_email, order, farmer_name, buyer_name, order, full_reason)
    return {'buyer_sent': b_ok, 'farmer_sent': f_ok}

def send_order_rejection_emails(buyer_email: str, farmer_email: str, order: dict, reason: str = "") -> dict:
    """Send rejection notifications to both buyer and farmer."""
    b_ok, b_msg = (False, "")
    f_ok, f_msg = (False, "")
    buyer_name = order.get('buyer_name', 'Buyer')
    farmer_name = order.get('farmer_name', 'Farmer')

    if buyer_email and is_valid_email_format(buyer_email):
        b_ok, b_msg = send_order_rejected_buyer_email(buyer_email, order, buyer_name, order, reason)
    if farmer_email and is_valid_email_format(farmer_email):
        f_ok, f_msg = send_order_rejected_farmer_email(farmer_email, order, farmer_name, buyer_name, order, reason)
    return {'buyer_sent': b_ok, 'farmer_sent': f_ok}

def send_buyer_purchase_otp_email(buyer_email: str, otp_code: str, livestock: dict) -> tuple[bool, str]:
    """Send single-use Purchase Verification OTP email to buyer (legacy)."""
    return send_purchase_otp(buyer_email, otp_code, livestock)

def notify_matching_buyers_for_new_listing(listing_id: int) -> int:
    """
    Checks all active buyer availability alerts against the newly published listing.
    Sends automatic emails to matching buyers without duplicate notifications.
    Returns the count of buyers notified.
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM listings WHERE id = ? AND is_deleted = 0 AND status = 'Available'", (listing_id,))
    listing = cursor.fetchone()
    if not listing:
        conn.close()
        return 0

    cursor.execute("SELECT * FROM livestock_availability_alerts WHERE active = 1")
    alerts = cursor.fetchall()

    notified_count = 0
    now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for alert in alerts:
        buyer_id = alert['buyer_id']
        buyer_email = alert['buyer_email']

        # Prevent duplicate notification for the same buyer email and listing
        cursor.execute("SELECT id FROM livestock_availability_notifications WHERE (email = ? OR (buyer_id IS NOT NULL AND buyer_id = ?)) AND listing_id = ?", (buyer_email, buyer_id, listing_id))
        if cursor.fetchone():
            continue

        # Match criteria
        match = True

        # Species check
        if alert['species'] and alert['species'].strip().lower() not in ['', 'all']:
            if str(listing['species']).strip().lower() != alert['species'].strip().lower():
                match = False

        # Breed check
        if match and alert['breed'] and alert['breed'].strip().lower() not in ['', 'all']:
            if str(listing['breed']).strip().lower() != alert['breed'].strip().lower():
                match = False

        # District check
        if match and alert['district'] and alert['district'].strip().lower() not in ['', 'all']:
            if str(listing['district']).strip().lower() != alert['district'].strip().lower():
                match = False

        # Milk production check
        if match and alert['minimum_milk_production']:
            try:
                min_m = float(alert['minimum_milk_production'])
                if float(listing['milk_production']) < min_m:
                    match = False
            except (ValueError, TypeError):
                pass

        # Maximum milk production check (if applicable)
        if match and alert['maximum_milk_production']:
            try:
                max_m = float(alert['maximum_milk_production'])
                if float(listing['milk_production']) > max_m:
                    match = False
            except (ValueError, TypeError):
                pass

        if match:
            # Send notification email
            ok, msg = send_livestock_available_notification(buyer_email, dict(listing))
            status = 'SENT' if ok else 'FAILED'
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO livestock_availability_notifications (buyer_id, listing_id, email, sent_at, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (buyer_id, listing_id, buyer_email, now_ts, status))
                conn.commit()
                if ok:
                    notified_count += 1
            except Exception as e:
                print(f"[Alert Notification Error] Failed to log alert notification: {e}")

    conn.close()
    return notified_count

def log_and_dispatch_notification(user_id: int, user_email: str, user_phone: str, order_id: int, subject: str, message: str) -> dict:
    """
    Record and dispatch multi-channel notification (Email + SMS) to user.
    Records delivery status in database notifications table.
    """
    conn = get_db()
    cursor = conn.cursor()

    email_status = "NOT_ATTEMPTED"
    sms_status = "NOT_ATTEMPTED"

    # 1. Send Email if email is valid
    if user_email and is_valid_email_format(user_email):
        html_body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px;">
            <h3 style="color: #1e293b; border-bottom: 2px solid #3b82f6; padding-bottom: 8px;">{subject}</h3>
            <p style="font-size: 15px; color: #334155; line-height: 1.5;">{message}</p>
            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 20px 0;">
            <p style="font-size: 11px; color: #94a3b8;">LivestockIQ Notification</p>
        </div>
        """
        ok, res_msg = send_email(user_email, f"[LivestockIQ] {subject}", html_body, message)
        email_status = "SENT" if ok else "FAILED"

    # 2. Dispatch SMS to Indian phone
    norm_phone = normalize_indian_phone(user_phone) if user_phone else ""
    if norm_phone and is_valid_indian_phone(norm_phone):
        try:
            safe_msg = message[:80].replace('₹', 'Rs. ').encode('ascii', errors='replace').decode('ascii')
            print(f"[SMS Gateway] Dispatched to {norm_phone}: {safe_msg}...")
            sms_status = "SENT"
        except Exception:
            sms_status = "FAILED"

    # 3. Store notification record in database
    cursor.execute('''
        INSERT INTO notifications (user_id, user_email, user_phone, order_id, subject, message, email_status, sms_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, user_email, norm_phone, order_id, subject, message, email_status, sms_status))
    conn.commit()
    conn.close()

    return {
        'email_status': email_status,
        'sms_status': sms_status
    }
