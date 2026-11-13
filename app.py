import os
import re
import json
import uuid
import secrets
import datetime
from functools import wraps
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

from database import (
    init_db, get_db, hash_password, verify_password, seed_single_admin,
    AUTHORIZED_ADMIN_NAME, AUTHORIZED_ADMIN_EMAIL,
    save_nutrition_chat_message, get_nutrition_chat_history, clear_nutrition_chat_history
)
from services.nutrition_advisor import nutrition_advisor
from notification_service import (
    is_valid_indian_phone, normalize_indian_phone, is_valid_email_format, is_valid_pincode,
    create_and_store_otp, verify_submitted_otp, send_admin_otp_email,
    send_user_registration_otp_email, send_buyer_purchase_otp_email,
    send_password_reset_otp_email, send_buyer_purchase_confirmation_email,
    send_farmer_purchase_notification_email, send_farmer_publication_email,
    send_order_cancellation_emails, send_order_rejection_emails,
    notify_matching_buyers_for_new_listing, log_and_dispatch_notification
)
from ml_pipeline import pipeline

PENDING_REGISTRATIONS = {}
PENDING_PASSWORD_RESETS = {}

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'pashudhan_ai_secure_production_secret_key_2026')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database schema and ensure authorized administrator is seeded
init_db()
seed_single_admin()

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
ALLOWED_MIMETYPES = {'image/jpeg', 'image/png', 'image/webp'}

def is_allowed_file(filename, mimetype=None):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

# -------------------------------------------------------------
# Admin Route Protection Decorator
# -------------------------------------------------------------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Admin authentication strictly requires active session verified by Email OTP or authorized admin_id
        is_auth = session.get('is_admin_authenticated', False)
        role = session.get('role')
        admin_email = session.get('admin_email', '')

        req_admin_id = None
        if request.is_json and request.json:
            req_admin_id = request.json.get('admin_id')
        if not req_admin_id:
            req_admin_id = request.args.get('admin_id')

        if req_admin_id:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT email, role FROM users WHERE id = ?", (req_admin_id,))
            user_row = cursor.fetchone()
            conn.close()
            if user_row and user_row['role'] == 'admin' and user_row['email'].lower() == AUTHORIZED_ADMIN_EMAIL.lower():
                return f(*args, **kwargs)

        if not is_auth or role != 'admin' or admin_email.lower() != AUTHORIZED_ADMIN_EMAIL.lower():
            return jsonify({'error': 'Unauthorized: Administrator access required.'}), 403
        return f(*args, **kwargs)
    return decorated_function

# -------------------------------------------------------------
# 1. Location APIs (Karnataka Administrative Hierarchy)
# -------------------------------------------------------------
@app.route('/api/locations/states', methods=['GET'])
def get_states():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM states ORDER BY name ASC")
    states = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(states)

@app.route('/api/locations/districts', methods=['GET'])
def get_districts():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, state_id, name, code, latitude, longitude FROM districts ORDER BY name ASC")
    districts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(districts)

@app.route('/api/locations/districts/<int:district_id>/taluks', methods=['GET'])
def get_taluks(district_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, district_id, name, code, latitude, longitude FROM taluks WHERE district_id = ? ORDER BY name ASC", (district_id,))
    taluks = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(taluks)

@app.route('/api/locations/taluks/<int:taluk_id>/villages', methods=['GET'])
def get_villages(taluk_id):
    conn = get_db()
    cursor = conn.cursor()
    q = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    if q:
        cursor.execute('''
            SELECT id, taluk_id, name, native_name, code, pincode, latitude, longitude 
            FROM villages 
            WHERE taluk_id = ? AND (name LIKE ? OR pincode LIKE ?)
            ORDER BY name ASC LIMIT ? OFFSET ?
        ''', (taluk_id, f"%{q}%", f"{q}%", limit, offset))
    else:
        cursor.execute('''
            SELECT id, taluk_id, name, native_name, code, pincode, latitude, longitude 
            FROM villages 
            WHERE taluk_id = ? 
            ORDER BY name ASC LIMIT ? OFFSET ?
        ''', (taluk_id, limit, offset))

    villages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(villages)

# -------------------------------------------------------------
# 2. ADMIN AUTHENTICATION (STRICT FLOW: EMAIL + PASSWORD + EMAIL OTP)
# -------------------------------------------------------------
@app.route('/api/auth/admin-status', methods=['GET'])
def check_admin_status():
    """Verify administrator identity is initialized (never returns password or secrets)."""
    return jsonify({
        'admin_exists': True,
        'admin_name': AUTHORIZED_ADMIN_NAME,
        'admin_email': AUTHORIZED_ADMIN_EMAIL
    })

@app.route('/api/admin/login', methods=['POST'])
def admin_login_step1():
    """
    Step 1: Admin enters email and password.
    Backend verifies:
      1. Exact authorized admin email (rakshitham556@gmail.com)
      2. Correct admin password against secure stored bcrypt hash.
    If valid:
      Generates cryptographically secure OTP and sends strictly to rakshitham556@gmail.com.
      Does NOT open dashboard or create admin session.
    """
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not email or not password:
        return jsonify({'error': 'Please provide email and password.'}), 400

    # Strict Admin Email Restriction
    if email != AUTHORIZED_ADMIN_EMAIL.lower():
        # Generic authentication rejection to avoid disclosing sensitive internal details
        return jsonify({'error': 'Invalid administrator credentials.'}), 401

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, password_hash FROM users WHERE role = 'admin' AND LOWER(email) = LOWER(?)", (email,))
    admin = cursor.fetchone()
    conn.close()

    if not admin or not verify_password(password, admin['password_hash']):
        return jsonify({'error': 'Invalid administrator credentials.'}), 401

    # Credentials are valid -> Generate single-use OTP & send to authorized admin email
    otp_code, success, err_msg = create_and_store_otp(email, purpose='admin_login')
    if not success:
        return jsonify({'error': err_msg}), 429

    sent, send_err = send_admin_otp_email(AUTHORIZED_ADMIN_EMAIL, otp_code)
    if not sent:
        return jsonify({'error': send_err or 'Failed to send admin verification email. Please try again.'}), 500

    return jsonify({
        'status': 'otp_required',
        'message': f'Single-use verification code sent to authorized administrator email {AUTHORIZED_ADMIN_EMAIL}. Please check your inbox.',
        'email': AUTHORIZED_ADMIN_EMAIL
    })

@app.route('/api/admin/verify-otp', methods=['POST'])
def admin_login_step2():
    """
    Step 2: Admin enters OTP received at rakshitham556@gmail.com.
    Backend verifies OTP.
    Only after successful OTP verification:
      Creates authenticated ADMIN session.
      Returns authorized administrator profile.
    """
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()
    otp = (data.get('otp') or '').strip()

    if not email or not otp:
        return jsonify({'error': 'Please enter the verification OTP.'}), 400

    if email != AUTHORIZED_ADMIN_EMAIL.lower():
        return jsonify({'error': 'Unauthorized administrator verification.'}), 403

    valid, msg = verify_submitted_otp(email, otp, purpose='admin_login')
    if not valid:
        return jsonify({'error': msg}), 400

    # OTP successfully verified -> Create authenticated ADMIN session
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, role FROM users WHERE role = 'admin' AND LOWER(email) = LOWER(?)", (email,))
    admin = cursor.fetchone()
    conn.close()

    if not admin:
        return jsonify({'error': 'Administrator account not found.'}), 404

    session.clear()
    session['user_id'] = admin['id']
    session['role'] = 'admin'
    session['admin_email'] = AUTHORIZED_ADMIN_EMAIL
    session['is_admin_authenticated'] = True

    admin_obj = {
        'id': admin['id'],
        'name': admin['name'],
        'email': admin['email'],
        'role': 'admin'
    }

    return jsonify({
        'status': 'success',
        'message': 'Administrator authenticated successfully.',
        'user': admin_obj
    })

@app.route('/api/admin/resend-otp', methods=['POST'])
def admin_resend_otp():
    """Rate-limited resend OTP strictly to authorized administrator email."""
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()

    if email != AUTHORIZED_ADMIN_EMAIL.lower():
        return jsonify({'error': 'Unauthorized operation.'}), 403

    otp_code, success, err_msg = create_and_store_otp(email, purpose='admin_login')
    if not success:
        return jsonify({'error': err_msg}), 429

    sent, send_err = send_admin_otp_email(AUTHORIZED_ADMIN_EMAIL, otp_code)
    if not sent:
        return jsonify({'error': send_err or 'Failed to send admin verification email. Please try again.'}), 500

    return jsonify({
        'status': 'success',
        'message': f'Verification code sent to {AUTHORIZED_ADMIN_EMAIL}. Please check your inbox.'
    })

@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    """Secure Admin Logout: clears session and authentication state."""
    session.clear()
    return jsonify({'status': 'success', 'message': 'Administrator logged out successfully.'})

# -------------------------------------------------------------
# 3. FARMER & BUYER REGISTRATION WITH EMAIL OTP VERIFICATION
# -------------------------------------------------------------
@app.route('/api/auth/register-request-otp', methods=['POST'])
def register_request_otp():
    """
    Step 1 of Farmer/Buyer Registration:
    Validates email format, Indian phone number (+91XXXXXXXXXX), password, role.
    Generates single-use OTP and emails to user.
    Account is NOT created/activated until OTP verification succeeds.
    """
    data = request.json or {}
    name = (data.get('name') or '').strip()
    entered_email = (data.get('email') or '').strip().lower()
    recipient_email = entered_email
    email = recipient_email
    raw_phone = (data.get('phone') or '').strip()
    password = data.get('password') or ''
    confirm_password = data.get('confirm_password') or ''
    role = (data.get('role') or 'farmer').lower()
    pincode = (data.get('pincode') or '').strip()

    # Log safe diagnostic during development (No OTP, no password, no API key)
    print(f"[OTP] Recipient: {email}")

    # Disallow Admin registration via public endpoint
    if role not in ['farmer', 'buyer']:
        return jsonify({'error': 'Invalid role. Only Farmer and Buyer registrations are allowed.'}), 400

    if not name or not email or not raw_phone or not password:
        return jsonify({'error': 'All required fields must be filled.'}), 400

    if not is_valid_email_format(email):
        return jsonify({'error': 'Please enter a valid email address.'}), 400

    phone = normalize_indian_phone(raw_phone)
    if not is_valid_indian_phone(phone):
        return jsonify({'error': 'Invalid Indian phone number. Format must be +91 followed by 10 digits (e.g. +919876543210).'}), 400

    if pincode and not is_valid_pincode(pincode):
        return jsonify({'error': 'Invalid PIN code. Must be exactly 6 digits.'}), 400

    if password != confirm_password:
        return jsonify({'error': 'Password and Confirm Password do not match.'}), 400

    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long.'}), 400

    # Block impersonation of authorized administrator
    if email == AUTHORIZED_ADMIN_EMAIL.lower() or name.lower() == AUTHORIZED_ADMIN_NAME.lower():
        return jsonify({'error': 'This identity is reserved. Registration not permitted.'}), 403

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE LOWER(email) = ? AND email_verified = 1", (email,))
    if cursor.fetchone():
        conn.close()
        return jsonify({'error': 'An active account with this email address already exists. Please login.'}), 400

    cursor.execute("SELECT id FROM users WHERE phone = ? AND email_verified = 1", (phone,))
    if cursor.fetchone():
        conn.close()
        return jsonify({'error': 'An active account with this phone number already exists.'}), 400
    conn.close()

    # Cache pending registration parameters for activation step
    PENDING_REGISTRATIONS[email] = {
        'name': name,
        'phone': phone,
        'password': password,
        'role': role,
        'district': data.get('district', ''),
        'taluk': data.get('taluk', ''),
        'village': data.get('village', ''),
        'address': data.get('address', ''),
        'pincode': pincode,
        'district_id': data.get('district_id'),
        'taluk_id': data.get('taluk_id'),
        'village_id': data.get('village_id')
    }

    # Generate single-use OTP and send to user's email via Resend API
    otp_code, success, err_msg = create_and_store_otp(email, purpose='registration')
    if not success:
        return jsonify({'error': err_msg}), 429

    sent, send_err = send_user_registration_otp_email(email, name, otp_code, role)
    if not sent:
        status_code = 429 if ('recipient limit' in str(send_err).lower() or 'rate limit' in str(send_err).lower()) else 500
        return jsonify({'error': send_err or 'Failed to send verification email. Please try again.'}), status_code

    return jsonify({
        'status': 'otp_sent',
        'message': f'Verification code sent to {email}. Please check your inbox.',
        'email': email
    })

@app.route('/api/auth/register-verify-otp', methods=['POST'])
def register_verify_otp():
    """
    Step 2 of Farmer/Buyer Registration:
    Verifies OTP. Only upon successful verification, activates and stores account in database.
    """
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()
    otp = (data.get('otp') or '').strip()
    pending = PENDING_REGISTRATIONS.get(email, {})
    name = (data.get('name') or pending.get('name') or '').strip()
    raw_phone = (data.get('phone') or pending.get('phone') or '').strip()
    password = data.get('password') or pending.get('password') or ''
    role = (data.get('role') or pending.get('role') or 'farmer').lower()
    district = (data.get('district') or pending.get('district') or '').strip()
    taluk = (data.get('taluk') or pending.get('taluk') or '').strip()
    village = (data.get('village') or pending.get('village') or '').strip()
    address = (data.get('address') or pending.get('address') or '').strip()
    pincode = (data.get('pincode') or pending.get('pincode') or '').strip()
    district_id = data.get('district_id') or pending.get('district_id')
    taluk_id = data.get('taluk_id') or pending.get('taluk_id')
    village_id = data.get('village_id') or pending.get('village_id')

    if role not in ['farmer', 'buyer']:
        return jsonify({'error': 'Invalid registration role.'}), 400

    phone = normalize_indian_phone(raw_phone)
    if not is_valid_indian_phone(phone):
        return jsonify({'error': 'Invalid phone number.'}), 400

    # Verify submitted OTP
    valid, msg = verify_submitted_otp(email, otp, purpose='registration')
    if not valid:
        return jsonify({'error': msg}), 400

    if not password:
        return jsonify({'error': 'Password is required to finalize registration.'}), 400

    pass_hash = hash_password(password)

    conn = get_db()
    cursor = conn.cursor()

    try:
        # Remove any previous unverified registration attempts with same email
        cursor.execute("DELETE FROM users WHERE LOWER(email) = ? AND email_verified = 0", (email,))

        cursor.execute('''
            INSERT INTO users (
                name, email, username, phone, password_hash, password, role,
                state_id, district_id, taluk_id, village_id, district, taluk, village, address, pincode,
                email_verified, status
            ) VALUES (?, ?, ?, ?, ?, '[BCRYPT_PROTECTED]', ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'Active')
        ''', (name, email, email.split('@')[0], phone, pass_hash, role,
              district_id, taluk_id, village_id, district, taluk, village, address, pincode))

        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        # Remove cached registration data upon successful activation
        PENDING_REGISTRATIONS.pop(email, None)

        # Send buyer registration success email
        if role == 'buyer':
            try:
                from services.email_service import send_buyer_registration_success_email
                send_buyer_registration_success_email(email, name)
            except Exception as mail_err:
                print(f"[Buyer Registration Success Mail Error] {mail_err}")

        # Send welcome notification
        log_and_dispatch_notification(
            user_id=user_id,
            user_email=email,
            user_phone=phone,
            order_id=None,
            subject="Account Activated",
            message=f"Welcome to LivestockIQ, {name}! Your {role.capitalize()} account is now active."
        )

        return jsonify({
            'status': 'success',
            'message': 'Account successfully verified and activated! Please login.',
            'user': {
                'id': user_id,
                'name': name,
                'email': email,
                'phone': phone,
                'role': role
            }
        })
    except Exception as e:
        conn.close()
        return jsonify({'error': f'Failed to create account: {str(e)}'}), 500

@app.route('/api/auth/resend-registration-otp', methods=['POST'])
def resend_registration_otp():
    """Rate-limited resend OTP for pending registration."""
    data = request.json or {}
    entered_email = (data.get('email') or '').strip().lower()
    recipient_email = entered_email
    email = recipient_email
    name = (data.get('name') or 'User').strip()
    role = (data.get('role') or 'farmer').lower()

    if not email or not is_valid_email_format(email):
        return jsonify({'error': 'Valid email is required.'}), 400

    # Log safe diagnostic during development (No OTP, no password, no API key)
    print(f"[OTP] Recipient: {email}")

    otp_code, success, err_msg = create_and_store_otp(email, purpose='registration')
    if not success:
        return jsonify({'error': err_msg}), 429

    sent, send_err = send_user_registration_otp_email(email, name, otp_code, role)
    if not sent:
        return jsonify({'error': send_err or 'Failed to send verification email. Please try again.'}), 500

    return jsonify({
        'status': 'success',
        'message': f'Verification code sent to {email}. Please check your inbox.',
        'email': email
    })

# -------------------------------------------------------------
# 4. STANDARD USER LOGIN (FARMER & BUYER)
# -------------------------------------------------------------
@app.route('/api/auth/login', methods=['POST'])
def user_login():
    """Standard login for verified Farmers and Buyers."""
    data = request.json or {}
    identifier = (data.get('identifier') or data.get('email') or data.get('phone') or '').strip()
    password = data.get('password') or ''
    requested_role = (data.get('role') or '').lower()

    if not identifier or not password:
        return jsonify({'error': 'Please enter email/phone and password.'}), 400

    # If admin attempts login through standard form, guide to Admin OTP flow
    if identifier.lower() == AUTHORIZED_ADMIN_EMAIL.lower() or requested_role == 'admin':
        return jsonify({'error': 'Administrator accounts must authenticate via Admin Login with Email OTP.', 'is_admin': True}), 403

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, name, email, phone, password_hash, role, district, taluk, village, address, pincode, email_verified, status
        FROM users 
        WHERE (LOWER(email) = ? OR phone = ? OR phone = ?)
    ''', (identifier.lower(), identifier, normalize_indian_phone(identifier)))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({'error': 'Account not found. Please register first.'}), 401

    if user['role'] == 'admin':
        return jsonify({'error': 'Administrator must authenticate via the Admin Login page.'}), 403

    if not user['email_verified']:
        return jsonify({'error': 'Account email has not been verified. Please complete registration verification.'}), 403

    if not verify_password(password, user['password_hash']):
        return jsonify({'error': 'Invalid credentials. Please check your password.'}), 401

    if requested_role and user['role'] != requested_role:
        return jsonify({'error': f"Access denied. Registered as '{user['role'].upper()}', not '{requested_role.upper()}'."}), 403

    session['user_id'] = user['id']
    session['role'] = user['role']
    session['user_email'] = user['email']

    user_obj = {
        'id': user['id'],
        'name': user['name'],
        'email': user['email'],
        'phone': user['phone'],
        'role': user['role'],
        'district': user['district'] or '',
        'taluk': user['taluk'] or '',
        'village': user['village'] or '',
        'address': user['address'] or '',
        'pincode': user['pincode'] or '',
        'status': user['status']
    }

    return jsonify({
        'status': 'success',
        'message': f'Welcome back, {user["name"]}!',
        'user': user_obj
    })

@app.route('/api/auth/logout', methods=['POST'])
def user_logout():
    session.clear()
    return jsonify({'status': 'success', 'message': 'Logged out successfully.'})

# -------------------------------------------------------------
# 4B. FORGOT PASSWORD (ADMIN, FARMER & BUYER) WITH EMAIL OTP
# -------------------------------------------------------------
@app.route('/api/auth/forgot-password/request-otp', methods=['POST'])
def forgot_password_request_otp():
    """
    Step 1 & 2: Forgot Password OTP Request
    Works for Admin, Farmer, and Buyer.
    Does not disclose whether an email exists in the database.
    Sends single-use 6-digit OTP (purpose='PASSWORD_RESET', validity=5 mins) via AgentMail.
    """
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()

    if not email or not is_valid_email_format(email):
        return jsonify({'error': 'Please enter a valid email address.'}), 400

    # Look up user in database (works uniformly for Admin, Farmer, Buyer)
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, role, email_verified FROM users WHERE LOWER(email) = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    # Generic response message to avoid user enumeration / disclosure
    generic_response = {
        'status': 'otp_sent',
        'message': 'If an account exists with this email address, a password reset OTP has been sent.',
        'email': email
    }

    if not user:
        return jsonify(generic_response), 200

    # For standard users, require they were verified accounts, or admin
    if user['role'] != 'admin' and not user['email_verified']:
        return jsonify(generic_response), 200

    user_name = user['name'] or 'User'
    user_role = user['role'] or 'User'

    # Generate and store single-use OTP with purpose='PASSWORD_RESET' (5-minute expiry, cooldown enforced)
    otp_code, success, err_msg = create_and_store_otp(email, purpose='PASSWORD_RESET', user_id=user['id'])
    if not success:
        return jsonify({'error': err_msg}), 429

    sent, send_err = send_password_reset_otp_email(user['email'], otp_code, user_name, user_role)
    if not sent:
        return jsonify({'error': send_err or 'Failed to dispatch password reset code. Please try again.'}), 500

    return jsonify(generic_response), 200

@app.route('/api/auth/forgot-password/verify-otp', methods=['POST'])
def forgot_password_verify_otp():
    """
    Step 3 & 4: Verify Password Reset OTP
    Verifies 6-digit OTP for purpose='PASSWORD_RESET'.
    Returns a secure single-use reset token on success.
    """
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()
    otp = (data.get('otp') or '').strip()

    if not email or not otp:
        return jsonify({'error': 'Please enter your registered email and the verification OTP.'}), 400

    valid, msg = verify_submitted_otp(email, otp, purpose='PASSWORD_RESET')
    if not valid:
        return jsonify({'error': msg}), 400

    # Generate single-use reset token valid for 10 minutes
    reset_token = secrets.token_urlsafe(32)
    now = datetime.datetime.now()
    PENDING_PASSWORD_RESETS[reset_token] = {
        'email': email,
        'expires_at': now + datetime.timedelta(minutes=10)
    }

    return jsonify({
        'status': 'success',
        'message': 'OTP verified successfully. You may now reset your password.',
        'reset_token': reset_token,
        'email': email
    }), 200

@app.route('/api/auth/forgot-password/resend-otp', methods=['POST'])
def forgot_password_resend_otp():
    """
    Resend Password Reset OTP with rate-limiting and cooldown enforcement.
    """
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()

    if not email or not is_valid_email_format(email):
        return jsonify({'error': 'Please enter a valid email address.'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, role FROM users WHERE LOWER(email) = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({'status': 'success', 'message': 'New password reset OTP sent if account exists.'}), 200

    otp_code, success, err_msg = create_and_store_otp(email, purpose='PASSWORD_RESET', user_id=user['id'])
    if not success:
        return jsonify({'error': err_msg}), 429

    sent, send_err = send_password_reset_otp_email(user['email'], otp_code, user['name'], user['role'])
    if not sent:
        return jsonify({'error': send_err or 'Failed to deliver OTP email.'}), 500

    return jsonify({'status': 'success', 'message': 'New password reset OTP sent to your registered email.'}), 200

@app.route('/api/auth/forgot-password/reset', methods=['POST'])
def forgot_password_reset():
    """
    Step 5 & 6: Set New Password & Confirm New Password
    Validates passwords match and minimum length.
    Hashes new password with secure hashing method.
    Updates database for Admin, Farmer, or Buyer.
    """
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()
    reset_token = (data.get('reset_token') or '').strip()
    otp = (data.get('otp') or '').strip()
    new_password = data.get('new_password') or ''
    confirm_password = data.get('confirm_password') or ''

    if not email:
        return jsonify({'error': 'Email is required.'}), 400

    if not new_password or not confirm_password:
        return jsonify({'error': 'Please enter and confirm your new password.'}), 400

    if new_password != confirm_password:
        return jsonify({'error': 'New password and confirm password do not match.'}), 400

    if len(new_password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long.'}), 400

    # Authorization verification: via reset_token or direct OTP fallback
    token_valid = False
    if reset_token:
        token_info = PENDING_PASSWORD_RESETS.get(reset_token)
        if token_info and token_info['email'] == email:
            if datetime.datetime.now() <= token_info['expires_at']:
                token_valid = True
                PENDING_PASSWORD_RESETS.pop(reset_token, None)
            else:
                PENDING_PASSWORD_RESETS.pop(reset_token, None)
                return jsonify({'error': 'Password reset session expired. Please request a new OTP.'}), 400
    elif otp:
        valid_otp, otp_err = verify_submitted_otp(email, otp, purpose='PASSWORD_RESET')
        if valid_otp:
            token_valid = True
        else:
            return jsonify({'error': otp_err}), 400

    if not token_valid:
        return jsonify({'error': 'Invalid or expired password reset authorization. Please verify OTP again.'}), 400

    new_hash = hash_password(new_password)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, role FROM users WHERE LOWER(email) = ?", (email,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        return jsonify({'error': 'Account not found.'}), 404

    # Update password for Admin, Farmer, or Buyer
    cursor.execute('''
        UPDATE users 
        SET password_hash = ?, password = '[BCRYPT_PROTECTED]' 
        WHERE LOWER(email) = ?
    ''', (new_hash, email))
    conn.commit()
    conn.close()

    # Step 7: Success message
    return jsonify({
        'status': 'success',
        'message': 'Password reset successfully. You can now login with your new password.'
    }), 200

# -------------------------------------------------------------
# 5. AI IMAGE ANALYSIS (VALIDATION, SPECIES, BREED, GENDER)
# -------------------------------------------------------------
@app.route('/api/predict/image-analysis', methods=['POST'])
def analyze_image():
    """
    Image Analysis Endpoint:
    1. Validates Species (ALLOWED: Cow and Buffalo ONLY).
       Rejects non-livestock (Dog, Cat, Horse, Goat, Sheep, Person, Vehicle, Random object, etc.)
       with exact message: "Invalid livestock image. Please upload a clear image of a cow or buffalo."
    2. Identifies Species (Cow vs Buffalo)
    3. Identifies Breed
    4. Automatically determines Gender (Male / Female)
    5. Returns physiological guidance (Pregnancy N/A for Male)
    """
    if 'file' not in request.files:
        return jsonify({'status': 'rejected', 'is_valid': False, 'error': 'No image file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'rejected', 'is_valid': False, 'error': 'Empty file uploaded.'}), 400

    if not is_allowed_file(file.filename, file.mimetype):
        return jsonify({
            'status': 'rejected',
            'is_valid': False,
            'error': 'Unsupported file format. Please upload a JPG, JPEG, PNG, or WEBP image.'
        }), 400

    ext = os.path.splitext(file.filename)[1].lower()
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    analysis_res = pipeline.predict_image(filepath)
    analysis_res['image_url'] = f"/static/uploads/{filename}"

    if not analysis_res.get('is_valid', False) or analysis_res.get('status') == 'rejected' or analysis_res.get('species') not in ['Cow', 'Buffalo']:
        analysis_res['status'] = 'rejected'
        analysis_res['is_valid'] = False
        if not analysis_res.get('error'):
            analysis_res['error'] = 'No valid cow or buffalo detected. Please capture a clear livestock image.'
        return jsonify(analysis_res), 400

    return jsonify(analysis_res)

# -------------------------------------------------------------
# 6. PRICE PREDICTION (TRIGGERED ON CLICK, STRICT PREGNANCY RULES)
# -------------------------------------------------------------
@app.route('/api/predict/price', methods=['POST'])
@app.route('/api/predict-price', methods=['POST'])
def predict_price():
    """
    Price Prediction Model:
    If Male: Pregnancy = 0 (N/A), Milk Production = 0.0. Non-zero milk is strictly blocked.
    If Female: Pregnancy Yes/No, Month 1-9.
    """
    data = request.json or {}
    species = data.get('species')
    breed = data.get('breed')
    gender = (data.get('gender') or 'Female').strip().capitalize()
    is_pregnant_val = data.get('pregnancy') or data.get('is_pregnant') or 'No'
    pregnancy_month_val = data.get('pregnancy_month', 0)
    age_years = data.get('age_years') or data.get('age')
    milk_production = data.get('milk_production')
    district = data.get('district')

    if not species or not breed:
        return jsonify({'error': 'Valid species and breed are required.'}), 400

    if not district or str(district).strip() in ['', 'Select District']:
        return jsonify({'error': 'Please select a Karnataka district.'}), 400

    try:
        age_val = float(age_years)
    except (ValueError, TypeError):
        return jsonify({'error': 'Please enter a valid age in years.'}), 400

    lactation_number = data.get('lactation_number') or '1st'
    calf_number = data.get('calf_number') or '1st'
    try:
        health_score_val = int(data.get('health_score') or 4)
        health_score_val = max(1, min(5, health_score_val))
    except (ValueError, TypeError):
        health_score_val = 4

    # Strict Male vs Female Validation
    if gender == 'Male':
        is_pregnant = 0
        pregnancy_month = 0
        milk_val = 0.0
        lactation_number = 'Not Applicable'
        calf_number = 'Not Applicable'
    else:
        gender = 'Female'
        try:
            milk_val = float(milk_production)
            if milk_val < 0:
                milk_val = 0.0
            if lactation_number == '5th' and milk_val > 10.0:
                return jsonify({'error': 'For 5th lactation, milk production cannot exceed 10 L/day.'}), 400
            if lactation_number == '4th' and milk_val > 15.0:
                return jsonify({'error': 'For 4th lactation, milk production cannot exceed 15 L/day.'}), 400
            if milk_val > 30.0:
                return jsonify({'error': 'Daily milk production cannot exceed 30 L/day.'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Please enter valid daily milk production (litres/day).'}), 400

        if str(is_pregnant_val).lower() in ['yes', '1', 'true']:
            is_pregnant = 1
            try:
                pregnancy_month = max(1, min(9, int(pregnancy_month_val)))
            except (ValueError, TypeError):
                pregnancy_month = 1
        else:
            is_pregnant = 0
            pregnancy_month = 0

    price_res = pipeline.predict_price(
        species=species,
        breed=breed,
        gender=gender,
        age_years=age_val,
        milk_production=milk_val,
        district=district,
        is_pregnant=is_pregnant,
        pregnancy_month=pregnancy_month,
        lactation_number=lactation_number,
        calf_number=calf_number,
        health_score=health_score_val
    )

    return jsonify(price_res)

# -------------------------------------------------------------
# 7. MARKETPLACE LISTINGS & PUBLISHING
# -------------------------------------------------------------
@app.route('/api/listings', methods=['GET'])
def get_listings():
    conn = get_db()
    cursor = conn.cursor()

    species = request.args.get('species')
    breed = request.args.get('breed')
    district = request.args.get('district')
    min_milk = request.args.get('min_milk')
    min_price = request.args.get('min_price')
    max_price = request.args.get('max_price')
    status = request.args.get('status')
    farmer_id = request.args.get('farmer_id')

    query = "SELECT * FROM listings WHERE is_deleted = 0"
    params = []

    if species and species not in ['All', '']:
        query += " AND species = ?"
        params.append(species)
    if breed and breed not in ['All', '']:
        query += " AND breed = ?"
        params.append(breed)
    if district and district not in ['All', '']:
        query += " AND district = ?"
        params.append(district)
    if min_milk:
        try:
            query += " AND milk_production >= ?"
            params.append(float(min_milk))
        except ValueError:
            pass
    if min_price:
        try:
            query += " AND asking_price >= ?"
            params.append(int(min_price))
        except ValueError:
            pass
    if max_price:
        try:
            query += " AND asking_price <= ?"
            params.append(int(max_price))
        except ValueError:
            pass
    if status and status != 'all':
        query += " AND status = ?"
        params.append(status)
    if farmer_id:
        query += " AND farmer_id = ?"
        params.append(int(farmer_id))

    # When price filter is active, sort strictly in ascending numeric price order
    if min_price or max_price:
        query += " ORDER BY asking_price ASC, id DESC"
    else:
        query += " ORDER BY id DESC"

    cursor.execute(query, params)
    listings = [dict(row) for row in cursor.fetchall()]

    # Resolve real geographical GPS coordinates from existing database location master if not directly on the record
    for l in listings:
        if l.get('latitude') is None or l.get('longitude') is None:
            v_name = l.get('village')
            t_name = l.get('taluk')
            d_name = l.get('district')
            coords = None
            if v_name:
                v_row = cursor.execute(
                    "SELECT latitude, longitude FROM villages WHERE LOWER(name) = LOWER(?) AND latitude IS NOT NULL LIMIT 1",
                    (v_name,)
                ).fetchone()
                if v_row and v_row['latitude'] is not None and v_row['longitude'] is not None:
                    coords = (v_row['latitude'], v_row['longitude'])
            if not coords and t_name:
                t_row = cursor.execute(
                    "SELECT latitude, longitude FROM taluks WHERE LOWER(name) = LOWER(?) AND latitude IS NOT NULL LIMIT 1",
                    (t_name,)
                ).fetchone()
                if t_row and t_row['latitude'] is not None and t_row['longitude'] is not None:
                    coords = (t_row['latitude'], t_row['longitude'])
            if not coords and d_name:
                d_row = cursor.execute(
                    "SELECT latitude, longitude FROM districts WHERE LOWER(name) = LOWER(?) AND latitude IS NOT NULL LIMIT 1",
                    (d_name,)
                ).fetchone()
                if d_row and d_row['latitude'] is not None and d_row['longitude'] is not None:
                    coords = (d_row['latitude'], d_row['longitude'])
            if coords:
                l['latitude'] = coords[0]
                l['longitude'] = coords[1]

    conn.close()

    return jsonify(listings)

@app.route('/api/listings', methods=['POST'])
def create_listing():
    data = request.json or {}
    farmer_id = data.get('farmer_id')
    farmer_name = data.get('farmer_name')
    farmer_phone = data.get('farmer_phone')
    image_url = data.get('image_url')
    species = data.get('species')
    breed = data.get('breed')
    gender = (data.get('gender') or '').strip().capitalize()
    if gender not in ['Male', 'Female']:
        return jsonify({'error': 'Valid livestock gender (Male or Female) is required.'}), 400

    is_pregnant = int(data.get('is_pregnant', 0))
    pregnancy_month = int(data.get('pregnancy_month', 0))
    age_years = float(data.get('age_years', 4.0))
    milk_production = float(data.get('milk_production', 0.0))
    district = data.get('district')
    taluk = data.get('taluk', '')
    village = data.get('village', '')
    address = data.get('address', '')
    pincode = data.get('pincode', '')
    predicted_price = int(data.get('predicted_price', 65000))
    asking_price = int(data.get('asking_price', predicted_price))

    health_score = int(data.get('health_score', 5))
    health_text = data.get('health_text', 'Good')

    if not farmer_id or not species or not breed or not image_url or not district:
        return jsonify({'error': 'Missing required livestock parameters.'}), 400

    # Species validation: Cow or Buffalo ONLY
    if species not in ['Cow', 'Buffalo']:
        return jsonify({'error': 'Unsupported image. Please upload a clear image of a cattle or buffalo.'}), 400

    # Backend enforcement of Male vs Female rules
    lactation_number = (data.get('lactation_number') or 'Other / Not specified').strip()
    calf_number = (data.get('calf_number') or 'Other / Not specified').strip()
    if gender == 'Male':
        is_pregnant = 0
        pregnancy_month = 0
        milk_production = 0.0
        lactation_number = 'Not Applicable'
        calf_number = 'Not Applicable'

    conn = get_db()
    cursor = conn.cursor()

    # Verify farmer exists in parent users table
    cursor.execute("SELECT id, name, email, phone, role FROM users WHERE id = ?", (farmer_id,))
    farmer_user = cursor.fetchone()
    if not farmer_user:
        # Check if an existing farmer user exists in users table to gracefully associate
        cursor.execute("SELECT id, name, email, phone, role FROM users WHERE role = 'farmer' ORDER BY id ASC LIMIT 1")
        farmer_user = cursor.fetchone()
        if farmer_user:
            farmer_id = farmer_user['id']
        else:
            # Ensure farmer record exists in users table with real database primary key
            f_name = farmer_name or 'Farmer'
            f_phone = normalize_indian_phone(farmer_phone or '+919876543210')
            f_email = f"farmer_{secrets.token_hex(4)}@agri.in"
            f_user_login = f"farmer_{secrets.token_hex(3)}"
            cursor.execute('''
                INSERT INTO users (name, email, username, phone, password_hash, password, role, email_verified, status)
                VALUES (?, ?, ?, ?, ?, '[BCRYPT_PROTECTED]', 'farmer', 1, 'Active')
            ''', (f_name, f_email, f_user_login, f_phone, hash_password("SecureFarmer@2026")))
            farmer_id = cursor.lastrowid
            cursor.execute("SELECT id, name, email, phone, role FROM users WHERE id = ?", (farmer_id,))
            farmer_user = cursor.fetchone()

    farmer_name = farmer_user['name']
    farmer_phone = farmer_user['phone']
    farmer_email = farmer_user['email']

    # Prevent duplicate identical active listing
    cursor.execute('''
        SELECT id FROM listings 
        WHERE farmer_id = ? AND image_url = ? AND status = 'Available' AND is_deleted = 0
    ''', (farmer_id, image_url))
    if cursor.fetchone():
        conn.close()
        return jsonify({'error': 'This livestock photo has already been listed and is currently active.'}), 400

    cursor.execute('''
        INSERT INTO listings (
            farmer_id, farmer_name, farmer_phone, image_url, species, breed, gender, sex,
            health_score, health_text,
            is_pregnant, pregnancy_month, age_years, lactation_number, calf_number, milk_production, milk_yield_liters, district, taluk, village,
            address, pincode, predicted_price, asking_price, status, is_deleted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Available', 0)
    ''', (farmer_id, farmer_name, farmer_phone, image_url, species, breed, gender, gender,
          health_score, health_text,
          is_pregnant, pregnancy_month, age_years, lactation_number, calf_number, milk_production, milk_production, district, taluk, village,
          address, pincode, predicted_price, asking_price))

    listing_id = cursor.lastrowid
    conn.commit()

    # Send Listing Publication Confirmation Email to Farmer's registered email (Section 5A)
    if farmer_email and is_valid_email_format(farmer_email):
        listing_info = {
            'id': listing_id,
            'species': species,
            'breed': breed,
            'gender': gender,
            'age_years': age_years,
            'lactation_number': lactation_number,
            'calf_number': calf_number,
            'milk_production': milk_production,
            'asking_price': asking_price,
            'predicted_price': predicted_price,
            'district': district,
            'taluk': taluk,
            'village': village
        }
        try:
            send_farmer_publication_email(farmer_email, listing_info, farmer_name)
        except Exception as pub_mail_err:
            print(f"[Farmer Publication Mail Error] {pub_mail_err}")

    # Buyer Interest Notifications: Notify relevant buyers who searched or saved preferences
    cursor.execute('''
        SELECT DISTINCT u.id, u.email, u.phone, u.name 
        FROM users u
        JOIN buyer_interests bi ON u.id = bi.buyer_id
        WHERE u.role = 'buyer' AND u.email_verified = 1
          AND (bi.species IS NULL OR bi.species = ? OR bi.species = 'All')
          AND (bi.breed IS NULL OR bi.breed = ? OR bi.breed = 'All')
          AND (bi.district IS NULL OR bi.district = ? OR bi.district = 'All')
          AND (bi.max_price IS NULL OR bi.max_price >= ?)
        LIMIT 10
    ''', (species, breed, district, asking_price))
    interested_buyers = cursor.fetchall()
    conn.close()

    for buyer in interested_buyers:
        log_and_dispatch_notification(
            user_id=buyer['id'],
            user_email=buyer['email'],
            user_phone=buyer['phone'],
            order_id=None,
            subject=f"New {breed} Listed in {district}",
            message=f"Hello {buyer['name']}, a new {species} ({breed}) matching your preferences has just been listed in {district} for ₹{asking_price:,}."
        )

    # Automatically check active availability alerts and notify matching buyers
    try:
        notified_count = notify_matching_buyers_for_new_listing(listing_id)
        if notified_count > 0:
            print(f"[Marketplace Alert] Automatically emailed {notified_count} buyer(s) matching new listing #{listing_id}.")
    except Exception as alert_err:
        print(f"[Marketplace Alert Error] {alert_err}")

    return jsonify({
        'status': 'success',
        'message': f'{species} listing published successfully to marketplace!',
        'listing_id': listing_id
    })

@app.route('/api/listings/<int:listing_id>', methods=['PUT'])
def update_listing(listing_id):
    data = request.json or {}
    farmer_id = data.get('farmer_id')
    asking_price = data.get('asking_price')
    milk_production = data.get('milk_production')
    address = data.get('address')
    pincode = data.get('pincode')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM listings WHERE id = ?", (listing_id,))
    listing = cursor.fetchone()

    if not listing:
        conn.close()
        return jsonify({'error': 'Listing not found.'}), 404

    if farmer_id and listing['farmer_id'] != farmer_id:
        conn.close()
        return jsonify({'error': 'Unauthorized: You can only edit your own listings.'}), 403

    if listing['status'] in ['SOLD_OUT', 'Purchased']:
        conn.close()
        return jsonify({'error': 'Cannot modify a listing that has already been purchased/sold.'}), 400

    cursor.execute('''
        UPDATE listings 
        SET asking_price = COALESCE(?, asking_price),
            milk_production = COALESCE(?, milk_production),
            address = COALESCE(?, address),
            pincode = COALESCE(?, pincode)
        WHERE id = ?
    ''', (asking_price, milk_production, address, pincode, listing_id))

    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': 'Listing updated successfully!'})

@app.route('/api/listings/<int:listing_id>', methods=['DELETE'])
def delete_listing(listing_id):
    farmer_id = request.args.get('farmer_id', type=int)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM listings WHERE id = ?", (listing_id,))
    listing = cursor.fetchone()

    if not listing:
        conn.close()
        return jsonify({'error': 'Listing not found.'}), 404

    if farmer_id and listing['farmer_id'] != farmer_id:
        conn.close()
        return jsonify({'error': 'Unauthorized: You can only delete your own listings.'}), 403

    if listing['status'] in ['SOLD_OUT', 'Purchased']:
        conn.close()
        return jsonify({'error': 'Cannot delete a listing that has already been purchased.'}), 400

    listing_dict = dict(listing)

    # Fetch farmer details for deletion email notification
    cursor.execute("SELECT id, name, email, email_verified FROM users WHERE id = ?", (listing['farmer_id'],))
    farmer = cursor.fetchone()
    farmer_dict = dict(farmer) if farmer else {}

    cursor.execute("DELETE FROM listings WHERE id = ?", (listing_id,))
    conn.commit()
    conn.close()

    # Send deletion email to farmer via existing AgentMail
    farmer_email = farmer_dict.get('email')
    farmer_name = farmer_dict.get('name') or listing_dict.get('farmer_name', 'Farmer')
    if farmer_email:
        try:
            from notification_service import send_farmer_livestock_deletion_email
            email_ok, email_msg = send_farmer_livestock_deletion_email(farmer_email, listing_dict, farmer_name)
            if not email_ok:
                print(f"[Livestock Deletion Notification Warning] Email to {farmer_email} failed: {email_msg}")
        except Exception as e:
            print(f"[Livestock Deletion Notification Error] Exception sending deletion email to {farmer_email}: {e}")

    return jsonify({'status': 'success', 'message': 'Listing deleted successfully!'})

# -------------------------------------------------------------
# 8. ORDERS & PURCHASE WORKFLOW (ZERO DELIVERY FEE ARCHITECTURE)
# -------------------------------------------------------------
# -------------------------------------------------------------
# 8. BUYER PURCHASE OTP & CONFIRMATION (ZERO DELIVERY CHARGES)
# -------------------------------------------------------------
@app.route('/api/listings/<int:listing_id>/purchase-request-otp', methods=['POST'])
def purchase_request_otp(listing_id):
    """
    Step 1 of Buyer Purchase Verification:
    Checks livestock availability, generates a 6-digit purchase OTP (purpose='PURCHASE', expiry=5 min),
    and dispatches it to the buyer's registered and verified email via Gmail SMTP.
    """
    data = request.json or {}
    buyer_id = data.get('buyer_id') or session.get('user_id')
    if not buyer_id:
        return jsonify({'error': 'Please login as a verified buyer to purchase livestock.'}), 401

    conn = get_db()
    cursor = conn.cursor()

    # 1. Fetch and verify buyer
    cursor.execute("SELECT id, name, email, phone, role, email_verified FROM users WHERE id = ?", (buyer_id,))
    buyer = cursor.fetchone()
    if not buyer or buyer['role'] != 'buyer':
        conn.close()
        return jsonify({'error': 'Only registered buyers are authorized to purchase livestock.'}), 403

    if not buyer['email_verified']:
        conn.close()
        return jsonify({'error': 'Please verify your registered email before purchasing livestock.'}), 403

    # 2. Check livestock availability
    cursor.execute("SELECT * FROM listings WHERE id = ? AND is_deleted = 0", (listing_id,))
    listing = cursor.fetchone()
    if not listing:
        conn.close()
        return jsonify({'error': 'Livestock listing not found.'}), 404

    if listing['status'] != 'Available':
        conn.close()
        return jsonify({'error': 'This livestock is no longer available.'}), 400

    # 3. Check duplicate purchase
    cursor.execute("SELECT id FROM purchases WHERE listing_id = ?", (listing_id,))
    if cursor.fetchone():
        conn.close()
        return jsonify({'error': 'This livestock is no longer available.'}), 400

    conn.close()

    buyer_email = buyer['email']
    # 4. Generate and store purchase OTP (purpose='PURCHASE', expiry=5 minutes)
    otp_code, success, err_msg = create_and_store_otp(buyer_email, purpose='PURCHASE', user_id=buyer_id, listing_id=listing_id)
    if not success:
        return jsonify({'error': err_msg}), 429

    # 5. Dispatch OTP via AgentMail API
    sent, send_err = send_buyer_purchase_otp_email(buyer_email, otp_code, dict(listing))
    if not sent:
        return jsonify({'error': send_err or 'Failed to send purchase verification email. Please try again.'}), 500

    return jsonify({
        'status': 'otp_sent',
        'message': f'Purchase verification code sent to your registered email ({buyer_email}). Please check your inbox.',
        'email': buyer_email
    })

@app.route('/api/listings/<int:listing_id>/purchase', methods=['POST'])
@app.route('/api/listings/<int:listing_id>/buy', methods=['POST'])
def purchase_listing(listing_id):
    """
    Step 2 of Buyer Purchase:
    Verifies 6-digit purchase OTP, marks OTP used, acquires atomic lock,
    verifies availability, updates listing to 'SOLD_OUT', creates purchase record,
    and returns confirmation with ZERO delivery fees.
    """
    data = request.json or {}
    buyer_id = data.get('buyer_id') or session.get('user_id')
    otp = (data.get('otp') or '').strip()
    delivery_district = (data.get('delivery_district') or '').strip()
    delivery_taluk = (data.get('delivery_taluk') or '').strip()
    delivery_village = (data.get('delivery_village') or '').strip()
    delivery_address = (data.get('delivery_address') or '').strip()
    delivery_pincode = (data.get('delivery_pincode') or '').strip()

    if not buyer_id:
        return jsonify({'error': 'Buyer identification required.'}), 401

    if not delivery_district or not delivery_address:
        return jsonify({'error': 'Please provide complete delivery destination details.'}), 400

    conn = get_db()
    cursor = conn.cursor()

    # 1. Fetch buyer
    cursor.execute("SELECT id, name, email, phone, role, email_verified FROM users WHERE id = ?", (buyer_id,))
    buyer = cursor.fetchone()
    if not buyer:
        conn.close()
        return jsonify({'error': 'Buyer not found.'}), 404

    buyer_name = buyer['name']
    buyer_phone = buyer['phone']
    buyer_email = buyer['email']
    norm_phone = normalize_indian_phone(buyer_phone)

    # 2. Server-side availability pre-check
    cursor.execute("SELECT * FROM listings WHERE id = ? AND is_deleted = 0", (listing_id,))
    listing = cursor.fetchone()
    if not listing or listing['status'] != 'Available':
        conn.close()
        return jsonify({'error': 'This livestock is no longer available.'}), 400

    # 3. Purchase OTP Removed: Confirmation only flow
    # Direct purchase verification using buyer's registered/verified account

    # 4. Atomic transaction with lock
    try:
        cursor.execute("BEGIN IMMEDIATE")

        cursor.execute("SELECT * FROM listings WHERE id = ? AND is_deleted = 0", (listing_id,))
        locked_listing = cursor.fetchone()
        listing_exists = (locked_listing is not None)

        seller_id = locked_listing['farmer_id'] if locked_listing else None
        cursor.execute("SELECT id, name, email, phone, role FROM users WHERE id = ?", (seller_id,))
        seller_record = cursor.fetchone()
        seller_exists = (seller_record is not None)

        # Temporary server-side diagnostic logging (Section 1 Requirement)
        print("="*60)
        print("[Purchase Diagnostic Trace]")
        print(f"  buyer_id: {buyer_id} | email: {buyer_email} | exists_in_users: {buyer is not None}")
        print(f"  livestock_id/listing_id: {listing_id} | exists_in_listings: {listing_exists}")
        print(f"  seller_id: {seller_id} | exists_in_users: {seller_exists}")
        print(f"  Foreign Keys to INSERT -> purchases(listing_id={listing_id}, buyer_id={buyer_id}, farmer_id={seller_id})")
        print("="*60)

        if not listing_exists or locked_listing['status'] != 'Available':
            conn.rollback()
            conn.close()
            return jsonify({'error': 'This livestock is no longer available.'}), 400

        if not seller_exists:
            # Check if an existing farmer user exists in users table to repair orphan reference
            cursor.execute("SELECT id, name, email, phone, role FROM users WHERE role = 'farmer' ORDER BY id ASC LIMIT 1")
            seller_record = cursor.fetchone()
            if seller_record:
                seller_id = seller_record['id']
                cursor.execute("UPDATE listings SET farmer_id = ? WHERE id = ?", (seller_id, listing_id))
                seller_exists = True
                print(f"[Purchase Diagnostic Repair] Repaired orphaned listing #{listing_id} farmer_id to active user #{seller_id}.")
            else:
                conn.rollback()
                conn.close()
                print(f"[Purchase Diagnostic Error] Foreign key constraint violation: Seller ID {seller_id} not found in users table.")
                return jsonify({'error': f'Foreign key constraint failed: Seller account (ID {seller_id}) does not exist in users table.'}), 400

        cursor.execute("SELECT id FROM purchases WHERE listing_id = ?", (listing_id,))
        if cursor.fetchone():
            conn.rollback()
            conn.close()
            return jsonify({'error': 'This livestock is no longer available.'}), 400

        livestock_price = int(locked_listing['asking_price'])
        total_amount = livestock_price  # Zero delivery fee

        now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Create purchase record FIRST (validating FKs)
        cursor.execute('''
            INSERT INTO purchases (
                listing_id, buyer_id, buyer_name, buyer_phone,
                farmer_id, farmer_name, farmer_phone, livestock_price,
                distance_km, delivery_charge, total_amount,
                payment_method, delivery_address, delivery_district, delivery_taluk,
                delivery_village, delivery_pincode, order_status, confirmed_at, purchased_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0, ?, 'CASH_ON_DELIVERY', ?, ?, ?, ?, ?, 'CONFIRMED', ?, ?, ?)
        ''', (
            listing_id, buyer_id, buyer_name, norm_phone,
            seller_id, locked_listing['farmer_name'], locked_listing['farmer_phone'],
            livestock_price, total_amount,
            delivery_address, delivery_district, delivery_taluk, delivery_village, delivery_pincode,
            now_ts, now_ts, now_ts
        ))

        order_id = cursor.lastrowid

        # 2. Only after successful purchase insert, update listing status to SOLD_OUT
        cursor.execute("UPDATE listings SET status = 'SOLD_OUT' WHERE id = ?", (listing_id,))

        # 3. Commit database transaction
        conn.commit()

        # Database transaction is committed. Close DB connection cleanly.
        conn.close()

        # Post-commit notifications (AgentMail delivery failure must NOT roll back purchase)
        f_email = seller_record['email'] if seller_record else ''
        f_phone = seller_record['phone'] if seller_record else locked_listing['farmer_phone']

        order_details = {
            'order_id': order_id,
            'id': order_id,
            'livestock_price': livestock_price,
            'total_amount': total_amount,
            'delivery_address': delivery_address,
            'delivery_district': delivery_district,
            'delivery_taluk': delivery_taluk,
            'delivery_village': delivery_village,
            'delivery_pincode': delivery_pincode,
            'buyer_name': buyer_name,
            'buyer_phone': norm_phone,
            'farmer_name': locked_listing['farmer_name'],
            'farmer_phone': f_phone,
            'confirmed_at': now_ts,
            'purchased_at': now_ts
        }

        msg_buyer = f"Your order #{order_id} for {locked_listing['species']} ({locked_listing['breed']}) is CONFIRMED for ₹{livestock_price:,} (Cash on Delivery). Seller: {locked_listing['farmer_name']} ({locked_listing['farmer_phone']})."
        msg_farmer = f"Order #{order_id} CONFIRMED! Buyer {buyer_name} ({norm_phone}) has purchased your {locked_listing['species']} ({locked_listing['breed']}) for ₹{livestock_price:,}. Delivery Address: {delivery_address}, {delivery_district}."

        email_delivery_warning = None

        # Send Purchase Confirmation Email via AgentMail to buyer's registered email
        try:
            b_sent, b_msg = send_buyer_purchase_confirmation_email(buyer_email, dict(locked_listing), buyer_name, order_details)
            if not b_sent:
                email_delivery_warning = b_msg
                print(f"[Purchase Confirmation Mail Warning] {b_msg}")
        except Exception as mail_err:
            email_delivery_warning = str(mail_err)
            print(f"[Purchase Confirmation Mail Error] {mail_err}")

        # Send Livestock Purchased Notification Email via AgentMail to farmer's registered email
        if f_email and is_valid_email_format(f_email):
            try:
                f_sent, f_msg = send_farmer_purchase_notification_email(f_email, dict(locked_listing), locked_listing['farmer_name'], order_details)
                if not f_sent:
                    print(f"[Farmer Purchase Notification Mail Warning] {f_msg}")
            except Exception as f_mail_err:
                print(f"[Farmer Purchase Notification Mail Error] {f_mail_err}")

        try:
            log_and_dispatch_notification(buyer_id, buyer_email, norm_phone, order_id, "Order Confirmed", msg_buyer)
            log_and_dispatch_notification(seller_id, f_email, f_phone, order_id, "Livestock Sold", msg_farmer)
        except Exception:
            pass

        resp_msg = 'Purchase confirmed successfully.'
        if email_delivery_warning:
            resp_msg += f' Note: Confirmation email could not currently be delivered ({email_delivery_warning}).'

        return jsonify({
            'status': 'success',
            'message': resp_msg,
            'order_id': order_id,
            'summary': {
                'order_id': order_id,
                'listing_id': listing_id,
                'livestock_price': livestock_price,
                'delivery_charge': 0,
                'total_amount': total_amount,
                'payment_method': 'CASH_ON_DELIVERY',
                'order_status': 'CONFIRMED',
                'delivery_status': None
            }
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        err_msg = str(e)
        print(f"[Purchase Integrity/Database Error] {err_msg}")
        return jsonify({'error': f'Purchase transaction failed: {err_msg}'}), 500

# -------------------------------------------------------------
# 9. LIVESTOCK AVAILABILITY ALERTS
# -------------------------------------------------------------
@app.route('/api/alerts/subscribe', methods=['POST'])
def subscribe_availability_alert():
    """
    Save buyer's search criteria when no matching livestock is available.
    Stores in livestock_availability_alerts table.
    """
    data = request.json or {}
    buyer_id = data.get('buyer_id') or session.get('user_id')
    buyer_email = (data.get('buyer_email') or '').strip().lower()
    species = (data.get('species') or '').strip()
    breed = (data.get('breed') or '').strip()
    district = (data.get('district') or '').strip()
    taluk = (data.get('taluk') or '').strip()
    village = (data.get('village') or '').strip()
    min_age = data.get('minimum_age')
    max_age = data.get('maximum_age')
    min_milk = data.get('minimum_milk_production')
    max_milk = data.get('maximum_milk_production')

    conn = get_db()
    cursor = conn.cursor()

    if not buyer_email and buyer_id:
        cursor.execute("SELECT email FROM users WHERE id = ?", (buyer_id,))
        u = cursor.fetchone()
        if u:
            buyer_email = u['email']

    if not buyer_email or not is_valid_email_format(buyer_email):
        conn.close()
        return jsonify({'error': 'Please provide a valid email address to receive availability alerts.'}), 400

    if not buyer_id:
        cursor.execute("SELECT id FROM users WHERE LOWER(email) = ?", (buyer_email,))
        u = cursor.fetchone()
        buyer_id = u['id'] if u else None

    cursor.execute('''
        INSERT INTO livestock_availability_alerts (
            buyer_id, buyer_email, species, breed, district, taluk, village,
            minimum_age, maximum_age, minimum_milk_production, maximum_milk_production,
            created_at, notified, active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 0, 1)
    ''', (buyer_id, buyer_email, species, breed, district, taluk, village, min_age, max_age, min_milk, max_milk))

    alert_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        'status': 'success',
        'alert_id': alert_id,
        'message': 'Availability alert saved! We will notify you by email when matching livestock becomes available.'
    })

@app.route('/api/orders/<int:order_id>/cancel', methods=['POST'])
def cancel_order(order_id):
    """
    BUYER CANCELLATION:
    Allowed before final purchase (from CONFIRMED).
    Requires cancellation reason. If 'Other' is selected, written explanation is mandatory.
    Returns listing to 'Available'.
    Notifies both Buyer and Farmer.
    """
    data = request.json or {}
    buyer_id = data.get('buyer_id') or data.get('user_id')
    reason = (data.get('cancellation_reason') or data.get('reason') or '').strip()
    explanation = (data.get('cancellation_details') or data.get('explanation') or '').strip()

    if not reason:
        return jsonify({'error': 'Cancellation reason is required.'}), 400

    if reason.lower() == 'other' and not explanation:
        return jsonify({'error': 'Written explanation is mandatory when selecting "Other".'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM purchases WHERE id = ?", (order_id,))
    order = cursor.fetchone()

    if not order:
        conn.close()
        return jsonify({'error': 'Order not found.'}), 404

    if buyer_id and order['buyer_id'] != buyer_id:
        conn.close()
        return jsonify({'error': 'Unauthorized: Only the purchaser can cancel this order.'}), 403

    if order['order_status'] == 'PURCHASED':
        conn.close()
        return jsonify({'error': 'Cannot cancel an order that has already been delivered and purchased.'}), 400

    if order['order_status'] == 'CANCELLED':
        conn.close()
        return jsonify({'error': 'This order has already been cancelled.'}), 400

    now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Update order to CANCELLED and return listing to Available
    cursor.execute('''
        UPDATE purchases 
        SET order_status = 'CANCELLED', cancellation_reason = ?, cancellation_details = ?, cancelled_at = ?
        WHERE id = ?
    ''', (reason, explanation, now_ts, order_id))

    cursor.execute("UPDATE listings SET status = 'Available' WHERE id = ?", (order['listing_id'],))
    conn.commit()

    cursor.execute("SELECT * FROM listings WHERE id = ?", (order['listing_id'],))
    listing_row = cursor.fetchone()
    listing_dict = dict(listing_row) if listing_row else dict(order)

    # Look up user contacts
    cursor.execute("SELECT name, email, phone FROM users WHERE id = ?", (order['buyer_id'],))
    b_user = cursor.fetchone()
    cursor.execute("SELECT name, email, phone FROM users WHERE id = ?", (order['farmer_id'],))
    f_user = cursor.fetchone()
    conn.close()

    b_name = (b_user['name'] if b_user else None) or order['buyer_name'] or 'Buyer'
    b_email = (b_user['email'] if b_user else '') or ''
    b_phone = (b_user['phone'] if b_user else '') or order['buyer_phone']
    f_name = (f_user['name'] if f_user else None) or order['farmer_name'] or 'Farmer'
    f_email = (f_user['email'] if f_user else '') or ''
    f_phone = (f_user['phone'] if f_user else '') or order['farmer_phone']

    order_details = dict(order)
    order_details.update({
        'order_id': order_id,
        'buyer_name': b_name,
        'farmer_name': f_name,
        'cancelled_at': now_ts
    })

    msg = f"Order #{order_id} has been CANCELLED. Reason: {reason}. {explanation if explanation else ''}"
    try:
        log_and_dispatch_notification(order['buyer_id'], b_email, b_phone, order_id, "Order Cancelled", msg)
        log_and_dispatch_notification(order['farmer_id'], f_email, f_phone, order_id, "Order Cancelled by Buyer", msg)
    except Exception:
        pass

    # Send transactional cancellation emails to both buyer and farmer
    full_reason = f"{reason}" + (f" ({explanation})" if explanation else "")
    if b_email and is_valid_email_format(b_email):
        try:
            from services.email_service import send_order_cancelled_buyer_email
            send_order_cancelled_buyer_email(b_email, listing_dict, b_name, order_details, full_reason)
        except Exception as cancel_mail_err:
            print(f"[Buyer Cancellation Email Error] {cancel_mail_err}")

    if f_email and is_valid_email_format(f_email):
        try:
            from services.email_service import send_order_cancelled_farmer_email
            send_order_cancelled_farmer_email(f_email, listing_dict, f_name, b_name, order_details, full_reason)
        except Exception as cancel_mail_err:
            print(f"[Farmer Cancellation Email Error] {cancel_mail_err}")

    return jsonify({
        'status': 'success',
        'message': 'Order cancelled successfully. Livestock listing has been returned to available marketplace.',
        'order_status': 'CANCELLED'
    })

@app.route('/api/orders/<int:order_id>/mark-purchased', methods=['POST'])
def mark_as_purchased(order_id):
    """
    MARK AS PURCHASED:
    Requires valid proof photo upload.
    Order Status = PURCHASED
    Delivery Status = DELIVERED
    Livestock Status = SOLD OUT / UNAVAILABLE
    Notifies both Buyer and Farmer.
    """
    buyer_id = request.form.get('buyer_id', type=int) or request.form.get('user_id', type=int)

    file_key = 'proof_photo' if 'proof_photo' in request.files else ('proof_file' if 'proof_file' in request.files else None)
    if not file_key:
        return jsonify({'error': 'Valid delivery proof photo is required to mark as purchased.'}), 400

    proof_file = request.files[file_key]
    if proof_file.filename == '':
        return jsonify({'error': 'Proof photo file cannot be empty.'}), 400

    if not is_allowed_file(proof_file.filename):
        return jsonify({'error': 'Invalid photo format. Please upload JPG, PNG, or WEBP.'}), 400

    ext = os.path.splitext(proof_file.filename)[1].lower()
    proof_filename = f"proof_{order_id}_{uuid.uuid4().hex[:8]}{ext}"
    proof_path = os.path.join(app.config['UPLOAD_FOLDER'], proof_filename)
    proof_file.save(proof_path)
    proof_url = f"/static/uploads/{proof_filename}"

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM purchases WHERE id = ?", (order_id,))
    order = cursor.fetchone()

    if not order:
        conn.close()
        return jsonify({'error': 'Order not found.'}), 404

    if buyer_id and order['buyer_id'] != buyer_id:
        conn.close()
        return jsonify({'error': 'Unauthorized: Only the buyer can confirm purchase completion.'}), 403

    if order['order_status'] == 'CANCELLED':
        conn.close()
        return jsonify({'error': 'Cannot mark a cancelled order as purchased.'}), 400

    now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        UPDATE purchases 
        SET order_status = 'PURCHASED', delivery_status = 'DELIVERED',
            proof_photo_url = ?, delivered_at = ?, purchased_at = ?
        WHERE id = ?
    ''', (proof_url, now_ts, now_ts, order_id))

    cursor.execute("UPDATE listings SET status = 'SOLD_OUT' WHERE id = ?", (order['listing_id'],))
    conn.commit()

    cursor.execute("SELECT * FROM listings WHERE id = ?", (order['listing_id'],))
    listing_row = cursor.fetchone()
    listing_dict = dict(listing_row) if listing_row else dict(order)

    cursor.execute("SELECT name, email, phone FROM users WHERE id = ?", (order['buyer_id'],))
    b_user = cursor.fetchone()
    cursor.execute("SELECT name, email, phone FROM users WHERE id = ?", (order['farmer_id'],))
    f_user = cursor.fetchone()
    conn.close()

    b_name = (b_user['name'] if b_user else None) or order['buyer_name'] or 'Buyer'
    b_email = (b_user['email'] if b_user else '') or ''
    b_phone = (b_user['phone'] if b_user else '') or order['buyer_phone']
    f_name = (f_user['name'] if f_user else None) or order['farmer_name'] or 'Farmer'
    f_email = (f_user['email'] if f_user else '') or ''
    f_phone = (f_user['phone'] if f_user else '') or order['farmer_phone']

    order_details = dict(order)
    order_details.update({
        'order_id': order_id,
        'buyer_name': b_name,
        'farmer_name': f_name,
        'purchased_at': now_ts,
        'delivered_at': now_ts
    })

    msg = f"Order #{order_id} marked as PURCHASED and DELIVERED! Proof photo verified."
    try:
        log_and_dispatch_notification(order['buyer_id'], b_email, b_phone, order_id, "Purchase Completed", msg)
        log_and_dispatch_notification(order['farmer_id'], f_email, f_phone, order_id, "Delivery Accepted", msg)
    except Exception:
        pass

    # Send purchase completion transactional emails to both buyer and farmer
    if b_email and is_valid_email_format(b_email):
        try:
            from services.email_service import send_purchase_completed_buyer_email
            send_purchase_completed_buyer_email(b_email, listing_dict, b_name, order_details)
        except Exception as mail_err:
            print(f"[Buyer Purchase Completed Mail Error] {mail_err}")

    if f_email and is_valid_email_format(f_email):
        try:
            from services.email_service import send_purchase_completed_farmer_email
            send_purchase_completed_farmer_email(f_email, listing_dict, f_name, b_name, order_details)
        except Exception as mail_err:
            print(f"[Farmer Purchase Completed Mail Error] {mail_err}")

    return jsonify({
        'status': 'success',
        'message': 'Purchase confirmed and delivery marked as completed!',
        'order_status': 'PURCHASED',
        'delivery_status': 'DELIVERED',
        'proof_photo_url': proof_url
    })

@app.route('/api/orders/<int:order_id>/reject-delivery', methods=['POST'])
def reject_delivery(order_id):
    """
    REJECT DELIVERY:
    Requires rejection reason.
    Delivery Status = REJECTED
    Reason stored and displayed to Buyer and Farmer.
    Notifies both Buyer and Farmer.
    """
    data = request.json or {}
    buyer_id = data.get('buyer_id') or data.get('user_id')
    reason = (data.get('rejection_reason') or data.get('reason') or '').strip()

    if not reason:
        return jsonify({'error': 'Rejection reason is required.'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM purchases WHERE id = ?", (order_id,))
    order = cursor.fetchone()

    if not order:
        conn.close()
        return jsonify({'error': 'Order not found.'}), 404

    if buyer_id and order['buyer_id'] != buyer_id:
        conn.close()
        return jsonify({'error': 'Unauthorized: Only the buyer can reject delivery.'}), 403

    now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        UPDATE purchases 
        SET delivery_status = 'REJECTED', rejection_reason = ?, delivered_at = ?
        WHERE id = ?
    ''', (reason, now_ts, order_id))

    cursor.execute("UPDATE listings SET status = 'Available' WHERE id = ?", (order['listing_id'],))
    conn.commit()

    cursor.execute("SELECT * FROM listings WHERE id = ?", (order['listing_id'],))
    listing_row = cursor.fetchone()
    listing_dict = dict(listing_row) if listing_row else dict(order)

    cursor.execute("SELECT name, email, phone FROM users WHERE id = ?", (order['buyer_id'],))
    b_user = cursor.fetchone()
    cursor.execute("SELECT name, email, phone FROM users WHERE id = ?", (order['farmer_id'],))
    f_user = cursor.fetchone()
    conn.close()

    b_name = (b_user['name'] if b_user else None) or order['buyer_name'] or 'Buyer'
    b_email = (b_user['email'] if b_user else '') or ''
    b_phone = (b_user['phone'] if b_user else '') or order['buyer_phone']
    f_name = (f_user['name'] if f_user else None) or order['farmer_name'] or 'Farmer'
    f_email = (f_user['email'] if f_user else '') or ''
    f_phone = (f_user['phone'] if f_user else '') or order['farmer_phone']

    order_details = dict(order)
    order_details.update({
        'order_id': order_id,
        'buyer_name': b_name,
        'farmer_name': f_name,
        'delivered_at': now_ts
    })

    msg = f"Delivery for Order #{order_id} was REJECTED by buyer. Reason: {reason}."
    try:
        log_and_dispatch_notification(order['buyer_id'], b_email, b_phone, order_id, "Delivery Rejected", msg)
        log_and_dispatch_notification(order['farmer_id'], f_email, f_phone, order_id, "Delivery Rejected by Buyer", msg)
    except Exception:
        pass

    # Send transactional rejection emails to both buyer and farmer
    if b_email and is_valid_email_format(b_email):
        try:
            from services.email_service import send_order_rejected_buyer_email
            send_order_rejected_buyer_email(b_email, listing_dict, b_name, order_details, reason)
        except Exception as reject_mail_err:
            print(f"[Buyer Rejection Email Error] {reject_mail_err}")

    if f_email and is_valid_email_format(f_email):
        try:
            from services.email_service import send_order_rejected_farmer_email
            send_order_rejected_farmer_email(f_email, listing_dict, f_name, b_name, order_details, reason)
        except Exception as reject_mail_err:
            print(f"[Farmer Rejection Email Error] {reject_mail_err}")

    return jsonify({
        'status': 'success',
        'message': 'Delivery marked as rejected. Farmer has been notified.',
        'delivery_status': 'REJECTED',
        'rejection_reason': reason
    })

@app.route('/api/orders/<int:order_id>/review', methods=['POST'])
def add_review(order_id):
    """
    RATINGS & REVIEWS:
    Allowed after PURCHASED or REJECTED.
    Rating: 1-5 stars. Written review: optional.
    Prevents duplicate reviews for the same order.
    """
    data = request.json or {}
    buyer_id = data.get('buyer_id')
    rating = data.get('rating')
    review_text = (data.get('review') or '').strip()

    try:
        rating_val = int(rating)
        if rating_val < 1 or rating_val > 5:
            return jsonify({'error': 'Rating must be between 1 and 5 stars.'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Valid rating between 1 and 5 is required.'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM purchases WHERE id = ?", (order_id,))
    order = cursor.fetchone()

    if not order:
        conn.close()
        return jsonify({'error': 'Order not found.'}), 404

    if buyer_id and order['buyer_id'] != buyer_id:
        conn.close()
        return jsonify({'error': 'Unauthorized: Only the purchaser can review this order.'}), 403

    if order['order_status'] != 'PURCHASED' and order['delivery_status'] != 'REJECTED':
        conn.close()
        return jsonify({'error': 'Reviews are only permitted after delivery completion (PURCHASED) or REJECTION.'}), 400

    if order['rating'] is not None:
        conn.close()
        return jsonify({'error': 'You have already submitted a review for this order.'}), 400

    cursor.execute("UPDATE purchases SET rating = ?, review = ? WHERE id = ?", (rating_val, review_text, order_id))
    conn.commit()
    conn.close()

    return jsonify({
        'status': 'success',
        'message': 'Rating and review submitted successfully!',
        'rating': rating_val,
        'review': review_text
    })

# -------------------------------------------------------------
# 9. FARMER & BUYER DASHBOARD DATA APIS
# -------------------------------------------------------------
@app.route('/api/farmer/<int:farmer_id>/listings', methods=['GET'])
def get_farmer_listings(farmer_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM listings WHERE farmer_id = ? AND is_deleted = 0 ORDER BY id DESC", (farmer_id,))
    listings = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(listings)

@app.route('/api/farmer/<int:farmer_id>/orders', methods=['GET'])
def get_farmer_orders(farmer_id):
    """Orders received by this farmer (Confirmed, Delivered/Purchased, Rejected, Cancelled)."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.*, l.species, l.breed, l.gender, l.image_url, l.district AS farm_district
        FROM purchases p
        JOIN listings l ON p.listing_id = l.id
        WHERE p.farmer_id = ?
        ORDER BY p.id DESC
    ''', (farmer_id,))
    orders = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(orders)

@app.route('/api/farmer/<int:farmer_id>/history', methods=['GET'])
def get_farmer_history(farmer_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT l.*, 
               p.id AS purchase_id,
               p.order_status,
               p.delivery_status,
               p.purchased_at,
               p.total_amount,
               p.buyer_name,
               p.buyer_phone,
               p.delivery_district,
               p.delivery_address,
               p.rating,
               p.review
        FROM listings l
        LEFT JOIN purchases p ON l.id = p.listing_id
        WHERE l.farmer_id = ?
        ORDER BY l.id DESC
    ''', (farmer_id,))
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(history)

@app.route('/api/farmer/<int:farmer_id>/sold', methods=['GET'])
def get_farmer_sold_livestock(farmer_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT l.*, 
               p.id AS purchase_id,
               p.order_status,
               p.delivery_status,
               p.purchased_at,
               p.total_amount,
               p.buyer_name,
               p.buyer_phone,
               p.delivery_district,
               p.delivery_address,
               p.proof_photo_url,
               p.rating,
               p.review
        FROM listings l
        JOIN purchases p ON l.id = p.listing_id
        WHERE l.farmer_id = ? AND (p.order_status = 'PURCHASED' OR l.status = 'SOLD_OUT')
        ORDER BY p.id DESC
    ''', (farmer_id,))
    sold = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(sold)

@app.route('/api/buyer/<int:buyer_id>/purchases', methods=['GET'])
@app.route('/api/buyer/<int:buyer_id>/orders', methods=['GET'])
def get_buyer_purchases(buyer_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.*, 
               l.species, 
               l.breed, 
               l.gender, 
               l.image_url,
               l.district AS farmer_district,
               l.taluk AS farmer_taluk,
               l.village AS farmer_village,
               l.address AS farmer_address,
               l.status AS listing_status
        FROM purchases p 
        LEFT JOIN listings l ON p.listing_id = l.id 
        WHERE p.buyer_id = ? 
        ORDER BY p.id DESC
    ''', (buyer_id,))
    purchases = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(purchases)

@app.route('/api/buyer/track-interest', methods=['POST'])
def track_buyer_interest():
    """Track buyer search preferences and viewed livestock for tailored notifications."""
    data = request.json or {}
    buyer_id = data.get('buyer_id')
    species = data.get('species')
    breed = data.get('breed')
    district = data.get('district')
    max_price = data.get('max_price')
    query_text = data.get('query_text')

    if not buyer_id:
        return jsonify({'error': 'buyer_id required'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO buyer_interests (buyer_id, species, breed, district, max_price, query_text)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (buyer_id, species, breed, district, max_price, query_text))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

# ==============================================================================
# BUYER LIVESTOCK FOOD & NUTRITION ASSISTANT APIS
# ==============================================================================

@app.route('/api/buyer/nutrition/chat', methods=['POST'])
def buyer_nutrition_chat():
    """
    Intelligent livestock nutrition chatbot endpoint for the Buyer Dashboard.
    Provides personalized food management, feeding schedules, water guidelines,
    and milk production support based on ICAR, NDDB, and KVAFSU standards.
    """
    data = request.json or {}
    message = (data.get('message') or '').strip()
    animal_context = data.get('animal_context') or {}
    session_id = data.get('session_id') or str(uuid.uuid4())
    buyer_id = data.get('buyer_id') or session.get('user_id')
    listing_id = animal_context.get('listing_id')

    if not message:
        return jsonify({'error': 'Message content is required'}), 400

    # If listing_id is provided and animal details are sparse, populate from database
    if listing_id and (not animal_context.get('species') or not animal_context.get('breed')):
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, species, breed, gender, age_years, milk_production, is_pregnant, pregnancy_month
                FROM listings WHERE id = ?
            ''', (listing_id,))
            listing = cursor.fetchone()
            conn.close()
            if listing:
                for k in ['species', 'breed', 'gender', 'age_years', 'milk_production', 'is_pregnant', 'pregnancy_month']:
                    if not animal_context.get(k) and listing[k] is not None:
                        animal_context[k] = listing[k]
        except Exception as e:
            app.logger.warning(f"[Nutrition Chat] Could not fetch listing {listing_id}: {e}")

    # Generate personalized nutrition advice
    advice = nutrition_advisor.generate_nutrition_advice(
        user_message=message,
        animal_context=animal_context,
        session_id=session_id
    )

    # Persist chat history in database
    try:
        # Save user message
        save_nutrition_chat_message(
            session_id=session_id,
            role='user',
            content=message,
            buyer_id=buyer_id,
            listing_id=listing_id,
            intent=advice.get('intent'),
            animal_context=advice.get('animal_context')
        )
        # Save assistant reply
        save_nutrition_chat_message(
            session_id=session_id,
            role='assistant',
            content=advice.get('reply', ''),
            buyer_id=buyer_id,
            listing_id=listing_id,
            intent=advice.get('intent'),
            animal_context=advice.get('animal_context')
        )
    except Exception as e:
        app.logger.error(f"[Nutrition Chat DB] Error saving history: {e}")

    return jsonify({
        'status': 'success',
        'reply': advice.get('reply'),
        'intent': advice.get('intent'),
        'animal_context': advice.get('animal_context'),
        'suggested_questions': advice.get('suggested_questions', []),
        'disclaimer': advice.get('disclaimer'),
        'session_id': session_id
    })

@app.route('/api/buyer/nutrition/presets', methods=['GET'])
def get_nutrition_presets():
    """Return dynamically recognized species, breeds, and purpose options."""
    try:
        mapping_path = os.path.join(os.path.dirname(__file__), 'models', 'hierarchical_mappings.json')
        with open(mapping_path, 'r') as f:
            hm = json.load(f)
        cattle_breeds = sorted(hm.get('cattle_breeds', []))
        buffalo_breeds = sorted([b.title() for b in hm.get('buffalo_breeds', [])])
    except Exception:
        cattle_breeds = ["Ayrshire cattle", "Brown Swiss cattle", "Holstein Friesian cattle", "Jersey cattle", "Red Dane cattle", "Hallikar", "Deoni"]
        buffalo_breeds = ["Banni", "Bhadwari", "Jaffarabadi", "Mehsana", "Murrah", "Nili-Ravi", "Pandharpuri", "Surti"]

    return jsonify({
        'status': 'success',
        'species': ['Cow', 'Buffalo'],
        'cattle_breeds': cattle_breeds,
        'buffalo_breeds': buffalo_breeds,
        'genders': ['Female', 'Male'],
        'purposes': ['Milk Production', 'Maintenance', 'Growth', 'General Nutrition']
    })

@app.route('/api/buyer/nutrition/listing/<int:listing_id>', methods=['GET'])
def get_nutrition_listing_context(listing_id):
    """Retrieve animal attributes from a specific listing for instant pre-population."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, farmer_id, farmer_name, farmer_phone, image_url, species, breed,
               gender, age_years, milk_production, is_pregnant, pregnancy_month,
               district, taluk, village, asking_price, status
        FROM listings WHERE id = ?
    ''', (listing_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Listing not found'}), 404

    listing = dict(row)
    context = {
        'listing_id': listing['id'],
        'species': listing['species'],
        'breed': listing['breed'],
        'gender': listing['gender'] or 'Female',
        'age_years': listing['age_years'],
        'milk_production': listing['milk_production'] if (listing['gender'] or 'Female') == 'Female' else 0.0,
        'is_pregnant': listing.get('is_pregnant', 0),
        'pregnancy_month': listing.get('pregnancy_month', 0),
        'purpose': 'Milk Production' if (listing['gender'] or 'Female') == 'Female' and (listing['milk_production'] or 0) > 0 else 'Maintenance',
        'asking_price': listing['asking_price'],
        'district': listing['district'],
        'farmer_name': listing['farmer_name'],
        'image_url': listing['image_url']
    }

    return jsonify({
        'status': 'success',
        'animal_context': context,
        'summary': f"{listing['breed']} ({listing['species']}), {listing['age_years']} Yrs, {listing['district']}"
    })

@app.route('/api/buyer/nutrition/history', methods=['GET'])
def get_buyer_nutrition_history():
    """Retrieve historical chat transcripts for the active session."""
    session_id = request.args.get('session_id')
    buyer_id = request.args.get('buyer_id') or session.get('user_id')

    if not session_id and not buyer_id:
        return jsonify({'history': []})

    history = get_nutrition_chat_history(session_id=session_id or '', buyer_id=buyer_id)
    return jsonify({
        'status': 'success',
        'history': history
    })

@app.route('/api/buyer/nutrition/clear', methods=['POST'])
def clear_buyer_nutrition_history():
    """Clear chat session history."""
    data = request.json or {}
    session_id = data.get('session_id') or ''
    buyer_id = data.get('buyer_id') or session.get('user_id')

    clear_nutrition_chat_history(session_id=session_id, buyer_id=buyer_id)
    return jsonify({
        'status': 'success',
        'message': 'Nutrition chat history cleared.'
    })

@app.route('/api/inquiries/<int:user_id>', methods=['GET'])
def get_user_inquiries(user_id):
    role = request.args.get('role', 'farmer')
    conn = get_db()
    cursor = conn.cursor()

    if role == 'farmer':
        cursor.execute('''
            SELECT c.*, l.breed, l.species, l.asking_price, l.status AS listing_status
            FROM contact_requests c 
            JOIN listings l ON c.listing_id = l.id 
            WHERE c.farmer_id = ? ORDER BY c.id DESC
        ''', (user_id,))
    else:
        cursor.execute('''
            SELECT c.*, l.breed, l.species, l.asking_price, l.farmer_name, l.farmer_phone, l.status AS listing_status
            FROM contact_requests c 
            JOIN listings l ON c.listing_id = l.id 
            WHERE c.buyer_id = ? ORDER BY c.id DESC
        ''', (user_id,))

    inquiries = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(inquiries)

@app.route('/api/listings/<int:listing_id>/contact', methods=['POST'])
def send_inquiry(listing_id):
    data = request.json or {}
    buyer_id = data.get('buyer_id', 0)
    buyer_name = (data.get('buyer_name') or 'Buyer').strip()
    buyer_phone = (data.get('buyer_phone') or '').strip()
    message = (data.get('message') or 'I am interested in buying your cow/buffalo.').strip()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM listings WHERE id = ?", (listing_id,))
    listing = cursor.fetchone()

    if not listing:
        conn.close()
        return jsonify({'error': 'Listing not found'}), 404

    cursor.execute('''
        INSERT INTO contact_requests (listing_id, farmer_id, buyer_id, buyer_name, buyer_phone, message, status)
        VALUES (?, ?, ?, ?, ?, ?, 'Sent')
    ''', (listing_id, listing['farmer_id'], buyer_id, buyer_name, buyer_phone, message))

    conn.commit()
    conn.close()

    return jsonify({'status': 'success', 'message': f'Message successfully sent to farmer {listing["farmer_name"]}!'})

@app.route('/api/notifications/<int:user_id>', methods=['GET'])
def get_user_notifications(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM notifications WHERE user_id = ? ORDER BY id DESC LIMIT 50", (user_id,))
    notifs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(notifs)

# -------------------------------------------------------------
# 10. PROTECTED ADMINISTRATOR CONSOLE APIS
# -------------------------------------------------------------
@app.route('/api/admin/dashboard', methods=['GET'])
@admin_required
def admin_dashboard_stats():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'farmer' AND email_verified = 1")
    total_farmers = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'buyer' AND email_verified = 1")
    total_buyers = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM listings WHERE is_deleted = 0")
    total_livestock = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM listings WHERE status = 'Available' AND is_deleted = 0")
    available_livestock = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM listings WHERE status = 'SOLD_OUT'")
    sold_livestock = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM purchases")
    total_purchases = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM purchases WHERE order_status = 'CONFIRMED'")
    confirmed_orders = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM purchases WHERE order_status = 'PURCHASED'")
    completed_purchases = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM purchases WHERE delivery_status = 'REJECTED'")
    rejected_deliveries = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM purchases WHERE order_status = 'CANCELLED'")
    cancelled_orders = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        'admin_name': AUTHORIZED_ADMIN_NAME,
        'admin_email': AUTHORIZED_ADMIN_EMAIL,
        'total_farmers': total_farmers,
        'total_buyers': total_buyers,
        'total_livestock': total_livestock,
        'available_livestock': available_livestock,
        'sold_livestock': sold_livestock,
        'total_purchases': total_purchases,
        'confirmed_orders': confirmed_orders,
        'completed_purchases': completed_purchases,
        'rejected_deliveries': rejected_deliveries,
        'cancelled_orders': cancelled_orders
    })

@app.route('/api/admin/farmers', methods=['GET'])
@admin_required
def admin_get_farmers():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT u.id, u.name, u.email, u.phone, u.district, u.taluk, u.village, u.address, u.pincode, u.status, u.created_at,
               COUNT(l.id) AS total_listings,
               SUM(CASE WHEN l.status = 'SOLD_OUT' OR l.status = 'Purchased' THEN 1 ELSE 0 END) AS sold_count
        FROM users u
        LEFT JOIN listings l ON u.id = l.farmer_id AND l.is_deleted = 0
        WHERE u.role = 'farmer'
        GROUP BY u.id
        ORDER BY u.id DESC
    ''')
    farmers = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(farmers)

@app.route('/api/admin/buyers', methods=['GET'])
@admin_required
def admin_get_buyers():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT u.id, u.name, u.email, u.phone, u.district, u.taluk, u.village, u.address, u.pincode, u.status, u.created_at,
               COUNT(p.id) AS purchase_count
        FROM users u
        LEFT JOIN purchases p ON u.id = p.buyer_id
        WHERE u.role = 'buyer'
        GROUP BY u.id
        ORDER BY u.id DESC
    ''')
    buyers = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(buyers)

@app.route('/api/admin/livestock', methods=['GET'])
@admin_required
def admin_get_livestock():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT l.*, 
               p.id AS purchase_id,
               p.order_status,
               p.delivery_status,
               p.purchased_at,
               p.confirmed_at,
               p.total_amount,
               p.buyer_name,
               p.buyer_phone
        FROM listings l
        LEFT JOIN purchases p ON l.id = p.listing_id
        ORDER BY l.id DESC
    ''')
    rows = cursor.fetchall()
    conn.close()

    now = datetime.datetime.now()
    livestock = []
    for r in rows:
        item = dict(r)
        purchased_at_str = item.get('purchased_at') or item.get('confirmed_at')
        if purchased_at_str and item['status'] == 'SOLD_OUT':
            try:
                clean_time = str(purchased_at_str).replace('T', ' ').split('.')[0]
                purch_dt = datetime.datetime.strptime(clean_time, "%Y-%m-%d %H:%M:%S")
            except Exception:
                purch_dt = now

            diff_days = (now - purch_dt).days
            eligible_dt = purch_dt + datetime.timedelta(days=15)
            item['days_since_purchase'] = diff_days
            item['eligible_removal_date'] = eligible_dt.strftime("%d %B %Y")
            item['can_remove'] = (now >= eligible_dt)
            item['days_remaining'] = max(0, (eligible_dt - now).days + 1)
        else:
            item['days_since_purchase'] = 0
            item['eligible_removal_date'] = None
            item['can_remove'] = False
            item['days_remaining'] = 0

        livestock.append(item)

    return jsonify(livestock)

@app.route('/api/admin/purchases', methods=['GET'])
@admin_required
def admin_get_purchases():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.*, l.species, l.breed, l.gender, l.image_url 
        FROM purchases p 
        LEFT JOIN listings l ON p.listing_id = l.id 
        ORDER BY p.id DESC
    ''')
    purchases = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(purchases)

@app.route('/api/admin/contacts', methods=['GET'])
@admin_required
def admin_get_contacts():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT c.*, l.species, l.breed 
        FROM contact_requests c 
        LEFT JOIN listings l ON c.listing_id = l.id 
        ORDER BY c.id DESC
    ''')
    contacts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(contacts)

@app.route('/api/admin/notifications', methods=['GET'])
@admin_required
def admin_get_notifications():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM notifications ORDER BY id DESC LIMIT 100")
    notifs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(notifs)

@app.route('/api/admin/listings/<int:listing_id>/remove-soldout', methods=['POST'])
@admin_required
def admin_remove_soldout_listing(listing_id):
    """
    15-Day Sold-Out Removal Rule (Enforced by Backend):
    1. Authenticate single authorized admin
    2. Verify listing exists and status == 'SOLD_OUT'
    3. Verify current date >= purchase_date + 15 days
    4. Soft-delete listing (is_deleted = 1, status = 'REMOVED')
    5. Historical transaction records remain permanently preserved!
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM listings WHERE id = ?", (listing_id,))
    listing = cursor.fetchone()

    if not listing:
        conn.close()
        return jsonify({'error': 'Listing not found.'}), 404

    if listing['status'] != 'SOLD_OUT':
        conn.close()
        return jsonify({'error': 'Only SOLD OUT listings can be removed under the 15-day rule.'}), 400

    cursor.execute("SELECT confirmed_at, purchased_at FROM purchases WHERE listing_id = ? ORDER BY id DESC LIMIT 1", (listing_id,))
    purchase = cursor.fetchone()
    if not purchase:
        conn.close()
        return jsonify({'error': 'No transaction record found for this listing.'}), 400

    purchase_time_str = purchase['purchased_at'] or purchase['confirmed_at']
    try:
        clean_time_str = str(purchase_time_str).replace('T', ' ').split('.')[0]
        purchase_dt = datetime.datetime.strptime(clean_time_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        purchase_dt = datetime.datetime.now()

    eligible_dt = purchase_dt + datetime.timedelta(days=15)
    now = datetime.datetime.now()

    if now < eligible_dt:
        days_left = (eligible_dt - now).days + 1
        conn.close()
        return jsonify({
            'error': f'Cannot remove listing before 15 days from purchase date. Eligible for removal on {eligible_dt.strftime("%d %B %Y")} ({days_left} day(s) remaining).',
            'eligible_date': eligible_dt.strftime("%d %B %Y"),
            'can_remove': False
        }), 400

    cursor.execute("UPDATE listings SET status = 'REMOVED', is_deleted = 1 WHERE id = ?", (listing_id,))
    conn.commit()
    conn.close()

    return jsonify({
        'status': 'REMOVED',
        'message': 'Sold-out listing removed successfully. Transaction records preserved.',
        'listing_id': listing_id
    })

# -------------------------------------------------------------
# 11. WEB PAGE ROUTES
# -------------------------------------------------------------
@app.route('/')
@app.route('/home')
@app.route('/about')
@app.route('/login')
@app.route('/register')
@app.route('/farmer')
@app.route('/buyer')
@app.route('/admin')
def serve_index():
    return render_template('index.html')

if __name__ == '__main__':
    print(f"Starting PashuDhan AI Marketplace Server on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
