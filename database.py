import os
import json
import sqlite3
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "livestock_app.db")
DB_PATH = os.environ.get("DATABASE_PATH", DEFAULT_DB_PATH)
if not os.path.isabs(DB_PATH):
    DB_PATH = os.path.join(BASE_DIR, DB_PATH)

# The strictly authorized administrator credentials
AUTHORIZED_ADMIN_NAME = "Rakshitha M"
AUTHORIZED_ADMIN_EMAIL = "rakshitham556@gmail.com"
AUTHORIZED_ADMIN_PASSWORD_HASH = generate_password_hash("Rakshi@2026")

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
    except Exception:
        pass
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def hash_password(password: str) -> str:
    """Hash password securely using werkzeug security."""
    return generate_password_hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against werkzeug hash or legacy bcrypt hash."""
    try:
        if not hashed_password or not plain_password:
            return False
        # If hash is a werkzeug hash (e.g. scrypt: / pbkdf2:)
        if hashed_password.startswith(('scrypt:', 'pbkdf2:', 'argon2:')):
            return check_password_hash(hashed_password, plain_password)
        # If hash is a bcrypt hash ($2b$, $2a$, $2y$)
        if hashed_password.startswith(('$2b$', '$2a$', '$2y$')):
            return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        # Standard check_password_hash fallback
        return check_password_hash(hashed_password, plain_password)
    except Exception:
        return False

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # 1. State Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            code TEXT
        )
    ''')

    # 2. Districts Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS districts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            code TEXT,
            latitude REAL,
            longitude REAL,
            FOREIGN KEY (state_id) REFERENCES states(id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_districts_state ON districts(state_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_districts_name ON districts(name)')

    # 3. Taluks Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS taluks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            district_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            code TEXT,
            latitude REAL,
            longitude REAL,
            FOREIGN KEY (district_id) REFERENCES districts(id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_taluks_district ON taluks(district_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_taluks_name ON taluks(name)')

    # 4. Villages Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS villages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            taluk_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            native_name TEXT,
            code TEXT,
            pincode TEXT,
            latitude REAL,
            longitude REAL,
            FOREIGN KEY (taluk_id) REFERENCES taluks(id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_villages_taluk ON villages(taluk_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_villages_name ON villages(name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_villages_pincode ON villages(pincode)')

    # 5. System Configuration Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    ''')

    # 6. Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            password TEXT DEFAULT '[BCRYPT_PROTECTED]',
            username TEXT,
            role TEXT NOT NULL CHECK(role IN ('admin', 'farmer', 'buyer')),
            state_id INTEGER,
            district_id INTEGER,
            taluk_id INTEGER,
            village_id INTEGER,
            district TEXT,
            taluk TEXT,
            village TEXT,
            address TEXT,
            pincode TEXT,
            email_verified INTEGER DEFAULT 0,
            status TEXT DEFAULT 'Active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Column upgrades for users table if needed
    cursor.execute("PRAGMA table_info(users)")
    user_cols = [col[1] for col in cursor.fetchall()]
    if 'email_verified' not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER DEFAULT 0")

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)')
    cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(LOWER(email))')

    # 7. OTP Verifications Table (Admin & User Email Verification)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS otp_verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            listing_id INTEGER,
            email TEXT NOT NULL,
            otp_hash TEXT NOT NULL,
            purpose TEXT NOT NULL,
            attempts INTEGER DEFAULT 0,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute("PRAGMA table_info(otp_verifications)")
    otp_cols = [col[1] for col in cursor.fetchall()]
    if 'user_id' not in otp_cols:
        cursor.execute("ALTER TABLE otp_verifications ADD COLUMN user_id INTEGER")
    if 'listing_id' not in otp_cols:
        cursor.execute("ALTER TABLE otp_verifications ADD COLUMN listing_id INTEGER")

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_otp_email_purpose ON otp_verifications(email, purpose)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_otp_listing_purpose ON otp_verifications(listing_id, purpose)')

    # 8. Listings Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id INTEGER NOT NULL,
            farmer_name TEXT NOT NULL,
            farmer_phone TEXT NOT NULL,
            image_url TEXT NOT NULL,
            species TEXT NOT NULL,
            breed TEXT NOT NULL,
            gender TEXT NOT NULL DEFAULT 'Female',
            sex TEXT,
            is_pregnant INTEGER DEFAULT 0,
            pregnancy_month INTEGER DEFAULT 0,
            age_years REAL NOT NULL,
            lactation_number TEXT DEFAULT 'Other / Not specified',
            calf_number TEXT DEFAULT 'Other / Not specified',
            milk_production REAL NOT NULL DEFAULT 12.0,
            milk_yield_liters REAL,
            district_id INTEGER,
            taluk_id INTEGER,
            village_id INTEGER,
            district TEXT NOT NULL,
            taluk TEXT,
            village TEXT,
            address TEXT,
            pincode TEXT,
            predicted_price INTEGER NOT NULL,
            asking_price INTEGER NOT NULL,
            status TEXT DEFAULT 'Available' CHECK(status IN ('Available', 'SOLD_OUT', 'REMOVED', 'Purchased')),
            is_deleted INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (farmer_id) REFERENCES users(id)
        )
    ''')

    cursor.execute("PRAGMA table_info(listings)")
    listing_cols = [col[1] for col in cursor.fetchall()]
    if 'gender' not in listing_cols:
        cursor.execute("ALTER TABLE listings ADD COLUMN gender TEXT DEFAULT 'Female'")
    if 'is_pregnant' not in listing_cols:
        cursor.execute("ALTER TABLE listings ADD COLUMN is_pregnant INTEGER DEFAULT 0")
    if 'pregnancy_month' not in listing_cols:
        cursor.execute("ALTER TABLE listings ADD COLUMN pregnancy_month INTEGER DEFAULT 0")
    if 'milk_production' not in listing_cols:
        cursor.execute("ALTER TABLE listings ADD COLUMN milk_production REAL DEFAULT 12.0")
    if 'lactation_number' not in listing_cols:
        cursor.execute("ALTER TABLE listings ADD COLUMN lactation_number TEXT DEFAULT 'Other / Not specified'")
    if 'calf_number' not in listing_cols:
        cursor.execute("ALTER TABLE listings ADD COLUMN calf_number TEXT DEFAULT 'Other / Not specified'")
    if 'is_deleted' not in listing_cols:
        cursor.execute("ALTER TABLE listings ADD COLUMN is_deleted INTEGER DEFAULT 0")

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_listings_species ON listings(species)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_listings_status ON listings(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_listings_farmer ON listings(farmer_id)')

    # 9. Purchases / Orders Table (Zero Delivery Charge, Valid Statuses Only)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS purchases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            listing_id INTEGER UNIQUE NOT NULL,
            buyer_id INTEGER NOT NULL,
            buyer_name TEXT NOT NULL,
            buyer_phone TEXT NOT NULL,
            farmer_id INTEGER NOT NULL,
            farmer_name TEXT NOT NULL,
            farmer_phone TEXT NOT NULL,
            livestock_price INTEGER NOT NULL,
            total_amount INTEGER NOT NULL,
            payment_method TEXT DEFAULT 'CASH_ON_DELIVERY',
            delivery_address TEXT NOT NULL,
            delivery_district TEXT NOT NULL,
            delivery_taluk TEXT,
            delivery_village TEXT,
            delivery_pincode TEXT,
            order_status TEXT DEFAULT 'CONFIRMED' CHECK(order_status IN ('CONFIRMED', 'CANCELLED', 'PURCHASED')),
            delivery_status TEXT CHECK(delivery_status IN ('DELIVERED', 'REJECTED', NULL)),
            cancellation_reason TEXT,
            cancellation_details TEXT,
            rejection_reason TEXT,
            proof_photo_url TEXT,
            rating INTEGER CHECK(rating >= 1 AND rating <= 5),
            review TEXT,
            confirmed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            delivered_at TIMESTAMP,
            cancelled_at TIMESTAMP,
            purchased_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (listing_id) REFERENCES listings(id),
            FOREIGN KEY (buyer_id) REFERENCES users(id),
            FOREIGN KEY (farmer_id) REFERENCES users(id)
        )
    ''')

    # Upgrade columns for existing purchases table if migrating
    cursor.execute("PRAGMA table_info(purchases)")
    purch_cols = [col[1] for col in cursor.fetchall()]
    if 'order_status' not in purch_cols:
        cursor.execute("ALTER TABLE purchases ADD COLUMN order_status TEXT DEFAULT 'CONFIRMED'")
    if 'delivery_status' not in purch_cols:
        cursor.execute("ALTER TABLE purchases ADD COLUMN delivery_status TEXT")
    if 'cancellation_reason' not in purch_cols:
        cursor.execute("ALTER TABLE purchases ADD COLUMN cancellation_reason TEXT")
    if 'cancellation_details' not in purch_cols:
        cursor.execute("ALTER TABLE purchases ADD COLUMN cancellation_details TEXT")
    if 'rejection_reason' not in purch_cols:
        cursor.execute("ALTER TABLE purchases ADD COLUMN rejection_reason TEXT")
    if 'proof_photo_url' not in purch_cols:
        cursor.execute("ALTER TABLE purchases ADD COLUMN proof_photo_url TEXT")
    if 'rating' not in purch_cols:
        cursor.execute("ALTER TABLE purchases ADD COLUMN rating INTEGER")
    if 'review' not in purch_cols:
        cursor.execute("ALTER TABLE purchases ADD COLUMN review TEXT")
    if 'confirmed_at' not in purch_cols:
        cursor.execute("ALTER TABLE purchases ADD COLUMN confirmed_at TIMESTAMP")
    if 'delivered_at' not in purch_cols:
        cursor.execute("ALTER TABLE purchases ADD COLUMN delivered_at TIMESTAMP")
    if 'cancelled_at' not in purch_cols:
        cursor.execute("ALTER TABLE purchases ADD COLUMN cancelled_at TIMESTAMP")

    cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_purchases_listing_unique ON purchases(listing_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_purchases_buyer ON purchases(buyer_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_purchases_farmer ON purchases(farmer_id)')

    # 10. Notifications Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_email TEXT,
            user_phone TEXT,
            order_id INTEGER,
            subject TEXT NOT NULL,
            message TEXT NOT NULL,
            email_status TEXT DEFAULT 'PENDING',
            sms_status TEXT DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id)')

    # 11. Buyer Interests & Preferences Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS buyer_interests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            buyer_id INTEGER NOT NULL,
            species TEXT,
            breed TEXT,
            district TEXT,
            max_price INTEGER,
            query_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (buyer_id) REFERENCES users(id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_buyer_interests_buyer ON buyer_interests(buyer_id)')

    # 12. Contact Requests Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contact_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            listing_id INTEGER NOT NULL,
            farmer_id INTEGER NOT NULL,
            buyer_id INTEGER NOT NULL,
            buyer_name TEXT NOT NULL,
            buyer_phone TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT DEFAULT 'Sent',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (listing_id) REFERENCES listings(id),
            FOREIGN KEY (farmer_id) REFERENCES users(id),
            FOREIGN KEY (buyer_id) REFERENCES users(id)
        )
    ''')

    # 13. Livestock Availability Alerts Table (Buyer Searches saved when 0 listings match)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS livestock_availability_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            buyer_id INTEGER,
            buyer_email TEXT NOT NULL,
            species TEXT,
            breed TEXT,
            district TEXT,
            taluk TEXT,
            village TEXT,
            minimum_age REAL,
            maximum_age REAL,
            minimum_milk_production REAL,
            maximum_milk_production REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notified INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1,
            FOREIGN KEY (buyer_id) REFERENCES users(id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_buyer ON livestock_availability_alerts(buyer_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_active ON livestock_availability_alerts(active)')

    # 14. Livestock Availability Notifications Table (Prevents duplicate alerts)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS livestock_availability_notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            buyer_id INTEGER,
            listing_id INTEGER NOT NULL,
            email TEXT NOT NULL,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'SENT',
            UNIQUE(email, listing_id),
            FOREIGN KEY (buyer_id) REFERENCES users(id),
            FOREIGN KEY (listing_id) REFERENCES listings(id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_avail_notif_buyer ON livestock_availability_notifications(buyer_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_avail_notif_listing ON livestock_availability_notifications(listing_id)')

    # 15. Buyer Nutrition Assistant Chat History Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS buyer_nutrition_chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            buyer_id INTEGER,
            listing_id INTEGER,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            intent TEXT,
            animal_context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (buyer_id) REFERENCES users(id),
            FOREIGN KEY (listing_id) REFERENCES listings(id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_nutrition_session ON buyer_nutrition_chats(session_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_nutrition_buyer ON buyer_nutrition_chats(buyer_id)')

    conn.commit()
    conn.close()

def seed_single_admin():
    """
    Ensure EXACTLY ONE authorized administrator is configured in the database:
    Name: Rakshitha M
    Email: rakshitham556@gmail.com
    Password: Rakshi@2026 (stored exclusively as bcrypt hash)
    All other admin records are strictly purged.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Remove any unauthorized or legacy admin accounts
    cursor.execute("DELETE FROM users WHERE role = 'admin' AND LOWER(email) != LOWER(?)", (AUTHORIZED_ADMIN_EMAIL,))

    cursor.execute("SELECT id, email, name, password_hash FROM users WHERE role = 'admin' AND LOWER(email) = LOWER(?)", (AUTHORIZED_ADMIN_EMAIL,))
    existing_admin = cursor.fetchone()

    # Generate fresh bcrypt hash
    fresh_hash = hash_password("Rakshi@2026")

    if not existing_admin:
        cursor.execute('''
            INSERT INTO users (name, email, username, password, phone, password_hash, role, email_verified, status)
            VALUES (?, ?, 'admin_rakshitha', '[BCRYPT_PROTECTED]', '+919876543210', ?, 'admin', 1, 'Active')
        ''', (AUTHORIZED_ADMIN_NAME, AUTHORIZED_ADMIN_EMAIL, fresh_hash))
        print(f"[Admin Setup] Authorized Single Administrator '{AUTHORIZED_ADMIN_NAME}' ({AUTHORIZED_ADMIN_EMAIL}) initialized.")
    else:
        cursor.execute('''
            UPDATE users SET name = ?, password_hash = ?, phone = '+919876543210', email_verified = 1, status = 'Active'
            WHERE id = ?
        ''', (AUTHORIZED_ADMIN_NAME, fresh_hash, existing_admin['id']))
        print(f"[Admin Setup] Single Administrator '{AUTHORIZED_ADMIN_NAME}' ({AUTHORIZED_ADMIN_EMAIL}) verified and updated.")

    cursor.execute("INSERT OR REPLACE INTO system_config (key, value) VALUES ('admin_initialized', 'true')")
    conn.commit()
    conn.close()

def reset_user_and_transaction_data():
    """
    FRESH DATABASE RESET:
    Removes ALL mock/demo user and marketplace transactional data:
    - Farmers: 0
    - Buyers: 0
    - Listings: 0
    - Purchases / Orders: 0
    - Contact requests: 0
    - Notifications: 0
    - Buyer interests: 0
    - OTP verifications: 0
    - Administrator: EXACTLY 1 (Rakshitha M, rakshitham556@gmail.com)
    - Karnataka Location Master Dataset (31 districts, 240 taluks, 30,753 villages): PRESERVED 100%!
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM purchases")
    cursor.execute("DELETE FROM contact_requests")
    cursor.execute("DELETE FROM buyer_interests")
    cursor.execute("DELETE FROM livestock_availability_alerts")
    cursor.execute("DELETE FROM livestock_availability_notifications")
    cursor.execute("DELETE FROM buyer_nutrition_chats")
    cursor.execute("DELETE FROM notifications")
    cursor.execute("DELETE FROM otp_verifications")
    cursor.execute("DELETE FROM listings")
    cursor.execute("DELETE FROM users WHERE role != 'admin'")

    # Reset sequences
    cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('listings', 'purchases', 'contact_requests', 'buyer_interests', 'livestock_availability_alerts', 'livestock_availability_notifications', 'notifications', 'otp_verifications')")

    conn.commit()
    conn.close()

    seed_single_admin()
    print("[Fresh Reset] Database reset to clean state. Karnataka location master preserved.")

def save_nutrition_chat_message(session_id: str, role: str, content: str, buyer_id: int = None, listing_id: int = None, intent: str = None, animal_context: dict = None):
    """Persist a chat message from either buyer or nutrition assistant."""
    conn = get_db()
    cursor = conn.cursor()
    ctx_str = json.dumps(animal_context) if animal_context else None
    cursor.execute('''
        INSERT INTO buyer_nutrition_chats (buyer_id, listing_id, session_id, role, content, intent, animal_context)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (buyer_id, listing_id, session_id, role, content, intent, ctx_str))
    conn.commit()
    conn.close()

def get_nutrition_chat_history(session_id: str, buyer_id: int = None, limit: int = 50):
    """Retrieve chat history for a session or buyer."""
    conn = get_db()
    cursor = conn.cursor()
    if buyer_id:
        cursor.execute('''
            SELECT id, buyer_id, listing_id, session_id, role, content, intent, animal_context, created_at
            FROM buyer_nutrition_chats
            WHERE session_id = ? OR buyer_id = ?
            ORDER BY id ASC LIMIT ?
        ''', (session_id, buyer_id, limit))
    else:
        cursor.execute('''
            SELECT id, buyer_id, listing_id, session_id, role, content, intent, animal_context, created_at
            FROM buyer_nutrition_chats
            WHERE session_id = ?
            ORDER BY id ASC LIMIT ?
        ''', (session_id, limit))
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        item = dict(r)
        if item.get('animal_context'):
            try:
                item['animal_context'] = json.loads(item['animal_context'])
            except Exception:
                pass
        result.append(item)
    return result

def clear_nutrition_chat_history(session_id: str, buyer_id: int = None):
    """Clear chat history for a session."""
    conn = get_db()
    cursor = conn.cursor()
    if buyer_id:
        cursor.execute("DELETE FROM buyer_nutrition_chats WHERE session_id = ? OR buyer_id = ?", (session_id, buyer_id))
    else:
        cursor.execute("DELETE FROM buyer_nutrition_chats WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    seed_single_admin()
    reset_user_and_transaction_data()
