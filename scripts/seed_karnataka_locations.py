import os
import csv
import json
import sqlite3
import sys
# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_db, init_db, seed_single_admin
CSV_PATH = os.path.join("data", "karnataka_villages.csv")
COORDS_PATH = os.path.join("data", "karnataka_coords.json")
def seed_karnataka_locations():
    init_db()
    seed_single_admin()

    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    # Load Coordinates if present
    coords_dict = {}
    if os.path.exists(COORDS_PATH):
        try:
            with open(COORDS_PATH, 'r', encoding='utf-8') as f:
                coords_dict = json.load(f)
        except Exception as e:
            print(f"Warning loading coords: {e}")

    conn = get_db()
    cursor = conn.cursor()

    # Clear existing location tables for clean seed
    cursor.execute("DELETE FROM villages")
    cursor.execute("DELETE FROM taluks")
    cursor.execute("DELETE FROM districts")
    cursor.execute("DELETE FROM states")

    # Insert Karnataka State
    cursor.execute("INSERT INTO states (name, code) VALUES ('Karnataka', 'KA')", ())
    state_id = cursor.lastrowid

    district_map = {} # name -> id
    district_codes = {}
    taluk_map = {} # (district_id, taluk_name) -> id
    taluk_codes = {}

    district_coords = {} # dist_id -> list of [lat, lng]
    taluk_coords = {} # taluk_id -> list of [lat, lng]

    village_records = []
    seen_villages = set()

    duplicates_removed = 0
    invalid_records = 0

    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist_name = (row.get('District') or '').strip()
            dist_code = (row.get('District Code') or '').strip()
            taluk_name = (row.get('Mandal') or '').strip() # Mandal/Taluk
            taluk_code = (row.get('Mandal Code') or '').strip()
            village_name = (row.get('Village') or '').strip()
            village_native = (row.get('Village (Native)') or '').strip()
            village_code = (row.get('Village Code') or '').strip()
            pincode = (row.get('Pincode') or '').strip()

            if not dist_name or not taluk_name or not village_name:
                invalid_records += 1
                continue

            # Standardize Title Case
            dist_name = dist_name.title()
            taluk_name = taluk_name.title()

            # Ensure District exists
            if dist_name not in district_map:
                cursor.execute(
                    "INSERT INTO districts (state_id, name, code) VALUES (?, ?, ?)",
                    (state_id, dist_name, dist_code)
                )
                dist_id = cursor.lastrowid
                district_map[dist_name] = dist_id
                district_codes[dist_id] = dist_code
                district_coords[dist_id] = []
            else:
                dist_id = district_map[dist_name]

            # Ensure Taluk exists
            taluk_key = (dist_id, taluk_name)
            if taluk_key not in taluk_map:
                cursor.execute(
                    "INSERT INTO taluks (district_id, name, code) VALUES (?, ?, ?)",
                    (dist_id, taluk_name, taluk_code)
                )
                taluk_id = cursor.lastrowid
                taluk_map[taluk_key] = taluk_id
                taluk_codes[taluk_id] = taluk_code
                taluk_coords[taluk_id] = []
            else:
                taluk_id = taluk_map[taluk_key]

            # Deduplication key
            v_key = (taluk_id, village_code if village_code else village_name.lower())
            if v_key in seen_villages:
                duplicates_removed += 1
                continue
            seen_villages.add(v_key)

            # Coordinates lookup
            lat, lng = None, None
            if village_code and village_code in coords_dict:
                coord = coords_dict[village_code]
                if isinstance(coord, list) and len(coord) == 2:
                    lat, lng = float(coord[0]), float(coord[1])
                    district_coords[dist_id].append((lat, lng))
                    taluk_coords[taluk_id].append((lat, lng))

            village_records.append((
                taluk_id, village_name, village_native, village_code, pincode, lat, lng
            ))

    # Bulk insert villages in batches
    cursor.executemany('''
        INSERT INTO villages (taluk_id, name, native_name, code, pincode, latitude, longitude)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', village_records)

    # Approximate District and Taluk Centroids from child village coordinates
    for dist_id, points in district_coords.items():
        if points:
            avg_lat = sum(p[0] for p in points) / len(points)
            avg_lng = sum(p[1] for p in points) / len(points)
            cursor.execute("UPDATE districts SET latitude = ?, longitude = ? WHERE id = ?", (avg_lat, avg_lng, dist_id))

    for taluk_id, points in taluk_coords.items():
        if points:
            avg_lat = sum(p[0] for p in points) / len(points)
            avg_lng = sum(p[1] for p in points) / len(points)
            cursor.execute("UPDATE taluks SET latitude = ?, longitude = ? WHERE id = ?", (avg_lat, avg_lng, taluk_id))

    conn.commit()

    # Query final counts from database
    cursor.execute("SELECT COUNT(*) FROM districts")
    dist_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM taluks")
    taluk_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM villages")
    village_count = cursor.fetchone()[0]

    conn.close()

    print("\n=============================================")
    print("KARNATAKA LOCATION IMPORT")
    print("=============================================")
    print(f"Districts imported: {dist_count}")
    print(f"Taluks imported: {taluk_count}")
    print(f"Villages imported: {village_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Invalid records: {invalid_records}")
    print("=============================================\n")
if __name__ == '__main__':
    seed_karnataka_locations() 
