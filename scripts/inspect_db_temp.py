import sqlite3
conn = sqlite3.connect('livestock_app.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [r[0] for r in c.fetchall()])
c.execute("SELECT id, species, breed, gender, image_path, created_at FROM listings ORDER BY id DESC LIMIT 10")
for r in c.fetchall():
    print(r)
