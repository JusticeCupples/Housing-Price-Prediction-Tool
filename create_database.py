import sqlite3
import os

db_name = 'state_populations.db'

# Remove the existing database file if it exists
if os.path.exists(db_name):
    os.remove(db_name)

conn = sqlite3.connect(db_name)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS state_populations (
    state TEXT PRIMARY KEY,
    population INTEGER
)
''')

state_populations = [
    ('AL', 4903185), ('AK', 731545), ('AZ', 7278717), ('AR', 3017804), ('CA', 39512223),
    ('CO', 5773714), ('CT', 3565287), ('DE', 989948), ('FL', 21538187), ('GA', 10711908),
    ('HI', 1415872), ('ID', 1787065), ('IL', 12671821), ('IN', 6732219), ('IA', 3155070),
    ('KS', 2913314), ('KY', 4467673), ('LA', 4648794), ('ME', 1344212), ('MD', 6045680),
    ('MA', 6892503), ('MI', 9986857), ('MN', 5639632), ('MS', 2976149), ('MO', 6137428),
    ('MT', 1068778), ('NE', 1934408), ('NV', 3080156), ('NH', 1359711), ('NJ', 8882190),
    ('NM', 2096829), ('NY', 19453561), ('NC', 10488084), ('ND', 762062), ('OH', 11689100),
    ('OK', 3956971), ('OR', 4217737), ('PA', 12801989), ('RI', 1059361), ('SC', 5148714),
    ('SD', 884659), ('TN', 6829174), ('TX', 28995881), ('UT', 3205958), ('VT', 623989),
    ('VA', 8535519), ('WA', 7614893), ('WV', 1792147), ('WI', 5822434), ('WY', 578759)
]

cursor.executemany('INSERT OR REPLACE INTO state_populations VALUES (?, ?)', state_populations)

conn.commit()
conn.close()

print(f"Database '{db_name}' created and populated successfully.")
print(f"Current directory: {os.getcwd()}")
print(f"Database file exists: {os.path.exists(db_name)}")

# Verify the database
conn = sqlite3.connect(db_name)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM state_populations")
count = cursor.fetchone()[0]
conn.close()
print(f"Number of records in the database: {count}")
