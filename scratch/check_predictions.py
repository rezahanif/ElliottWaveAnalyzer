import sqlite3
conn = sqlite3.connect("data/predictions.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in data/predictions.db:", tables)
conn.close()
