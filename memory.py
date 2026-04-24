import sqlite3
import time

conn = sqlite3.connect("memory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    role TEXT,
    content TEXT,
    timestamp REAL
)
""")
conn.commit()

def save_message(user_id, role, content):
    cursor.execute(
        "INSERT INTO chat (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, role, content, time.time())
    )
    conn.commit()

def load_history(user_id, limit=5):
    cursor.execute("""
        SELECT role, content FROM chat
        WHERE user_id=?
        ORDER BY timestamp ASC
        LIMIT ?
    """, (user_id, limit))

    rows = cursor.fetchall()
    return [{"role": r, "content": c} for r, c in rows]

def clear_history(user_id):
    cursor.execute("DELETE FROM chat WHERE user_id=?", (user_id,))
    conn.commit()