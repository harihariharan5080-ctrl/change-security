import os
import sqlite3
import pickle
import logging
import numpy as np
from datetime import datetime

DATABASE_PATH = "database.db"

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(face_handler):
    """Initializes SQLite database and populates FaceHandler cache from DB records."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("encodings", exist_ok=True)
    os.makedirs("attendance", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create Employees table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            EmployeeID TEXT PRIMARY KEY,
            Name TEXT NOT NULL,
            Department TEXT,
            Phone TEXT,
            Email TEXT,
            ImageFolder TEXT,
            FaceEncoding BLOB,
            CreatedAt TEXT
        )
    ''')
    
    # Run dynamic schema migrations for new columns
    cursor.execute("PRAGMA table_info(employees)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'Designation' not in columns:
        cursor.execute("ALTER TABLE employees ADD COLUMN Designation TEXT")
        logging.info("Schema migration: Added Designation column to employees table.")
    if 'DateOfJoining' not in columns:
        cursor.execute("ALTER TABLE employees ADD COLUMN DateOfJoining TEXT")
        logging.info("Schema migration: Added DateOfJoining column to employees table.")
    
    
    # Create Attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            AttendanceID INTEGER PRIMARY KEY AUTOINCREMENT,
            EmployeeID TEXT,
            Name TEXT,
            Date TEXT,
            Time TEXT,
            Status TEXT,
            FOREIGN KEY (EmployeeID) REFERENCES employees (EmployeeID)
        )
    ''')
    
    # Create Admins table for Email/Password login security
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            Email TEXT PRIMARY KEY,
            PasswordHash TEXT NOT NULL,
            Name TEXT NOT NULL,
            CreatedAt TEXT
        )
    ''')
    
    # Initialize default admin profile if empty
    cursor.execute("SELECT COUNT(*) FROM admins")
    if cursor.fetchone()[0] == 0:
        from werkzeug.security import generate_password_hash
        default_email = "hari123"
        default_pwd_hash = generate_password_hash("hari@123")
        default_name = "System Administrator"
        cursor.execute(
            "INSERT INTO admins (Email, PasswordHash, Name, CreatedAt) VALUES (?, ?, ?, ?)",
            (default_email, default_pwd_hash, default_name, datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        )
        logging.info("Initialized default admin security profile (hari123 / hari@123).")
    else:
        from werkzeug.security import generate_password_hash
        # Clean up old default admin account if it exists
        cursor.execute("DELETE FROM admins WHERE Email = 'admin@crt.com'")
        # Ensure the custom credential login ID 'hari123' is active
        cursor.execute("SELECT 1 FROM admins WHERE Email = 'hari123'")
        if not cursor.fetchone():
            default_email = "hari123"
            default_pwd_hash = generate_password_hash("hari@123")
            default_name = "System Administrator"
            cursor.execute(
                "INSERT INTO admins (Email, PasswordHash, Name, CreatedAt) VALUES (?, ?, ?, ?)",
                (default_email, default_pwd_hash, default_name, datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
            )
            logging.info("Migrated administrator credentials login ID to: hari123")
    
    conn.commit()
    
    # Load all face encodings from Database into FaceHandler
    cursor.execute("SELECT EmployeeID, Name, FaceEncoding FROM employees WHERE FaceEncoding IS NOT NULL")
    rows = cursor.fetchall()
    
    encodings = []
    names = []
    ids = []
    
    for row in rows:
        try:
            encoding_vec = pickle.loads(row['FaceEncoding'])
            encodings.append(np.array(encoding_vec))
            names.append(row['Name'])
            ids.append(row['EmployeeID'])
        except Exception as e:
            logging.error(f"Error loading face encoding from DB for employee {row['Name']}: {str(e)}")
            
    face_handler.known_face_encodings = encodings
    face_handler.known_face_names = names
    face_handler.known_face_ids = ids
    face_handler.save_known_faces()  # Keep faces.pkl synced
    
    logging.info(f"Database initialized. Synchronized {len(names)} employees in FaceHandler.")
    conn.close()
