import os
import cv2
import numpy as np
import pandas as pd
import time
import threading
import pickle
import logging
from datetime import datetime
from face_handler import FaceHandler
from database import get_db_connection

# Initialize Face Handler
face_handler = FaceHandler()

# Global variables for Webcam and Registration State
camera_lock = threading.Lock()
active_streams = 0

# System modes: 'passive' (just detect), 'register' (capture for employee), 'attendance' (detect and check in)
system_mode = 'passive'

# Registration State
register_data = {
    'employee_id': '',
    'name': '',
    'department': '',
    'designation': '',
    'phone': '',
    'email': '',
    'date_of_joining': '',
    'captured_frames': []
}
register_target_count = 30
register_is_capturing = False
register_status = {
    'count': 0,
    'status': 'idle',  # 'idle', 'capturing', 'completed', 'error'
    'message': ''
}

# Latest verified check-in
latest_match = {
    'employee_id': None,
    'name': None,
    'department': None,
    'time': None,
    'date': None,
    'status': None,
    'timestamp': 0
}

# ----------------- Camera Class -----------------
class VideoCamera:
    def __init__(self):
        self.video = None
        self.active = False
        self.lock = threading.Lock()
        self.thread = None
        self.latest_frame = None

    def start(self):
        with self.lock:
            if not self.active:
                # Try opening camera 0
                self.video = cv2.VideoCapture(0)
                if not self.video.isOpened():
                    # Check camera index 1 if 0 is blocked
                    self.video = cv2.VideoCapture(1)
                    if not self.video.isOpened():
                        logging.error("Failed to open webcam (indices 0 & 1).")
                        return False
                self.active = True
                self.thread = threading.Thread(target=self._update, daemon=True)
                self.thread.start()
                logging.info("Webcam feed thread started.")
                return True
        return False

    def stop(self):
        with self.lock:
            if self.active:
                self.active = False
                if self.video:
                    self.video.release()
                    self.video = None
                logging.info("Webcam feed thread stopped.")

    def _update(self):
        while self.active:
            ret, frame = self.video.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue
            with self.lock:
                self.latest_frame = frame.copy()
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

# Instantiate global camera object
camera = VideoCamera()

# ----------------- Frame Processors -----------------
def process_passive_frame(frame):
    """Detects faces passively and overlays basic boxes."""
    boxes = face_handler.detect_faces(frame)
    for (top, right, bottom, left) in boxes:
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (243, 150, 33), 2)
        cv2.putText(frame, "Face Detected", (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (243, 150, 33), 1, cv2.LINE_AA)
    return frame

def process_register_frame(frame):
    """Handles face image capture frames during registration."""
    global register_is_capturing, register_data, register_status
    
    boxes = face_handler.detect_faces(frame)
    h_img, w_img = frame.shape[:2]
    
    # Overlay scanning guide circle
    cv2.circle(frame, (w_img // 2, h_img // 2), 140, (243, 150, 33), 1, cv2.LINE_AA)
    
    if len(boxes) == 0:
        cv2.putText(frame, "Align your face in the guide", (w_img // 2 - 120, h_img // 2 - 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        # Draw the target box
        largest_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[1]-b[3]))
        top, right, bottom, left = largest_box
        cv2.rectangle(frame, (left, top), (right, bottom), (33, 150, 243), 2)
        
        if register_is_capturing:
            # Capture frame
            current_count = len(register_data['captured_frames'])
            if current_count < register_target_count:
                register_data['captured_frames'].append(frame.copy())
                register_status['count'] = current_count + 1
                register_status['message'] = f"Capturing: {current_count + 1}/{register_target_count}"
                
                # Overlay scanning animation indicator
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"SCANNING {current_count + 1}", (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                register_is_capturing = False
                register_status['status'] = 'completed'
                register_status['message'] = "Capture completed. Saving face..."
    return frame

def log_attendance_to_csv(emp_id, name, date_str, time_str, status):
    """Saves attendance records into attendance.csv."""
    csv_file = os.path.join("attendance", "attendance.csv")
    new_entry = pd.DataFrame([{
        "Employee ID": emp_id,
        "Name": name,
        "Date": date_str,
        "Time": time_str,
        "Status": status
    }])
    
    try:
        if not os.path.exists(csv_file):
            new_entry.to_csv(csv_file, index=False)
        else:
            new_entry.to_csv(csv_file, mode='a', header=False, index=False)
        logging.info(f"CSV logged: {name} at {time_str}")
    except Exception as e:
        logging.error(f"Failed to log attendance to CSV: {str(e)}")

def log_attendance_to_excel_book(emp_id, date_str, time_str, status):
    """Appends attendance logs to dataset/book.xlsx with custom lowercase headers."""
    excel_path = os.path.join("dataset", "book.xlsx")
    new_row = {
        "employe id": emp_id,
        "date": date_str,
        "time": time_str,
        "present or absent": status
    }
    
    try:
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        if os.path.exists(excel_path):
            try:
                df = pd.read_excel(excel_path, engine='openpyxl')
                required_cols = ["employe id", "date", "time", "present or absent"]
                for col in required_cols:
                     if col not in df.columns:
                        df = pd.DataFrame(columns=required_cols)
                        break
            except Exception as e:
                logging.warning(f"Could not read existing book.xlsx, initializing new: {str(e)}")
                df = pd.DataFrame(columns=["employe id", "date", "time", "present or absent"])
        else:
            df = pd.DataFrame(columns=["employe id", "date", "time", "present or absent"])
            
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_excel(excel_path, index=False, engine='openpyxl')
        logging.info(f"Excel book.xlsx logged check-in for Employee ID: {emp_id}")
    except Exception as e:
        logging.error(f"Failed to log attendance to book.xlsx: {str(e)}")

def process_attendance_frame(frame):
    """Processes attendance matching in real-time."""
    global latest_match
    boxes = face_handler.detect_faces(frame)
    
    for box in boxes:
        top, right, bottom, left = box
        
        # Match faces
        emp_id, emp_name, dist = face_handler.identify_face(frame, box)
        
        if emp_id:
            # Check database for today's check-in
            today_date = datetime.now().strftime("%d-%m-%Y")
            now_time = datetime.now().strftime("%H:%M:%S")
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM attendance WHERE EmployeeID = ? AND Date = ?", (emp_id, today_date))
            already_marked = cursor.fetchone() is not None
            
            if not already_marked:
                # Retrieve department info
                cursor.execute("SELECT Department FROM employees WHERE EmployeeID = ?", (emp_id,))
                dept_row = cursor.fetchone()
                dept = dept_row['Department'] if dept_row else "Unknown"
                
                # Insert check-in log
                cursor.execute(
                    "INSERT INTO attendance (EmployeeID, Name, Date, Time, Status) VALUES (?, ?, ?, ?, ?)",
                    (emp_id, emp_name, today_date, now_time, "Present")
                )
                conn.commit()
                
                # Append to CSV
                log_attendance_to_csv(emp_id, emp_name, today_date, now_time, "Present")
                
                # Append to custom excel book.xlsx
                log_attendance_to_excel_book(emp_id, today_date, now_time, "Present")
                
                # Set latest verified match for JS polling alerts
                latest_match = {
                    'employee_id': emp_id,
                    'name': emp_name,
                    'department': dept,
                    'time': now_time,
                    'date': today_date,
                    'status': 'Present',
                    'timestamp': time.time()
                }
                
                logging.info(f"Verified & Checked In: {emp_name} ({emp_id}) at {now_time}")
            else:
                # Set duplicate alert state if expired
                if time.time() - latest_match['timestamp'] > 2.0:
                    latest_match = {
                        'employee_id': emp_id,
                        'name': emp_name,
                        'department': 'N/A',
                        'time': now_time,
                        'date': today_date,
                        'status': 'Duplicate',
                        'timestamp': time.time()
                    }
            
            conn.close()
            
            # Draw green verified box
            cv2.rectangle(frame, (left, top), (right, bottom), (46, 125, 50), 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (46, 125, 50), cv2.FILLED)
            cv2.putText(frame, f"{emp_name} - VERIFIED", (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Set unknown alert state if expired
            if time.time() - latest_match['timestamp'] > 2.0:
                latest_match = {
                    'employee_id': 'unknown',
                    'name': 'Unknown Person',
                    'department': 'N/A',
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'date': datetime.now().strftime("%d-%m-%Y"),
                    'status': 'Unknown',
                    'timestamp': time.time()
                }
            
            # Draw red unknown box
            cv2.rectangle(frame, (left, top), (right, bottom), (211, 47, 47), 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (211, 47, 47), cv2.FILLED)
            cv2.putText(frame, "Unknown Face", (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
    return frame

# ----------------- Video Feed Endpoint -----------------
def gen_frames():
    global active_streams, system_mode
    with camera_lock:
        active_streams += 1
        if active_streams == 1:
            success = camera.start()
            if not success:
                active_streams -= 1
                return
            
    try:
        while True:
            if not camera.active:
                break
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue
                
            processed = frame.copy()
            if system_mode == 'register':
                processed = process_register_frame(processed)
            elif system_mode == 'attendance':
                processed = process_attendance_frame(processed)
            else:
                processed = process_passive_frame(processed)
                
            ret, jpeg = cv2.imencode('.jpg', processed)
            if not ret:
                time.sleep(0.03)
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)  # roughly 30 FPS cap
            
    finally:
        with camera_lock:
            active_streams -= 1
            if active_streams == 0:
                camera.stop()

def save_registered_employee():
    """Processes captured frames, generates face encodings, and saves to database."""
    global register_data, register_status, system_mode
    
    emp_id = register_data['employee_id']
    name = register_data['name']
    frames = register_data['captured_frames']
    
    if len(frames) < 10:
        logging.warning(f"Registration failed: only {len(frames)} frames captured.")
        return False
        
    # Save frames to dataset directory
    sanitized_name = "".join([c for c in name if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')
    folder_name = f"{sanitized_name}_{emp_id}"
    target_dir = os.path.join("dataset", folder_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # Save images to folder
    saved_frames_paths = []
    for idx, f in enumerate(frames):
        filepath = os.path.join(target_dir, f"image{idx+1}.jpg")
        cv2.imwrite(filepath, f)
        saved_frames_paths.append(filepath)
        
    # Generate and save encodings
    encoding = face_handler.add_employee_face(emp_id, name, frames)
    if encoding is None:
        # Cleanup folder
        for p in saved_frames_paths:
            try:
                os.remove(p)
            except:
                pass
        try:
            os.rmdir(target_dir)
        except:
            pass
        return False
        
    # Save to SQLite DB
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO employees (EmployeeID, Name, Department, Designation, Phone, Email, DateOfJoining, ImageFolder, FaceEncoding, CreatedAt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (emp_id, name, register_data['department'], register_data.get('designation', ''), register_data['phone'], register_data['email'], 
             register_data.get('date_of_joining', ''), target_dir, sqlite3.Binary(pickle.dumps(encoding)), datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        )
        conn.commit()
        conn.close()
        logging.info(f"Database insertion successful for {name} ({emp_id}).")
        return True
    except Exception as e:
        logging.error(f"Failed to insert employee into SQLite: {str(e)}")
        return False
