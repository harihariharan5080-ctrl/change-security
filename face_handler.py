import os
import cv2
import numpy as np
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "app.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Attempt to import face_recognition
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
    logging.info("face_recognition library loaded successfully.")
except ImportError:
    HAS_FACE_RECOGNITION = False
    logging.warning("face_recognition library NOT found. Falling back to OpenCV-based detection and custom feature descriptor recognition.")

class FaceHandler:
    def __init__(self, encodings_path=os.path.join("encodings", "faces.pkl")):
        self.encodings_path = encodings_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        # Load OpenCV Face Detector Haar Cascade in case fallback is needed
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logging.error(f"Failed to load OpenCV Haar Cascade from {cascade_path}")
            
        self.load_known_faces()

    def load_known_faces(self):
        """Loads known face encodings and names from pickle file."""
        if os.path.exists(self.encodings_path):
            try:
                with open(self.encodings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get("encodings", [])
                    self.known_face_names = data.get("names", [])
                    self.known_face_ids = data.get("ids", [])
                logging.info(f"Loaded {len(self.known_face_names)} face encodings from {self.encodings_path}")
            except Exception as e:
                logging.error(f"Error loading face encodings: {str(e)}")
        else:
            logging.info("No face encodings file found. Starting fresh.")
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_ids = []

    def save_known_faces(self):
        """Saves current known face encodings and names to pickle file."""
        try:
            os.makedirs(os.path.dirname(self.encodings_path), exist_ok=True)
            data = {
                "encodings": self.known_face_encodings,
                "names": self.known_face_names,
                "ids": self.known_face_ids
            }
            with open(self.encodings_path, 'wb') as f:
                pickle.dump(data, f)
            logging.info(f"Saved {len(self.known_face_names)} face encodings to {self.encodings_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving face encodings: {str(e)}")
            return False

    def detect_faces(self, frame):
        """
        Detects faces in a frame and returns coordinates in (top, right, bottom, left) format.
        Args:
            frame: OpenCV BGR image
        Returns:
            List of tuples representing face boxes: [(top, right, bottom, left), ...]
        """
        if frame is None or frame.size == 0:
            return []

        if HAS_FACE_RECOGNITION:
            try:
                # face_recognition requires RGB images
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Using Hog model by default for speed on CPU
                boxes = face_recognition.face_locations(rgb_frame, model="hog")
                return boxes
            except Exception as e:
                logging.error(f"face_recognition detection failed, falling back: {str(e)}")
                # Fall back to Haar if it fails
                pass

        # OpenCV Fallback Detection (Haar Cascades)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detectMultiScale returns (x, y, w, h)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            boxes = []
            for (x, y, w, h) in faces:
                # Convert to (top, right, bottom, left)
                top = int(y)
                right = int(x + w)
                bottom = int(y + h)
                left = int(x)
                boxes.append((top, right, bottom, left))
            return boxes
        except Exception as e:
            logging.error(f"OpenCV face detection failed: {str(e)}")
            return []

    def compute_encoding(self, frame, box):
        """
        Generates 128D encoding vector for a face bounding box in a frame.
        Args:
            frame: OpenCV BGR image
            box: Coordinate tuple (top, right, bottom, left)
        Returns:
            numpy array of 128 float elements, or None
        """
        if frame is None or frame.size == 0 or not box:
            return None

        if HAS_FACE_RECOGNITION:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_frame, [box])
                if encodings:
                    return encodings[0]
            except Exception as e:
                logging.error(f"face_recognition encoding failed, falling back: {str(e)}")
                # Fall back to custom encoding if it fails
                pass

        # Custom Fallback Encoding
        try:
            top, right, bottom, left = box
            h_img, w_img = frame.shape[:2]
            
            # Bound check box coordinates
            top = max(0, int(top))
            left = max(0, int(left))
            bottom = min(h_img, int(bottom))
            right = min(w_img, int(right))
            
            if bottom <= top or right <= left:
                return None
                
            face = frame[top:bottom, left:right]
            if face.size == 0:
                return None
                
            # Resize cropped face to 64x64 for uniform representation
            face_resized = cv2.resize(face, (64, 64))
            
            # Part 1: Grayscale spatial features (64 elements)
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            gray_tiny = cv2.resize(gray, (8, 8))  # 8x8 = 64 dimensions
            gray_flat = gray_tiny.flatten().astype(np.float32)
            
            # Part 2: HSV Color Histogram features (64 elements: H=32, S=16, V=16)
            hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
            color_flat = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
            
            # Concatenate spatial and color flat representation (128 elements)
            vector = np.concatenate([gray_flat, color_flat])
            
            # Normalize to unit length (L2 norm)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
                
            return vector
        except Exception as e:
            logging.error(f"Fallback custom face encoding failed: {str(e)}")
            return None

    def add_employee_face(self, emp_id, emp_name, images_list):
        """
        Processes registered face images and computes an average encoding.
        Args:
            emp_id: String employee ID
            emp_name: String employee name
            images_list: List of OpenCV BGR frames
        Returns:
            The computed face encoding as a list, or None if failed
        """
        encodings = []
        for img in images_list:
            boxes = self.detect_faces(img)
            if boxes:
                # Take the largest face in the frame if multiple are detected
                largest_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[1]-b[3]))
                encoding = self.compute_encoding(img, largest_box)
                if encoding is not None:
                    encodings.append(encoding)
                    
        if not encodings:
            logging.warning(f"Could not extract any valid face encodings for employee {emp_name} ({emp_id})")
            return None
            
        # Average the encodings to get a robust representative encoding
        avg_encoding = np.mean(encodings, axis=0)
        # Re-normalize to unit length
        norm = np.linalg.norm(avg_encoding)
        if norm > 0:
            avg_encoding = avg_encoding / norm
            
        # Remove existing if updates
        self.remove_employee_face(emp_id)
        
        # Append to our local cache
        self.known_face_encodings.append(avg_encoding)
        self.known_face_names.append(emp_name)
        self.known_face_ids.append(emp_id)
        
        self.save_known_faces()
        logging.info(f"Registered face encoding for {emp_name} ({emp_id}) based on {len(encodings)} captured frames.")
        return avg_encoding.tolist()

    def remove_employee_face(self, emp_id):
        """Removes employee face encoding from local lists."""
        indices_to_remove = [i for i, x in enumerate(self.known_face_ids) if x == emp_id]
        if indices_to_remove:
            for index in sorted(indices_to_remove, reverse=True):
                self.known_face_encodings.pop(index)
                self.known_face_names.pop(index)
                self.known_face_ids.pop(index)
            self.save_known_faces()
            logging.info(f"Removed face encoding cache for Employee ID: {emp_id}")
            return True
        return False

    def identify_face(self, frame, box, tolerance=None):
        """
        Matches a detected face box against known employee faces.
        Args:
            frame: OpenCV BGR image
            box: Coordinate tuple (top, right, bottom, left)
            tolerance: Match threshold (defaults: 0.6 for face_recognition, 0.45 for fallback)
        Returns:
            Tuple of (EmployeeID, Name, match_distance) or (None, None, None)
        """
        if not self.known_face_encodings:
            return None, None, None

        encoding = self.compute_encoding(frame, box)
        if encoding is None:
            return None, None, None

        if tolerance is None:
            tolerance = 0.6 if HAS_FACE_RECOGNITION else 0.45

        # Compute distances to all known face encodings
        if HAS_FACE_RECOGNITION:
            # face_recognition.face_distance returns euclidean distance
            distances = face_recognition.face_distance(self.known_face_encodings, encoding)
        else:
            # Manual Euclidean distance for numpy arrays
            distances = np.array([np.linalg.norm(np.array(known) - encoding) for known in self.known_face_encodings])

        if len(distances) == 0:
            return None, None, None

        best_match_idx = np.argmin(distances)
        best_distance = distances[best_match_idx]

        if best_distance <= tolerance:
            emp_id = self.known_face_ids[best_match_idx]
            emp_name = self.known_face_names[best_match_idx]
            logging.info(f"Face matched: {emp_name} ({emp_id}), Distance: {best_distance:.4f}")
            return emp_id, emp_name, float(best_distance)
            
        return None, None, None
