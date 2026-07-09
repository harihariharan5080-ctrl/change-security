import os
import logging
from flask import Flask
from database import init_db
from camera import face_handler
from routes import register_routes

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "app.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Create Flask application
app = Flask(__name__)
app.secret_key = "crt_industries_chemical_lab_secure_key_1298"

# Register routes on app
register_routes(app)

if __name__ == '__main__':
    # Initialize database and synchronize face encodings in FaceHandler
    init_db(face_handler)
    # Bind to localhost port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
